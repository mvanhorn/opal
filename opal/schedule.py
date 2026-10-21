"""When the agent is asked: wrap(decisions=..., schedule=...).

schedule (the decision updates; it has nothing to do with the learning-rate schedule):
  "front" (default)   a fifth of the decisions evenly over the first 5% of training (where our agents made most of
                      their code changes: SiT, 49 of 125 decisions before update 50K changed a file, 6 of 310 after),
                      the rest evenly over the remainder. The same as [(0.05, a fifth), (1.0, the rest)].
  "uniform"           every total // decisions updates, as in the v1 harness (646, 1292, ... on 64,626 updates with
                      100 decisions).
  [(0.05, 20), (1.0, 30)]
                      segments, each (end, count) with end a fraction of training: 20 decisions evenly over the first
                      5%, then 30 evenly over the rest. Within a segment the gaps are equal and the last decision falls
                      on the segment's end, except in a segment that ends the run, whose last decision falls one gap
                      before the end (an answer at the last update could not act on the run).
  [500, 1000, 5000]   these updates.
decisions is the count for "front" and "uniform" (50 by default); a list gives its own count (the sum of the segments'
counts, or the number of updates), and a decisions= that disagrees with it is an error.

Decisions are asked only at checks (every `every` updates), so each update is moved up to the next check, two that
land on the same check become one, and none falls in the last `every` updates (an answer could no longer act); the run
has as many scheduled decisions as are left. Besides the schedule, the agent's own analysis can ask for a decision
({"wake": reason}), and a non-finite loss does.
"""

from __future__ import annotations

DEFAULT_DECISIONS = 50
NAMED = ("front", "uniform")
EARLY_FRACTION, EARLY_SHARE = 0.05, 0.2  # "front": a fifth of the decisions in the first 5% of training

HELP = (
    'schedule must be "front", "uniform", a list of (end_fraction, count) segments such as [(0.05, 20), (1.0, 30)], '
    "or a list of update numbers such as [500, 1000, 5000]"
)


def kind(schedule) -> str:
    """'front', 'uniform', 'segments' or 'updates'; raises ValueError for anything else."""
    if isinstance(schedule, str):
        if schedule in NAMED:
            return schedule
        raise ValueError(f"opal: {HELP}, got {schedule!r}")
    if not isinstance(schedule, (list, tuple)):
        raise ValueError(f"opal: {HELP}, got {schedule!r}")
    if all(_is_int(u) for u in schedule):
        return "updates"
    if all(isinstance(s, (list, tuple)) and len(s) == 2 for s in schedule):
        return "segments"
    raise ValueError(f"opal: {HELP}, got {schedule!r}")


def _is_int(x):
    return isinstance(x, int) and not isinstance(x, bool)


def _segments(schedule):
    """Checked (end_fraction, count) pairs."""
    out, prev = [], 0.0
    for s in schedule:
        end, count = s
        if isinstance(end, bool) or not isinstance(end, (int, float)) or not prev < end <= 1:
            raise ValueError(
                f"opal: schedule segment {tuple(s)!r}: each end must be a fraction of training in (0, 1], larger than "
                "the previous segment's end"
            )
        if not _is_int(count) or count < 0:
            raise ValueError(f"opal: schedule segment {tuple(s)!r}: the count must be an int >= 0")
        out.append((float(end), count))
        prev = end
    return out


def count(schedule, decisions=None) -> int:
    """How many decisions the schedule asks for before snapping to checks. decisions=None means the default (50) for
    "front" and "uniform" and the list's own count otherwise; a decisions that disagrees with a list is an error."""
    k = kind(schedule)
    if decisions is not None and (not _is_int(decisions) or decisions < 0):
        raise ValueError(f"opal: decisions must be an int >= 0, got {decisions!r}")
    if k in NAMED:
        return DEFAULT_DECISIONS if decisions is None else decisions
    if k == "segments":
        given = sum(c for _, c in _segments(schedule))
        what = " + ".join(str(c) for _, c in schedule) if len(schedule) > 1 else str(given)
    else:
        given, what = len(schedule), f"{len(schedule)} updates"
    if decisions is not None and decisions != given:
        raise ValueError(
            f"opal: decisions={decisions} but schedule gives {given} ({what}); leave decisions out, or make them agree"
        )
    return given


def decision_updates(total: int, n: int, cadence=1, schedule="front") -> list:
    """Scheduled decision updates, snapped up to the check cadence (decisions can only happen at check steps) and
    without the last `cadence` updates, so the count is the number that can actually happen. n: the count for "front"
    and "uniform" (ignored for a list, which gives its own)."""
    return [u for seg in decision_segments(total, n, cadence, schedule) for u in seg]


def decision_segments(total: int, n: int, cadence=1, schedule="front") -> list:
    """decision_updates, as one list per segment: two for "front" (the first 5% and the rest), one for "uniform" and
    for a list of updates, one per (end, count) segment. A check claimed by an earlier segment is not repeated."""
    k = kind(schedule)
    if k == "updates":
        for u in schedule:
            if not 1 <= u <= total:
                raise ValueError(f"opal: schedule update {u} is outside the run (1 to total_steps={total})")
    if total <= 1:
        return [[]]
    if k == "uniform":
        return [_snap([max(1, total // n) * (i + 1) for i in range(n)], total, cadence) if n > 0 else []]
    if k == "updates":
        return [_snap(list(schedule), total, cadence)]
    if k == "front":
        if n <= 0:
            return [[]]
        n_early = max(1, round(n * EARLY_SHARE))
        segments = [(EARLY_FRACTION, n_early), (1.0, max(0, n - n_early))]
    else:
        segments = _segments(schedule)
    out, seen, start = [], set(), 0
    for end_fraction, c in segments:
        end = total if end_fraction == 1.0 else max(1, int(total * end_fraction))
        span, parts = end - start, c + 1 if end == total else c
        raw = [start + round(span * (i + 1) / parts) for i in range(c)] if c else []
        seg = [u for u in _snap(raw, total, cadence) if u not in seen]
        seen.update(seg)
        out.append(seg)
        start = end
    return out


def _snap(updates, total, cadence):
    snapped = {-(-u // cadence) * cadence for u in updates}
    return sorted(u for u in snapped if 0 < u and total - u >= cadence)
