"""Before a run: how long training will wait for the agent, and what the agent will cost.

`opal estimate --total-steps N --sec-per-step S` prints it before launch; a run prints the same at its first check,
measured from its own time per update. Both use the per-decision times and costs below, which come from our runs.

When `decisions` is not set, a blocking run schedules at most as many decisions as keep the expected waiting within
MAX_WAIT_FACTOR (4) times the run's own training time, at least one; a `decisions` the user sets is never changed, and a line says when its waiting is above that.

Per decision of claude-opus-5-5 at effort high, the agent's `seconds` and `cost_usd`, 567 decisions in all:
  the first decision, which reads the training code and writes the first program: 101 to 432 s over 20 runs (median
  251 s), $0.33 to $1.62 (median $1.08);
  each later decision, averaged over a run: 10 to 51 s over the 15 runs with at least 10 of them (median 27 s), $0.058
  to $0.377 (median $0.23).
Through a mailbox, add the broker's poll and the ssh transfer, a few seconds per decision. Other models and efforts were not measured.
"""

from __future__ import annotations

MODEL, EFFORT = "claude-opus-5-5", "high"  # what the numbers below were measured with
FIRST_SECONDS = (100.0, 250.0, 430.0)  # (low, typical, high) seconds of the first decision
LATER_SECONDS = (10.0, 27.0, 50.0)  # seconds of each later decision (a run's average)
FIRST_COST = (0.30, 1.08, 1.60)  # dollars
LATER_COST = (0.05, 0.23, 0.38)
MAX_WAIT_FACTOR = 4.0  # with decisions not set: expected waiting at most this many times the run's own training time


def waiting(n, blocking=True, sec_per_step=None, max_lag=50, which=1, first=None, later=None):
    """Seconds training waits for n decisions: which=0 low, 1 typical, 2 high. blocking=False: none (training waits
    only when the loss turns non-finite). "auto": the first decision, and the later ones whose answer would land more
    than max_lag updates late."""
    if n <= 0 or blocking is False:
        return 0.0
    f = FIRST_SECONDS[which] if first is None else first
    g = LATER_SECONDS[which] if later is None else later
    if blocking == "auto" and sec_per_step and g / sec_per_step <= max_lag:
        g = 0.0
    return f + (n - 1) * g


def cost(n, which=1):
    return 0.0 if n <= 0 else FIRST_COST[which] + (n - 1) * LATER_COST[which]


def capped_decisions(n, run_seconds, blocking=True, sec_per_step=None, max_lag=50, factor=MAX_WAIT_FACTOR,
                     first=None, later=None):  # fmt: skip
    """The most decisions, at most n and at least 1, whose typical waiting stays within factor x run_seconds."""
    if n <= 1:
        return n
    budget = factor * run_seconds
    k = n
    while k > 1 and waiting(k, blocking, sec_per_step, max_lag, 1, first, later) > budget:
        k -= 1
    return k


def _t(s):
    if s < 90:
        return f"{s:.0f} s"
    if s < 90 * 60:
        return f"{s / 60:.0f} min"
    return f"{s / 3600:.1f} h"


def _x(a, b):
    if b <= 0:
        return "?"
    r = a / b
    return f"{r:.2f}x" if r < 1 else f"{r:.1f}x" if r < 10 else f"{r:,.0f}x"


def _usd(x):
    return f"${x:.2f}" if x < 10 else f"${x:.0f}"


def lines(total_steps, sec_per_step, decisions, blocking=True, max_lag=50, model=MODEL, effort=EFFORT, mailbox=False,
          capped_from=None, explicit=True):  # fmt: skip
    """The estimate, a few lines. decisions: the scheduled count. capped_from: the count before the cap of
    MAX_WAIT_FACTOR (decisions not set). explicit: the user set decisions (then a waiting above the factor is said)."""
    run = total_steps * sec_per_step
    lo, typ, hi = (waiting(decisions, blocking, sec_per_step, max_lag, w) for w in (0, 1, 2))
    mode = (
        "training waits for each (blocking=True)"
        if blocking is True
        else "training goes on while the agent thinks (blocking=False)"
        if blocking is False
        else 'training waits at the first and whenever an answer would land late (blocking="auto")'
    )
    out = [f"training without the agent: {total_steps:,} updates x {sec_per_step:.3g} s = {_t(run)}"]
    if blocking is False:
        wait = "no waiting for the agent, except when the training loss turns non-finite"
    else:
        wait = (
            f"about {_t(typ)} of waiting for the agent ({_t(lo)} to {_t(hi)}), {_x(typ, run)} the run's own time "
            f"({_x(lo, run)} to {_x(hi, run)})"
            + (", plus a few seconds per decision through the mailbox" if mailbox else "")
        )
    out.append(f"{decisions} decision{'s' * (decisions != 1)}, {mode}: {wait}")
    out.append(f"agent cost: about {_usd(cost(decisions))} ({_usd(cost(decisions, 0))} to {_usd(cost(decisions, 2))})")
    over = blocking is not False and typ > MAX_WAIT_FACTOR * run
    if capped_from is not None and capped_from != decisions:
        out.append(
            f"decisions not set: {decisions} scheduled instead of {capped_from}, so that the expected waiting stays "
            f"within {MAX_WAIT_FACTOR:g}x the run's own time (set decisions= to choose)"
        )
    fewer = capped_decisions(decisions, run, blocking, sec_per_step, max_lag) if over else decisions
    # a run shorter than one typical answer ends before the answer arrives when training does not wait for it
    no_wait = (
        "blocking=False lets training go on while the agent thinks"
        if run >= FIRST_SECONDS[1]
        else "blocking=False would not help: training would end before the first answer arrives"
    )
    if over and (
        decisions <= 1 or not explicit or waiting(fewer, blocking, sec_per_step, max_lag) > MAX_WAIT_FACTOR * run
    ):
        n = decisions if not explicit and decisions > 1 else 1
        out.append(
            f"even {'one decision' if n <= 1 else f'{n} decisions'} wait{'s' * (n <= 1)} more "
            f"than {MAX_WAIT_FACTOR:g}x the run's own time; {no_wait}"
        )
    elif over:
        out.append(
            f"the expected waiting is more than {MAX_WAIT_FACTOR:g}x the run's own time: ask fewer times "
            f"(decisions={fewer} keeps it within {MAX_WAIT_FACTOR:g}x) or let training go on while the agent thinks "
            f"(blocking=False)"
        )
    if (model, effort) != (MODEL, EFFORT):
        out.append(
            f"these numbers are from {MODEL} at effort {EFFORT}; {model} at effort {effort} was not measured"
        )
    return out
