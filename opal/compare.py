"""Set two OPAL run folders, or two result.json files, beside each other.

`opal compare a b` prints the metric and command when those files exist, the decision counts, cost and waiting,
and one line for each decision whose outcome differs. A run folder is one with `events.jsonl`. A result.json
written by an example script (`--out`) is enough for the metric and command alone.
"""

from __future__ import annotations

import json
from pathlib import Path

from .report import (
    OUTCOMES,
    _finished,
    _index,
    _jsonl,
    _list,
    _number,
    _object,
    _response,
    _text,
    decision_counts,
    describe_counts,
)


def load_side(path):
    """A run folder (events.jsonl) and/or a result.json. Raises FileNotFoundError or ValueError with an opal: prefix."""
    p = Path(path).expanduser()
    if p.is_file():
        return _from_result_file(p)
    if not p.is_dir():
        raise FileNotFoundError(f"opal: not a run folder or result.json: {p}")
    if (p / "events.jsonl").exists():
        result = _object(p / "result.json") if (p / "result.json").exists() else {}
        return snapshot(p, result=result if isinstance(result, dict) else {})
    if (p / "result.json").exists():
        return _from_result_file(p / "result.json")
    inside = sorted(d.name for d in p.iterdir() if (d / "events.jsonl").exists())
    hint = f"; runs inside: {', '.join(inside[-5:])}" if inside else ""
    raise FileNotFoundError(f"opal: not a run folder: {p} (no events.jsonl){hint}")


def _from_result_file(path: Path):
    try:
        obj = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise ValueError(f"opal: not a result.json: {path} ({exc})") from exc
    if not isinstance(obj, dict) or "metric" not in obj:
        raise ValueError(f"opal: not a result.json: {path} (missing metric)")
    run = None
    listed = obj.get("opal_run")
    if isinstance(listed, str) and listed and (Path(listed) / "events.jsonl").exists():
        run = Path(listed)
    elif (path.parent / "events.jsonl").exists():
        run = path.parent
    if run is not None:
        return snapshot(run, result=obj)
    return _result_only(path, obj)


def snapshot(run, result=None) -> dict:
    """JSON-serialisable summary of one run folder. `result` is the example script's result.json when present."""
    run = Path(run)
    events = _jsonl(run / "events.jsonl")
    start = next((e for e in events if e.get("kind") == "start"), {})
    end = next((e for e in reversed(events) if e.get("kind") in ("end", "interrupted")), {})
    result = result if isinstance(result, dict) else {}
    counts = decision_counts(events) if events else {}
    wait = result.get("agent_wait_seconds")
    if not _number(wait):
        wait = end.get("agent_wait_seconds")
        if not _number(wait):
            wait = next(
                (e.get("agent_wait_seconds") for e in reversed(events) if _number(e.get("agent_wait_seconds"))),
                None,
            )
    cost = result.get("agent_cost_usd")
    if not _number(cost):
        cost = end.get("cost_usd") if _number(end.get("cost_usd")) else None
    return dict(
        path=str(run),
        kind="run",
        goal=_text(start.get("goal")) or None,
        backend=start.get("backend"),
        blocking=start.get("blocking"),
        total_updates=start.get("total_updates"),
        update=end.get("update"),
        finished=_finished(events),
        counts=counts,
        cost_usd=cost,
        agent_wait_seconds=wait if _number(wait) else None,
        last_loss=_last_loss(run),
        metric=result.get("metric") if isinstance(result.get("metric"), dict) else None,
        command=result.get("command"),
        hardware=result.get("hardware"),
        opal_commit=result.get("opal_commit"),
        decisions=_decisions(run, events),
    )


def _result_only(path: Path, obj: dict) -> dict:
    metric = obj.get("metric") if isinstance(obj.get("metric"), dict) else None
    dec = obj.get("decisions") if isinstance(obj.get("decisions"), dict) else None
    counts = {}
    if dec:
        counts = dict(
            scheduled=None,
            asked=dec.get("asked"),
            answered=None,
            applied=dec.get("applied"),
            failed=dec.get("failed"),
        )
    return dict(
        path=str(path),
        kind="result",
        goal=None,
        backend=None,
        blocking=None,
        total_updates=None,
        update=None,
        finished=True,
        counts=counts,
        cost_usd=obj.get("agent_cost_usd"),
        agent_wait_seconds=obj.get("agent_wait_seconds"),
        last_loss=None,
        metric=metric,
        command=obj.get("command"),
        hardware=obj.get("hardware"),
        opal_commit=obj.get("opal_commit"),
        decisions=[],
    )


def _last_loss(run: Path):
    last = None
    for r in _jsonl(run / "losses.jsonl"):
        for v in _list(r.get("losses")):
            if _number(v):
                last = v
    return last


def _decisions(run: Path, events) -> list:
    req = {e["k"]: e for e in events if e.get("kind") == "requested" and _index(e.get("k"))}
    out = {
        e["k"]: e
        for e in events
        if e.get("kind") in (*OUTCOMES, "cancelled") and _index(e.get("k"))
    }
    rows = []
    for k in sorted(set(req) | set(out)):
        o = out.get(k, {})
        r = req.get(k, {})
        resp = _response(run, k)
        rows.append(
            dict(
                k=k,
                asked_at=r.get("update", o.get("requested_at")),
                outcome=_outcome(events, k, o),
                applied_at=o.get("applied_at"),
                seconds=resp.get("seconds"),
                cost_usd=resp.get("cost_usd"),
                changed=_list(o.get("changed")),
            )
        )
    return rows


def _outcome(events, k, o) -> str:
    kind = o.get("kind", "pending")
    req = next((e for e in events if e.get("kind") == "requested" and e.get("k") == k), {})
    if kind == "pending" and any(
        e.get("kind") in ("resumed", "end") and e.get("utc", "") > req.get("utc", "") for e in events
    ):
        kind = "lost"
    if kind == "applied":
        kind = "changed"
    elif o.get("failed") and not o.get("gave_up"):
        kind = "failed"
    return kind


def compare(a, b) -> dict:
    """Two snapshots and the decisions whose outcomes differ (empty when either side has no decision list)."""
    left, right = load_side(a), load_side(b)
    differ = []
    if left["kind"] == "run" and right["kind"] == "run":
        by_b = {d["k"]: d for d in right["decisions"]}
        by_a = {d["k"]: d for d in left["decisions"]}
        for k in sorted(set(by_a) | set(by_b)):
            da, db = by_a.get(k, {}), by_b.get(k, {})
            oa, ob = da.get("outcome", "absent"), db.get("outcome", "absent")
            if oa != ob:
                differ.append(dict(k=k, a=oa, b=ob, asked_at=da.get("asked_at", db.get("asked_at"))))
    return dict(a=left, b=right, differing_decisions=differ)


def _cell(value, kind="plain"):
    if value is None or value == "":
        return "—"
    if kind == "cost":
        return f"${value:.2f}" if _number(value) else "—"
    if kind == "wait":
        return f"{value:g} s" if _number(value) else "—"
    if kind == "loss":
        return f"{value:.5g}" if _number(value) else "—"
    if kind == "updates":
        return f"{value:,}" if _number(value) else "—"
    if kind == "mode":
        if value == "auto":
            return "auto-blocking"
        if value is True or value == "blocking":
            return "blocking"
        if value is False:
            return "non-blocking"
        return str(value)
    if kind == "metric":
        if not isinstance(value, dict) or not value.get("name"):
            return "—"
        direction = value.get("direction") or ""
        shown = f"{value['name']} {value.get('value')}"
        if direction in ("lower", "higher"):
            shown += f" ({direction} is better)"
        return shown
    return str(value)


def _counts_cell(side):
    c = side.get("counts") or {}
    if side["kind"] == "result":
        parts = []
        if _number(c.get("asked")):
            parts.append(f"{c['asked']} asked")
        if _number(c.get("applied")):
            parts.append(f"{c['applied']} applied")
        if _number(c.get("failed")):
            parts.append(f"{c['failed']} failed")
        return ", ".join(parts) or "—"
    if not c:
        return "—"
    return (
        f"{c.get('scheduled', 0)} scheduled, {c.get('asked', 0)} asked, {c.get('applied', 0)} applied"
    )


def lines(cmp) -> list:
    """Human lines for `opal compare`, in the same register as `opal show`."""
    a, b = cmp["a"], cmp["b"]
    out = [f"a: {a['path']}", f"b: {b['path']}"]
    if a.get("goal") or b.get("goal"):
        out.append(f"goal a: {a.get('goal') or '—'}")
        out.append(f"goal b: {b.get('goal') or '—'}")
    rows = [
        ("updates", _cell(a.get("total_updates"), "updates"), _cell(b.get("total_updates"), "updates")),
        ("backend", _cell(a.get("backend")), _cell(b.get("backend"))),
        ("mode", _cell(a.get("blocking"), "mode"), _cell(b.get("blocking"), "mode")),
        ("decisions", _counts_cell(a), _counts_cell(b)),
        ("cost", _cell(a.get("cost_usd"), "cost"), _cell(b.get("cost_usd"), "cost")),
        ("wait", _cell(a.get("agent_wait_seconds"), "wait"), _cell(b.get("agent_wait_seconds"), "wait")),
        ("last loss", _cell(a.get("last_loss"), "loss"), _cell(b.get("last_loss"), "loss")),
        ("metric", _cell(a.get("metric"), "metric"), _cell(b.get("metric"), "metric")),
    ]
    if a.get("command") or b.get("command"):
        rows.append(("command", _cell(a.get("command")), _cell(b.get("command"))))
    width = max(len(name) for name, _, _ in rows)
    col = max(24, max(len(x) for row in rows for x in row[1:]))
    out.append(f"  {'':{width}}  {'a':<{col}}  b")
    for name, left, right in rows:
        mark = "*" if left != right else " "
        out.append(f"{mark} {name:{width}}  {left:<{col}}  {right}")
    for side, label in ((a, "a"), (b, "b")):
        c = side.get("counts") or {}
        if side["kind"] == "run" and c:
            out.append(f"{label}: {describe_counts(c)}")
    if a["kind"] != "run" or b["kind"] != "run":
        missing = "a" if a["kind"] != "run" else "b"
        out.append(
            f"decision-by-decision comparison needs two run folders (events.jsonl); {missing} is a result.json only."
        )
        return out
    differ = cmp["differing_decisions"]
    n = len(set(d["k"] for d in a["decisions"]) | set(d["k"] for d in b["decisions"]))
    if not n:
        out.append("no decisions were asked in either run.")
        return out
    if not differ:
        out.append(f"all {n} asked decisions have the same outcome.")
        return out
    out.append("decisions that differ:")
    out.append(f"{'k':>4}  {'asked':>9}  {'a':<11}  b")
    for row in differ:
        asked = f"{row['asked_at']:,}" if _number(row.get("asked_at")) else ""
        out.append(f"{row['k']:>4}  {asked:>9}  {row['a']:<11}  {row['b']}")
    same = n - len(differ)
    if same:
        out.append(f"{same} other decision{'s' if same != 1 else ''} match.")
    return out
