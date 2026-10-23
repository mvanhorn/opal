"""Print a run's events as they are written.

`opal watch <run>` prints one line per event, starting from the beginning of events.jsonl, then waits for more.
It returns when the run has finished (`end` or `interrupted`, and no later `resumed`). `--timeout 0` prints what is
already there and returns; a positive `--timeout` is idle seconds after the last new event.
"""

from __future__ import annotations

import time
from pathlib import Path

from .report import OUTCOMES, _finished, _jsonl, _number, _response, _text


def _mode(start) -> str:
    mode = start.get("blocking")
    if mode == "auto":
        return "auto-blocking"
    if mode:
        return "blocking"
    return "non-blocking"


def _cost(value):
    return f"${value:.2f}" if _number(value) else ""


def _secs(value):
    if not _number(value):
        return ""
    return f"{value:.0f} s" if float(value) >= 1 else f"{value:g} s"


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


def event_line(run, event, events=None) -> str:
    """One line for one events.jsonl record, in the same register as `opal show`."""
    kind = event.get("kind") or "?"
    update = event.get("update", event.get("applied_at"))
    at = f"{update:,}" if _number(update) else ""
    if kind == "start":
        total = event.get("total_updates", "?")
        total = f"{total:,}" if _number(total) else total
        backend = event.get("backend") or "none"
        return f"start: {total} updates; backend {backend}; {_mode(event)}"
    if kind == "requested":
        return f"requested  k={event.get('k')}  at update {at}"
    if kind in (*OUTCOMES, "cancelled"):
        events = events or []
        label = _outcome(events, event.get("k"), event)
        resp = _response(run, event["k"]) if _number(event.get("k")) else {}
        said = " ".join(_text(resp.get("text")).split())[:70]
        cost, secs = _cost(resp.get("cost_usd")), _secs(resp.get("seconds"))
        extra = "  ".join(x for x in (secs, cost, said) if x)
        applied = f"{event.get('applied_at'):,}" if _number(event.get("applied_at")) else at
        return f"{label:<10} k={event.get('k')}  at update {applied}" + (f"  {extra}" if extra else "")
    if kind == "estimate":
        n = event.get("decisions", "?")
        extra = f"; capped from {event['capped_from']}" if _number(event.get("capped_from")) else ""
        return f"estimate at update {at}: {n} scheduled{extra}"
    if kind == "end":
        n = event.get("decisions")
        n = f"{n} decisions; " if _number(n) else ""
        cost = _cost(event.get("cost_usd")) or "cost not recorded"
        return f"finished at update {at}; {n}{cost}"
    if kind == "interrupted":
        return f"interrupted at update {at}"
    if at:
        return f"{kind} at update {at}"
    return kind


def follow(run, *, interval=1.0, timeout=None, sleep=time.sleep, now=time.monotonic, write=print):
    """Print new events until the run finishes, or until `timeout` idle seconds (0: do not wait). Returns 0."""
    run = Path(run)
    write(f"watching {run}")
    seen = 0
    last_new = now()
    events = []
    while True:
        events = _jsonl(run / "events.jsonl")
        if len(events) < seen:
            seen = 0
        fresh = events[seen:]
        seen = len(events)
        if fresh:
            last_new = now()
        for e in fresh:
            if not isinstance(e, dict) or not isinstance(e.get("kind"), str):
                continue
            line = event_line(run, e, events)
            write(line)
            if e.get("kind") == "start" and e.get("goal"):
                write(f"goal: {e['goal']}")
        if _finished(events):
            return 0
        if timeout is not None and (timeout <= 0 or now() - last_new >= timeout):
            last = next((e.get("update") for e in reversed(events) if _number(e.get("update"))), None)
            at = f" at update {last:,}" if last is not None else ""
            write(f"still running{at}")
            return 0
        sleep(interval)
