"""blocking="auto": training waits at a decision only when the answer would land more than max_lag_updates late (and
at the first decision); the report counts scheduled, asked, answered and applied decisions in every mode."""

import json
import time

import pytest
import torch

import opal
from opal.report import decision_counts, write_report
from test_core import MOMENTUM_UPDATER, adamw, events, flat, make
from test_review2 import plain_run


def run(tmp_path, agent, steps, step_sleep=0.0, **kw):
    m, data = make()
    o, s = adamw(m, steps)
    kw = dict(dict(decisions=4, every=10, blocking="auto", verbose=False), **kw)
    w = opal.wrap(o, m, scheduler=s, total_steps=steps, goal="t", run_dir=tmp_path / "r", backend=agent, **kw)
    for i in range(steps):
        x, y = data[i % len(data)]
        loss = torch.nn.functional.mse_loss(m(x), y)
        w.zero_grad()
        loss.backward()
        w.step(loss=loss)
        s.step()
        if step_sleep:
            time.sleep(step_sleep)
    ev = events(tmp_path / "r")
    return m, w, [e for e in ev if e["kind"] == "requested"], [e for e in ev if e["kind"] in ("applied", "kept")], ev


def thinking(seconds, first=None):
    def agent(ws, msg, sys_):
        time.sleep(seconds)
        if first and "Decision 1." in msg:
            (ws / "updater.py").write_text(first)
        return "x"

    return agent


def test_auto_waits_when_steps_are_fast_relative_to_the_answer(tmp_path):
    # about 1 ms per update and 0.3 s per answer: an answer would land about 300 updates late, above 50
    m, w, req, out, ev = run(tmp_path, thinking(0.3), 300, decisions=6)
    assert [r["wait"] for r in req] == ["waited"] * 6 and len(req) == decision_counts(ev)["scheduled"] == 6
    assert req[0]["wait_reason"] == "no answer time measured yet"
    assert all("> 50" in r["wait_reason"] and r["predicted_lag_updates"] > 50 for r in req[1:])
    assert all(o["lag_updates"] == 0 for o in out)


def test_auto_does_not_wait_when_steps_are_slow(tmp_path):
    # 30 ms per update and 0.3 s per answer: about 10 updates late, within 50
    m, w, req, out, ev = run(tmp_path, thinking(0.3), 200, step_sleep=0.03, decisions=3)
    assert [r["wait"] for r in req] == ["waited", "did not wait", "did not wait"]
    assert all("<= 50" in r["wait_reason"] and 0 < r["predicted_lag_updates"] <= 50 for r in req[1:])
    lags = {o["k"]: o["lag_updates"] for o in out}
    assert lags[1] == 0 and all(0 < lags[k] <= 50 for k in (2, 3))


def test_auto_waits_at_the_first_decision(tmp_path):
    m, w, req, out, ev = run(tmp_path, thinking(0.05, first=MOMENTUM_UPDATER), 100, step_sleep=0.03, decisions=1)
    applied = [o for o in out if o["kind"] == "applied"]
    assert req[0]["wait"] == "waited" and applied[0]["applied_at"] == applied[0]["requested_at"] == req[0]["update"]
    assert w.version == 1
    assert "Training is paused until you finish" in (tmp_path / "r" / "decisions" / "001" / "message.txt").read_text()


def test_summary_counts_scheduled_asked_answered_applied(tmp_path):
    calls = []

    def agent(ws, msg, sys_):
        calls.append(msg)
        if len(calls) == 1:
            (ws / "updater.py").write_text(MOMENTUM_UPDATER)  # applied
        if len(calls) == 2:
            raise RuntimeError("backend down")  # asked, not answered
        if len(calls) == 3:
            (ws / "analysis.py").write_text("def analyze(:\n")  # answered, rejected
        return "x"

    m, w, req, out, ev = run(tmp_path / "a", agent, 300, blocking=True)
    c = decision_counts(ev)
    assert (c["scheduled"], c["asked"], c["answered"], c["applied"]) == (4, 4, 3, 1)
    page = write_report(tmp_path / "a" / "r").read_text()
    assert "Decisions: 4 scheduled, 4 asked, 3 answered, 1 applied (changed the programs)" in page
    # the default, non-blocking: a slow agent on fast steps is asked about few of the scheduled decisions
    m, w, req, out, ev = run(tmp_path / "b", thinking(0.2), 300, step_sleep=0.002, blocking=False, decisions=10)
    c = decision_counts(ev)
    assert c["scheduled"] == 10 and c["asked"] == len(req) < 10 and c["answered"] <= c["asked"]
    assert c["waited"] == c["did_not_wait"] == 0 and all("wait" not in r for r in req)
    assert f"Decisions: 10 scheduled, {c['asked']} asked" in write_report(tmp_path / "b" / "r").read_text()


def test_auto_with_keep_is_bitwise_equal_to_plain_training(tmp_path):
    steps = 200
    m, w, req, out, ev = run(tmp_path, "keep", steps)
    assert torch.equal(flat(m), plain_run(steps)) and req and req[0]["wait"] == "waited"
    assert json.loads((tmp_path / "r" / "events.jsonl").read_text().splitlines()[0])["blocking"] == "auto"
    m2, _ = make()
    with pytest.raises(ValueError, match="blocking must be"):
        opal.wrap(adamw(m2, 10)[0], m2, total_steps=10, goal="t", blocking="sometimes", backend=None)


def test_retry_documentation_distinguishes_unavailable_agents(capsys):
    from pathlib import Path
    from opal import cli

    with pytest.raises(SystemExit) as stopped:
        cli.main(["broker", "--help"])
    assert stopped.value.code == 0
    help_text = " ".join(capsys.readouterr().out.split())
    retries = help_text.split("--retries RETRIES", 2)[-1].split("--unavailable-timeout", 1)[0]
    assert "usage or spend limit" not in retries
    root = Path(__file__).resolve().parents[1]
    readme, guide = ((root / p).read_text() for p in ("README.md", "docs/GUIDE.md"))
    assert "unavailable_timeout" in next(line for line in readme.splitlines() if line.startswith("| `backend_options`"))
    assert "the run stops waiting for that decision" not in guide


@pytest.mark.parametrize("every", [None, 2.0])
def test_unavailable_wait_backoff_remains_bounded_after_many_polls(every):
    from opal.backends import unavailable

    wait = unavailable.Wait(lambda _: None, every=every)
    for i in range(2000):
        delay = wait.note("broker", "no broker answering")
        assert 0 < delay <= unavailable.RECHECK_MAX
        if i < 10:
            assert delay == min(unavailable.RECHECK * 2 ** i, unavailable.RECHECK_MAX)
