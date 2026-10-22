"""The estimate of the waiting and the cost (opal estimate, and at a run's first check), and the default number of
decisions: with decisions not set, a blocking run keeps the expected waiting within 4x its own training time."""

import json

import pytest

import opal
from opal import cli, estimate
from opal.backends.base import Backend
from opal.schedule import decision_updates
from test_claude_backend import claude_run, fake_cli
from test_core import adamw, events, make, train


def test_the_cap_keeps_the_waiting_within_four_times_the_run():
    # a run of 66 s whose decisions take 90 s each: 4 x 66 = 264 s leaves room for 2 of them
    assert estimate.capped_decisions(50, 66, first=90, later=90) == 2
    assert estimate.capped_decisions(50, 5 * 3600) == 50  # a 5-hour run keeps its 50
    assert estimate.capped_decisions(50, 0.1) == 1  # never below one
    assert estimate.capped_decisions(50, 66, blocking=False) == 50  # training does not wait: nothing to cap
    assert estimate.capped_decisions(0, 66) == 0
    k = estimate.capped_decisions(50, 300)  # a 5-minute run
    assert 1 < k < 50 and estimate.waiting(k) <= 4 * 300 < estimate.waiting(k + 1)


def test_opal_estimate_prints_the_multiple_and_the_cap(capsys):
    assert cli.main(["estimate", "--total-steps", "1950", "--sec-per-step", "0.034"]) == 0
    out = capsys.readouterr().out
    assert "training without the agent: 1,950 updates x 0.034 s = 66 s" in out
    assert "1 decision, training waits for each (blocking=True): about 4 min of waiting" in out
    assert "x the run's own time" in out and "decisions not set: 1 scheduled instead of 38" in out
    assert "These are estimates from our runs" in out
    cli.main(["estimate", "--total-steps", "1950", "--sec-per-step", "0.034", "--decisions", "50"])
    out = capsys.readouterr().out
    assert "38 decisions" in out and "more than 4x the run's own time: ask fewer times (decisions=1" in out
    cli.main(["estimate", "--total-steps", "72000", "--sec-per-step", "0.25"])
    out = capsys.readouterr().out
    assert "50 decisions" in out and "more than 4x" not in out and "not set" not in out
    cli.main(["estimate", "--total-steps", "1950", "--sec-per-step", "0.034", "--no-blocking"])
    assert "no waiting for the agent" in capsys.readouterr().out


def test_a_run_with_decisions_not_set_is_capped_at_its_first_check(tmp_path, capsys):
    exe = fake_cli(tmp_path)
    m, data, s, w = claude_run(tmp_path, exe, steps=200, decisions=None, verbose=True)
    before = list(w.schedule)
    assert before == decision_updates(200, 50, 10)
    train(m, w, data, 200, sched=s)
    w.close()
    assert len(w.schedule) == 1 and w.max_decisions == 2  # this run trains in well under a second
    ev = [e for e in events(tmp_path / "r") if e["kind"] == "estimate"]
    assert len(ev) == 1 and ev[0]["capped_from"] == len(before) and ev[0]["schedule"] == w.schedule
    assert [e["update"] for e in events(tmp_path / "r") if e["kind"] == "requested"] == w.schedule
    system = (tmp_path / "r" / "system.md").read_text()
    assert f"You are called once, at update {w.schedule[0]:,}." in system
    err = capsys.readouterr().err
    assert "[opal] estimate from this run's updates 2 to 9 (" in err
    assert f"decisions not set: 1 scheduled instead of {len(before)}" in err and "even one decision waits" in err


def test_decisions_set_are_kept_and_warned_about(tmp_path, capsys):
    exe = fake_cli(tmp_path)
    m, data, s, w = claude_run(tmp_path, exe, steps=200, decisions=4, verbose=True)
    before = list(w.schedule)
    train(m, w, data, 200, sched=s)
    w.close()
    assert w.schedule == before and len(before) == 4
    err = capsys.readouterr().err
    # this run trains in well under a second: even one decision waits more than 4x, so no smaller count is offered
    assert "even one decision waits more than 4x the run's own time" in err and "keeps it within" not in err
    assert not [e for e in events(tmp_path / "r") if e["kind"] == "estimate" and "capped_from" in e]


def test_explicit_decisions_offer_a_count_only_when_it_fits():
    from opal import estimate

    fits = estimate.lines(10_000, 0.05, 80)  # 8 min of training: 65 decisions keep the waiting within 4x
    assert any("ask fewer times (decisions=65 keeps it within 4x)" in line for line in fits)
    short = estimate.lines(1_000, 0.00055, 2)  # 1 s of training: no count fits
    assert any("even one decision waits more than 4x" in line for line in short)
    assert not any("keeps it within" in line for line in short)
    assert any("training would end before the first answer arrives" in line for line in short)


def test_a_resumed_run_keeps_the_capped_schedule(tmp_path):
    exe = fake_cli(tmp_path)
    m, data, s, w = claude_run(tmp_path, exe, steps=200, decisions=None)
    train(m, w, data, 60, sched=s)
    state = w.state_dict()
    assert len(w.schedule) == 1 and state["estimated"] and state["decision_schedule"]
    w.close(interrupted=True)
    m2, data2, s2, w2 = claude_run(tmp_path, exe, steps=200, decisions=None, run="r2")
    w2.load_state_dict(state)
    assert w2.schedule == w.schedule and not w2._estimate_due
    assert f"You are called once, at update {w.schedule[0]:,}." in w2.worker.system
    w2.close()


def test_no_estimate_and_no_cap_without_an_agent(tmp_path, capsys):
    m, data = make()
    o, s = adamw(m, 200)
    w = opal.wrap(o, m, scheduler=s, total_steps=200, goal="g", backend="keep", every=10, run_dir=tmp_path / "r")
    before = list(w.schedule)
    train(m, w, data, 200, sched=s)
    assert w.schedule == before == decision_updates(200, 50, 10)
    assert "estimate from this run" not in capsys.readouterr().err


def test_capping_a_uniform_schedule_keeps_at_least_one_decision(tmp_path):
    exe = fake_cli(tmp_path)
    m, data, s, w = claude_run(tmp_path, exe, steps=200, decisions=None, schedule="uniform")
    assert w.schedule
    train(m, w, data, 200, sched=s)
    requested = [e for e in events(tmp_path / "r") if e["kind"] == "requested"]
    assert len(w.schedule) == len(requested) == 1


@pytest.mark.parametrize("every, sec_per_step, asked", [(10, 1.0, 4), (1, 1.0, 4), (1, 0.01, 3)])
def test_the_cap_counts_early_decisions_and_excludes_their_waiting(tmp_path, monkeypatch, every, sec_per_step, asked):
    import time
    from opal.agent import AgentWorker

    clock = [0.0]
    original_wait = AgentWorker.wait

    def timed_wait(worker):
        clock[0] += 90.0
        original_wait(worker)

    monkeypatch.setattr(time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(AgentWorker, "wait", timed_wait)
    monkeypatch.setattr(estimate, "FIRST_SECONDS", (90.0, 90.0, 90.0))
    monkeypatch.setattr(estimate, "LATER_SECONDS", (90.0, 90.0, 90.0))

    class TimedBackend(Backend):
        name = "claude"  # requests the first timing estimate, without a real CLI or model call

        def decide(self, workspace, message, system, record_dir):
            return dict(text="keep", cost_usd=0.0, returncode=0, error=None, seconds=90.0)

    m, data = make()
    o, s = adamw(m, 100)
    w = opal.wrap(o, m, scheduler=s, total_steps=100, goal="g", every=every, backend=TimedBackend(),
                  summarize=False, verbose=False, run_dir=tmp_path / "r")
    before = list(w.schedule)
    for i in range(100):
        clock[0] += sec_per_step
        train(m, w, data, i + 1, sched=s, start=i)
    state = w.state_dict()
    w.close()
    ev = events(tmp_path / "r")
    measured = next(e for e in ev if e["kind"] == "estimate")
    assert measured["sec_per_update"] == sec_per_step
    assert measured["capped_from"] == len(before)
    requested = [e["update"] for e in ev if e["kind"] == "requested"]
    assert len(requested) == asked and requested == w.schedule
    assert w.max_decisions == 2 * len(w.schedule)
    assert state["decision_schedule"] == w._schedule_segments
    if every == 10:
        assert w.schedule == decision_updates(100, 4, 10)  # normal first-check grid is unchanged
    else:
        assert requested[:3] == before[:3] == [1, 2, 3]
        assert all(u >= measured["update"] for u in w.schedule[3:])
    if sec_per_step == 1.0:
        assert estimate.waiting(asked) <= 4 * 100 * sec_per_step < estimate.waiting(asked + 1)
