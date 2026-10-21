"""Regressions from the review of 2026-10-18, what the agent is told: a decision asked under the non-finite pause says
training waits (M2, prompts#10) and keeps its reason (B9); the wall time per update leaves out the pauses (M3, F7);
the decision budget (prompts#7, F17); the starting program is timed (prompts#9, F10); report counts past 400
(prompts#16, F12); the non-finite pause sentence (prompts#11); clipping in the starting summary (prompts#4); no lambda
source without share_code (prompts#13); no claim of a copied script (prompts#14); no shell without the sandbox
(prompts#15); a failed agent call is said and counted as failed (B4); the budget a backend may cap a call with (D1)."""

import re
from pathlib import Path
import time

import torch

import opal
from opal.backends.base import Backend
from opal.report import decision_counts, describe_counts
from test_core import adamw, events, make, train


def run(tmp_path, agent, steps=200, nan_at=None, sched=True, **kw):
    m, data = make()
    o, s = adamw(m, steps)
    kw = dict(dict(decisions=4, every=10, verbose=False), **kw)
    w = opal.wrap(o, m, scheduler=s if sched else None, total_steps=steps, goal="t", run_dir=tmp_path / "r",
                     backend=agent, **kw)  # fmt: skip
    train(m, w, data, steps, sched=s if sched else None, nan_at=nan_at)
    return w


def recorder(edit=None):
    msgs = []

    def agent(ws, msg, sys_):
        msgs.append(msg)
        if edit:
            edit(ws, msg, len(msgs))
        return "x"

    agent.msgs = msgs
    return agent


def test_m2_a_decision_asked_under_the_nonfinite_pause_says_training_waits(tmp_path):
    for blocking in (False, "auto"):

        def edit(ws, msg, k):
            if "non-finite" in msg:
                (ws / "analysis.py").write_text((ws / "analysis.py").read_text() + "\n# looked at the NaN\n")

        agent = recorder(edit)
        run(tmp_path / str(blocking), agent, nan_at=54, decisions=1, blocking=blocking)
        nan_msg = next(x for x in agent.msgs if "non-finite" in x)
        assert "Training is paused until you finish; your edits take effect from update 60." in nan_msg
        ev = [e for e in events(tmp_path / str(blocking) / "r") if e["kind"] == "applied"]
        assert ev[-1]["requested_at"] == ev[-1]["applied_at"] == 60


def test_b9_the_nonfinite_reason_survives_many_wake_ups():
    from opal.wrapper import WAKE, _reasons_shown

    reasons = [f"{WAKE}look at update {u}" for u in range(40, 130, 10)] + ["scheduled decision"]
    reasons += [f"{WAKE}look at update {u}" for u in range(130, 160, 10)]
    reasons += ["non-finite training loss before update 160 (0 of 4 parameter tensors are non-finite now)"]
    shown = _reasons_shown(reasons)
    assert len(shown) == 11 and shown[:2] == [reasons[9], reasons[-1]]  # kept first, whatever came before
    assert shown[2:10] == [r for r in reasons if r.startswith(WAKE)][-8:]  # then the most recent wake-ups
    assert shown[-1] == "4 earlier requests from your analysis, not listed"
    assert _reasons_shown(reasons[:5]) == reasons[:5]


def test_m3_the_wall_time_per_update_leaves_out_the_pauses(tmp_path):
    def slow(ws, msg, sys_):
        time.sleep(1.0)
        return "x"

    agent_msgs = []

    def agent(ws, msg, sys_):
        agent_msgs.append(msg)
        return slow(ws, msg, sys_)

    run(tmp_path, agent, steps=120, decisions=10, schedule="uniform", every=10, blocking=True)
    shown = [float(m) for x in agent_msgs for m in re.findall(r"Timing: ([0-9.]+) ms of wall time", x)]
    assert shown and max(shown) < 50  # a 1 s pause over 10 updates used to show as ~100 ms per update


def test_prompts7_f17_the_budget_counts_what_can_still_come(tmp_path):
    agent = recorder()
    w = run(tmp_path, agent, steps=500, decisions=10, schedule="uniform", max_decisions=3)
    assert len(agent.msgs) == 3
    assert "This is your last decision in this run." in agent.msgs[-1]
    assert "scheduled decisions left: 0" in agent.msgs[-1] and "scheduled decisions left: 1" in agent.msgs[-2]
    # an analysis that wakes at every check leaves room for the scheduled decisions
    waking = 'EVERY = 10\ndef analyze(obs, memory):\n    return {"wake": "again"}\n'

    def edit(ws, msg, k):
        if k == 1:
            (ws / "analysis.py").write_text(waking)

    agent = recorder(edit)
    w = run(tmp_path / "w", agent, steps=400, decisions=3, schedule="uniform")
    asked = [e["update"] for e in events(tmp_path / "w" / "r") if e["kind"] == "requested"]
    assert len(w.schedule) == 2 and len(asked) == w.max_decisions == 4 and set(w.schedule) <= set(asked)


def test_prompts9_the_starting_program_is_timed_before_a_first_edit(tmp_path):
    def edit(ws, msg, k):
        if k == 1:
            (ws / "analysis.py").write_text((ws / "analysis.py").read_text() + "\n# an analysis-only edit\n")

    agent = recorder(edit)
    run(tmp_path, agent, steps=300, decisions=3, every=50)
    timing = [line for x in agent.msgs for line in x.splitlines() if line.startswith("Timing:")]
    assert timing and all("the starting program: not measured yet" not in t for t in timing)


def test_prompts16_reports_past_400_are_counted(tmp_path):
    every1 = 'EVERY = 1\ndef analyze(obs, memory):\n    return {"report": {"u": obs.update}}\n'

    def edit(ws, msg, k):
        if k == 1:
            (ws / "analysis.py").write_text(every1)

    agent = recorder(edit)
    run(tmp_path, agent, steps=1500, decisions=3, schedule="uniform", every=50)  # decisions at 500, 1000
    line = next(x for x in agent.msgs[1].splitlines() if x.startswith("Reports from your analysis"))
    made = int(re.search(r"decision: (\d+);", line).group(1))
    assert made > 400 and line.endswith("the most recent 400 are in inbox/decision-002.json.")


def system_of(tmp_path, **kw):
    m, _ = make()
    o, s = adamw(m, 200)
    kw = dict(dict(decisions=4, every=10, verbose=False), **kw)
    w = opal.wrap(o, m, scheduler=s, total_steps=200, goal="t", run_dir=tmp_path / "r",
                     backend=kw.pop("backend", lambda ws, msg, sy: "x"), **kw)  # fmt: skip
    return w, w.worker.system


def test_prompts11_the_pause_sentence_only_when_the_pause_is_on(tmp_path):
    for prompt in ("neutral",):
        _, on = system_of(tmp_path / f"{Path(prompt).name}-on", prompt=prompt, blocking=False)
        _, off = system_of(tmp_path / f"{Path(prompt).name}-off", prompt=prompt, blocking=False, pause_on_nonfinite=False)
        assert "A non-finite training loss pauses training until you answer." in on
        assert "pauses training until you answer" not in off


def test_prompts4_clipping_is_in_the_starting_summary_and_the_reference(tmp_path):
    w, system = system_of(tmp_path, clip_grad_norm=1.0)
    assert "gradients clipped to a global norm of 1 before each update" in system
    ref = (tmp_path / "r" / "workspace" / "reference" / "starting_optimizer.txt").read_text()
    assert "clips the gradients of all these parameters to a global norm of 1" in ref


def test_prompts13_no_lambda_source_without_share_code(tmp_path):
    m, _ = make()
    o = torch.optim.AdamW(m.parameters(), lr=1e-3)
    s = torch.optim.lr_scheduler.LambdaLR(o, lambda step: 1.0)  # SECRET-COMMENT
    for prompt in ("neutral",):
        w = opal.wrap(o, m, scheduler=s, total_steps=100, goal="t", run_dir=tmp_path / Path(prompt).name, prompt=prompt,
                         backend=lambda ws, msg, sy: "x", share_code=False, verbose=False)  # fmt: skip
        shown = [w.worker.system, w.files["updater.py"]] + [
            p.read_text() for p in (tmp_path / Path(prompt).name / "workspace" / "reference").glob("*")
        ]
        assert not any("SECRET-COMMENT" in t for t in shown)
        assert "a LambdaLR schedule (its function is not shared)" in w.worker.system
        w.close()


def test_prompts14_no_claim_of_a_training_script_that_was_not_copied(tmp_path, monkeypatch):
    monkeypatch.setattr(opal.context, "reference_files", lambda *a, **k: {})  # as in a notebook
    w, system = system_of(tmp_path)
    assert "the training script itself could not be copied" in system
    assert "and the training script itself is in reference/" not in system
    assert "the training script, the source files" not in system


class NoShell(Backend):
    name = "noshell"
    sandbox = False  # as the Claude backend where the OS sandbox cannot start

    def decide(self, workspace, message, system, record_dir):
        return dict(text="x", cost_usd=0.0, returncode=0, error=None, seconds=0.0)


def test_prompts15_no_shell_sentences_without_a_shell(tmp_path):
    _, with_shell = system_of(tmp_path / "a")
    _, without = system_of(tmp_path / "b", backend=NoShell())
    assert "Your shell runs the training's Python" in with_shell
    assert "shell" not in without.split("Your tools work inside this workspace")[1].split("\n\n")[0]
    assert (
        "Your programs run inside the training process, which reports import and runtime errors back to you." in without
    )


class Fails(Backend):
    """A call that fails with exit code 0 and an error, as Claude Code's usage limit does."""

    name = "fails"

    def __init__(self):
        self.calls, self.budgets = [], []

    def decide(self, workspace, message, system, record_dir):
        self.calls.append(message)
        self.budgets.append(getattr(self, "budget_left", None))
        if len(self.calls) == 1:
            return dict(text="", cost_usd=0.25, returncode=0, error="You've hit your limit", seconds=0.1)
        return dict(text="x", cost_usd=0.25, returncode=0, error=None, seconds=0.1)


def test_b4_d1_a_failed_call_is_said_and_counted_and_the_budget_is_passed(tmp_path):
    be = Fails()
    run(tmp_path, be, decisions=3, schedule="uniform", max_cost_usd=2.0)
    second = be.calls[1]
    assert (
        "Decision 1 was not answered: the agent call failed (You've hit your limit). The programs were not changed"
        in second
    )
    assert "you kept the programs" not in second and "No errors since your previous decision." not in second
    ev = events(tmp_path / "r")
    assert [e.get("failed", False) for e in ev if e["kind"] in ("kept", "applied", "rejected")][0] is True
    c = decision_counts(ev)
    assert c["failed"] == 1 and c["answered"] == c["asked"] - 1 and "1 agent call failed" in describe_counts(c)
    assert be.budgets[:2] == [2.0, 1.75]  # max_cost_usd minus what was spent, before each decision
