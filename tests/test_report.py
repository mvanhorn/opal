"""The report: the table of code changes (files, constants, summaries, tags), the live page, the extra curves."""

import json
import time

import pytest

import opal
from opal import cli, summaries
from opal.backends.claude import ClaudeBackend
from opal.report import constants, decision_rows, report_series, summarize_missing, write_report
from test_core import SGD_UPDATER, adamw, make, train

CONST_UPDATER = SGD_UPDATER.replace("RAN = []", 'RAN = []\nLR = 0.05\nBETAS = (0.9, 0.99)\nSCALE = float("2")').replace(
    "torch.tensor(0.05)", "torch.tensor(LR)"
)
CONST_UPDATER_2 = CONST_UPDATER.replace("LR = 0.05", "LR = 0.02\nWARMUP = 10")
RATIONALE = (
    "Observation: the loss plateaued near update 40. Change: SGD with a lower LR on every tensor. Fixed schedule."
)


def scripted(*answers):
    """A fake agent writing updater.py at its 1st, 2nd, ... decision (None keeps)."""
    calls = []

    def agent(ws, msg, system):
        i = len(calls)
        calls.append(msg)
        if i < len(answers) and answers[i]:
            (ws / "updater.py").write_text(answers[i])
        return RATIONALE

    return agent


def run_with(tmp_path, backend, steps=120, decisions=3, **kw):
    m, data = make()
    o, s = adamw(m, steps)
    w = opal.wrap(
        o,
        m,
        scheduler=s,
        total_steps=steps,
        goal="report test",
        decisions=decisions,
        every=10,
        run_dir=tmp_path / "r",
        backend=backend,
        blocking=True,
        verbose=False,
        **kw,
    )
    train(m, w, data, steps, sched=s)
    w._live.finish(timeout=30)
    return w, tmp_path / "r"


def fake_cli(reply=None, code=0, cost=0.0004):
    """Stands in for summaries._run_cli; records the summary calls (not the --help probe)."""
    calls = []

    def run(cmd, text, cwd, env, timeout):
        if cmd[1:] == ["--help"]:
            return 0, "", ""
        calls.append(dict(cmd=cmd, text=text, cwd=cwd))
        body = reply if reply is not None else json.dumps(dict(summary="Lowers the LR. Because of a plateau.", tags=[]))
        return code, json.dumps(dict(type="result", result=body, total_cost_usd=cost, is_error=code != 0)), ""

    run.calls = calls
    return run


def test_table_rows_files_and_constants(tmp_path):
    w, run = run_with(tmp_path, scripted(CONST_UPDATER, CONST_UPDATER_2, None))
    rows = decision_rows(run)
    assert [r["k"] for r in rows] == [1, 2] and w.version == 2
    second = rows[1]
    assert second["files"] == [dict(name="updater.py", added=2, removed=1, status="")]
    assert second["constants"] == [("updater.py", "LR", "0.05", "0.02"), ("updater.py", "WARMUP", None, "10")]
    assert ("updater.py", "BETAS", None, "(0.9, 0.99)") in rows[0]["constants"]
    assert not any(c[1] == "SCALE" for r in rows for c in r["constants"])  # not a literal
    assert second["requested"] is not None and second["applied"] == second["requested"]  # blocking
    assert second["summary"] is None and second["rationale"] == RATIONALE
    page = (run / "report.html").read_text()
    assert "answered 3 and applied 2 changes (2 to the updater, 0 only to the analysis)" in page
    assert "WARMUP = 10" in page and "0.05 → 0.02" in page
    assert "summary unavailable" in page and 'href="#d2"' in page and RATIONALE.split(":")[0] in page
    assert 'http-equiv="refresh"' not in page  # the run has ended


def test_constants_parsing():
    src = 'import math\nA = 1\nB: float = -2e-3\nC = math.pi\nD = 1\nD = len("x")\nlower = 3\nE = {"x": (1, 2)}\n'
    assert constants(src) == dict(A=1, B=-2e-3, E={"x": (1, 2)})
    assert constants("def broken(:\n") == {}


def test_summaries_in_the_run_cached_and_tags_filtered(tmp_path, monkeypatch):
    reply = json.dumps(
        dict(
            summary="Switches to plain SGD at LR 0.05.  Cites a plateau.",
            tags=["Block specific", "bold idea", "observation-originated", "Fix or Revert"],
        )
    )
    fake = fake_cli(reply)
    monkeypatch.setattr(summaries, "_run_cli", fake)

    class FakeClaude(ClaudeBackend):
        def preflight(self):
            pass

        def decide(self, workspace, message, system, record_dir):
            if "Decision 1." in message:
                (workspace / "updater.py").write_text(CONST_UPDATER)
            return dict(text=RATIONALE, cost_usd=0.1, returncode=0, error=None, seconds=0.0)

    w, run = run_with(tmp_path, FakeClaude(sandbox=False))
    assert len(fake.calls) == 1  # one call per code change, none for kept decisions
    cmd = fake.calls[0]["cmd"]
    assert cmd[cmd.index("--tools") + 1] == "" and cmd[cmd.index("--model") + 1] == summaries.DEFAULT_MODEL
    assert "--no-session-persistence" in cmd and "--safe-mode" in cmd
    assert RATIONALE in fake.calls[0]["text"] and "+LR = 0.05" in fake.calls[0]["text"]
    assert str(run) not in fake.calls[0]["cwd"]  # an empty temporary folder
    s = json.loads((run / "decisions" / "001" / "summary.json").read_text())
    assert s["summary"] == "Switches to plain SGD at LR 0.05. Cites a plateau."
    assert s["tags"] == ["block-specific", "observation-originated", "fix or revert"]
    assert s["dropped_tags"] == ["bold idea"]
    assert s["cost_usd"] == 0.0004 and s["model"] == summaries.DEFAULT_MODEL
    page = (run / "report.html").read_text()
    assert "Switches to plain SGD" in page and "block-specific, observation-originated, fix or revert" in page
    assert 'class="tag ' not in page and "saved summaries cost $0.0004 separately" in page
    assert summarize_missing(run, summaries.Summarizer()) == 0  # cached: nothing to do
    assert cli.main(["report", str(run), "--summarize"]) == 0
    assert len(fake.calls) == 1
    w2, _ = run_with(tmp_path / "off", FakeClaude(sandbox=False), steps=30, decisions=1, summarize=False)
    assert w2._live.summarizer is None


def test_failed_summary_is_not_cached_and_retried(tmp_path, monkeypatch):
    _, run = run_with(tmp_path, scripted(CONST_UPDATER))
    monkeypatch.setattr(summaries, "_run_cli", fake_cli("I would rather not answer in JSON."))
    assert summarize_missing(run, summaries.Summarizer()) == 1
    rec = run / "decisions" / "001"
    assert not (rec / "summary.json").exists()
    assert "no JSON object" in json.loads((rec / "summary-error.json").read_text())["error"]
    monkeypatch.setattr(summaries, "_run_cli", fake_cli("not even", code=1))
    assert summarize_missing(run, summaries.Summarizer()) == 1
    good = fake_cli('```json\n{"summary": "Plain SGD.", "tags": "common practice"}\n```')
    monkeypatch.setattr(summaries, "_run_cli", good)
    assert summarize_missing(run, summaries.Summarizer()) == 0
    assert summaries.load(rec)["tags"] == ["common practice"] and not (rec / "summary-error.json").exists()
    assert "Plain SGD." in write_report(run).read_text()


def test_live_report_after_each_decision(tmp_path):
    seen = []

    def agent(ws, msg, system):
        if "Decision 2." in msg:
            page = ws.parent / "report.html"
            for _ in range(100):  # written by the report thread, not the training step
                if page.exists() and "applied 1 change" in page.read_text():
                    break
                time.sleep(0.05)
            seen.append(page.read_text())
        if "Decision 1." in msg:
            (ws / "updater.py").write_text(CONST_UPDATER)
        return "x"

    _, run = run_with(tmp_path, agent)
    assert seen and 'http-equiv="refresh" content="60"' in seen[0] and "Live: the run is at update" in seen[0]
    assert "applied 1 change" in seen[0]
    assert 'http-equiv="refresh"' not in (run / "report.html").read_text()


def test_extra_curves_from_analysis_reports(tmp_path):
    run = tmp_path / "r"
    run.mkdir()
    (run / "events.jsonl").write_text(json.dumps(dict(kind="start", total_updates=100, goal="g")) + "\n")
    with open(run / "reports.jsonl", "w") as f:
        for u in (10, 20, 30, 40):
            rep = dict(loss=1.0 / u, flag=True, note="text", grad_norm=dict(embed=u * 0.1, head=2.0, deep=dict(x=1)))
            if u > 20:
                rep["late"] = 1.0  # in 2 reports only
            rep.update({f"m{i}": float(i) * u for i in range(15)})  # m0 stays 0
            f.write(json.dumps(dict(update=u, version=0, report=rep)) + "\n")
        f.write('{"update": 50, "report": ')  # a line being written
    series, n = report_series(run)
    keys = [k for k, _ in series]
    # constant series (grad_norm.head, m0) are left out, and so is the update itself (review F15)
    assert keys[:3] == ["loss", "grad_norm.embed", "m1"] and len(keys) == 12 and n == 16
    assert dict(series)["grad_norm.embed"] == [[10, 1.0], [20, 2.0], [30, 3.0], [40, 4.0]]
    assert not {"flag", "note", "late", "grad_norm.deep", "grad_norm.deep.x", "grad_norm.head", "m0"} & set(keys)
    page = write_report(run).read_text()
    assert "grad_norm.embed (reported by the agent" in page and "4 more series not shown" in page


def test_nothing_breaks_without_claude(tmp_path, monkeypatch):
    _, run = run_with(tmp_path, scripted(CONST_UPDATER))
    assert summarize_missing(run, summaries.Summarizer(binary="no-such-claude-binary")) == 1
    err = json.loads((run / "decisions" / "001" / "summary-error.json").read_text())
    assert "not found" in err["error"]
    monkeypatch.setenv("PATH", str(tmp_path))  # no claude on PATH
    assert cli.main(["report", str(run), "--summarize"]) == 0
    assert "summary unavailable" in (run / "report.html").read_text()


@pytest.mark.parametrize("resumed", [False, True])
def test_reports_count_the_capped_schedule(tmp_path, resumed):
    from opal.report import decision_counts
    from test_claude_backend import claude_run, fake_cli as fake_agent_cli
    from test_core import events

    exe = fake_agent_cli(tmp_path)
    m, data, s, w = claude_run(tmp_path, exe, steps=200, decisions=None)
    train(m, w, data, 60, sched=s)
    state = w.state_dict()
    w.close(interrupted=True)
    if resumed:
        m, data, s, w = claude_run(tmp_path, exe, steps=200, decisions=None, run="r2")
        w.load_state_dict(state)
        w.close(interrupted=True)
    assert decision_counts(events(w.run.dir))["scheduled"] == len(w.schedule) == 1
    import re

    page = write_report(w.run.dir).read_text()
    shown = re.search(r"Schedule \(updates\).*?</dt><dd>(.*?)</dd>", page).group(1)
    assert shown == ", ".join(str(v) for v in w.schedule)
