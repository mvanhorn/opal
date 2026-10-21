"""The Claude Code backend against a fake `claude` CLI (CPU only, no model call).

The fake follows Claude Code 2.1.302 where it matters to the backend (read from the installed binary and checked with
the real CLI against a local mock API, tests/test_claude_real_cli.py): `--session-id X` for an X that exists in the
working directory's project fails with "Error: Session ID X is already in use." and exit 1; `--resume X` for an unknown
X fails with "No conversation found with session ID: X" and exit 1; sessions are kept per working directory. The fake
reads its settings from fake.json next to it (the backend passes the CLI only an allowlisted environment).
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest
import torch

import opal
import opal.backends.claude as claude
from opal.backends.base import agent_env
from opal.backends.claude import ClaudeBackend, _login_prefixes
from test_core import adamw, events, make, train

FAKE_CLI = textwrap.dedent(
    """\
    #!{py}
    import hashlib, json, os, signal, subprocess, sys, time
    cfg = json.load(open(os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), "fake.json")))
    argv = sys.argv[1:]
    with open(cfg["log"], "a") as f:
        f.write(json.dumps(dict(argv=argv, cwd=os.getcwd(), mode=cfg.get("mode"))) + "\\n")
    if "--help" in argv:
        print(cfg.get("help", "--safe-mode --restricted --no-chrome --permission-prompts "
              "--setting-sources --strict-mcp-config --effort --max-budget-usd"))
        sys.exit(0)
    proj = os.path.join(cfg["state"], hashlib.sha1(os.getcwd().encode()).hexdigest()[:10])
    os.makedirs(proj, exist_ok=True)
    def val(flag):
        return argv[argv.index(flag) + 1] if flag in argv else None
    sid, rid = val("--session-id"), val("--resume")
    if sid:
        if os.path.exists(os.path.join(proj, sid)):
            sys.stderr.write(f"Error: Session ID {{sid}} is already in use.\\n")
            sys.exit(1)
        s = sid
    elif rid:
        if not os.path.exists(os.path.join(proj, rid)):
            sys.stderr.write(f"No conversation found with session ID: {{rid}}\\n")
            sys.exit(1)
        s = rid
    else:
        s = "none"
    message = sys.stdin.read()
    if cfg.get("msglog"):
        with open(cfg["msglog"], "a") as f:
            f.write(json.dumps(message) + "\\n")
    sf = os.path.join(proj, s)
    prev = json.load(open(sf)) if os.path.exists(sf) else dict(cost=0.0, turns=0)
    prev["turns"] += 1
    prev["cost"] += 0.25
    json.dump(prev, open(sf, "w"))
    def emit(e):
        sys.stdout.write(json.dumps(e) + "\\n")
        sys.stdout.flush()
    emit(dict(type="system", subtype="init", session_id=s))
    mode = cfg.get("mode", "ok")
    if mode == "budget" and float(val("--max-budget-usd") or "inf") < 0.25:
        emit(dict(type="result", subtype="error_max_budget_usd", is_error=True,
                  result=f"Reached maximum budget (${{val('--max-budget-usd')}})", total_cost_usd=0))
        sys.exit(1)
    pidfile = cfg.get("pidfile")
    secs = float(cfg.get("seconds", 20))
    if mode == "apierror":
        emit(dict(type="result", subtype="success", is_error=True, result="API Error: 529 overloaded",
                  total_cost_usd=prev["cost"]))
        sys.exit(1)
    if mode == "nologin":  # as Claude Code 2.1.305 without a login: a result event, exit 1, nothing on stderr
        emit(dict(type="result", subtype="success", is_error=True, result="Not logged in \\u00b7 Please run /login",
                  total_cost_usd=0))
        sys.exit(1)
    if mode == "sigkill":
        os.kill(os.getpid(), signal.SIGKILL)
    if mode == "flaky":  # its first `fails` calls edit updater.py halfway (unless half=0), then fail with `say` (exit 1)
        nf = os.path.join(cfg["state"], "flaky-calls")
        n = int(open(nf).read()) + 1 if os.path.exists(nf) else 1
        open(nf, "w").write(str(n))
        if n <= int(cfg.get("fails", 1)):
            if int(cfg.get("half", 1)):
                with open("updater.py", "a") as f:
                    f.write("\\n# half an edit\\n")
            print(cfg.get("say", "You've hit your org's monthly spend limit \\u00b7 your session limit resets 5am"),
                  flush=True)
            sys.exit(1)
    limit = "You've hit your limit \\u00b7 resets 3pm (America/New_York)"
    if mode == "limit":  # a usage limit reported in the stream; exit 1, nothing on stderr
        emit(dict(type="assistant", message=dict(content=[dict(type="text", text=limit)])))
        emit(dict(type="result", subtype="success", is_error=True, result=limit, total_cost_usd=prev["cost"]))
        sys.exit(1)
    if mode == "limit_plain":  # the limit as a plain line, no result event; exit 1, nothing on stderr
        print(limit, flush=True)
        sys.exit(1)
    if mode == "limit_ok":  # the limit as the result text of a call that exits 0
        emit(dict(type="result", subtype="success", is_error=False, result=limit, total_cost_usd=prev["cost"]))
        sys.exit(0)
    if mode in ("holder", "bg"):
        quiet = dict(stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        if mode == "bg":
            quiet["stderr"] = subprocess.DEVNULL
        p = subprocess.Popen([sys.executable, "-c", f"import time; time.sleep({{secs}})"], **quiet)
        if pidfile:
            open(pidfile, "w").write(str(p.pid))
    if mode == "slow":
        if pidfile:
            open(pidfile, "w").write(str(os.getpid()))
        time.sleep(secs)
    if mode == "envdump":
        json.dump(dict(os.environ), open(cfg["dump"], "w"))
    if mode == "edit":
        with open("updater.py", "a") as f:
            f.write("\\n# edited by the fake agent\\n")
    if cfg.get("updater"):  # an answer that changes training: this updater
        with open("updater.py", "w") as f:
            f.write(cfg["updater"])
    if mode == "notes":
        with open("NOTES.md", "a") as f:
            f.write(message.splitlines()[0] + "\\n")
    emit(dict(type="result", subtype="success", is_error=False, result=f"done ({{mode}})",
              total_cost_usd=prev["cost"]))
    sys.exit(0)
    """
)


def fake_cli(tmp_path, mode="ok", **cfg) -> Path:
    """Write the fake CLI (and its fake.json) into tmp_path/fakebin. Returns the binary's path."""
    b = tmp_path / "fakebin"
    b.mkdir(exist_ok=True)
    exe = b / "claude"
    exe.write_text(FAKE_CLI.format(py=sys.executable))
    exe.chmod(0o755)
    cfg = dict(state=str(tmp_path / "fake-state"), log=str(tmp_path / "fake-calls.jsonl"), mode=mode, **cfg)
    (b / "fake.json").write_text(json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in cfg.items()}))
    return exe


def set_fake(exe, **cfg):
    f = Path(exe).with_name("fake.json")
    f.write_text(json.dumps({**json.loads(f.read_text()), **{k: str(v) for k, v in cfg.items()}}))


def calls(tmp_path):
    f = tmp_path / "fake-calls.jsonl"
    rows = [json.loads(line) for line in open(f)] if f.exists() else []
    return [c for c in rows if "--help" not in c["argv"]]


def backend(tmp_path, exe, run="run", ws="ws", **kw):
    kw.setdefault("sandbox", False)
    kw.setdefault("retries", 0)  # one call per decision, unless a test makes failed calls again
    kw.setdefault("unavailable_timeout", 0)  # an unavailable agent fails the decision at once, unless a test waits
    be = ClaudeBackend(binary=str(exe), **kw)
    be.preflight()
    w = tmp_path / ws
    w.mkdir(parents=True, exist_ok=True)
    be.setup(tmp_path / run, w)
    return be, w


def decide(be, ws, tmp_path, message="m", name=None):
    rec = tmp_path / (name or f"rec-{time.time_ns()}")
    rec.mkdir(parents=True, exist_ok=True)
    return be.decide(ws, message, "system", rec)


def alive(pid):
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    try:  # a zombie (exited, not reaped yet) counts as gone
        out = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
        return bool(out) and not out.startswith("Z")
    except OSError:
        return True


def wait_gone(pid, seconds=10):
    t0 = time.time()
    while alive(pid) and time.time() - t0 < seconds:
        time.sleep(0.1)
    return not alive(pid)


def claude_run(tmp_path, exe, steps=200, run="r", **kw):
    kw = {
        **dict(total_steps=steps, goal="t", backend="claude",
               backend_options=dict(binary=str(exe), sandbox=False, retries=0, unavailable_timeout=0),
               decisions=4, every=10, run_dir=tmp_path / run, summarize=False, verbose=False),
        **kw,
    }  # fmt: skip
    m, data = make()
    o, s = adamw(m, steps)
    return m, data, s, opal.wrap(o, m, scheduler=s, **kw)


# ----- git context ------------------------------------------------------------------------------------------------


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_agent_env_hides_the_enclosing_git_repository(tmp_path):
    """Claude Code 2.1.302 puts a gitStatus block (branch, `git config user.name`, untracked file names, recent
    commits) of the repository around its working directory into the context unless CLAUDE_CODE_DISABLE_GIT_INSTRUCTIONS
    is set; GIT_CEILING_DIRECTORIES keeps the agent's own git commands out of that repository too."""
    outer = tmp_path / "outer"
    outer.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=outer, check=True)
    dump = tmp_path / "env.json"
    exe = fake_cli(tmp_path, mode="envdump", dump=dump)
    be, ws = backend(outer, exe, run="opal_runs/r1", ws="opal_runs/r1/workspace")
    out = decide(be, ws, tmp_path)
    assert out["returncode"] == 0, out
    env = json.loads(dump.read_text())
    assert env["CLAUDE_CODE_DISABLE_GIT_INSTRUCTIONS"] == "1"
    assert env["GIT_CEILING_DIRECTORIES"] == str(ws.resolve().parent)
    seen = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=ws, env=env, capture_output=True, text=True)
    assert seen.returncode != 0, seen.stdout  # the agent's git finds no repository
    plain = {k: v for k, v in env.items() if k != "GIT_CEILING_DIRECTORIES"}
    found = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=ws, env=plain, capture_output=True, text=True)
    assert Path(found.stdout.strip()).resolve() == outer.resolve()  # the control: without it, git finds outer/


# ----- B1: the session survives a resume ----------------------------------------------------------------------------


def test_session_id_already_in_use_is_resumed(tmp_path):
    """A checkpoint saved started=False although the session had been used (requeued after its first decision):
    the next call is refused with 'already in use' and must --resume it, not fail for the rest of the run."""
    exe = fake_cli(tmp_path)
    be, ws = backend(tmp_path, exe)
    assert decide(be, ws, tmp_path)["returncode"] == 0
    saved = dict(be.state(), started=False)
    again, _ = backend(tmp_path, exe)
    again.restore(saved)
    outs = [decide(again, ws, tmp_path) for _ in range(3)]
    assert [o["returncode"] for o in outs] == [0, 0, 0] and all(o["error"] is None for o in outs), outs
    flags = ["--session-id" if "--session-id" in c["argv"] else "--resume" for c in calls(tmp_path)]
    assert flags == ["--session-id", "--session-id", "--resume", "--resume", "--resume"]


def test_session_id_is_chosen_at_the_first_call(tmp_path):
    exe = fake_cli(tmp_path)
    be, ws = backend(tmp_path, exe)
    assert be.state()["session"] is None  # a checkpoint before the first decision holds no id to collide with
    decide(be, ws, tmp_path)
    assert be.state()["session"] and be.state()["started"] and be.state()["cwd"] == str(ws.resolve())


def test_resumed_run_keeps_its_agent_after_a_checkpoint_before_the_first_decision(tmp_path):
    """The review's R01 scenario through opal.wrap: checkpoint at update 0, first decisions, requeue, resume."""
    exe = fake_cli(tmp_path)
    m, data, s, w = claude_run(tmp_path, exe)
    ck = dict(opt=w.state_dict(), sched=s.state_dict())
    train(m, w, data, 60, sched=s)
    w.worker.wait()
    m2, data2, s2, w2 = claude_run(tmp_path, exe)
    w2.load_state_dict(ck["opt"])
    s2.load_state_dict(ck["sched"])
    train(m2, w2, data2, 200, sched=s2)
    w2.close()
    ev = events(tmp_path / "r")
    resumed = max(i for i, e in enumerate(ev) if e["kind"] == "resumed")
    after = [e for e in ev[resumed:] if e["kind"] in ("applied", "kept", "rejected", "load_failed")]
    assert after and not any(e.get("failed") for e in after), after


def test_resume_into_another_folder_starts_a_new_session_with_a_true_note(tmp_path):
    """Claude Code keys sessions by working directory: from another run folder the old one cannot be resumed. The
    backend starts a new session at once (no failed call) and says what the new folder really holds."""
    exe = fake_cli(tmp_path)
    be, ws = backend(tmp_path, exe, run="A", ws="A/workspace")
    (ws / "inbox").mkdir()
    for k in (1, 2):
        (ws / "inbox" / f"decision-{k:03d}.json").write_text("{}")
        decide(be, ws, tmp_path, message=f"Decision {k}.")
    other, ws_b = backend(tmp_path, exe, run="B", ws="B/workspace")
    other.restore(be.state())
    (ws_b / "inbox").mkdir()
    (ws_b / "inbox" / "decision-003.json").write_text("{}")
    sent = []
    real = other._decide
    other._decide = lambda w, msg, sys_, rec, *a: sent.append(msg) or real(w, msg, sys_, rec, *a)
    out = decide(other, ws_b, tmp_path, message="Decision 3.")
    assert out["returncode"] == 0 and len(sent) == 1
    assert sent[0].startswith(
        "(Your earlier conversation could not be resumed, so this is a new conversation: the inputs of earlier "
        "decisions are not in this folder, and your current programs are in this folder.)\n\nDecision 3."
    ), sent[0]
    (ws_b / "inbox" / "decision-001.json").write_text("{}")
    (ws_b / "inbox" / "decision-002.json").write_text("{}")
    (ws_b / "NOTES.md").write_text("n")
    assert "inbox/ holds the inputs of 2 earlier decisions (1 to 2), and your current programs and your notes" in (
        ClaudeBackend.lost_note(ws_b)
    )


def test_session_lost_on_another_machine_starts_a_new_one(tmp_path):
    exe = fake_cli(tmp_path)
    be, ws = backend(tmp_path, exe)
    decide(be, ws, tmp_path)
    shutil.rmtree(tmp_path / "fake-state")  # the conversation is not on this machine
    out = decide(be, ws, tmp_path)
    assert out["returncode"] == 0 and out["error"] is None
    assert calls(tmp_path)[-1]["argv"].count("--session-id") == 1


# ----- B2: the Linux sandbox needs bubblewrap and socat, and must be able to start --------------------------------


def _linux(monkeypatch, tmp_path, found, bwrap_works=True):
    monkeypatch.setattr(claude.platform, "system", lambda: "Linux")
    fake = tmp_path / "linuxbin"
    fake.mkdir(exist_ok=True)
    tools = {}
    for name, where in found.items():
        if where is None:
            continue
        d = tmp_path / where.strip("/")
        d.mkdir(parents=True, exist_ok=True)
        exe = d / name
        body = (
            "exit 0"
            if (name != "bwrap" or bwrap_works)
            else "echo 'bwrap: No permissions to create new namespace' >&2; exit 1"
        )
        exe.write_text("#!/bin/sh\n" + body + "\n")
        exe.chmod(0o755)
        tools[name] = str(exe)
    real_which = shutil.which
    monkeypatch.setattr(
        claude.shutil, "which", lambda n, *a, **k: tools.get(n) if n in found else real_which(n, *a, **k)
    )
    return tools


@pytest.mark.parametrize(
    "found, works, problem",
    [
        (dict(bwrap="/usr/bin", socat=None), True, "socat not found on PATH"),
        (dict(bwrap=None, socat="/usr/bin"), True, "bwrap not found on PATH"),
        (dict(bwrap="/usr/bin", socat="/usr/bin"), False, "No permissions to create new namespace"),
    ],
)
def test_linux_sandbox_that_cannot_start_falls_back_to_file_tools(tmp_path, monkeypatch, capsys, found, works, problem):
    _linux(monkeypatch, tmp_path, found, bwrap_works=works)
    exe = fake_cli(tmp_path)
    be, ws = backend(tmp_path, exe, sandbox="auto")
    assert not be.sandbox and problem in be.sandbox_problem
    assert "file tools only" in capsys.readouterr().err
    cmd = be._cmd("sys")
    assert cmd[cmd.index("--tools") + 1] == "Edit,Read,Write"
    assert "sandbox" not in json.loads((be.control / "settings.json").read_text())
    with pytest.raises(RuntimeError, match="cannot start here"):
        backend(tmp_path, exe, sandbox=True)


def test_linux_sandbox_tools_outside_the_system_folders_are_on_the_agents_path(tmp_path, monkeypatch):
    tools = _linux(monkeypatch, tmp_path, dict(bwrap="/usr/local/bin", socat="/opt/socat/bin"))
    exe = fake_cli(tmp_path)
    be, ws = backend(tmp_path, exe, sandbox="auto")
    assert be.sandbox and be.sandbox_problem is None
    s = json.loads((be.control / "settings.json").read_text())
    path = s["env"]["PATH"].split(":")
    for t in tools.values():
        assert os.path.dirname(t) in path and os.path.dirname(t) in s["sandbox"]["filesystem"]["allowRead"]
    assert s["sandbox"]["failIfUnavailable"] is True


# ----- B3, B13: the timeout bounds the decision; nothing the agent started outlives it ---------------------------


def test_descendant_holding_stderr_does_not_stretch_the_timeout(tmp_path):
    pidf = tmp_path / "holder.pid"
    exe = fake_cli(tmp_path, mode="holder", seconds=60, pidfile=pidf)
    be, ws = backend(tmp_path, exe, timeout=5)
    t0 = time.time()
    out = decide(be, ws, tmp_path)
    took = time.time() - t0
    assert took < 15, took
    assert out["returncode"] == 0 and out["text"] == "done (holder)" and out["error"] is None, out
    assert wait_gone(int(pidf.read_text()), 5)


def test_timeout_after_the_answer_keeps_the_answer(tmp_path, monkeypatch):
    """The CLI wrote its result and then did not exit: the decision times out, and the answer counts."""
    exe = fake_cli(tmp_path)
    be, ws = backend(tmp_path, exe, timeout=1)

    def run(cmd, input, stdout, cwd, timeout, env=None):
        stdout.write(json.dumps(dict(type="system", subtype="init")) + "\n")
        stdout.write(json.dumps(dict(type="result", result="finished", total_cost_usd=0.1)) + "\n")
        stdout.flush()
        raise subprocess.TimeoutExpired(cmd, timeout)

    monkeypatch.setattr(be, "_run", run)
    out = decide(be, ws, tmp_path)
    assert out["returncode"] == 0 and out["text"] == "finished" and out["error"] is None, out


def test_slow_agent_is_stopped_at_the_timeout(tmp_path):
    pidf = tmp_path / "slow.pid"
    exe = fake_cli(tmp_path, mode="slow", seconds=60, pidfile=pidf)
    be, ws = backend(tmp_path, exe, timeout=2)
    t0 = time.time()
    out = decide(be, ws, tmp_path)
    assert time.time() - t0 < 10
    assert out["returncode"] == -1 and out["error"] == "timed out after 2 s"
    assert wait_gone(int(pidf.read_text()), 5)


def test_background_job_of_the_agent_is_stopped_after_the_decision(tmp_path):
    pidf = tmp_path / "bg.pid"
    exe = fake_cli(tmp_path, mode="bg", seconds=60, pidfile=pidf)
    be, ws = backend(tmp_path, exe)
    out = decide(be, ws, tmp_path)
    assert out["returncode"] == 0
    assert wait_gone(int(pidf.read_text()), 5)


PARENT = """
import sys
from pathlib import Path
sys.path[:0] = [sys.argv[3]]
from opal.backends.claude import ClaudeBackend
be = ClaudeBackend(binary=sys.argv[1], sandbox=False, timeout=600)
be.preflight()
ws = Path(sys.argv[2]) / "ws"; ws.mkdir(exist_ok=True)
be.setup(Path(sys.argv[2]) / "run", ws)
rec = Path(sys.argv[2]) / "d1"; rec.mkdir(exist_ok=True)
be.decide(ws, "m", "s", rec)
"""


def test_watchdog_stops_the_agent_when_training_is_killed(tmp_path):
    pidf = tmp_path / "slow.pid"
    exe = fake_cli(tmp_path, mode="slow", seconds=120, pidfile=pidf)
    script = tmp_path / "parent.py"
    script.write_text(PARENT)
    root = str(Path(opal.__file__).resolve().parent.parent)
    p = subprocess.Popen([sys.executable, str(script), str(exe), str(tmp_path), root])
    for _ in range(200):
        if pidf.exists() and pidf.read_text():
            break
        time.sleep(0.05)
    agent = int(pidf.read_text())
    p.send_signal(signal.SIGKILL)
    p.wait()
    gone = wait_gone(agent, 15)
    if not gone:
        os.kill(agent, signal.SIGKILL)
    assert gone


# ----- B4: a failed call always carries an error -----------------------------------------------------------------


LIMIT = "You've hit your limit · resets 3pm (America/New_York)"


@pytest.mark.parametrize(
    "mode, error",
    [
        ("apierror", "API Error: 529 overloaded"),
        ("sigkill", "killed by signal 9 (SIGKILL)"),
        ("limit", LIMIT),
        ("limit_plain", LIMIT),
        ("limit_ok", LIMIT),
    ],
)
def test_failed_call_with_nothing_on_stderr_has_an_error(tmp_path, mode, error):
    exe = fake_cli(tmp_path, mode=mode)
    be, ws = backend(tmp_path, exe)
    out = decide(be, ws, tmp_path)
    assert out["error"] and error in out["error"], out
    if mode != "limit_ok":
        assert out["returncode"] != 0


def test_usage_limit_is_a_failed_decision_not_kept(tmp_path):
    """A failed call must be recorded as failed, and the next message must say it was not answered."""
    exe = fake_cli(tmp_path, mode="limit")
    m, data, s, w = claude_run(tmp_path, exe, steps=120, decisions=2)
    train(m, w, data, 120, sched=s)
    w.close()
    r1 = json.loads((tmp_path / "r" / "decisions" / "001" / "response.json").read_text())
    assert r1["returncode"] == 1 and LIMIT in r1["backend_error"], r1
    out = [e for e in events(tmp_path / "r") if e["kind"] in ("applied", "kept", "rejected")]
    assert out and all(e.get("failed") for e in out), out
    msg2 = (tmp_path / "r" / "decisions" / "002" / "message.txt").read_text()
    assert "Decision 1 was not answered" in msg2 and "you kept the programs" not in msg2


@pytest.mark.parametrize("mode", ["apierror", "sigkill"])
def test_failed_call_is_reported_as_failed_not_kept(tmp_path, mode):
    exe = fake_cli(tmp_path, mode=mode)
    m, data, s, w = claude_run(tmp_path, exe, steps=120, decisions=2)
    train(m, w, data, 120, sched=s)
    w.close()
    out = [e for e in events(tmp_path / "r") if e["kind"] in ("applied", "kept", "rejected")]
    assert out and out[0].get("failed"), out
    msg2 = (tmp_path / "r" / "decisions" / "002" / "message.txt").read_text()
    assert "Decision 1 was not answered: the agent call failed" in msg2


# ----- B5: the sandbox reads the interpreter's real installation, never a folder around the run ------------------


def test_interpreter_reached_through_a_link_in_home_does_not_open_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    (home / "bin").mkdir(parents=True)
    link = home / "bin" / "python3"
    link.symlink_to(os.path.realpath(sys.executable))
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(sys, "executable", str(link))
    monkeypatch.setattr(claude.platform, "system", lambda: "Darwin")
    run = home / "project" / "opal_runs" / "20261018-run"
    be = ClaudeBackend(sandbox=True)
    be.setup(run, run / "workspace")
    allow = json.loads((be.control / "settings.json").read_text())["sandbox"]["filesystem"]["allowRead"]
    for secret in (run / "metrics.jsonl", home / ".ssh" / "id_rsa", run.parent / "other-run" / "metrics.jsonl"):
        covering = [a for a in allow if str(secret).startswith(a.rstrip("/") + "/")]
        assert covering == [], (secret, covering)
    assert str(home) not in allow and str(home / "bin") in allow
    real_prefix = os.path.dirname(os.path.dirname(os.path.realpath(sys.executable)))
    assert real_prefix in allow or os.path.realpath(real_prefix) in allow


# ----- B11: the CLI gets an allowlisted environment; login variables stay out of the agent's shell --------------

PLANTED = {
    "DATABASE_URL": "postgres://admin:hunter2@db.internal/prod",
    "AZURE_STORAGE_CONNECTION_STRING": "DefaultEndpointsProtocol=https;AccountName=x;AccountKey=abc==",
    "PIP_INDEX_URL": "https://user:pypi-pass@pypi.internal/simple",
    "DB_PASS": "hunter2",
    "SMTP_PWD": "hunter2",
    "GH_PAT": "ghp_xxx",
    "SENTRY_DSN": "https://key@o1.ingest.sentry.io/1",
    "SSH_AUTH_SOCK": "/tmp/ssh-agent.sock",
    "CODEX_HOME": "/Users/me/.codex",
    "WANDB_API_KEY": "w",
    "HF_TOKEN": "h",
    "LC_ALL": "postgres://u:p@x/y",  # an allowed name with a password in a URL
}


def test_secrets_do_not_reach_the_cli(tmp_path, monkeypatch):
    for k, v in PLANTED.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-xxx")
    monkeypatch.setenv("LANG", "C.UTF-8")
    dump = tmp_path / "env.json"
    exe = fake_cli(tmp_path, mode="envdump", dump=dump)
    be, ws = backend(tmp_path, exe)
    decide(be, ws, tmp_path)
    seen = json.loads(dump.read_text())
    assert sorted(k for k in PLANTED if k in seen) == []
    assert seen["ANTHROPIC_API_KEY"] == "sk-ant-xxx"  # the CLI's own login
    assert seen["LANG"] == "C.UTF-8" and seen["PATH"] == os.environ["PATH"] and seen["HOME"] == os.environ["HOME"]


def test_login_variables_are_unset_in_the_agents_shell(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-xxx")
    monkeypatch.setenv("CLAUDE_CODE_USE_BEDROCK", "1")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "aws-secret")
    monkeypatch.setattr(claude.platform, "system", lambda: "Darwin")
    assert "AWS_SECRET_ACCESS_KEY" in agent_env(_login_prefixes())  # Bedrock needs it in the CLI
    be = ClaudeBackend(sandbox=True)
    be.setup(tmp_path / "run", tmp_path / "ws")
    denied = json.loads((be.control / "settings.json").read_text())["sandbox"]["credentials"]["envVars"]
    names = {d["name"] for d in denied if d["mode"] == "deny"}
    assert {"ANTHROPIC_API_KEY", "AWS_SECRET_ACCESS_KEY"} <= names


# ----- B14 (backend side), D1 ------------------------------------------------------------------------------------


def test_cancel_before_the_cli_starts_returns_an_error(tmp_path):
    exe = fake_cli(tmp_path)
    be, ws = backend(tmp_path, exe)
    be.cancel()
    out = decide(be, ws, tmp_path)
    assert out["returncode"] == -1 and "before the agent started" in out["error"]
    assert calls(tmp_path) == []


def test_budget_is_passed_to_the_cli(tmp_path):
    exe = fake_cli(tmp_path)
    be, ws = backend(tmp_path, exe, max_budget_usd=2.0)
    be.budget_left = 0.75
    decide(be, ws, tmp_path)
    argv = calls(tmp_path)[-1]["argv"]
    assert argv[argv.index("--max-budget-usd") + 1] == "0.7500"
    set_fake(exe, help="--safe-mode")  # an older CLI without the flag: not passed
    old, ws2 = backend(tmp_path, exe, ws="ws2", max_budget_usd=2.0)
    decide(old, ws2, tmp_path)
    assert "--max-budget-usd" not in calls(tmp_path)[-1]["argv"]
    assert "--restricted" not in calls(tmp_path)[-1]["argv"]  # flags --help does not list are not passed


def test_default_timeout_is_an_hour_and_known_to_be_the_default():
    assert ClaudeBackend().timeout == 3600 and not ClaudeBackend().timeout_given
    assert ClaudeBackend(timeout=60).timeout_given


# ----- failed calls are made again while the decision's time allows -----------------------------------------------

SPEND = "You've hit your org's monthly spend limit"
CRASH = "Error: the agent's tool crashed"  # a failure of the agent's own turn, not an unavailable agent


def test_a_failed_call_is_made_again_from_the_programs_the_decision_started_with(tmp_path):
    """A crash (exit 1 within seconds, after a half-made edit) twice, then an answer: the third call answers, from the
    workspace as the decision started; each failure is logged and recorded with the CLI's words."""
    exe = fake_cli(tmp_path, mode="flaky", fails=2, say=CRASH)
    be, ws = backend(tmp_path, exe, retries=None, backoff=0.2, timeout=60)
    (ws / "updater.py").write_text("x = 1\n")
    said = []
    be.log = said.append
    t0 = time.time()
    out = decide(be, ws, tmp_path, name="rec")
    assert out["returncode"] == 0 and out["error"] is None and out["text"] == "done (flaky)", out
    assert 0.6 <= time.time() - t0 < 30  # waited 0.2 s, then 0.4 s
    assert len(calls(tmp_path)) == 3 and (ws / "updater.py").read_text() == "x = 1\n"
    assert len(said) == 2 and all(CRASH in s and "calling again in" in s for s in said), said
    tried = [json.loads(line) for line in open(tmp_path / "rec" / "attempts.jsonl")]
    assert [t["call"] for t in tried] == [1, 2] and all(CRASH in t["error"] for t in tried)
    assert (tmp_path / "rec" / "backend-call1.jsonl").exists() and (tmp_path / "rec" / "backend.jsonl").exists()


def test_retries_off_and_a_cap_on_retries(tmp_path):
    exe = fake_cli(tmp_path, mode="flaky", fails=5, say=CRASH)
    be, ws = backend(tmp_path, exe, backoff=0.1)  # retries=0
    assert decide(be, ws, tmp_path)["returncode"] == 1 and len(calls(tmp_path)) == 1
    be.retries = 2
    out = decide(be, ws, tmp_path)
    assert out["returncode"] == 1 and CRASH in out["error"] and "(3 calls over" in out["error"], out
    assert len(calls(tmp_path)) == 4


def test_calls_that_fail_until_the_deadline_give_one_failed_decision_with_the_error(tmp_path):
    """Every call fails: training waits for the decision's whole time (4 s), and the decision is reported as failed
    once, with the CLI's words; the next message says so."""
    exe = fake_cli(tmp_path, mode="flaky", fails=10_000, say=CRASH)
    opts = dict(binary=str(exe), sandbox=False, retries=None, backoff=0.2, timeout=4)
    m, data, s, w = claude_run(tmp_path, exe, steps=120, decisions=2, backend_options=opts)
    t0 = time.time()
    train(m, w, data, 120, sched=s)
    w.close()
    assert time.time() - t0 >= 3.5
    r1 = json.loads((tmp_path / "r" / "decisions" / "001" / "response.json").read_text())
    assert r1["returncode"] == 1 and CRASH in r1["backend_error"] and " calls over " in r1["backend_error"], r1
    out = [e for e in events(tmp_path / "r") if e.get("k") == 1 and e["kind"] in ("applied", "kept", "rejected")]
    assert len(out) == 1 and out[0].get("failed"), out
    assert len(calls(tmp_path)) >= 6  # a call each 0.2, 0.4, 0.8, ... s, in two decisions
    msg2 = (tmp_path / "r" / "decisions" / "002" / "message.txt").read_text()
    assert "Decision 1 was not answered: the agent call failed" in msg2 and CRASH in msg2


def test_a_login_problem_never_switches_the_agent_off(tmp_path, capsys):
    """Up to 0.3.3 a run whose Claude Code was not logged in stopped asking for good. Now every decision is asked: with
    unavailable_timeout=0 each one fails at once (and says why); by default each one waits (the tests below)."""
    exe = fake_cli(tmp_path, mode="nologin")
    m, data, s, w = claude_run(tmp_path, exe, steps=120, decisions=4, verbose=True)
    train(m, w, data, 120, sched=s)
    w.close()
    asked = [e["k"] for e in events(tmp_path / "r") if e["kind"] == "requested"]
    assert len(asked) >= 3 and len(calls(tmp_path)) >= len(asked)
    err = capsys.readouterr().err
    assert "Claude Code is unavailable (Not logged in \u00b7 Please run /login); training waits for the agent" in err
    assert "run `claude`, then /login" in err and "no more agent calls" not in err


def test_reaching_max_cost_usd_is_said_once_on_stdout(tmp_path, capsys):
    exe = fake_cli(tmp_path)  # each call costs $0.25
    m, data, s, w = claude_run(tmp_path, exe, steps=120, decisions=4, max_cost_usd=0.5, verbose=True)
    train(m, w, data, 120, sched=s)
    w.close()
    assert [e["k"] for e in events(tmp_path / "r") if e["kind"] == "requested"] == [1, 2]
    out = capsys.readouterr().out
    assert out.count("max_cost_usd reached ($0.50 spent, cap $0.50)") == 1 and "no more agent calls" in out, out


@pytest.mark.parametrize("cap, asked", [(0.001, 0), (0.251, 1), (0.49, 1), (0.5, 2), (0.75, 3)])
def test_a_budget_too_small_for_another_call_stops_without_a_backend_failure(tmp_path, capsys, cap, asked):
    exe = fake_cli(tmp_path, mode="budget")  # $0.25 per answer; rejects a smaller per-call budget
    m, data, s, w = claude_run(tmp_path, exe, steps=120, decisions=4, max_cost_usd=cap, verbose=True)
    train(m, w, data, 120, sched=s)
    w.close()
    ev = events(tmp_path / "r")
    assert len(calls(tmp_path)) == asked
    assert len([e for e in ev if e["kind"] == "requested"]) == asked
    assert not any(e.get("failed") for e in ev)
    captured = capsys.readouterr()
    assert captured.out.count("max_cost_usd reached") == 1 and "no more agent calls" in captured.out
    assert "backend failed" not in captured.err and "Reached maximum budget" not in captured.err
    if cap in (0.001, 0.251, 0.49):
        assert "remaining is below" in captured.out


def test_each_decision_says_when_it_is_sent_and_what_became_of_it(tmp_path, capsys):
    exe = fake_cli(tmp_path)
    m, data, s, w = claude_run(tmp_path, exe, steps=120, decisions=2, verbose=True)
    train(m, w, data, 120, sched=s)
    w.close()
    lines = [line for line in capsys.readouterr().err.splitlines() if line.startswith("[opal] decision ")]
    said = [line.split(": ", 1)[1].split(" (")[0].split(" at ")[0] for line in lines]
    assert said == ["sent to the agent", "programs kept"] * 2, lines
    assert lines[0].endswith("(training waits for the answer)"), lines


def test_the_default_agent_is_the_blogs_and_another_one_is_warned_about(tmp_path, capsys):
    exe = fake_cli(tmp_path)
    be, ws = backend(tmp_path, exe)
    assert be.reproduction_note() is None
    decide(be, ws, tmp_path)
    argv = calls(tmp_path)[-1]["argv"]
    assert argv[argv.index("--model") + 1] == "claude-opus-5-5" and argv[argv.index("--effort") + 1] == "high"
    other, _ = backend(tmp_path, exe, run="run2", ws="ws2", model="claude-sonnet-x", effort="low")
    note = other.reproduction_note()
    assert "(https://xingjianbai.com/reactive-optimizers/) were made with claude-opus-5-5 at effort high; with claude-sonnet-x at effort low" in note
    m, data, s, w = claude_run(
        tmp_path, exe, backend_options=dict(binary=str(exe), sandbox=False, retries=0, effort="medium"), verbose=True
    )
    w.close()
    err = capsys.readouterr().err
    assert err.count("may not reproduce them") == 1 and "claude-opus-5-5 at effort medium" in err
    m, data, s, w = claude_run(tmp_path, exe, run="r2", verbose=True)
    w.close()
    assert "may not reproduce" not in capsys.readouterr().err


# ----- an unavailable agent is waited for, not skipped -----------------

NOLOGIN = "Not logged in · Please run /login"
SPEND_RESET = (
    "You've hit your org's monthly spend limit · run /usage-credits to ask your admin for a higher limit · "
    "your session limit resets 1:10am (America/New_York)"
)


@pytest.fixture
def quick_waits(monkeypatch):
    from opal.backends import unavailable

    monkeypatch.setattr(unavailable, "RECHECK", 0.2)
    monkeypatch.setattr(unavailable, "RECHECK_MAX", 0.4)
    monkeypatch.setattr(unavailable, "REMIND", 0.5)


def outage_run(tmp_path, name, fails, say=NOLOGIN, steps=120, **opts):
    """A blocking run whose agent's first `fails` calls fail with `say` (no edit, as an unavailable CLI does); every call
    that runs answers with SGD_UPDATER."""
    from test_core import SGD_UPDATER

    d = tmp_path / name
    d.mkdir()
    exe = fake_cli(d, mode="flaky", fails=fails, half=0, say=say, updater=SGD_UPDATER, msglog=d / "messages.jsonl")
    opts = dict(dict(binary=str(exe), sandbox=False, retries=0), **opts)
    m, data, s, w = claude_run(d, exe, steps=steps, decisions=2, backend_options=opts, verbose=True)
    train(m, w, data, steps, sched=s)
    w.close()
    messages = [json.loads(line) for line in open(d / "messages.jsonl")]
    return torch.cat([p.detach().flatten() for p in m.parameters()]), events(d / "r"), messages



@pytest.mark.parametrize("say", [NOLOGIN, SPEND_RESET], ids=["not-logged-in", "spend-limit"])
def test_an_unavailable_agent_is_waited_for_and_the_run_is_the_same(tmp_path, capsys, quick_waits, say):
    """The calls of decision 1 fail (not logged in; a spend limit with a reset time hours away) four times, then the
    agent is back: the decision is answered, not failed, the agent's message is the same at every call (no note), and
    the run's parameters are bitwise those of a run whose agent never went away."""
    calm, calm_events, _ = outage_run(tmp_path, "calm", fails=0)
    capsys.readouterr()
    params, evs, messages = outage_run(tmp_path, "outage", fails=4, say=say)
    assert torch.equal(params, calm)
    first = [e for e in evs if e.get("k") == 1 and e["kind"] in ("applied", "kept", "rejected")]
    assert [e["kind"] for e in first] == ["applied"] and not first[0].get("failed") and first[0]["unavailable_seconds"] > 1
    assert [e["kind"] for e in evs if e["kind"] in ("applied", "kept")] == [
        e["kind"] for e in calm_events if e["kind"] in ("applied", "kept")
    ]
    assert len(messages) == 6 and len(set(messages[:5])) == 1  # four failed calls and the answer of decision 1, then 2
    assert not any("stopped before it finished" in m for m in messages)
    err = capsys.readouterr().err
    start = f"[opal] decision 1: Claude Code is unavailable ({say}); training waits for the agent and tries again"
    assert err.count(start) == 1 and "To go on: " in err
    assert "[opal] decision 1: still waiting for Claude Code (" in err  # a reminder (every 0.5 s here)
    assert err.count("[opal] decision 1: Claude Code is available again after ") == 1
    assert "s waiting for the agent to be available" in err


def test_unavailable_timeout_fails_the_decision_and_the_next_one_is_still_asked(tmp_path, capsys, quick_waits):
    t0 = time.time()
    params, evs, messages = outage_run(tmp_path, "t", fails=10_000, unavailable_timeout=0.5)
    assert time.time() - t0 < 30
    out = [e for e in evs if e["kind"] in ("applied", "kept", "rejected")]
    assert [e["k"] for e in out] == [1, 2] and all(e.get("failed") for e in out), out
    assert all(e["unavailable_seconds"] >= 0.5 for e in out)
    r1 = json.loads((tmp_path / "t" / "r" / "decisions" / "001" / "response.json").read_text())
    assert NOLOGIN in r1["backend_error"]
    assert capsys.readouterr().err.count("stayed unavailable for") == 2


def test_a_genuine_agent_error_still_fails_at_once(tmp_path):
    """A crash in the agent's own turn is not an unavailable agent: with the default (no unavailable_timeout) it fails as
    before, within seconds."""
    exe = fake_cli(tmp_path, mode="sigkill")
    opts = dict(binary=str(exe), sandbox=False, retries=0)
    m, data, s, w = claude_run(tmp_path, exe, steps=120, decisions=2, backend_options=opts)
    t0 = time.time()
    train(m, w, data, 120, sched=s)
    w.close()
    assert time.time() - t0 < 30
    out = [e for e in events(tmp_path / "r") if e["kind"] in ("applied", "kept", "rejected")]
    assert out and all(e.get("failed") and "unavailable_seconds" not in e for e in out), out


def test_without_waiting_training_goes_on_and_the_line_says_so(tmp_path, quick_waits):
    exe = fake_cli(tmp_path, mode="flaky", fails=2, half=0, say=NOLOGIN)
    be, ws = backend(tmp_path, exe, unavailable_timeout=None)
    said = []
    be.log, be.training_waits = said.append, False
    out = decide(be, ws, tmp_path)
    assert out["returncode"] == 0 and out["error"] is None and out["unavailable_seconds"] > 0, out
    assert said[0].startswith(f"Claude Code is unavailable ({NOLOGIN}); training continues with the current programs")
    assert said[-1].startswith("Claude Code is available again after")


def test_the_reset_time_of_a_limit_message():
    from opal.backends.unavailable import RESET_MARGIN, classify, seconds_to_reset

    assert classify([NOLOGIN])[0] == "login" and classify([SPEND_RESET])[0] == "limit"
    assert classify(["API Error: 529 overloaded"])[0] == "down" and classify(["Reached maximum budget ($0.01)"]) is None
    assert classify(["killed by signal 9 (SIGKILL)"]) is None and classify(["x" * 600 + " rate limit"]) is None
    import datetime
    from zoneinfo import ZoneInfo

    now = datetime.datetime(2026, 10, 20, 21, 4, tzinfo=ZoneInfo("America/New_York")).timestamp()
    assert seconds_to_reset(SPEND_RESET, now) == pytest.approx(4 * 3600 + 6 * 60 + RESET_MARGIN)
    assert seconds_to_reset("resets 3pm", now) is not None and seconds_to_reset(NOLOGIN, now) is None
