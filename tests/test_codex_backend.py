"""Codex protocol, persistence, isolation and OPAL integration, using an offline fake CLI."""

import json
import os
import sys
import threading
import time
from pathlib import Path

import pytest
import torch

import opal
from opal.backends import make_backend
from opal.backends.codex import CodexBackend, isolated_home
from opal.backends.claude import ClaudeBackend
from test_core import SGD_UPDATER, adamw, events, flat, make, train


def fake_cli(tmp_path, **config):
    exe = tmp_path / "fake-codex"
    exe.write_text("#!" + sys.executable + "\n" + Path(__file__).with_name("fake_codex.py").read_text())
    exe.chmod(0o755)
    (tmp_path / "fake.json").write_text(json.dumps(dict(root=str(tmp_path / "provider"), **config)))
    return str(exe)


def backend(tmp_path, **config):
    opts = config.pop("options", {})
    exe = fake_cli(tmp_path, **config)
    b = CodexBackend(binary=exe, codex_home=tmp_path / "isolated", retries=0, unavailable_timeout=0, **opts)
    ws = tmp_path / "run" / "workspace"
    ws.mkdir(parents=True)
    b.setup(ws.parent, ws)
    b.preflight()
    return b, ws


def decide(b, ws, name="001", message="new observation", system="OPAL system"):
    rec = ws.parent / name
    rec.mkdir(exist_ok=True)
    return b.decide(ws, message, system, rec)


def provider_state(b):
    return json.loads((Path(b.rollout).with_suffix(".json")).read_text())


def test_persistent_session_new_messages_and_cumulative_token_deltas(tmp_path):
    b, ws = backend(tmp_path)
    first = decide(b, ws, message="observation one")
    state = b.state()
    second = decide(b, ws, name="002", message="observation two")
    assert first["error"] is None and second["error"] is None, (first, second)
    assert first["session"] == second["session"]
    assert [t["message"] for t in provider_state(b)["turns"]] == ["observation one", "observation two"]
    assert (
        first["usage"]
        == second["usage"]
        == dict(input_tokens=100, cached_input_tokens=40, output_tokens=30, reasoning_output_tokens=20)
    )
    assert second["cost_usd"] is None and b.tokens["input_tokens"] == 200
    assert state["session"] == b.session
    assert state["tokens"]["input_tokens"] == 100  # saved state must not mutate with future calls
    assert json.loads((ws.parent / "001" / "codex-audit.json").read_text())["passed"]


def test_restore_resumes_the_same_thread(tmp_path):
    b, ws = backend(tmp_path)
    assert decide(b, ws)["error"] is None
    other = CodexBackend(binary=b.binary, codex_home=b.codex_home, retries=0, unavailable_timeout=0)
    other.setup(ws.parent, ws)
    other.restore(b.state())
    out = decide(other, ws, name="002", message="second")
    assert out["error"] is None and other.session == b.session
    assert len(provider_state(other)["turns"]) == 2


@pytest.mark.parametrize("mode", ["injected", "duplicate", "hidden"])
def test_context_or_hidden_tool_injection_fails_closed(tmp_path, mode):
    b, ws = backend(tmp_path, mode=mode)
    out = decide(b, ws)
    assert out["error"] and out["returncode"] != 0 and b.poisoned
    calls = (tmp_path / "provider" / "launches").read_text()
    decide(b, ws, name="002")
    assert (tmp_path / "provider" / "launches").read_text() == calls


def test_instruction_sources_rejected_before_inference(tmp_path):
    b, ws = backend(tmp_path, sources=["/private/AGENTS.md"])
    out = decide(b, ws)
    assert "instruction sources" in out["error"] and b.poisoned
    assert provider_state(b)["turns"] == []


def test_auth_is_separate_and_not_linked(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    default = tmp_path / ".codex"
    default.mkdir()
    (default / "auth.json").write_text("not-a-real-login")
    with pytest.raises(RuntimeError, match="own CODEX_HOME"):
        isolated_home(default)
    alias = tmp_path / "alias"
    alias.symlink_to(default)
    with pytest.raises(RuntimeError):
        isolated_home(alias)
    home = tmp_path / "own"
    home.mkdir()
    (home / "auth.json").symlink_to(default / "auth.json")
    with pytest.raises(RuntimeError, match="symlink"):
        isolated_home(home)
    (home / "auth.json").unlink()
    os.link(default / "auth.json", home / "auth.json")
    with pytest.raises(RuntimeError, match="link/copy"):
        isolated_home(home)


def test_workspace_profile_env_and_identical_system_user(tmp_path, monkeypatch):
    monkeypatch.setenv("SECRET_TOKEN", "secret")
    monkeypatch.setenv("CODEX_HOME", str(Path.home() / ".codex"))
    b, ws = backend(tmp_path)
    out = decide(b, ws)
    assert out["error"] is None
    assert b.codex_home == (tmp_path / "isolated").resolve()
    f = b.flags
    fs = f["permissions.opal.filesystem"]
    assert fs[":root"] == "deny" and fs[str(ws)] == "write"
    assert fs[str(b.codex_home)] == "deny" and fs[str(b.control)] == "deny"
    assert not f["permissions.opal.network.enabled"] and f["approval_policy"] == "never"
    shell = f["shell_environment_policy.set"]
    assert "SECRET_TOKEN" not in shell and "CODEX_HOME" not in shell and shell["CUDA_VISIBLE_DEVICES"] == ""
    calls = [json.loads(x) for x in (tmp_path / "provider" / "calls.jsonl").read_text().splitlines()]
    start = next(x["params"] for x in calls if x["method"] == "thread/start")
    claude = ClaudeBackend(sandbox=True)
    claude.setup(ws.parent / "claude-control", ws)
    argv = claude._cmd("OPAL system")
    assert start["baseInstructions"] == argv[argv.index("--append-system-prompt") + 1]
    assert start["developerInstructions"] == ""
    assert start["permissions"] == "opal" and start["runtimeWorkspaceRoots"] == [str(ws)]
    assert start["config"]["mcp_servers"]["planted"]["enabled"] is False


@pytest.fixture
def quick_waits(monkeypatch):
    from opal.backends import unavailable

    monkeypatch.setattr(unavailable, "RECHECK", 0.05)
    monkeypatch.setattr(unavailable, "RECHECK_MAX", 0.1)
    monkeypatch.setattr(unavailable, "REMIND", 0.1)


@pytest.mark.parametrize(
    "options",
    [
        dict(login_fails=2),
        dict(turn_fails=2),
        dict(turn_fails=2, kind="rateLimitExceeded", error="429 too many requests"),
        dict(turn_fails=2, kind="httpConnectionFailed", error="connection refused"),
    ],
)
def test_unavailable_waits_do_not_duplicate_the_observation(tmp_path, quick_waits, options):
    b, ws = backend(tmp_path, **options)
    b.unavailable_timeout = 5
    said = []
    b.log, b.training_waits = said.append, True
    out = decide(b, ws)
    assert out["returncode"] == 0 and out["error"] is None and out["unavailable_seconds"] > 0, out
    assert len(provider_state(b)["turns"]) == 1
    assert provider_state(b)["turns"][0]["message"] == "new observation"
    assert "Codex CLI is unavailable" in said[0] and "available again" in said[-1]


def test_unavailable_timeout_fails_then_later_decision_is_still_possible(tmp_path, quick_waits):
    b, ws = backend(tmp_path, login_fails=1000)
    b.unavailable_timeout = 0.2
    out = decide(b, ws)
    assert out["error"] and out["unavailable_seconds"] >= 0.2 and not b.poisoned
    cfg = tmp_path / "fake.json"
    cfg.write_text(json.dumps(dict(json.loads(cfg.read_text()), login_fails=0)))
    assert decide(b, ws, name="002")["returncode"] == 0


def test_failed_rollback_fails_closed(tmp_path, quick_waits):
    b, ws = backend(tmp_path, turn_fails=1, revert_fails=True)
    b.unavailable_timeout = 5
    out = decide(b, ws)
    assert out["error"] and b.poisoned
    assert (tmp_path / "provider" / "launches").read_text() == "1"


def test_timeout_and_cancel_stop_processes(tmp_path):
    b, ws = backend(tmp_path, mode="slow", seconds=20, options=dict(timeout=0.8))
    t0 = time.time()
    out = decide(b, ws)
    assert out["error"] and out["returncode"] != 0 and time.time() - t0 < 8, out
    assert out["usage"] is None  # no provider report is not evidence of zero token consumption
    b2 = CodexBackend(binary=b.binary, codex_home=b.codex_home, timeout=20, retries=0, unavailable_timeout=0)
    b2.setup(ws.parent, ws)
    timer = threading.Timer(0.4, b2.cancel)
    timer.start()
    out = decide(b2, ws, name="cancelled")
    timer.join()
    assert out["error"] and out["returncode"] != 0 and b2.proc is None


def test_cancel_before_start_and_soft_token_cap(tmp_path):
    b, ws = backend(tmp_path, options=dict(max_tokens=100))
    assert decide(b, ws)["returncode"] == 0
    out = decide(b, ws, name="cap")
    assert "max_tokens reached" in out["error"]
    assert (tmp_path / "provider" / "launches").read_text() == "1"
    b2 = CodexBackend(binary=b.binary, codex_home=b.codex_home, retries=0)
    b2.setup(ws.parent, ws)
    b2.cancel()
    assert "before the agent started" in decide(b2, ws, name="cancel")["error"]


def test_codex_edits_are_applied_by_opal_and_usage_is_recorded(tmp_path):
    exe = fake_cli(tmp_path, updater=SGD_UPDATER, code_mode=True)
    m, data = make()
    opt, sched = adamw(m, 120)
    w = opal.wrap(
        opt,
        m,
        scheduler=sched,
        total_steps=120,
        goal="test",
        backend="codex",
        every=20,
        decisions=2,
        run_dir=tmp_path / "run",
        verbose=False,
        backend_options=dict(binary=exe, codex_home=tmp_path / "isolated", retries=0, unavailable_timeout=0),
    )
    train(m, w, data, 120, sched=sched)
    w.close()
    assert w.version >= 1 and any(e["kind"] == "applied" for e in events(tmp_path / "run"))
    response = json.loads((tmp_path / "run" / "decisions" / "001" / "response.json").read_text())
    assert response["usage"]["input_tokens"] == 100 and response["session"]
    assert w.state_dict()["usage"] == w.worker.usage
    assert (tmp_path / "run" / "programs" / "v001" / "updater.py").read_text() == SGD_UPDATER
    assert "fake Codex edit" in (tmp_path / "run" / "workspace" / "analysis.py").read_text()


def test_keep_still_matches_plain_and_default_stays_claude(tmp_path):
    a, data = make()
    opt, sched = adamw(a, 120)
    train(a, opt, data, 120, sched=sched, pass_loss=False)
    b, _ = make()
    opt, sched = adamw(b, 120)
    w = opal.wrap(
        opt,
        b,
        scheduler=sched,
        total_steps=120,
        goal="test",
        backend="keep",
        decisions=2,
        run_dir=tmp_path / "run",
        verbose=False,
    )
    train(b, w, data, 120, sched=sched)
    w.close()
    assert torch.equal(flat(a), flat(b))
    assert make_backend("claude").model == "claude-opus-5-5"
    assert make_backend("codex").model == "gpt-6.1-sol"
    assert make_backend("codex", effort="xhigh").effort == "xhigh"


def test_compaction_discards_edits_and_stops_further_inference(tmp_path):
    exe = fake_cli(tmp_path, mode="compacted", updater=SGD_UPDATER, code_mode=True)
    plain, data = make()
    opt, sched = adamw(plain, 120)
    train(plain, opt, data, 120, sched=sched, pass_loss=False)
    model, _ = make()
    opt, sched = adamw(model, 120)
    w = opal.wrap(
        opt, model, scheduler=sched, total_steps=120, goal="test", backend="codex", every=20,
        decisions=2, run_dir=tmp_path / "run", verbose=False,
        backend_options=dict(binary=exe, codex_home=tmp_path / "isolated", retries=0, unavailable_timeout=0),
    )
    train(model, w, data, 120, sched=sched)
    w.close()
    assert "compacted persistent history" in w.worker.backend.poisoned
    assert (tmp_path / "provider" / "launches").read_text() == "1"
    assert w.worker.k == 2 and w.version == 0
    assert torch.equal(flat(plain), flat(model))
    assert not any(e["kind"] == "applied" for e in events(tmp_path / "run"))
    response = json.loads((tmp_path / "run" / "decisions" / "001" / "response.json").read_text())
    assert response["files"] is None and "compacted" in response["backend_error"]
    assert response["usage"]["input_tokens"] == 100  # rejected edits do not erase measured consumption


def test_codex_mailbox_preserves_metadata_and_release_budget_forwarding(tmp_path):
    from test_mailbox_backend import mailbox_for, serve, workspace

    exe = fake_cli(tmp_path, updater=SGD_UPDATER, code_mode=True)
    b = CodexBackend(binary=exe, codex_home=tmp_path / "isolated", retries=0, unavailable_timeout=0)
    box = tmp_path / "box"
    mb = mailbox_for(tmp_path, box, timeout=15)
    ws = workspace(tmp_path)
    mb.budget_left = 0.01
    broker = serve(box, b, tmp_path / "broker", run=mb.run_id)
    try:
        first = mb.decide(ws, "first observation", "OPAL system", tmp_path)
        second = mb.decide(ws, "second observation", "OPAL system", tmp_path)
        assert first["error"] is None and second["error"] is None, (first, second)
        assert first["session"] == second["session"] == b.session
        assert second["rollout"] == b.rollout and second["usage"]["input_tokens"] == 100
        assert second["cost_usd"] is None and b.tokens["input_tokens"] == 200
        assert b.budget_left == 0.01 and b._budget() is None
        assert (ws / "updater.py").read_text() == SGD_UPDATER
    finally:
        mb.finish()
        broker.join(10)
    assert not broker.is_alive() and broker.error is None
