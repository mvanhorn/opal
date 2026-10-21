"""The real Claude Code CLI through ClaudeBackend, against a local mock of the Messages API: no model is called and
nothing is billed (the CLI gets a placeholder API key, a fresh config folder and ANTHROPIC_BASE_URL on 127.0.0.1).
The mock can also answer with Bash tool calls, to see what the agent's sandboxed shell sees.

Opt-in, since it needs Claude Code installed and starts it a few times: OPAL_REAL_CLI=1 pytest tests/test_claude_real_cli.py
Checked with Claude Code 2.1.302 on macOS.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from opal.backends.claude import ClaudeBackend

pytestmark = pytest.mark.skipif(
    os.environ.get("OPAL_REAL_CLI") != "1" or shutil.which("claude") is None,
    reason="opt-in: OPAL_REAL_CLI=1 and the claude CLI on PATH",
)


def _sse(kind, data):
    return f"event: {kind}\ndata: {json.dumps(data)}\n\n".encode()


class _Mock(BaseHTTPRequestHandler):
    bodies: list = []
    commands: list = []  # Bash commands to run, one tool call per main-agent request, then a text answer
    served = 0

    def log_message(self, *a):
        pass

    def do_POST(self):
        raw = self.rfile.read(int(self.headers.get("content-length") or 0))
        try:
            body = json.loads(raw)
        except ValueError:
            body = {}
        cls = type(self)
        cls.bodies.append(body)  # bodies only: request headers carry credentials
        main = "Bash" in [t.get("name") for t in body.get("tools") or []]
        i = cls.served
        if main:
            cls.served += 1
        msg = dict(
            id=f"msg_{i}",
            type="message",
            role="assistant",
            model=body.get("model", "m"),
            content=[],
            stop_reason=None,
            stop_sequence=None,
            usage=dict(input_tokens=10, output_tokens=1),
        )
        if main and i < len(cls.commands):
            cmd = json.dumps(dict(command=cls.commands[i], description="check"))
            block, delta, stop = (
                dict(type="tool_use", id=f"toolu_{i:04d}", name="Bash", input={}),
                dict(type="input_json_delta", partial_json=cmd),
                "tool_use",
            )
        else:
            block, delta, stop = dict(type="text", text=""), dict(type="text_delta", text="ok"), "end_turn"
        if not body.get("stream"):
            full = (
                dict(block, input=json.loads(delta["partial_json"])) if stop == "tool_use" else dict(block, text="ok")
            )
            msg.update(content=[full], stop_reason=stop)
            out = json.dumps(msg).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)
            return
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.end_headers()
        for kind, data in (
            ("message_start", dict(type="message_start", message=msg)),
            ("content_block_start", dict(type="content_block_start", index=0, content_block=block)),
            ("content_block_delta", dict(type="content_block_delta", index=0, delta=delta)),
            ("content_block_stop", dict(type="content_block_stop", index=0)),
            ("message_delta", dict(type="message_delta", delta=dict(stop_reason=stop), usage=dict(output_tokens=1))),
            ("message_stop", dict(type="message_stop")),
        ):
            self.wfile.write(_sse(kind, data))
        self.wfile.flush()


@pytest.fixture
def mock_api(tmp_path, monkeypatch):
    _Mock.bodies, _Mock.commands, _Mock.served = [], [], 0
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Mock)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setenv("ANTHROPIC_BASE_URL", f"http://127.0.0.1:{srv.server_address[1]}")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-placeholder-not-a-key")
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude-config"))
    monkeypatch.setenv("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", "1")
    yield _Mock
    srv.shutdown()


def _tool_results(bodies):
    out = {}
    for b in bodies:
        for m in b.get("messages", []):
            for c in m.get("content") if isinstance(m.get("content"), list) else []:
                if c.get("type") == "tool_result":
                    t = c.get("content")
                    out[c["tool_use_id"]] = "".join(x.get("text", "") for x in t) if isinstance(t, list) else str(t)
    return out


def _setup(tmp_path, sandbox=False):
    outer = tmp_path / "outer"
    outer.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "probe-branch-xyz"], cwd=outer, check=True)
    subprocess.run(["git", "config", "user.name", "Probe User Name"], cwd=outer, check=True)
    (outer / "UNTRACKED_PROBE_NAME.txt").write_text("x")
    run = outer / "opal_runs" / "r1"
    ws = run / "workspace"
    ws.mkdir(parents=True)
    (ws / "updater.py").write_text("x = 1\n")
    (run / "metrics.jsonl").write_text('{"test_acc": 0.9123}\n')
    be = ClaudeBackend(timeout=180, sandbox="auto" if sandbox else False)
    be.preflight()
    be.setup(run, ws)
    return be, run, ws


def _decide(be, run, ws, k, message="say ok"):
    rec = run / "decisions" / f"{k:03d}"
    rec.mkdir(parents=True, exist_ok=True)
    return be.decide(ws, message, "You are a test.", rec)


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_no_git_status_of_the_enclosing_repository(tmp_path, mock_api):
    be, run, ws = _setup(tmp_path)
    out = _decide(be, run, ws, 1)
    assert out["returncode"] == 0, out
    sent = json.dumps(mock_api.bodies)
    assert mock_api.bodies, "the CLI made no request"
    for leak in ("git status at the start", "probe-branch-xyz", "Probe User Name", "UNTRACKED_PROBE_NAME"):
        assert leak not in sent, leak


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_a_session_id_already_used_is_resumed(tmp_path, mock_api):
    """B1 with the real CLI: the checkpoint said started=False, but the conversation exists."""
    be, run, ws = _setup(tmp_path)
    assert _decide(be, run, ws, 1)["returncode"] == 0
    be.started = False
    out = _decide(be, run, ws, 2)
    assert out["returncode"] == 0 and out["error"] is None, out
    assert be.started
    assert _decide(be, run, ws, 3)["returncode"] == 0


@pytest.mark.skipif(platform.system() != "Darwin" or shutil.which("git") is None, reason="macOS sandbox")
def test_sandboxed_shell_sees_no_secrets_and_not_the_run_folder(tmp_path, mock_api, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgres://u:hunter2@db/x")
    monkeypatch.setenv("GH_PAT", "ghp_probe")
    mock_api.commands = ["env", "cat ../metrics.jsonl", 'python -c "import sys, torch; print(sys.prefix)"']
    be, run, ws = _setup(tmp_path, sandbox=True)
    assert be.sandbox
    out = _decide(be, run, ws, 1)
    assert out["returncode"] == 0, out
    res = _tool_results(mock_api.bodies)
    env = res["toolu_0000"]
    assert "CUDA_VISIBLE_DEVICES=" in env  # the command ran
    for leak in ("ANTHROPIC_API_KEY", "sk-ant-placeholder", "hunter2", "ghp_probe"):
        assert leak not in env, leak
    assert "0.9123" not in res["toolu_0001"] and "not permitted" in res["toolu_0001"]
    assert sys.prefix in res["toolu_0002"], res["toolu_0002"]  # the training interpreter runs, with torch
