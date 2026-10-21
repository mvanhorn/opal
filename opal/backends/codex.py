"""OPAL Sol: isolated Codex app-server, one durable thread per run.

Only OPAL's new message is sent on each turn. The subprocess transport is short lived; thread/resume loads the
provider's durable conversation. See _codex_client.py for protocol handling and the fail-closed rollout audit.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

from .._version import __version__
from ._privacy import redact_secrets

from .base import Cancelled, agent_env
from .claude import ClaudeBackend, NO_GPU, SANDBOX_NOTE, _contains, _python_paths

DEFAULT_MODEL, DEFAULT_EFFORT = "gpt-6.1-sol", "high"
TOKEN_FIELDS = ("input_tokens", "cached_input_tokens", "output_tokens", "reasoning_output_tokens")


def isolated_home(value=None):
    """Never borrow the coding agent's home or login, including via a symlink/hardlink."""
    home = Path(value).expanduser() if value is not None else Path.home() / ".codex-opal"
    default = (Path.home() / ".codex").resolve()
    home = home.resolve()
    if home == default or _contains(default, home) or _contains(home, default):
        raise RuntimeError("opal: Codex requires its own CODEX_HOME, outside ~/.codex; use ~/.codex-opal")
    auth = home / "auth.json"
    if auth.is_symlink():
        raise RuntimeError("opal: isolated Codex auth.json must not be a symlink; log in separately")
    original = default / "auth.json"
    if auth.exists() and original.exists() and os.path.samefile(auth, original):
        raise RuntimeError("opal: do not link/copy the default Codex login; log in separately in the isolated home")
    return home


class CodexBackend(ClaudeBackend):
    """Reuse Claude's retry/wait deadline policy, without changing the Claude backend or its defaults.

    max_tokens is a soft run cap on input + output tokens (cached input and reasoning are subsets). A turn may
    exceed it; no subsequent inference starts. Dollar budgets cannot be enforced by this backend.
    """

    name = "codex"

    def __init__(
        self,
        model=DEFAULT_MODEL,
        effort=DEFAULT_EFFORT,
        binary="codex",
        codex_home=None,
        timeout=None,
        retries=None,
        backoff=15.0,
        backoff_max=600.0,
        unavailable_timeout=None,
        max_tokens=None,
    ):
        if effort not in ("high", "xhigh"):
            raise ValueError("opal: Codex effort must be 'high' or 'xhigh'")
        if max_tokens is not None and (isinstance(max_tokens, bool) or int(max_tokens) != max_tokens or max_tokens < 1):
            raise ValueError("opal: Codex max_tokens must be a positive integer")
        super().__init__(
            model=model,
            effort=effort,
            binary=binary,
            sandbox=False,
            timeout=timeout,
            retries=retries,
            backoff=backoff,
            backoff_max=backoff_max,
            unavailable_timeout=unavailable_timeout,
        )
        self.codex_home = isolated_home(codex_home)
        self.max_tokens = max_tokens
        self.tokens = dict.fromkeys(TOKEN_FIELDS, 0)
        self.provider_total = {}
        self.rollout = None
        self.audit_offset = 0
        self.poisoned = None
        self.sandbox = True

    def preflight(self):
        isolated_home(self.codex_home)
        if not shutil.which(self.binary):
            raise RuntimeError(f"opal: Codex CLI ({self.binary!r}) not found; install Codex and log in in its own home")
        if platform.system() not in ("Darwin", "Linux"):
            raise RuntimeError("opal: Codex backend requires the macOS or Linux OS sandbox")
        p = subprocess.run(
            [self.binary, "--version"],
            capture_output=True,
            text=True,
            timeout=30,
            env=agent_env(extra={"CODEX_HOME": str(self.codex_home)}),
        )
        if p.returncode:
            raise RuntimeError("opal: Codex CLI could not start: " + redact_secrets(p.stderr)[-300:])
        self.version = p.stdout.strip()

    def login_status(self):
        """No inference and no credential content returned. Missing login is waited for by decide()."""
        isolated_home(self.codex_home)
        p = subprocess.run(
            [self.binary, "login", "status"],
            capture_output=True,
            text=True,
            timeout=30,
            env=agent_env(extra={"CODEX_HOME": str(self.codex_home)}),
        )
        return p.returncode == 0

    def reproduction_note(self):
        return (
            f"OPAL Sol uses {self.model} at effort {self.effort}; Claude remains the default backend. "
            "Codex reports tokens, not USD; max_cost_usd does not cap this backend."
        )

    def setup(self, run_dir, workspace):
        self.run_dir, self.workspace = Path(run_dir).resolve(), Path(workspace).resolve()
        self.control = self.run_dir / "control"
        self.control.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.flags = self._config()
        (self.control / "codex-config.json").write_text(json.dumps(self.flags, indent=1))

    def _config(self):
        py, dropped = _python_paths(protect=(str(self.workspace), str(self.run_dir)))
        fs = {":root": "deny", ":minimal": "read", ":tmpdir": "deny", ":slash_tmp": "deny"}
        fs.update({p: "read" for p in py})
        # macOS must be able to resolve a symlinked Python installation's directory entry. Codex canonicalizes
        # permission paths; grant that installation parent, while denying its other children explicitly.
        for p in py:
            if Path(p).is_symlink() and Path(p).is_dir():
                parent = Path(p).parent
                if any(_contains(parent, x) for x in (self.workspace, self.run_dir, Path.home(), self.codex_home)):
                    continue
                fs[str(parent)] = "read"
                allowed = {Path(x).resolve() for x in py}
                for child in parent.iterdir():
                    if child.resolve() not in allowed:
                        fs[str(child)] = "deny"
        fs[str(self.workspace)] = "write"
        fs[str(self.control)] = "deny"
        fs[str(self.codex_home)] = "deny"
        shell = agent_env(extra=NO_GPU)
        shell["PATH"] = os.pathsep.join(
            dict.fromkeys([str(Path(sys.executable).parent), "/usr/bin", "/bin", "/usr/sbin", "/sbin"])
        )
        shell["TMPDIR"] = str(self.workspace / "scratch")
        flags = {
            "approval_policy": "never",
            "default_permissions": "opal",
            "permissions.opal.extends": ":workspace",
            "permissions.opal.filesystem": fs,
            "permissions.opal.network.enabled": False,
            "model_provider": "openai",
            "model": self.model,
            "model_reasoning_effort": self.effort,
            "include_environment_context": False,
            "include_permissions_instructions": False,
            "include_apps_instructions": False,
            "include_collaboration_mode_instructions": False,
            "project_doc_max_bytes": 0,
            "skills.include_instructions": False,
            "web_search": "disabled",
            "tools.update_plan.enabled": False,
            "shell_environment_policy.inherit": "none",
            "shell_environment_policy.set": shell,
        }
        for feature in (
            "apps",
            "plugins",
            "hooks",
            "memories",
            "multi_agent",
            "multi_agent_v2",
            "shell_snapshot",
            "shell_snapshot_v2",
            "browser_use",
            "computer_use",
            "image_generation",
            "view_image",
            "goals",
            "tool_suggest",
            "current_time_reminder",
            "request_permissions_tool",
            "sleep_tool",
            "standalone_web_search",
            "unbounded_connection_retries",
            "code_mode",
            "in_app_browser",
            "skill_search",
            "workspace_dependencies",
        ):
            flags[f"features.{feature}"] = False
        # The model catalog uses code_mode_only: exec coordinates sandboxed tools in an OS/network-free V8 isolate.
        flags["features.code_mode_host"] = True
        # Preserve model metadata, except its generic multi-agent instruction selector. No account data is copied.
        cache = self.codex_home / "models_cache.json"
        if cache.exists():
            catalog = json.loads(cache.read_text())
            selected = next((m for m in catalog.get("models", []) if m.get("slug") == self.model), None)
            if selected:
                target = self.control / "codex-model-catalog.json"
                target.write_text(json.dumps({"models": [dict(selected, multi_agent_version=None)]}, indent=1))
                flags["model_catalog_json"] = str(target)
        if dropped:
            self._say("Python read paths containing the run or home were excluded: " + ", ".join(dropped))
        return flags

    def _budget(self):
        return None  # Codex gives no dollar cost; never pretend the caller's USD cap is enforced.

    def _say(self, text):
        text = text.replace("Claude Code", "Codex CLI").replace(
            "run `claude`, then /login", "run CODEX_HOME=~/.codex-opal codex login --device-auth"
        )
        super()._say(text)

    def decide(self, workspace, message, system, record_dir):
        self._decision_usage = dict.fromkeys(TOKEN_FIELDS, 0)
        self._usage_reported = False
        # Match the exact OPAL-specific text ClaudeBackend._cmd appends, including its existing shell note.
        # Do not edit the Claude note just to adapt its provider-specific bookkeeping sentence to Codex.
        backend_system = system + "\n\n" + SANDBOX_NOTE if system else ""
        (self.control / "codex-system.md").write_text(backend_system)
        out = super().decide(workspace, message, backend_system, record_dir)
        out.update(
            usage=dict(self._decision_usage) if self._usage_reported else None,
            session=self.session,
            rollout=self.rollout,
        )
        return out

    def _call(self, workspace, message, system, record_dir, timeout):
        t0 = time.time()
        if self.poisoned:
            return dict(text="", cost_usd=None, returncode=-1, error=self.poisoned, seconds=0, final=True)
        if (
            self.max_tokens is not None
            and self.tokens["input_tokens"] + self.tokens["output_tokens"] >= self.max_tokens
        ):
            return dict(
                text="",
                cost_usd=None,
                returncode=-1,
                error="Codex max_tokens reached (soft run cap)",
                seconds=0,
                final=True,
            )
        if self.session_cwd and self.session_cwd != str(Path(workspace).resolve()):
            return dict(
                text="", cost_usd=None, returncode=-1, error="Codex thread workspace changed", seconds=0, final=True
            )
        payload = dict(
            client_version=__version__,
            binary=self.binary,
            home=str(isolated_home(self.codex_home)),
            flags=self.flags,
            workspace=str(Path(workspace).resolve()),
            model=self.model,
            effort=self.effort,
            stderr=str(self.control / "codex-stderr.txt"),
            session=self.session,
            system=system or "",
            message=message,
            rollout=self.rollout,
            audit_offset=self.audit_offset,
            provider_total=self.provider_total,
            timeout=max(0.1, timeout - 4),
        )
        transcript = Path(record_dir) / "backend.jsonl"
        failed = None
        with transcript.open("w") as output:
            try:
                p = self._run(
                    [sys.executable, str(Path(__file__).with_name("_codex_client.py"))],
                    input=json.dumps(payload),
                    stdout=output,
                    cwd=workspace,
                    timeout=timeout,
                    env=agent_env(extra={"CODEX_HOME": str(self.codex_home), **NO_GPU}),
                )
                if p.returncode:
                    failed = f"Codex transport exited {p.returncode}: {p.stderr[-500:]}"
            except subprocess.TimeoutExpired:
                failed = f"timed out after {timeout:.0f} s"
            except Cancelled as exc:
                failed = str(exc)
        result = None
        for line in transcript.read_text().splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if event.get("type") == "opal_thread":
                self.session = event["session"]
                self.session_cwd = payload["workspace"]
                self.rollout = event.get("rollout")
                self.started = True
            if event.get("type") == "opal_result":
                result = event["result"]
        if result is None:
            # A killed turn may have appended context; never replay an observation into uncertain history.
            self.poisoned = failed or "Codex transport ended without a result; thread history is uncertain"
            return dict(
                text="", cost_usd=None, returncode=-1, error=self.poisoned, seconds=time.time() - t0, final=True
            )
        self.audit_offset = result.pop("audit_offset", self.audit_offset)
        self.provider_total = result.pop("provider_total", self.provider_total)
        self._usage_reported = self._usage_reported or bool(result.get("usage"))
        for k in TOKEN_FIELDS:
            n = result.get("usage", {}).get(k, 0)
            self.tokens[k] += n
            self._decision_usage[k] += n
        audit = result.pop("audit", None)
        if audit:
            (Path(record_dir) / "codex-audit.json").write_text(json.dumps(audit, indent=1))
        if result.pop("poisoned", False):
            self.poisoned = result.get("error") or "Codex isolation audit failed"
            result["final"] = True
            result.pop("unavailable", None)
        return dict(result, cost_usd=None, seconds=time.time() - t0)

    def _worked(self, transcript, workspace, start):
        # The client reverts failed turns before a retry, and records whether any tool actually ran.
        return '"worked": true' in Path(transcript).read_text() or super()._worked(transcript, workspace, start)

    def state(self):
        return dict(
            session=self.session,
            started=self.started,
            cwd=self.session_cwd,
            rollout=self.rollout,
            audit_offset=self.audit_offset,
            provider_total=self.provider_total,
            tokens=dict(self.tokens),
            poisoned=self.poisoned,
        )

    def restore(self, state):
        self.session, self.started, self.session_cwd = (
            state.get("session"),
            state.get("started", False),
            state.get("cwd"),
        )
        self.rollout, self.audit_offset = state.get("rollout"), state.get("audit_offset", 0)
        self.provider_total = state.get("provider_total", {})
        self.tokens = dict(state.get("tokens") or dict.fromkeys(TOKEN_FIELDS, 0))
        self.poisoned = state.get("poisoned")
