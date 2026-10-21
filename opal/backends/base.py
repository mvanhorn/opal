from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

# The agent CLI's environment is built from an allowlist: what a CLI needs to run and log in, and what the training
# interpreter needs to start in the agent's shell. Everything else of the user's (database URLs, tokens in oddly named
# variables, ssh agents, other tools' homes) stays out, since the agent's shell could print it into the provider's
# context.
_ALLOWED = frozenset(
    (
        "PATH",
        "HOME",
        "USER",
        "LOGNAME",
        "SHELL",
        "LANG",
        "LANGUAGE",
        "TERM",
        "TZ",
        "TMPDIR",
        "NO_COLOR",
        "COLORTERM",
        "__CF_USER_TEXT_ENCODING",
        # network access of the CLI itself (the agent's shell has none)
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "no_proxy",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "NODE_EXTRA_CA_CERTS",
        "REQUESTS_CA_BUNDLE",
        "CURL_CA_BUNDLE",
        # the training interpreter in the agent's shell (module systems, conda, thread caps)
        "LD_LIBRARY_PATH",
        "DYLD_LIBRARY_PATH",
        "PYTHONPATH",
        "PYTHONHOME",
        "PYTHONNOUSERSITE",
        "VIRTUAL_ENV",
        "CONDA_PREFIX",
        "CONDA_DEFAULT_ENV",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
    )
)
_ALLOWED_PREFIXES = ("LC_", "XDG_")
# a kept variable whose value holds a password in a URL (user:pass@host) is still dropped, proxies excepted
_URL_PASSWORD = re.compile(r"://[^/\s:@]+:[^/\s@]+@")
_PROXIES = frozenset(n for n in _ALLOWED if n.lower().endswith("_proxy"))
_SECRET = re.compile(r"sk-[A-Za-z0-9_\-]{8,}|Bearer\s+\S+")


def error_tail(text, n=300) -> str:
    """The end of an agent CLI's error text on one line, for logs and records: at most n characters, with anything
    shaped like an API key or a bearer token masked."""
    return _SECRET.sub("<hidden>", " ".join(str(text or "").split()))[-n:]

# Started next to each agent process: stops the agent (its whole process group) if the training process dies
# without close() (SIGTERM from a scheduler, SIGKILL, a crash), so no agent keeps spending after its run is gone.
_WATCHDOG = """
import os, signal, sys, time
parent, child = int(sys.argv[1]), int(sys.argv[2])
def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False
while alive(child):
    if not alive(parent):
        try:
            os.killpg(child, signal.SIGTERM)
            time.sleep(5)
            os.killpg(child, signal.SIGKILL)
        except OSError:
            pass
        break
    time.sleep(2)
"""


class Cancelled(RuntimeError):
    """The run ended (or cancelled this decision) before the agent started."""


def agent_env(keep_prefixes=(), extra=None):
    """The environment for an agent CLI: an allowlist of the user's variables (PATH, HOME, locale, proxies and CA
    files, what the Python interpreter needs) plus the backend's own login variables, kept by prefix. Variables the
    agent's shell could otherwise print into the provider's context (keys, tokens, database URLs, ssh agents) are
    left out."""
    env = {}
    for k, v in os.environ.items():
        if k.startswith(tuple(keep_prefixes)):
            env[k] = v
        elif (k in _ALLOWED or k.startswith(_ALLOWED_PREFIXES)) and (k in _PROXIES or not _URL_PASSWORD.search(v)):
            env[k] = v
    env.update(extra or {})
    return env


def kill_group(pgid, grace=3.0):
    """SIGTERM, then SIGKILL, to a process group; quiet if it is gone already."""
    try:
        os.killpg(pgid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return
    t0 = time.time()
    while time.time() - t0 < grace:
        try:
            os.killpg(pgid, 0)
        except (ProcessLookupError, PermissionError):
            return
        time.sleep(0.05)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


class Backend:
    name = "backend"
    log = None  # fn(text): where a backend reports what it does about a failed call (the run's or the broker's log)

    def setup(self, run_dir: Path, workspace: Path) -> None:
        """Called before the first decision (and again after a resume)."""

    def preflight(self) -> None:
        """Called when the run starts: fail now, not at the first decision, if the agent cannot run."""

    def decide(self, workspace: Path, message: str, system: str | None, record_dir: Path) -> dict:
        raise NotImplementedError

    def state(self) -> dict:
        return {}

    def restore(self, state: dict) -> None:
        pass

    def finish(self, interrupted=False) -> None:
        """The run ended, or stopped early without close() (a mailbox tells its broker)."""

    def _run(self, cmd, input, stdout, cwd, timeout, env=None):
        """Run the agent CLI in its own process group and wait for the CLI itself, at most `timeout` seconds (then
        subprocess.TimeoutExpired). Whatever is left of its process group afterwards (a background job of the agent's
        shell, a process holding its output open) is stopped, after a normal exit too; cancel() and a watchdog (if
        this process dies) stop the whole group as well. stderr goes to a temporary file, so no descendant holding it
        open can make this wait."""
        if getattr(self, "cancelled", False):
            raise Cancelled("the run ended before the agent started")
        with tempfile.TemporaryFile(mode="w+") as err:
            proc = self.proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=stdout,
                stderr=err,
                text=True,
                cwd=cwd,
                env=env,
                start_new_session=True,
            )
            try:
                subprocess.Popen(
                    [sys.executable, "-c", _WATCHDOG, str(os.getpid()), str(proc.pid)],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
                if getattr(self, "cancelled", False):  # cancel() came while the CLI was starting
                    kill_group(proc.pid)

                def feed():
                    try:
                        proc.stdin.write(input or "")
                        proc.stdin.close()
                    except (BrokenPipeError, OSError, ValueError):
                        pass

                threading.Thread(target=feed, name="opal-agent-stdin", daemon=True).start()
                try:
                    proc.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    kill_group(proc.pid)
                    proc.wait()
                    raise
            finally:
                kill_group(proc.pid, grace=1.0)  # the CLI's leftovers, also after a normal exit
                if self.proc is proc:
                    self.proc = None
            err.seek(0)
            text = err.read()
        return subprocess.CompletedProcess(cmd, proc.returncode, None, text)

    def _kill(self):
        p = getattr(self, "proc", None)
        if p is not None:
            kill_group(p.pid)

    def cancel(self) -> None:
        self.cancelled = True
        self._kill()


class KeepBackend(Backend):
    """No agent: every decision keeps the programs. For smoke tests and for measuring the wrapper's own cost."""

    name = "keep"

    def decide(self, workspace, message, system, record_dir):
        return dict(text="kept (backend='keep')", cost_usd=0.0, returncode=0, error=None, seconds=0.0)


class CallableBackend(Backend):
    """fn(workspace: Path, message: str, system: str | None) -> str | None. For tests and custom agents."""

    name = "callable"

    def __init__(self, fn):
        self.fn = fn

    def decide(self, workspace, message, system, record_dir):
        t0 = time.time()
        text = self.fn(Path(workspace), message, system)
        return dict(text=text or "", cost_usd=0.0, returncode=0, error=None, seconds=time.time() - t0)
