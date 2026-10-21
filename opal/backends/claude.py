"""Claude Code, headless, one persistent session (--session-id, then --resume), confined to the workspace.

The CLI runs with customisations off (--safe-mode: no CLAUDE.md, skills, hooks, plugins of the user), only the file
tools and Bash, and Bash inside the OS sandbox with no network and reads limited to the workspace and the Python
installation. Where the OS sandbox cannot start (Linux without bubblewrap and socat, or where bubblewrap may not create
namespaces), Bash is removed and the agent keeps only Read/Edit/Write.
"""

from __future__ import annotations

import glob
import json
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

from .. import programs
from . import unavailable
from .base import Backend, Cancelled, agent_env, error_tail
from ._privacy import redact_secrets

_SYSTEM_READ_MAC = (
    "/System",
    "/bin",
    "/sbin",
    "/usr",
    "/Library/Developer/CommandLineTools",
    "/private/var/select",
    "/dev",
)
_SYSTEM_READ_LINUX = ("/bin", "/sbin", "/usr", "/lib", "/lib64", "/etc", "/proc", "/dev", "/opt")


_OPTIONAL_FLAGS = (
    "--safe-mode",
    "--restricted",
    "--no-chrome",
    "--permission-prompts",
    "--setting-sources",
    "--strict-mcp-config",
    "--effort",
    "--max-budget-usd",
)

SANDBOX_NOTE = (
    "Your Bash tool runs in a sandbox with no network; it can read only this folder and the Python installation of "
    "the training run (the same interpreter, with torch, on CPU only: no GPU is visible to it), for small checks. "
    "Each Bash call may end with "
    "'zsh:1: operation not permitted: /tmp/claude-.../cwd-...' and exit code 1: that is the sandbox's own "
    "bookkeeping, not your command; judge a command by its output."
)
# the agent's shell sees no GPU, whatever machine runs it (the training process, or a broker on a GPU node)
NO_GPU = {"CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": ""}
NO_BASH_NOTE = "You have file tools only (Read, Edit, Write) in this folder; there is no shell on this machine."

# Claude Code 2.1.302's answers when a session id cannot be used (read from the binary, checked with the real CLI)
_IN_USE = "is already in use"
_NOT_FOUND = "No conversation found"
# a usage or spend limit, as Claude Code words it ("You've hit your limit · resets 3pm"): the turn did not run, also
# when the CLI exits 0 with it as its answer
_LIMIT = unavailable.LIMIT
# a failed call of the agent's own turn (an exit code, a crash) is made again after BACKOFF s, then twice as long, ...
# (at most BACKOFF_MAX s per wait), while the decision's time allows; an unavailable agent (no login, a usage or spend
# limit, an overloaded API, no network: backends/unavailable.py) is waited for instead, without a deadline
BACKOFF, BACKOFF_MAX = 15.0, 600.0
MIN_CALL = 30.0  # seconds a call made again must have before the decision's time runs out (at most a tenth of it)
# the agent of the blog's runs; another model or effort is allowed, with a warning when the run starts
DEFAULT_MODEL, DEFAULT_EFFORT = "claude-opus-5-5", "high"
BLOG_URL = "https://xingjianbai.com/reactive-optimizers/"  # the blog post


def _login_prefixes():
    """Environment variables Claude Code needs to log in: its own, and the cloud's when it runs on Bedrock or Vertex."""
    keep = ["ANTHROPIC_", "CLAUDE_"]
    if os.environ.get("CLAUDE_CODE_USE_BEDROCK"):
        keep += ["AWS_"]
    if os.environ.get("CLAUDE_CODE_USE_VERTEX"):
        keep += ["GOOGLE_", "CLOUD_ML_", "VERTEX_"]
    return tuple(keep)


def _shell_denied_env(env):
    """Login variables the CLI needs and the agent's shell must not see: unset for sandboxed commands
    (sandbox.credentials.envVars, mode deny)."""
    login = ("ANTHROPIC_", "AWS_", "GOOGLE_", "CLOUD_ML_", "VERTEX_")
    names = [k for k in env if k.startswith(login) or k == "CLAUDE_CODE_OAUTH_TOKEN"]
    return sorted(n for n in names if not n.startswith("ANTHROPIC_BASE_URL"))


def _real(p):
    return os.path.realpath(p)


def _contains(folder, path):
    """True if `path` is `folder` or lies inside it (real paths)."""
    folder, path = _real(folder), _real(path)
    return path == folder or path.startswith(folder.rstrip(os.sep) + os.sep)


def _looks_like_python_prefix(d):
    """A folder that holds a Python installation or a venv, not just any parent of a bin/ folder (such as $HOME)."""
    return (
        os.path.isfile(os.path.join(d, "pyvenv.cfg"))
        or os.path.isdir(os.path.join(d, "conda-meta"))
        or bool(glob.glob(os.path.join(d, "lib", "python3*")))
        or os.path.isdir(os.path.join(d, "Lib"))
    )


def _python_paths(protect=()):
    """The folders the agent's shell may read to run the training interpreter, with torch: the interpreter's
    prefixes (venv and base installation, as given and resolved), the folder of each link from sys.executable to the
    real binary (the shell starts `python` from the first), the prefix above each of them when it holds a Python
    installation, and the site-packages folders. Never a folder that contains one of `protect` (the workspace, the run
    folder) or the home folder. Returns (paths, dropped)."""
    cand = []
    for p in (sys.prefix, sys.base_prefix, sys.exec_prefix, getattr(sys, "base_exec_prefix", sys.base_prefix)):
        cand += [p, _real(p)]
    p = sys.executable
    for _ in range(20):
        for d in {os.path.dirname(p), os.path.dirname(_real(p))}:
            cand.append(d)
            up = os.path.dirname(d)
            if _looks_like_python_prefix(up):
                cand += [up, _real(up)]
        if not os.path.islink(p):
            break
        p = os.path.join(os.path.dirname(p), os.readlink(p))
    for entry in sys.path:
        if entry and "site-packages" in entry and os.path.isdir(entry):
            cand += [entry, _real(entry)]
    home = os.path.expanduser("~")
    out, dropped = set(), set()
    for c in cand:
        if not c or c == os.sep:
            continue
        if _real(c) == _real(home) or any(_contains(c, x) for x in (*protect, home)):
            dropped.add(c)
        else:
            out.add(c)
    return sorted(out), sorted(dropped - out)


def linux_sandbox_problem():
    """None if Claude Code's Linux sandbox can start here, else why not. Claude Code 2.1.302 needs bubblewrap and
    socat, found on its PATH, and bubblewrap must be allowed to create namespaces; with failIfUnavailable it exits at
    startup otherwise. The check runs bwrap once with the flags Claude Code uses."""
    missing = [n for n in ("bwrap", "socat") if shutil.which(n) is None]
    if missing:
        return " and ".join(missing) + " not found on PATH"
    cmd = [shutil.which("bwrap"), "--new-session", "--die-with-parent", "--ro-bind", "/", "/", "--dev", "/dev"]
    cmd += ["--proc", "/proc", "--unshare-pid", "--unshare-net", "true"]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"bubblewrap could not run ({exc})"
    if p.returncode != 0:
        return "bubblewrap cannot create a sandbox here: " + (error_tail(p.stderr) or f"exit {p.returncode}")
    return None


def _sandbox_available() -> bool:
    """A quick check (no process started): macOS always; Linux when bwrap and socat are on PATH. preflight() runs the
    full check."""
    if platform.system() == "Darwin":
        return True
    return platform.system() == "Linux" and all(shutil.which(n) for n in ("bwrap", "socat"))


class ClaudeBackend(Backend):
    name = "claude"

    def __init__(
        self,
        model=DEFAULT_MODEL,
        effort=DEFAULT_EFFORT,
        sandbox="auto",
        timeout=None,
        binary="claude",
        extra_args=(),
        shell_timeout=None,
        max_budget_usd=None,
        retries=None,
        backoff=BACKOFF,
        backoff_max=BACKOFF_MAX,
        unavailable_timeout=None,
    ):
        self.model, self.effort, self.binary = model, effort, binary
        # seconds an unavailable agent is waited for before the decision fails (None: no deadline)
        self.unavailable_timeout = unavailable_timeout
        self.training_waits = None  # set by the run before each decision: whether training waits for it (the wait line)
        self.on_unavailable = None  # fn(reason or None), called when a wait for availability starts and ends (broker)
        self.retries = retries  # calls made again after a failed one: None while the decision's time allows; 0: none
        self.backoff, self.backoff_max = float(backoff), float(backoff_max)
        self._spent = 0.0  # what this decision's calls have cost so far: the next call's budget is what is left
        self.timeout_given = timeout is not None  # a broker may then lengthen it to what the agent was told
        self.timeout = 3600 if timeout is None else timeout
        self.shell_timeout = shell_timeout  # seconds per shell command of the agent (Claude Code's own cap otherwise)
        self.max_budget_usd = max_budget_usd  # per decision, passed as --max-budget-usd
        self.budget_left = None  # set by the caller before a decision: what the run may still spend
        self.extra_args = list(extra_args)
        self.sandbox_auto = sandbox == "auto"
        self.sandbox = _sandbox_available() if self.sandbox_auto else bool(sandbox)
        self.sandbox_problem = None
        self.session = None
        self.started = False
        self.session_cwd = None  # the folder the session belongs to: Claude Code keys sessions by working directory
        self.session_cost = 0.0  # Claude Code reports total_cost_usd for the whole session so far (2.1.302)

    def setup(self, run_dir, workspace):
        self.control = (Path(run_dir) / "control").resolve()
        self.control.mkdir(mode=0o700, parents=True, exist_ok=True)
        # the Bash tool keeps its cwd file in a temp dir; give it one the sandbox may read and write
        self.tmpdir = (Path(run_dir) / "agent-tmp").resolve()
        self.tmpdir.mkdir(exist_ok=True)
        self.run_dir = Path(run_dir).resolve()
        self.workspace = Path(workspace).resolve()
        (self.control / "settings.json").write_text(json.dumps(self._settings(), indent=1))

    def _tool_dirs(self):
        """Folders of bwrap and socat when they are not in the system folders: Claude Code looks them up on the PATH
        its settings give it, and socat runs inside the sandbox."""
        if platform.system() != "Linux":
            return []
        out = []
        for n in ("bwrap", "socat"):
            exe = shutil.which(n)
            if exe:
                for d in (os.path.dirname(exe), os.path.dirname(_real(exe))):
                    if d not in out and d not in ("/usr/bin", "/bin", "/usr/sbin", "/sbin"):
                        out.append(d)
        return out

    def _settings(self):
        ws, control = str(self.workspace), str(self.control.resolve())
        allow = [f"{t}(/{ws}/**)" for t in ("Read", "Write", "Edit")]
        s = {
            "permissions": {
                "allow": allow + (["Bash"] if self.sandbox else []),
                "deny": [f"{t}(/{control}/**)" for t in ("Read", "Write", "Edit")],
            }
        }
        s["env"] = dict(NO_GPU)  # also without the sandbox (no Bash then), so the env is the same everywhere
        if self.sandbox:
            py, dropped = _python_paths(protect=(ws, str(self.run_dir)))
            if dropped and not getattr(self, "_warned_dropped", False):
                self._warned_dropped = True
                print(
                    "[opal] the agent's shell may not read "
                    + ", ".join(dropped)
                    + " (it holds your home folder or the run folder), so the training interpreter may not start in "
                    "its shell; a venv or conda environment of its own avoids this.",
                    file=sys.stderr,
                )
            tools = [d for d in self._tool_dirs() if not any(_contains(d, x) for x in (ws, str(self.run_dir)))]
            tmp = [str(self.tmpdir)]
            system = _SYSTEM_READ_MAC if platform.system() == "Darwin" else _SYSTEM_READ_LINUX
            path = [os.path.dirname(sys.executable), *tools, "/usr/bin", "/bin", "/usr/sbin", "/sbin"]
            s["env"]["PATH"] = ":".join(dict.fromkeys(path))
            s["sandbox"] = {
                "enabled": True,
                "failIfUnavailable": True,
                "autoAllowBashIfSandboxed": True,
                "allowUnsandboxedCommands": False,
                "filesystem": {
                    "denyRead": ["/", control],
                    "allowRead": [ws, *py, *tools, *system, *tmp],
                    "allowWrite": tmp,
                    "denyWrite": [control],
                },
                "network": {"allowedDomains": [], "strictAllowlist": True, "allowLocalBinding": False},
            }
            hidden = _shell_denied_env(agent_env(_login_prefixes()))
            if hidden:
                s["sandbox"]["credentials"] = {"envVars": [dict(name=n, mode="deny") for n in hidden]}
        return s

    def preflight(self):
        exe = shutil.which(self.binary)
        if exe is None:
            raise RuntimeError(
                f"opal: the Claude Code CLI ({self.binary!r}) is not on PATH. Install Claude Code and run `claude` "
                "once to log in, or use backend='mailbox:<dir>' with `opal broker` on a machine "
                "that has an agent."
            )
        try:
            text = subprocess.run([exe, "--help"], capture_output=True, text=True, timeout=60).stdout
        except (OSError, subprocess.TimeoutExpired):
            text = ""
        self.supported = (
            {f for f in _OPTIONAL_FLAGS if f in text} if text else set(_OPTIONAL_FLAGS) - {"--max-budget-usd"}
        )
        missing = sorted(set(_OPTIONAL_FLAGS) - self.supported - {"--max-budget-usd"})
        if missing:
            print(
                f"[opal] this Claude Code version lacks {', '.join(missing)}; running without them "
                "(the agent is less isolated from your Claude Code settings). Updating Claude Code fixes this.",
                file=sys.stderr,
            )
        if platform.system() == "Linux" and (self.sandbox or self.sandbox_auto):
            self.sandbox_problem = linux_sandbox_problem()
            if self.sandbox_problem and not self.sandbox_auto:
                raise RuntimeError(
                    f"opal: sandbox=True, but the Linux sandbox for the agent's shell cannot start here "
                    f"({self.sandbox_problem}). Install bubblewrap and socat, or use sandbox=False (file tools only)."
                )
            self.sandbox = self.sandbox_problem is None
            if self.sandbox_problem:
                print(
                    f"[opal] the sandbox for the agent's shell cannot start here ({self.sandbox_problem}); the "
                    "agent gets file tools only (Read, Edit, Write), no shell. Install bubblewrap and socat to give "
                    "it a sandboxed shell.",
                    file=sys.stderr,
                )

    def reproduction_note(self):
        """One line when this agent is not the blog's (claude-opus-5-5 at effort high), else None. Call after
        preflight(), which learns whether this Claude Code takes --effort."""
        model = self.model or "the CLI's default model"
        effort = self.effort if "--effort" in getattr(self, "supported", {"--effort"}) else None
        if self.model == DEFAULT_MODEL and effort == DEFAULT_EFFORT:
            return None
        effort = (
            f"effort {effort}"
            if effort
            else ("the CLI's default effort" + ("" if not self.effort else " (this Claude Code has no --effort)"))
        )
        return (
            f"the blog's results ({BLOG_URL}) were made with {DEFAULT_MODEL} at effort {DEFAULT_EFFORT}; with "
            f"{model} at {effort} the agent may not reproduce them"
        )

    def _budget(self):
        caps = [c for c in (self.max_budget_usd, self.budget_left) if c is not None]
        return max(0.0, min(caps) - self._spent) if caps else None

    def _cmd(self, system):
        ok = self.supported if hasattr(self, "supported") else set(_OPTIONAL_FLAGS) - {"--max-budget-usd"}
        tools = "Bash,Edit,Read,Write" if self.sandbox else "Edit,Read,Write"
        cmd = [self.binary, "--print"]
        cmd += [f for f in ("--safe-mode", "--restricted", "--no-chrome") if f in ok]
        cmd += ["--tools", tools, "--permission-mode", "dontAsk"]
        if "--permission-prompts" in ok:
            cmd += ["--permission-prompts", "none"]
        if "--setting-sources" in ok:
            cmd += ["--setting-sources", ""]
        cmd += ["--settings", str(self.control / "settings.json")]
        if "--strict-mcp-config" in ok:
            cmd += ["--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}']
        cmd += ["--output-format", "stream-json", "--verbose"]
        if self.model:
            cmd += ["--model", self.model]
        if self.effort and "--effort" in ok:
            cmd += ["--effort", self.effort]
        budget = self._budget()
        if budget is not None and "--max-budget-usd" in ok:
            cmd += ["--max-budget-usd", f"{budget:.4f}"]
        # the system prompt goes with every call: on --resume it would otherwise be lost once the session compacts
        cmd += ["--session-id", self.session] if not self.started else ["--resume", self.session]
        if system:
            note = SANDBOX_NOTE if self.sandbox else NO_BASH_NOTE
            cmd += ["--append-system-prompt", system + "\n\n" + note]
        return cmd + self.extra_args

    def _new_session(self):
        self.session, self.started, self.session_cost, self.session_cwd = str(uuid.uuid4()), False, 0.0, None

    @staticmethod
    def lost_note(workspace) -> str:
        """The first lines of a message that starts a new conversation mid-run: what the folder still holds (the
        earlier decisions' inputs are there only if inbox/ has them)."""
        inbox = Path(workspace) / "inbox"
        ks = sorted(
            int(m.group(1)) for f in inbox.glob("decision-*.json") if (m := re.match(r"decision-(\d+)", f.name))
        )
        earlier = ks[:-1]  # the newest file is this decision's input
        if earlier:
            held = (
                f"inbox/ holds the inputs of decision {earlier[0]}"
                if len(earlier) == 1
                else f"inbox/ holds the inputs of {len(earlier)} earlier decisions ({earlier[0]} to {earlier[-1]})"
            )
        else:
            held = "the inputs of earlier decisions are not in this folder"
        notes = any(f.is_file() and f.name.lower().startswith("notes") for f in Path(workspace).iterdir())
        mine = "your current programs and your notes are" if notes else "your current programs are"
        return (
            f"(Your earlier conversation could not be resumed, so this is a new conversation: {held}, and {mine} "
            "in this folder.)\n\n"
        )

    def decide(self, workspace, message, system, record_dir):
        """One decision, within `timeout` seconds. A call that fails in the agent's own turn (an exit code, a crash) is
        made again after `backoff` seconds, then twice as long, ... (at most `backoff_max` per wait) while the
        decision's time allows, up to `retries` times (None: no cap), each time from the programs the decision started
        with: a failed call's edits are dropped, as when a decision fails. A call that fails because the agent is
        unavailable (no login, a usage or spend limit, an overloaded API, no network, no CLI: backends/unavailable.py)
        is made again after 60 s, then at most every 5 min, for as long as it takes (at most `unavailable_timeout`
        seconds when given); that time does not count against `timeout`, and the agent's message is the same unless
        the failed call had started to work (then the folder is set back and a note says so). Every failure is logged
        with the CLI's own words and kept in attempts.jsonl next to the transcript."""
        t0 = time.time()
        deadline, start = t0 + self.timeout, programs.read_dir(workspace)
        self._spent, costs, n, failures, note = 0.0, [], 0, 0, ""
        waiting, waited = None, 0.0  # the wait for an unavailable agent (backends/unavailable.py), its seconds
        while True:
            n += 1
            t_call, was_started, session_cost = time.time(), self.started, self.session_cost
            out = self._call(workspace, note + message, system, record_dir, max(1.0, deadline - time.time()))
            final = out.pop("final", False)
            down = out.pop("unavailable", None)
            if out.get("cost_usd") is not None:
                costs.append(out["cost_usd"])
                self._spent += out["cost_usd"]
            if out["returncode"] == 0 and not out.get("error"):
                break
            why = error_tail(out.get("error") or f"exit code {out['returncode']}")
            transcript = Path(record_dir) / "backend.jsonl"
            if down is not None and not getattr(self, "cancelled", False):
                worked = self._worked(transcript, workspace, start)
                if not worked:  # the turn never ran: the session and its cost are as they were
                    self.started = was_started
                    if not out.get("cost_usd"):
                        self.session_cost = session_cost
                if waiting is None:
                    waiting = unavailable.Wait(self._say, "Claude Code", self.training_waits, self.unavailable_timeout)
                    if self.on_unavailable is not None:
                        self.on_unavailable(down[1])
                pause = waiting.note(*down)
                line = dict(call=n, returncode=out["returncode"], error=why, unavailable=down[0], next_call_in=pause)
                with open(Path(record_dir) / "attempts.jsonl", "a") as f:
                    f.write(json.dumps(dict(line, seconds=round(out.get("seconds") or 0.0, 1))) + "\n")
                if pause is None:
                    break
                if transcript.exists():
                    transcript.replace(Path(record_dir) / f"backend-call{n}.jsonl")
                ok = self._pause(pause)
                deadline += time.time() - t_call  # the time without an agent does not count against `timeout`
                if not ok:
                    break
                if worked:
                    programs.write_dir(workspace, start)
                    note = self._retry_note(why)
                continue
            if waiting is not None:  # available again (this call failed in the agent's own turn)
                waited += self._end_wait(waiting)
                waiting = None
            failures += 1
            wait = None if final else self._next_wait(failures, deadline)
            line = dict(call=n, returncode=out["returncode"], error=why, seconds=round(out.get("seconds") or 0.0, 1))
            with open(Path(record_dir) / "attempts.jsonl", "a") as f:
                f.write(json.dumps(dict(line, next_call_in=wait)) + "\n")
            if wait is None:
                break
            self._say(f"agent call {n} failed (exit code {out['returncode']}): {why}; calling again in {wait:.0f} s")
            if transcript.exists():
                transcript.replace(Path(record_dir) / f"backend-call{n}.jsonl")
            if not self._pause(wait):
                break
            programs.write_dir(workspace, start)
            note = self._retry_note(why)
        if waiting is not None:  # answered after a wait, or given up (unavailable_timeout, the run cancelled it)
            if out["returncode"] == 0 and not out.get("error"):
                waited += self._end_wait(waiting)
            else:
                waited += waiting.seconds
                if self.on_unavailable is not None:
                    self.on_unavailable(None)
        out = dict(out, cost_usd=sum(costs) if costs else None, seconds=time.time() - t0 - waited)
        if waited:
            out["unavailable_seconds"] = round(waited, 1)
        if n > 1 and (out["returncode"] != 0 or out.get("error")):
            out["error"] = f"{out.get('error') or ''} ({n} calls over {time.time() - t0:.0f} s)".strip()
        return out

    def _end_wait(self, waiting):
        if self.on_unavailable is not None:
            self.on_unavailable(None)
        return waiting.end()

    @staticmethod
    def _retry_note(why):
        return (
            f"(The previous call for this decision stopped before it finished ({why[:200]}). This folder was set "
            "back to the programs as they were when the decision started; edits made in that call were not "
            "kept.)\n\n"
        )

    @staticmethod
    def _worked(transcript, workspace, start):
        """Whether a failed call had started to work: it changed the programs or used a tool."""
        try:
            if programs.read_dir(workspace) != start:
                return True
            for line in open(transcript):
                if '"tool_use"' in line:
                    return True
        except OSError:
            pass
        return False

    def _next_wait(self, n, deadline):
        """Seconds to wait before the call after the n-th failed one, or None: no further call (retries used up, the
        run cancelled the decision, its budget spent, or too little of its time left)."""
        if getattr(self, "cancelled", False) or (self.retries is not None and n > self.retries):
            return None
        budget = self._budget()
        if budget is not None and budget <= 0:
            return None
        left = deadline - time.time() - min(MIN_CALL, self.timeout / 10)
        if left <= 0:
            return None
        return min(self.backoff * 2 ** (n - 1), self.backoff_max, left)

    def _pause(self, seconds):
        """Wait, unless the run cancels the decision meanwhile (then False)."""
        end = time.time() + seconds
        while time.time() < end and not getattr(self, "cancelled", False):
            time.sleep(min(0.5, max(0.0, end - time.time())))
        return not getattr(self, "cancelled", False)

    def _say(self, text):
        text = redact_secrets(text)
        if self.log is not None:
            self.log(text)
        else:
            print(f"[opal] {text}", file=sys.stderr, flush=True)

    def _call(self, workspace, message, system, record_dir, timeout):
        """One call of the CLI in the run's session (resumed, or a new one when it cannot be)."""
        end = time.time() + timeout
        ws = str(Path(workspace).resolve())
        note = ""
        if self.started and self.session_cwd and self.session_cwd != ws:
            # Claude Code keys sessions by working directory: one started in another folder cannot be resumed here
            self._new_session()
            note = self.lost_note(workspace)
        out = self._decide(workspace, note + message, system, record_dir, timeout)
        err = out.get("error") or ""
        if out["returncode"] != 0 and not self.started and _IN_USE in err:
            # the session exists already (saved before its first call had started it, then requeued): resume it
            self.started = True
            out = self._decide(workspace, note + message, system, record_dir, max(1.0, end - time.time()))
            err = out.get("error") or ""
        if out["returncode"] != 0 and self.started and _NOT_FOUND in err:
            # the session lives on another machine (a requeued job), or in a folder that is gone: start a new one
            self._new_session()
            msg = self.lost_note(workspace) + message
            out = self._decide(workspace, msg, system, record_dir, max(1.0, end - time.time()))
        return out

    def _decide(self, workspace, message, system, record_dir, timeout=None):
        timeout = self.timeout if timeout is None else timeout
        t0 = time.time()
        transcript = Path(record_dir) / "backend.jsonl"
        if self.session is None:
            self._new_session()
        timed_out = final = missing = False  # final: calling again cannot help (the run cancelled the decision)
        with open(transcript, "w") as out:
            try:
                p = self._run(
                    self._cmd(system),
                    input=message,
                    stdout=out,
                    cwd=workspace,
                    timeout=timeout,
                    env=agent_env(_login_prefixes(), self._env()),
                )
                code, err = p.returncode, p.stderr[-2000:]
            except subprocess.TimeoutExpired:
                code, err, timed_out = -1, f"timed out after {timeout:.0f} s", True
            except FileNotFoundError:
                code, err, missing = -1, f"{self.binary!r} not found: install Claude Code and log in", True
            except Cancelled as exc:
                code, err, final = -1, str(exc), True
        events, plain = [], []
        for line in open(transcript):
            if line.strip().startswith("{"):
                try:
                    events.append(json.loads(line))
                    continue
                except json.JSONDecodeError:
                    pass
            if line.strip():
                plain.append(line.strip())  # the CLI's own words outside the stream (a usage limit, a crash)
        result = next((e for e in reversed(events) if e.get("type") == "result"), {})
        init = any(e.get("type") == "system" and e.get("subtype") == "init" for e in events)
        if code == 0 or result or init:
            self.started = True  # the session exists from its init event on: later calls must --resume it
            self.session_cwd = str(Path(workspace).resolve())
        if timed_out and result and not result.get("is_error"):
            code, err = 0, ""  # the answer was complete; the CLI (or a process it left) only failed to exit in time
        total, cost = result.get("total_cost_usd"), None
        if total is not None:  # this decision's cost: the session total minus what earlier decisions reported
            total = float(total)
            cost = total - self.session_cost if total >= self.session_cost else total
            self.session_cost = total
        error = None
        said = str(result.get("result") or "").strip()
        limit = next((t for t in [*plain, said] if _LIMIT.search(t) and len(t) < 500), None)
        if code != 0 or result.get("is_error") or result.get("subtype", "success") != "success" or limit:
            # a failed call always carries an error, also when stderr is empty (an API error or usage limit reported
            # in the stream, the CLI killed by a signal), so it is never taken for a decision that kept the programs
            errors = "; ".join(str(e) for e in result.get("errors") or [])  # "Reached maximum budget ($0.01)"
            error = err.strip() or limit or said or errors or " ".join(plain)[-500:]
            if not error:
                error = (
                    f"the Claude Code CLI was killed by signal {-code} ({_signame(-code)})"
                    if code < 0
                    else f"the Claude Code CLI exited with code {code} and gave no reason"
                )
        # an agent that cannot run a turn at all (no login, a limit, an overloaded API, no network, no CLI) is waited for
        down = None
        if error is not None and not final:
            down = ("missing", error) if missing else unavailable.classify([*plain, said, err.strip(), errors])
        return dict(
            text=result.get("result", "") or "",
            cost_usd=cost,
            returncode=code,
            error=error,
            seconds=time.time() - t0,
            **({"final": True} if final else {}),
            **({"unavailable": down} if down else {}),
        )

    def _env(self):
        """The agent's environment: its own temp dir; no git context from folders around the workspace, no CLAUDE.md,
        memory or background shells (checked with Claude Code 2.1.302 against a mock API: the gitStatus block (branch,
        user name, untracked files, commits) leaves the context; the account e-mail and 'Is a git repository' stay)."""
        env = {
            "CLAUDE_CODE_TMPDIR": str(self.tmpdir),
            "TMPDIR": str(self.tmpdir),
            "GIT_CEILING_DIRECTORIES": str(self.workspace.parent),
            "CLAUDE_CODE_DISABLE_GIT_INSTRUCTIONS": "1",
            "CLAUDE_CODE_DISABLE_CLAUDE_MDS": "1",
            "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
            "CLAUDE_CODE_DISABLE_BACKGROUND_TASKS": "1",
            **NO_GPU,
        }
        if self.shell_timeout:
            ms = str(int(self.shell_timeout * 1000))
            env.update(BASH_DEFAULT_TIMEOUT_MS=ms, BASH_MAX_TIMEOUT_MS=ms)
        return env

    def state(self):
        return dict(session=self.session, started=self.started, session_cost=self.session_cost, cwd=self.session_cwd)

    def restore(self, state):
        self.session, self.started = state.get("session"), state.get("started", False)
        self.session_cost = float(state.get("session_cost", 0.0))
        self.session_cwd = state.get("cwd")


def _signame(n):
    try:
        return signal.Signals(n).name
    except ValueError:
        return "unknown"
