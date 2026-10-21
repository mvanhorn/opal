"""A short description and tags for each code change, written by a cheap model through the Claude Code CLI with no
tools. One call per applied decision, cached in decisions/NNN/summary.json; the report shows them in its table."""

from __future__ import annotations

import datetime
import difflib
import json
import re
import subprocess
import tempfile
import time
from pathlib import Path

from .backends.base import agent_env

TAGS = (
    "common practice",
    "niche practice",
    "block-specific",
    "observation-originated",
    "time-step-specific",
    "fix or revert",
)
DEFAULT_MODEL = (
    "claude-opus-5-5"  # at effort low: on 95 decisions it mislabelled less than Haiku
)
DEFAULT_EFFORT = "low"
_KEY = {re.sub(r"[^a-z]", "", t): t for t in TAGS}

SYSTEM = """You describe a change that a coding agent made to the optimizer program of a neural-network training run.
You get the agent's own rationale and the unified diff of the program files. Reply with one JSON object and nothing
else: {"summary": "...", "tags": ["..."]}.

summary: one paragraph of 2 to 4 plain sentences. Start with what the change does (not with the decision number or
the update), concretely (which technique, which
parameters, groups or layers, which values, from or until which update), then the reason the agent gives for it. Use
only what the rationale and the diff show. Describe; do not judge whether the change is good, new or likely to help.

tags: every label below that applies to the change, spelled exactly as written: none, one or several. Judge each label
on its own; they overlap.
- "common practice": the change uses a technique that is standard in current training recipes (for example Muon on
  matrix parameters, learning-rate warmup, cosine or linear decay, gradient clipping, AdamW betas or weight decay).
- "niche practice": the change uses a technique that is published or known but uncommon (for example SOAP or Shampoo,
  per-tensor learning-rate controllers, NorMuon).
- "block-specific": some parameter groups, layers or blocks are treated differently from the others (one rule applied
  the same way to every tensor, even with per-tensor state, does not count).
- "observation-originated": triggered by something the agent saw in this run and cites in its rationale (for example a
  loss spike, a plateau, a gradient statistic).
- "time-step-specific": the program treats a particular update or phase of training differently (for example "start
  Muon at update 801", a warmup, a switch at some update). Every change is made at some update; that alone does not
  count.
- "fix or revert": the change repairs or undoes the agent's own earlier change (for example "reverted the per-tensor
  LR controller from decision 4", "fixed the NaN guard I added")."""

_OPTIONAL = ("--safe-mode", "--no-chrome", "--setting-sources", "--strict-mcp-config", "--max-budget-usd", "--effort")


def unified_diff(old: dict, new: dict) -> str:
    out = []
    for name in sorted(set(old) | set(new)):
        a, b = old.get(name, "").splitlines(), new.get(name, "").splitlines()
        if a != b:
            out += difflib.unified_diff(a, b, f"a/{name}", f"b/{name}", lineterm="", n=2)
    return "\n".join(out)


def message(k, rationale, diff, total=None, max_rationale=6000, max_diff=12000) -> str:
    """The input for one summary. The update at which the decision was made is left out: the table shows it, and
    given to the model it makes every change look tied to a time step."""
    run = f" The training run has {total:,} updates." if total else ""
    rationale = (rationale or "").strip() or "(the agent gave no rationale)"
    if len(rationale) > max_rationale:
        rationale = "[...]\n" + rationale[-max_rationale:]
    if len(diff) > max_diff:
        diff = diff[:max_diff] + f"\n[... diff truncated: {len(diff) - max_diff:,} more characters]"
    return (
        f"The agent's decision number {k}.{run}\n\n<rationale>\n{rationale}\n</rationale>\n\n<diff>\n{diff}\n</diff>"
        "\n\nReply with the JSON object only."
    )


def parse(text: str):
    """The model's reply as (summary, tags, dropped tags); raises ValueError if it is not the asked-for JSON."""
    text = (text or "").strip()
    i, j = text.find("{"), text.rfind("}")
    if i < 0 or j <= i:
        raise ValueError("no JSON object in the reply")
    obj = json.loads(text[i : j + 1])
    summary = obj.get("summary") if isinstance(obj, dict) else None
    if not isinstance(summary, str) or not summary.strip():
        raise ValueError("the reply has no summary")
    raw = obj.get("tags") or []
    raw = [raw] if isinstance(raw, str) else raw if isinstance(raw, list) else []
    tags, dropped = [], []
    for t in raw:
        tag = _KEY.get(re.sub(r"[^a-z]", "", str(t).lower()))
        (tags if tag else dropped).append(tag or str(t))
    return " ".join(summary.split()), [t for t in TAGS if t in tags], dropped


def _run_cli(cmd, text, cwd, env, timeout):
    """One CLI call: (return code, stdout, stderr). Tests replace this."""
    p = subprocess.run(cmd, input=text, capture_output=True, text=True, cwd=cwd, env=env, timeout=timeout)
    return p.returncode, p.stdout, p.stderr


class Summarizer:
    """Calls `binary --print` with no tools, no session file and the user's customisations off, in an empty folder."""

    def __init__(self, model=None, binary="claude", timeout=180, max_budget_usd=0.5):
        self.model, self.binary, self.timeout, self.budget = model or DEFAULT_MODEL, binary, timeout, max_budget_usd
        self._flags = None

    def _env(self, cwd):
        """The claude backend's environment: secrets removed except the login, no git context, CLAUDE.md or memory."""
        from .backends.claude import _login_prefixes

        isolate = {
            "TMPDIR": str(cwd),
            "GIT_CEILING_DIRECTORIES": str(Path(cwd).parent),
            "CLAUDE_CODE_DISABLE_GIT_INSTRUCTIONS": "1",
            "CLAUDE_CODE_DISABLE_CLAUDE_MDS": "1",
            "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
            "CLAUDE_CODE_DISABLE_BACKGROUND_TASKS": "1",
        }
        return agent_env(_login_prefixes(), isolate)

    def _supported(self, cwd):
        if self._flags is None:
            try:
                out = _run_cli([self.binary, "--help"], "", cwd, self._env(cwd), 60)[1]
            except (OSError, subprocess.TimeoutExpired):
                out = ""
            self._flags = {f for f in _OPTIONAL if f in out} if out else set(_OPTIONAL)
        return self._flags

    def command(self, cwd) -> list:
        ok = self._supported(cwd)
        cmd = [self.binary, "--print", "--output-format", "json", "--model", self.model, "--tools", ""]
        cmd += ["--no-session-persistence", "--system-prompt", SYSTEM]
        cmd += [f for f in ("--safe-mode", "--no-chrome") if f in ok]
        if "--setting-sources" in ok:
            cmd += ["--setting-sources", ""]
        if "--strict-mcp-config" in ok:
            cmd += ["--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}']
        if "--effort" in ok and self.model == DEFAULT_MODEL:
            cmd += ["--effort", DEFAULT_EFFORT]
        if "--max-budget-usd" in ok and self.budget:
            cmd += ["--max-budget-usd", str(self.budget)]
        return cmd

    def __call__(self, text: str) -> dict:
        """One description: dict(summary, tags, dropped_tags, cost_usd, seconds) or dict(error, cost_usd, seconds)."""
        t0, cost, reply = time.time(), None, ""
        try:
            with tempfile.TemporaryDirectory(prefix="opal-summary-") as tmp:
                code, out, err = _run_cli(self.command(tmp), text, tmp, self._env(tmp), self.timeout)
            result = _result(out)
            cost = result.get("total_cost_usd")
            reply = result.get("result") or ""
            if code != 0 or result.get("is_error"):
                raise RuntimeError(f"return code {code}: {(reply or err or out or '')[-500:]}")
            summary, tags, dropped = parse(reply)
            return dict(summary=summary, tags=tags, dropped_tags=dropped, cost_usd=cost, seconds=_since(t0))
        except FileNotFoundError:
            return dict(error=f"{self.binary!r} not found", cost_usd=cost, seconds=_since(t0))
        except Exception as exc:  # noqa: BLE001 - a summary must never fail the caller
            return dict(
                error=f"{type(exc).__name__}: {exc}"[:800], reply=reply[:2000], cost_usd=cost, seconds=_since(t0)
            )


def _result(out):
    """The CLI's result object: one JSON object, a JSON list of messages, or JSON lines (versions differ)."""
    out = (out or "").strip()
    try:
        items = json.loads(out)
    except ValueError:
        items = []
        for line in out.splitlines():
            try:
                items.append(json.loads(line))
            except ValueError:
                pass
    items = items if isinstance(items, list) else [items]
    return next((x for x in reversed(items) if isinstance(x, dict) and x.get("type") == "result"), {})


def _since(t0):
    return round(time.time() - t0, 1)


def load(record_dir: Path):
    f = Path(record_dir) / "summary.json"
    try:
        return json.loads(f.read_text()) if f.exists() else None
    except (OSError, ValueError):
        return None


def ensure(record_dir, summarizer, k, rationale, old: dict, new: dict, total=None):
    """The decision's summary from its cache, or from one call that fills the cache. A failed call is not cached (the
    next `opal report --summarize` tries again); its error goes to summary-error.json. Returns the summary,
    or None."""
    record_dir = Path(record_dir)
    cached = load(record_dir)
    if cached is not None or summarizer is None:
        return cached
    out = summarizer(message(k, rationale, unified_diff(old, new), total))
    out.update(model=summarizer.model, utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"))
    record_dir.mkdir(parents=True, exist_ok=True)
    if "error" in out:
        (record_dir / "summary-error.json").write_text(json.dumps(out, indent=1))
        return None
    (record_dir / "summary.json").write_text(json.dumps(out, indent=1))
    (record_dir / "summary-error.json").unlink(missing_ok=True)
    return out
