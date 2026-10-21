"""Agent backends. Each one runs one decision: the agent reads and edits the files in `workspace` and answers in text.

decide(workspace, message, system, record_dir) -> dict(text, cost_usd, returncode, error)
`system` is the system prompt, passed every time; each backend keeps one persistent conversation across decisions.
"""

from __future__ import annotations

from .base import Backend, CallableBackend, KeepBackend


def make_backend(spec, **kw) -> Backend:
    if isinstance(spec, Backend):
        return spec
    if callable(spec):
        return CallableBackend(spec)
    if spec == "keep":
        return KeepBackend()
    if spec == "claude":
        from .claude import ClaudeBackend

        return ClaudeBackend(**kw)
    if spec == "codex":
        from .codex import CodexBackend

        return CodexBackend(**kw)
    if isinstance(spec, str) and spec.startswith("mailbox:"):
        from .mailbox import MailboxBackend

        return MailboxBackend(spec, **kw)
    raise ValueError(f"unknown backend {spec!r}: use 'claude', 'codex', 'mailbox:<dir>', 'keep', or a callable")
