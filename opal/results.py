"""The fields every example writes to its result.json, so that a run can be set beside ours (docs/GUIDE.md, "Comparing
with our runs"): the metric (name, value, direction), the training time and the time training waited for the agent,
the agent's cost, the decisions asked, applied and failed, the hardware, the OPAL commit, the full command and the OPAL
run folder. A plain run has the same fields, with the OPAL ones None. save_weights() writes final weights for --save."""

from __future__ import annotations

import platform
import shlex
import subprocess
import sys
from pathlib import Path

import torch


def hardware(device=None) -> str:
    dev = torch.device(device) if device is not None else None
    if dev is not None and dev.type == "cuda":
        n = torch.cuda.device_count()
        return f"{n} x {torch.cuda.get_device_name(dev)}"
    return f"{platform.machine()} CPU ({platform.processor() or platform.system()})" + (
        f", {dev.type}" if dev is not None and dev.type not in ("cpu", "cuda") else ""
    )


def opal_commit() -> str | None:
    """The git commit of the OPAL checkout that ran (with "+changes" when it has local edits), else its version."""
    from . import __version__

    here = Path(__file__).parent
    try:
        head = subprocess.run(["git", "-C", str(here), "rev-parse", "--short=12", "HEAD"], capture_output=True,
                              text=True, timeout=10)  # fmt: skip
        if head.returncode != 0:
            return __version__
        dirty = subprocess.run(["git", "-C", str(here), "status", "--porcelain", "--", "."], capture_output=True,
                               text=True, timeout=10).stdout.strip()  # fmt: skip
        return head.stdout.strip() + ("+changes" if dirty else "")
    except (OSError, subprocess.TimeoutExpired):
        return __version__


def summary(metric, value, direction, train_seconds, opt=None, device=None) -> dict:
    """The uniform fields. direction: "higher" or "lower" is better. opt: the AgentOptimizer of an OPAL run."""
    out = dict(
        metric=dict(name=metric, value=value, direction=direction),
        train_seconds=round(float(train_seconds), 1),
        agent_wait_seconds=None,
        agent_cost_usd=None,
        decisions=None,
        hardware=hardware(device),
        opal_commit=None,
        command=" ".join(shlex.quote(x) for x in [sys.executable, *sys.argv]),
        opal_run=None,
    )
    run = getattr(opt, "run", None)
    if run is not None:
        from .report import _jsonl, decision_counts

        c = decision_counts(_jsonl(run.dir / "events.jsonl"))
        out.update(
            agent_wait_seconds=round(getattr(opt, "agent_wait_seconds", 0.0), 1),
            agent_cost_usd=opt.worker.recorded_cost if opt.worker is not None else 0.0,
            decisions=dict(asked=c["asked"], applied=c["applied"], failed=c["failed"]),
            opal_commit=opal_commit(),
            opal_run=str(run.dir),
        )
        if opt.worker is not None and (opt.worker.usage is not None or opt.worker.backend.name == "codex"):
            out.update(agent_cost_usd=None, agent_tokens=opt.worker.usage)
    return out


def save_weights(folder, **state_dicts):
    """--save: each state dict to <folder>/<name>.pt (model.pt, ema.pt, ...)."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for name, state in state_dicts.items():
        torch.save(state, folder / f"{name}.pt")
    return folder
