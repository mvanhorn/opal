"""What the agent is told about the run, built from the live objects so the user only writes a goal."""

from __future__ import annotations

import inspect
import re
import sys
from pathlib import Path

import torch

_PREFIXES = ("module.", "_orig_mod.", "_fsdp_wrapped_module.")


def clean_name(name: str) -> str:
    """Strip wrapper prefixes (DDP's module., torch.compile's _orig_mod.) so names match the bare model."""
    changed = True
    while changed:
        changed = False
        for p in _PREFIXES:
            if name.startswith(p):
                name, changed = name[len(p) :], True
    return name.replace("._orig_mod.", ".")


def named_params(model, optimizers):
    """(name, parameter) for every parameter the user's optimizers update, in the model's order."""
    in_opt = {id(p) for o in optimizers for g in o.param_groups for p in g["params"]}
    out, seen = [], set()
    for name, p in model.named_parameters():
        if id(p) in in_opt and id(p) not in seen:
            out.append((clean_name(name), p))
            seen.add(id(p))
    extra = [p for o in optimizers for g in o.param_groups for p in g["params"] if id(p) not in seen]
    for i, p in enumerate(extra):
        if id(p) not in seen:
            out.append((f"extra.{i}", p))
            seen.add(id(p))
    return out


def param_table(named, names_by_group=None) -> str:
    lines = []
    for n, p in named:
        g = f"  group {names_by_group[n]}" if names_by_group and n in names_by_group else ""
        lines.append(f"  {n} {tuple(p.shape)} {str(p.dtype).replace('torch.', '')}{g}")
    return "\n".join(lines)


def _wrappers(model):
    """The model and every wrapper inside it, outermost first: DDP and Accelerate (.module), torch.compile
    (._orig_mod), in any number and order (Accelerate on top of a script's own DDP gives two)."""
    chain, m = [model], model
    for _ in range(16):
        inner = getattr(m, "_orig_mod", None)
        if not isinstance(inner, torch.nn.Module):
            inner = getattr(m, "module", None)
        if not isinstance(inner, torch.nn.Module) or inner is m:
            break
        chain.append(inner)
        m = inner
    return chain


def bare_model(model):
    return _wrappers(model)[-1]


def model_summary(model, named, world_size, device) -> str:
    cls = type(bare_model(model))
    n = sum(p.numel() for _, p in named)
    compiled = any(hasattr(m, "_orig_mod") for m in _wrappers(model))
    gpu = torch.cuda.get_device_name(device) if device.type == "cuda" else str(device)
    return (
        f"Model class {cls.__module__}.{cls.__qualname__}; {n:,} parameters updated by the optimizer in {len(named)} "
        f"tensors; data-parallel world size {world_size}; device {gpu}; torch.compile on the model: "
        f"{'yes' if compiled else 'no'}; torch {torch.__version__}."
    )


def group_index(optimizers, named):
    by_id = {id(p): n for n, p in named}
    out = {}
    for i, o in enumerate(optimizers):
        for j, g in enumerate(o.param_groups):
            for p in g["params"]:
                if id(p) in by_id:
                    out[by_id[id(p)]] = f"{i}.{j}" if len(optimizers) > 1 else str(j)
    return out


def reference_files(model, optimizers, schedulers, max_bytes=300_000) -> dict:
    """Read-only copies for the agent: the training script and the source of user-defined classes."""
    out = {}
    main = sys.modules.get("__main__")
    script = getattr(main, "__file__", None)
    if script and Path(script).is_file() and Path(script).stat().st_size <= max_bytes:
        out["training_script_" + Path(script).name] = Path(script).read_text(errors="replace")
    objs = [getattr(getattr(model, "module", model), "_orig_mod", getattr(model, "module", model))]
    objs += list(optimizers) + list(schedulers)
    for obj in objs:
        mod = type(obj).__module__ or ""
        if mod == "torch" or mod.startswith("torch.") or mod == "__main__":
            continue
        try:
            f = inspect.getsourcefile(type(obj))
        except TypeError:
            f = None
        if f and Path(f).is_file() and Path(f).stat().st_size <= max_bytes:
            out["source_" + re.sub(r"[^A-Za-z0-9_.]", "_", mod) + ".py"] = Path(f).read_text(errors="replace")
    return out
