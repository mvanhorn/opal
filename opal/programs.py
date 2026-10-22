"""Program files: hashing, syntax checks, loading as fresh modules, and the generated starting program."""

from __future__ import annotations

import ast
import builtins
import contextlib
import hashlib
import inspect
import io
import json
import re
import sys
import types
from collections import Counter, OrderedDict
from pathlib import Path

import torch
import numpy as np

PROGRAM_FILES = ("updater.py", "analysis.py")
DATA_FILES = ()  # files besides modules that count as program files (the updater side): none since knobs mode went
_MODULE_FILE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\.py")
AGENT_ERRORS = (Exception, SystemExit)  # what the agent's code may raise without ending the run (not KeyboardInterrupt)


def digest(files: dict) -> str:
    h = hashlib.sha256()
    for name in sorted(files):
        h.update(name.encode() + b"\0" + files[name].encode() + b"\0")
    return h.hexdigest()[:16]


def read_dir(path: Path) -> dict:
    """The program files in a folder. Symbolic links are skipped: the training process (outside the agent's sandbox)
    would otherwise copy whatever file a link names into the program, and so into the agent's folder."""
    out = {}
    for p in sorted(Path(path).glob("*.py")) + [Path(path) / n for n in DATA_FILES if (Path(path) / n).is_file()]:
        if p.is_symlink() or not p.is_file():
            continue
        out[p.name] = p.read_text(errors="replace")
    return out


def write_dir(path: Path, files: dict) -> None:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    for p in list(path.glob("*.py")) + [path / n for n in DATA_FILES if (path / n).is_file() or (path / n).is_symlink()]:
        if p.name not in files or p.is_symlink():
            p.unlink()
    for name, text in files.items():
        if not (_MODULE_FILE.fullmatch(name) or name in DATA_FILES):
            raise ValueError(f"not a program file name: {name!r}")
        if (path / name).is_symlink():  # never write through a link the agent made
            (path / name).unlink()
        (path / name).write_text(text)


def _imports(text: str) -> set:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return set()
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            out.add(node.module.split(".")[0])
    return out


def closure(files: dict, root: str) -> dict:
    """`root` and the workspace modules it imports, transitively; the updater's side also holds the data files."""
    keep, todo = set(), [root] if root in files else []
    while todo:
        f = todo.pop()
        if f in keep:
            continue
        keep.add(f)
        todo += [m + ".py" for m in _imports(files[f]) if m + ".py" in files and m + ".py" not in keep]
    if root == "updater.py" and keep:
        keep.update(n for n in DATA_FILES if n in files)
    return {f: files[f] for f in sorted(keep)}


def program(files: dict) -> dict:
    """The program in a folder: updater.py, analysis.py and the modules they import. Other .py files (scratch
    scripts the agent ran) are not part of it."""
    out = {}
    for root in PROGRAM_FILES:
        out.update(closure(files, root))
    return out


def side_digests(files: dict):
    return digest(closure(files, "updater.py")), digest(closure(files, "analysis.py"))


def check_syntax(files: dict) -> list:
    """Errors that can be found without running anything: missing updater, syntax errors, helper modules named like
    standard-library modules (they would shadow them inside the program)."""
    errors = []
    if "updater.py" not in files:
        errors.append("updater.py is missing")
    stdlib = getattr(sys, "stdlib_module_names", ())
    for name in files:
        if name[:-3] in stdlib:
            errors.append(f"{name}: a helper module may not be named like the standard-library module {name[:-3]!r}")
    for name, text in files.items():
        if name in DATA_FILES:
            continue
        try:
            compile(text, name, "exec")
        except SyntaxError as exc:
            errors.append(f"{name}: SyntaxError: {exc}")
    return errors


def load(files: dict, root: str, tag: str, folder: Path):
    """Execute `root` and the workspace modules it imports as fresh module objects. Imports of workspace modules
    are resolved by a __import__ private to these modules: nothing goes on sys.path, and the user's own modules with
    the same names are neither shadowed nor used. Returns None if `root` is not among the files."""
    if root not in files:
        return None
    real_import = builtins.__import__
    cache = {}
    scope = dict(builtins.__dict__)
    package = f"opal_program_{tag}"

    def get(stem):
        if stem not in cache:
            mod = types.ModuleType(f"{package}.{stem}")
            mod.__file__ = str(Path(folder) / f"{stem}.py")
            mod.__builtins__ = scope
            cache[stem] = mod
            sys.modules[mod.__name__] = mod
            exec(compile(files[stem + ".py"], mod.__file__, "exec"), mod.__dict__)
        return cache[stem]

    def private_import(name, globals=None, locals=None, fromlist=(), level=0):
        top = name.split(".")[0]
        if level == 0 and top + ".py" in files and "." not in name:
            return get(top)
        return real_import(name, globals, locals, fromlist, level)

    scope["__import__"] = private_import
    return get(root[:-3])


def forget(tag: str) -> None:
    """Drop a version's modules from sys.modules (they were registered for tracebacks and introspection), so that
    what their globals hold is freed once nothing else refers to them."""
    prefix = f"opal_program_{tag}."
    for name in [n for n in sys.modules if n.startswith(prefix)]:
        del sys.modules[name]


def normalize_state(obj):
    """Copy plain containers, converting NumPy numbers/arrays before the safe-state check.

    Inspect only exact builtin/library types: rejected objects and serialization hooks are never called.
    Leave cycles and unsupported values for unpicklable() to reject.
    """
    memo = {}

    def convert(value, depth=0):
        kind = type(value)
        if depth > 100:
            return value
        for base, cast in ((np.integer, int), (np.floating, float), (np.complexfloating, complex), (np.bool_, bool)):
            if any(kind is t for t in np.sctypeDict.values()) and issubclass(kind, base):
                return cast(value)
        if kind is np.ndarray:
            try:
                return torch.from_numpy(value.copy())
            except (TypeError, ValueError):
                return value  # e.g. object/string arrays: reject through the same allowlist
        if id(value) in memo:
            return memo[id(value)]
        if any(kind is t for t in (dict, OrderedDict, Counter)):
            attrs = vars(value) if kind is not dict else None
            if attrs is not None and type(attrs) is not dict:
                return value  # the guard rejects custom attribute dictionaries without calling their methods
            out = kind()
            memo[id(value)] = out
            for key, item in dict.items(value):
                key = convert(key, depth + 1)
                if unpicklable(key):  # inserting a rejected key would call its __hash__
                    memo[id(value)] = value
                    return value
                out[key] = convert(item, depth + 1)
            if attrs is not None:
                attrs = convert(attrs, depth + 1)
                if unpicklable(attrs):
                    memo[id(value)] = value
                    return value
                vars(out).update(attrs)
            return out
        if kind is list:
            out = []
            memo[id(value)] = out
            out.extend(convert(item, depth + 1) for item in value)
            return out
        if any(kind is t for t in (tuple, set, frozenset)):
            memo[id(value)] = value
            items = [convert(item, depth + 1) for item in value]
            if kind is not tuple and any(unpicklable(item) for item in items):
                return value  # do not hash unsupported set members either
            out = kind(items)
            memo[id(value)] = out
            return out
        return value

    return convert(obj)


def unpicklable(obj) -> list:
    """Reject agent state outside a closed allowlist without invoking serialization hooks.

    Pickleability alone does not imply compatibility with torch.load(weights_only=True). The user's starting
    optimizers remain exempt at the call sites; this contract is for state returned by agent programs.
    """
    active = set()

    def plain(value, depth=0):
        kind = type(value)
        if depth > 100:
            raise TypeError("state nesting exceeds 100 levels")
        if any(kind is allowed for allowed in (set, frozenset, torch.memory_format)):
            # Probe only library-owned values, never a live object's reducer. Loader support differs
            # across torch versions; accepting an unsupported builtin would still break safe resume.
            stream = io.BytesIO()
            torch.save(kind() if kind is set or kind is frozenset else value, stream)
            stream.seek(0)
            torch.load(stream, weights_only=True)
        if any(kind is allowed for allowed in (
            type(None), bool, int, float, complex, str, bytes,
            torch.Size, torch.dtype, torch.device, torch.layout, torch.memory_format,
        )):
            return
        if not any(kind is allowed for allowed in (
            torch.Tensor, torch.nn.Parameter, dict, Counter, OrderedDict, list, tuple, set, frozenset,
        )):
            # Read the builtin type slot directly: ordinary attribute access can invoke a metaclass.
            name = type.__dict__["__qualname__"].__get__(kind)
            raise TypeError("non-plain state type " + str.__str__(name))
        if id(value) in active:
            raise TypeError("cyclic state container")
        active.add(id(value))
        try:
            if kind is torch.Tensor or kind is torch.nn.Parameter:
                plain(vars(value), depth + 1)  # Python attributes are serialized too
            elif kind is dict or kind is Counter or kind is OrderedDict:
                for key, item in dict.items(value):
                    plain(key, depth + 1)
                    plain(item, depth + 1)
                if kind is Counter or kind is OrderedDict:
                    plain(vars(value), depth + 1)
            else:
                for item in value:
                    plain(item, depth + 1)
        finally:
            active.remove(id(value))

    try:
        plain(obj)  # Never pickle live state here: even a rejected reducer can execute user code.
    except Exception as exc:
        return [f"{type(exc).__name__}: {exc}"[:300]]
    return []


@contextlib.contextmanager
def keep_param_group_identity():
    """While agent code runs, make torch.optim.Optimizer.load_state_dict keep its param_group dict objects. Torch
    replaces them, so a program that holds references to them (self.groups = opt.param_groups) silently stops
    controlling the learning rate after a carry-over or a resume. Patched only for the duration of our calls."""
    original = torch.optim.Optimizer.load_state_dict

    def load_state_dict(self, state_dict):
        old = list(self.param_groups)
        original(self, state_dict)
        if len(old) == len(self.param_groups):
            for i, (o, n) in enumerate(zip(old, self.param_groups)):
                if o is not n:
                    o.clear()
                    o.update(n)
                    self.param_groups[i] = o

    torch.optim.Optimizer.load_state_dict = load_state_dict
    try:
        yield
    finally:
        torch.optim.Optimizer.load_state_dict = original


# ----- the starting program ----------------------------------------------------------------------------------


def _hyper(group):
    out = {}
    for k, v in group.items():
        if k == "params":
            continue
        if torch.is_tensor(v):
            v = v.item() if v.numel() == 1 else f"tensor{tuple(v.shape)}"
        try:
            json.dumps(v)
        except TypeError:
            v = repr(v)
        out[k] = v
    return out


def _name_ranges(names):
    """Compact list of parameter names: blocks.0.attn.qkv.weight ... blocks.11.attn.qkv.weight -> blocks.*.attn.qkv.weight x12."""
    counts = {}
    for n in names:
        key = re.sub(r"\.\d+\.", ".*.", n)
        counts[key] = counts.get(key, 0) + 1
    return ", ".join(k if c == 1 else f"{k} x{c}" for k, c in counts.items())


def describe_user_objects(optimizers, schedulers, names_by_id, sources=True) -> str:
    lines = []
    for i, opt in enumerate(optimizers):
        cls = type(opt)
        lines.append(f"optimizer {i}: {cls.__module__}.{cls.__qualname__}")
        for j, g in enumerate(opt.param_groups):
            names = [names_by_id.get(id(p), "?") for p in g["params"]]
            lines.append(f"  group {j} ({len(names)} tensors: {_name_ranges(names)})")
            lines.append(f"    {json.dumps(_hyper(g))}")
    for i, s in enumerate(schedulers):
        cls = type(s)
        try:
            st = {k: v for k, v in s.state_dict().items() if k not in ("_get_lr_called_within_step", "lr_lambdas")}
            st = json.dumps(st, default=repr)
        except Exception:
            st = "(state not serialisable)"
        if len(st) > 1500:
            st = st[:1500] + " ..."
        lines.append(f"scheduler {i}: {cls.__module__}.{cls.__qualname__} {st}")
        for j, fn in enumerate((getattr(s, "lr_lambdas", None) or []) if sources else []):
            try:
                src = inspect.getsource(fn).strip()
            except (OSError, TypeError):
                continue
            lines.append(f"  lr_lambdas[{j}] (source): {src[:600]}")
        lines.append(
            "  stepped whenever the training script calls scheduler.step(), while follow_user_scheduler is True"
        )
    return "\n".join(lines)


STARTING_UPDATER = '''"""Updater: turns gradients into parameter updates. Runs on every rank after the gradients are averaged.

Starting program = the training script's own optimizer, unchanged:

{description}

You may replace any part of it.
"""

import torch

# Gradient clipping by global norm before the update, moved here from the training script with
# wrap(clip_grad_norm=...); None: no clipping in this program.
CLIP_GRAD_NORM = {clip}


class Updater:
    # True: the training script's scheduler.step() calls still step the user's schedulers (as in the plain run).
    # A program that sets its own learning rates should leave this False.
    follow_user_scheduler = True

    def __init__(self, named_params, previous=None, spec=None):
        self.named = named_params
        self.opts = spec["user_optimizers"]
        self.scheds = spec["user_schedulers"]

    @property
    def param_groups(self):
        return [g for o in self.opts for g in o.param_groups]

    def step(self, grads, update):
        # grads[name] is the same tensor as param.grad (None if the parameter got no gradient this update).
        if CLIP_GRAD_NORM is not None:
            torch.nn.utils.clip_grad_norm_([p for _, p in self.named], CLIP_GRAD_NORM)
        for o in self.opts:
            o.step()

    def state_dict(self):
        return dict(opts=[o.state_dict() for o in self.opts], scheds=[s.state_dict() for s in self.scheds])

    def load_state_dict(self, state):
        for o, s in zip(self.opts, state["opts"]):
            o.load_state_dict(s)
        for o, s in zip(self.scheds, state["scheds"]):
            o.load_state_dict(s)


def build(named_params, previous=None, spec=None):
    return Updater(named_params, previous, spec)
'''

STARTING_ANALYSIS = '''"""Analysis: reports what you want to see at your next decision. Runs on rank 0 only, every EVERY updates,
after the backward pass and before the update. It never changes the update; the updater computes its own signals.

obs.update, obs.total_updates, obs.losses (mean loss of each update since the previous call), obs.params and
obs.grads (name -> tensor), obs.updater (the running updater object), obs.metrics (values the training script chose
to show you since the previous call, often none), obs.program_ms_per_update (measured cost of the updater),
obs.activations (only if enabled). memory is a dict that persists across calls.
Return {{"report": <small JSON>}} to keep it for your next decision, {{"wake": "<reason>"}} to ask for a decision now.

Starting program: loss summary, and gradient and weight norms per parameter role (block indices merged) and per block.
"""
import math
import re

import torch

EVERY = {every}


def _role(name):
    return re.sub(r"\\.\\d+\\.", ".*.", name)


def _block(name):
    m = re.search(r"\\.(\\d+)\\.", name)
    return m.group(1) if m else None


def analyze(obs, memory):
    names = [n for n, g in obs.grads.items() if g is not None]
    if not names:
        return None
    gn = torch.stack(torch._foreach_norm([obs.grads[n].float() for n in names])).tolist()
    wn = torch.stack(torch._foreach_norm([obs.params[n].detach().float() for n in names])).tolist()
    roles, blocks = {{}}, {{}}
    for n, g, w in zip(names, gn, wn):
        r = roles.setdefault(_role(n), [0.0, 0.0])
        r[0] += g * g
        r[1] += w * w
        b = _block(n)
        if b is not None:
            blocks[b] = blocks.get(b, 0.0) + g * g
    losses = [x for x in obs.losses if math.isfinite(x)]
    report = dict(
        update=obs.update,
        loss_mean=round(sum(losses) / len(losses), 5) if losses else None,
        loss_min=round(min(losses), 5) if losses else None,
        loss_max=round(max(losses), 5) if losses else None,
        nonfinite_losses=len(obs.losses) - len(losses),
        grad_norm_total=round(math.sqrt(sum(g * g for g in gn)), 6),
        grad_norm_by_role={{k: round(math.sqrt(v[0]), 6) for k, v in roles.items()}},
        weight_norm_by_role={{k: round(math.sqrt(v[1]), 4) for k, v in roles.items()}},
        grad_norm_by_block={{k: round(math.sqrt(v), 6) for k, v in sorted(blocks.items(), key=lambda kv: int(kv[0]))}},
        program_ms_per_update=obs.program_ms_per_update,
    )
    return {{"report": report}}
'''


def starting_files(optimizers, schedulers, names_by_id, every, clip=None, sources=True) -> dict:
    desc = describe_user_objects(optimizers, schedulers, names_by_id, sources=sources)
    return {
        "updater.py": STARTING_UPDATER.format(description=desc, clip=repr(None if clip is None else float(clip))),
        "analysis.py": STARTING_ANALYSIS.format(every=every),
    }


def source_of(obj) -> str | None:
    """Source of a user class that is not part of torch, for the agent's reference folder."""
    cls = type(obj)
    mod = cls.__module__ or ""
    if mod.startswith("torch."):
        return None
    try:
        return inspect.getsource(sys.modules[mod]) if mod in sys.modules else inspect.getsource(cls)
    except (OSError, TypeError):
        return None
