"""The precision contract: the agent's program may not change the dtype or device of parameters or gradients, and
keeps the state it creates in the parameters' dtype or in float32. Earlier runs ran Newton-Schulz in float32 and stored
bfloat16 moments of float32 parameters, which changed speed and numerics outside the algorithm itself.

Checked by comparing tensor attributes only (no device work, no synchronisation): around build(); the dtype and device
of every parameter and gradient after every update of an agent's updater; its state (and the training script's
optimizers' state, which may not change dtype under it) after each update of its trial window and at check steps. A
program that breaks the contract fails like one that raised (the dtypes are put back first). The starting program is
the user's and is not checked. The agent's analysis may not change parameters or gradients at all: checked around
each call by their version counters, dtypes and devices."""

from __future__ import annotations

import contextlib

import torch

from .. import programs


class PrecisionMixin:
    def _record_dtypes(self):
        """At wrap time: each parameter's dtype and device and its gradient's dtype (None: torch allows any), each
        parameter's dtype by name, and which dtypes each parameter shape has."""
        self._dtypes = [p.dtype for _, p in self.named]
        self._devices = [p.device for _, p in self.named]
        self._grad_dtypes = [
            p.grad.dtype if p.grad is not None else getattr(p, "grad_dtype", p.dtype) for _, p in self.named
        ]
        self._by_name = {n: (tuple(p.shape), p.dtype) for n, p in self.named}
        self._by_shape = {}
        for n, p in self.named:
            self._by_shape.setdefault(tuple(p.shape), {}).setdefault(p.dtype, n)

    def _changed_dtypes(self, grads=None):
        """Parameters whose dtype or device, or whose gradient's (p.grad, not the dict handed to the program, which the
        program may use as scratch), differs from what the training script gave them."""
        out = []
        for (n, p), d, gd, dev in zip(self.named, self._dtypes, self._grad_dtypes, self._devices):
            g = p.grad
            bad_g = g is not None and ((gd is not None and g.dtype is not gd) or g.device != dev)
            if p.dtype is not d or bad_g or p.device != dev:
                out.append((n, p, g, d, gd, dev))
        return out

    @staticmethod
    def _describe_changes(changed):
        parts = []
        for n, p, g, d, gd, dev in changed[:8]:
            if p.dtype is not d:
                parts.append(f"parameter {n} {_name(d)} -> {_name(p.dtype)}")
            if p.device != dev:
                parts.append(f"parameter {n} on {dev} -> {p.device}")
            if g is not None and gd is not None and g.dtype is not gd:
                parts.append(f"gradient of {n} {_name(gd)} -> {_name(g.dtype)}")
            if g is not None and g.device != dev:
                parts.append(f"gradient of {n} on {dev} -> {g.device}")
        more = f" ({len(changed)} tensors in all)" if len(changed) > 8 else ""
        return "; ".join(parts) + more

    @contextlib.contextmanager
    def _dtypes_kept(self):
        """Around build(): a build that changes the dtype or device of a parameter or gradient is rejected, and the
        original tensors are put back exactly (references only, no copies)."""
        held = [(p.data, None if p.grad is None else p.grad.data) for _, p in self.named]
        try:
            yield
        finally:
            changed = self._changed_dtypes()
            if changed:
                what = self._describe_changes(changed)
                for (_, p), (w, g) in zip(self.named, held):
                    if p.dtype is not w.dtype or p.device != w.device:
                        p.data = w
                    if (
                        g is not None
                        and p.grad is not None
                        and (p.grad.dtype is not g.dtype or p.grad.device != g.device)
                    ):
                        p.grad.data = g
        if changed:
            raise TypeError(
                f"precision contract: build() changed dtypes or devices, which a program may not do: {what}"
            )

    def _precision_error(self, grads, first, state=True):
        """After an update of the agent's program: None, or why it broke the precision contract (changed dtypes and
        devices are put back, so the revert that follows restores the values). state: also check the updater's state
        and the training script's optimizers' state (the trial window and check steps)."""
        if self.digests[0] == _starting_digest(self):
            return None
        changed = self._changed_dtypes()
        if changed:
            what = self._describe_changes(changed)
            self._put_back_placement()
            for n, p, g, d, gd, dev in changed:
                if getattr(p, "grad_dtype", gd) is not gd and gd is not None:
                    p.grad_dtype = gd
            return (
                f"precision contract: your program changed dtypes or devices, which it may not do: {what}. Parameters "
                "and gradients keep the dtypes and devices the training script gives them; they were put back."
            )
        if not (first or state):
            return None
        bad = self._state_violations()
        if not bad:
            return None
        more = f" ({len(bad)} tensors in all)" if len(bad) > 8 else ""
        return (
            "precision contract: your updater keeps parameter-shaped state in a dtype that is neither the parameter's "
            "nor float32: " + "; ".join(bad[:8]) + more + ". Keep the state you create in the parameters' dtype, or in "
            "float32 where the standard algorithm does so."
        )

    def _user_state_dtypes(self):
        """The dtype of every floating tensor in the training script's optimizers' state, keyed by optimizer, parameter
        and name: taken before a new updater is built, so that a program which casts that state is caught."""
        return {
            (i, id(p), k): v.dtype
            for i, o in enumerate(self.user_optimizers)
            for p, s in o.state.items()
            if isinstance(s, dict)
            for k, v in s.items()
            if torch.is_tensor(v) and v.is_floating_point()
        }

    def _state_violations(self):
        bad = []
        rec = getattr(self, "_install_rec", None) or {}
        before = rec.get("user_dtypes") or {}
        names = {id(p): n for n, p in self.named}
        user = set()
        for i, o in enumerate(self.user_optimizers):
            for p, s in o.state.items():
                if not isinstance(s, dict):
                    continue
                for k, v in s.items():
                    if not torch.is_tensor(v):
                        continue
                    user.add(id(v))
                    was = before.get((i, id(p), k))
                    if was is not None and v.is_floating_point() and v.dtype is not was:
                        bad.append(
                            f"the training script's optimizer {i} state {k!r} of {names.get(id(p), '?')} was "
                            f"{_name(was)}, now {_name(v.dtype)}"
                        )
        try:
            state = self.updater.state_dict()
        except programs.AGENT_ERRORS:
            return bad  # a failing state_dict() surfaces when the run is checkpointed
        _walk_state(state, "state_dict()", self._by_name, self._by_shape, user, bad, None)
        return bad

    def _analysis_marks(self):
        """What analyze() may not change: each parameter's and gradient's version counter (bumped by any in-place
        change), the gradient object, dtypes and devices."""
        return [
            (p._version, p.dtype, p.device, id(p.grad), None if p.grad is None else (p.grad._version, p.grad.dtype))
            for _, p in self.named
        ]

    def _analysis_changes(self, marks):
        out = []
        for (n, p), m in zip(self.named, marks):
            now = (
                p._version,
                p.dtype,
                p.device,
                id(p.grad),
                None if p.grad is None else (p.grad._version, p.grad.dtype),
            )
            if now != m:
                out.append(n)
        return out


def _name(dtype):
    return str(dtype).replace("torch.", "")


def _starting_digest(opt):
    d = getattr(opt, "_start_digest", None)
    if d is None:
        d = opt._start_digest = programs.side_digests(opt._starting_files)[0]
    return d


def _walk_state(obj, path, by_name, by_shape, skip, out, owner):
    """Floating tensors shaped like a parameter whose dtype is neither that parameter's nor float32. A tensor under a
    key that names a parameter is compared with that parameter; otherwise with the parameters of its shape. The
    training script's optimizers' own state is checked separately (skip)."""
    if torch.is_tensor(obj):
        if not obj.is_floating_point() or obj.dtype is torch.float32 or id(obj) in skip:
            return
        shape = tuple(obj.shape)
        if owner is not None and by_name[owner][0] == shape:
            d = by_name[owner][1]
            if obj.dtype is not d:
                out.append(f"{path} is {_name(obj.dtype)}, the state of parameter {owner} ({_name(d)})")
            return
        same = by_shape.get(shape)
        if same and obj.dtype not in same:
            d, n = next(iter(same.items()))
            out.append(f"{path} is {_name(obj.dtype)}, shaped like parameter {n} ({_name(d)})")
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _walk_state(v, f"{path}[{k!r}]", by_name, by_shape, skip, out, k if k in by_name else owner)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _walk_state(v, f"{path}[{i}]", by_name, by_shape, skip, out, owner)
