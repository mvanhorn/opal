"""Skipping an update: a program that sets can_skip = True gets a copy of the parameters and of its state before each
update, and when its step returns True that update is undone. The program also gets this update's loss (the mean over
ranks, so every rank decides the same) when its step takes a third argument, ctx."""

from __future__ import annotations

import inspect
import types

import torch
import torch.distributed as dist

from .. import programs
from .nested import devices, restore_devices

MAX_SKIPS_IN_A_ROW = 10  # after this many, the next update is applied anyway and the agent is woken


def takes_ctx(updater):
    try:
        params = [
            p
            for p in inspect.signature(updater.step).parameters.values()
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ]
    except (TypeError, ValueError):
        return False
    return len(params) >= 3


def _tensor_ids(obj, out=None):
    out = set() if out is None else out
    if torch.is_tensor(obj):
        out.add(id(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            _tensor_ids(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _tensor_ids(v, out)
    return out


def user_optimizer_pre_step(ref):
    """A step pre-hook for each of the training script's optimizers: while a can_skip program's step runs, copy that
    optimizer's state just before it steps (unless the program's state_dict() already holds it), so that a skipped
    update also leaves the user's optimizer as it was. Does nothing otherwise."""

    def hook(opt, args, kwargs):
        w = ref()
        cap = getattr(w, "_skip_capture", None) if w is not None else None
        if cap is None or id(opt) in cap["opts"]:
            return None
        tensors = [v for st in opt.state.values() for v in st.values() if torch.is_tensor(v)]
        if tensors and all(id(t) in cap["ids"] for t in tensors):
            cap["opts"][id(opt)] = None  # already copied with the program's state
            return None
        cap["opts"][id(opt)] = (opt, [(p, st, {k: (v, clone(v)) for k, v in st.items()}) for p, st in opt.state.items()])
        return None

    return hook


def _restore_optimizer_state(opt, saved):
    opt.state.clear()
    for p, st, vals in saved:
        st.clear()
        for k, (v, c) in vals.items():
            if torch.is_tensor(v):
                v.copy_(c)
                st[k] = v
            else:
                st[k] = c
        opt.state[p] = st


def clone(obj):
    if torch.is_tensor(obj):
        return obj.detach().clone()
    if isinstance(obj, dict):
        return {k: clone(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(clone(v) for v in obj)
    return obj


class SkipMixin:
    def _ctx(self, u):
        """What a step(grads, update, ctx) receives: ctx.loss, this update's loss as a 0-d tensor on the device (the
        mean over ranks; None if the training script passes no loss), and ctx.update."""
        loss = self._step_loss
        if loss is not None:
            loss = loss.detach().float().mean() if torch.is_tensor(loss) else torch.tensor(float(loss))
            loss = loss.reshape(()).to(self.device, torch.float32).clone()
        if self.dist:
            loss = self._mean_over_ranks(loss)
        return types.SimpleNamespace(loss=loss, update=u, skipped_in_a_row=self._skips_in_a_row)

    def _mean_over_ranks(self, loss):
        """Every rank joins, whether or not the training script passed it a loss (some scripts pass it on rank 0
        only): the mean over the ranks that did. Whether any rank passes one is agreed once per installed updater
        (one gloo all-reduce); afterwards one all-reduce of (sum, count) on the device stream per update, no host
        sync. None on every rank when no rank passes a loss."""
        if getattr(self, "_ctx_any", None) is None:
            t = torch.tensor([0 if loss is None else 1])
            dist.all_reduce(t, group=self.ctrl)
            self._ctx_any = int(t.item())
        if not self._ctx_any:
            return None
        v = torch.zeros(2, device=self.device, dtype=torch.float32)
        if loss is not None:
            v[0] = loss
            v[1] = 1.0
        dist.all_reduce(v)  # default group
        return v[0] / v[1]

    def _call_updater(self, grads, u):
        """Run the updater's step for update u, undoing it when it asks to skip."""
        up = self.updater
        args = (grads, u, self._ctx(u)) if self._step_ctx else (grads, u)
        can_skip = bool(getattr(up, "can_skip", False))
        if can_skip:
            with programs.keep_param_group_identity():
                state = up.state_dict()
                before = (clone([p for _, p in self.named]), clone(state), devices(state))
            # the user's optimizers this step drives are copied just before they step (user_optimizer_pre_step)
            self._skip_capture = dict(ids=_tensor_ids(state), opts={})
        try:
            ret = up.step(*args)
        finally:
            captured, self._skip_capture = getattr(self, "_skip_capture", None), None
        if ret is None or ret is False:
            self._skips_in_a_row = 0
            return
        if not can_skip:
            self._warn(
                "skip-without-can_skip", "the updater's step returned a value; set can_skip = True to skip updates"
            )
            return
        if not bool(ret):  # one host read of the flag (a 0-d tensor), only for programs that can skip
            self._skips_in_a_row = 0
            return
        if self._skips_in_a_row >= MAX_SKIPS_IN_A_ROW:
            self._skips_in_a_row = 0
            self.errors.append(
                dict(
                    update=u,
                    where="skipping updates",
                    error=f"your program asked to skip more than {MAX_SKIPS_IN_A_ROW} updates in a row; update {u} was "
                    "applied anyway",
                )
            )
            self._skip_alarm = True
            return
        params, state, devs = before
        for (_, p), w in zip(self.named, params):
            p.data.copy_(w)
        with programs.keep_param_group_identity():
            up.load_state_dict(restore_devices(state, devs, self.device))
        for saved in (captured or {}).get("opts", {}).values():
            if saved is not None:
                _restore_optimizer_state(*saved)
        self._skips_in_a_row += 1
        self._skipped.append(u)
        del self._skipped[:-1000]
