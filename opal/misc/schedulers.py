"""The training script's LR schedulers: their step() is ignored while the agent's program owns the schedule, and the
user's schedule is brought up to date when control returns to a program that follows it.

While a program that does not follow the user's schedule runs (follow_user_scheduler = False), the training script's
scheduler.step() calls do nothing; the wrapper keeps a ledger instead: the user's schedule as it stood when that program
took over (every group setting of the user's optimizers, every scheduler's state) and the scheduler.step() calls made
since. When a program that follows the user's schedule takes control again (the agent restores the starting program, or
a failing program is reverted to one), the ledger is replayed: the group settings and scheduler states go back to the
handover and the missed calls run, so the schedule continues as if it had never stopped (bitwise, as the scheduler's own
arithmetic is replayed). The wrapper also records every training-script call since each new updater was installed, so
that a revert puts the schedule where the previous program would have it."""

from __future__ import annotations

import copy
import math
import warnings
import weakref

import torch

OWNER = weakref.WeakKeyDictionary()  # user scheduler -> weakref to the AgentOptimizer that owns it


_FOLLOW_CLASSES = {}


def _rebuild_scheduler(base, state):
    obj = base.__new__(base)
    obj.__dict__.update(state)
    return obj


def follows(updater):
    return bool(getattr(updater, "follow_user_scheduler", False))


def follow_class(cls):
    """A subclass of the user's scheduler class whose step() runs while the running program follows the user's
    schedule (follow_user_scheduler = True) or when the program itself calls it; the training script's calls do nothing
    while the program owns the schedule (they are recorded, see ScheduleMixin). The patch lives on the class, so
    state_dict() stays picklable, and a pickled scheduler object comes back as the plain base class. When the program
    owns the schedule, get_last_lr() reports the program's rates where it can."""
    if cls in _FOLLOW_CLASSES:
        return _FOLLOW_CLASSES[cls]

    def step(self, *args, **kwargs):
        ref = OWNER.get(self)
        owner = ref() if ref is not None else None
        if owner is None or owner._in_updater:
            return cls.step(self, *args, **kwargs)
        return owner._user_scheduler_step(self, cls, args, kwargs)

    def reduce_ex(self, protocol):
        return (_rebuild_scheduler, (cls, dict(self.__dict__)))

    sub = type(cls.__name__, (cls,), {"step": step, "__reduce_ex__": reduce_ex, "_opal_base": cls})
    sub.__module__, sub.__qualname__ = cls.__module__, cls.__qualname__
    _FOLLOW_CLASSES[cls] = sub
    return sub


def _plain_arg(v):
    """A scheduler.step() argument as it is kept for a replay (ReduceLROnPlateau's metric: a float)."""
    if torch.is_tensor(v) and v.numel() == 1:
        return float(v.detach())
    return v


def _record(calls, i, args, kwargs):
    """Append one call to a run-length list [[scheduler index, args, kwargs, count], ...]."""
    args, kwargs = tuple(_plain_arg(a) for a in args), {k: _plain_arg(v) for k, v in kwargs.items()}
    if calls and calls[-1][0] == i and calls[-1][1] == args and calls[-1][2] == kwargs:
        calls[-1][3] += 1
    else:
        calls.append([i, args, kwargs, 1])


class ScheduleMixin:
    def _schedule_state(self):
        """The user's schedule as it stands: every user optimizer group's settings (all but the parameters) and every
        user scheduler's state_dict(), as copies."""
        return dict(
            hypers=[
                [{k: copy.deepcopy(v) for k, v in g.items() if k != "params"} for g in o.param_groups]
                for o in self.user_optimizers
            ],
            scheds=[copy.deepcopy(s.state_dict()) for s in self.user_schedulers],
        )

    def _set_schedule_state(self, state):
        for o, hs in zip(self.user_optimizers, state["hypers"]):
            for g, h in zip(o.param_groups, hs):
                for k in [k for k in g if k != "params" and k not in h]:
                    del g[k]
                g.update(copy.deepcopy(h))
        for s, st in zip(self.user_schedulers, state["scheds"]):
            s.load_state_dict(copy.deepcopy(st))

    def _changed_settings(self, before):
        """User group settings that differ from `before` (a _schedule_state): {(optimizer, group, key): value}."""
        out = {}
        for i, (o, hs) in enumerate(zip(self.user_optimizers, before["hypers"])):
            for j, (g, h) in enumerate(zip(o.param_groups, hs)):
                for k, v in g.items():
                    if k != "params" and (k not in h or not _same(h[k], v)):
                        out[(i, j, k)] = copy.deepcopy(v)
        return out

    def _reapply(self, changes):
        for (i, j, k), v in (changes or {}).items():
            self.user_optimizers[i].param_groups[j][k] = copy.deepcopy(v)

    def _replay(self, calls):
        """Run recorded training-script calls on the user's schedulers, as the plain run would have."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for i, args, kwargs, n in calls:
                s = self.user_schedulers[i]
                base = getattr(type(s), "_opal_base", type(s))
                for _ in range(n):
                    base.step(s, *args, **kwargs)

    def _schedule_sync(self):
        """A program that follows the user's schedule takes control: the schedule continues as if it never stopped."""
        ledger, self._ledger = self._ledger, None
        if ledger is not None:
            self._set_schedule_state(ledger)
            self._replay(ledger["missed"])

    def _schedule_handover(self, old, new, before, build_changes):
        """The running updater changes from `old` to `new` (built after `before` was taken)."""
        if follows(new) and self._ledger is not None:
            self._schedule_sync()
            self._reapply(build_changes)  # what the new program's build() set on purpose stays
        elif not follows(new) and self._ledger is None:
            self._ledger = dict(copy.deepcopy(before), missed=[])

    def _new_install_record(self, before=None):
        """What a revert of the updater installed now needs: the user's schedule just before it was built, the ledger
        then, and the training script's scheduler.step() calls from now on."""
        self._install_rec = dict(
            before=self._schedule_state() if before is None else before,
            ledger=copy.deepcopy(self._ledger),
            calls=[],
            user_dtypes=self._user_state_dtypes(),  # misc/precision.py
        )

    def _schedule_revert(self, rec, previous_follows, build_changes=None):
        """The updater installed with `rec` failed and the previous program runs again: put the user's schedule where
        that program would have it now."""
        self._set_schedule_state(rec["before"])
        self._ledger = copy.deepcopy(rec["ledger"])
        if previous_follows:
            self._schedule_sync()
            self._replay(rec["calls"])
        else:
            if self._ledger is None:
                self._ledger = dict(copy.deepcopy(rec["before"]), missed=[])
            for i, args, kwargs, n in rec["calls"]:
                for _ in range(n):
                    _record(self._ledger["missed"], i, args, kwargs)
        self._reapply(build_changes)
        self._new_install_record()

    def _user_scheduler_step(self, sch, cls, args, kwargs):
        """The training script called sch.step(*args, **kwargs)."""
        try:
            i = next(j for j, s in enumerate(self.user_schedulers) if s is sch)
        except StopIteration:
            return cls.step(sch, *args, **kwargs)
        rec = getattr(self, "_install_rec", None)
        if rec is not None:
            _record(rec["calls"], i, args, kwargs)
        if follows(self.updater):
            if self._ledger is not None:  # a program that started following the schedule on its own
                self._schedule_sync()
            return cls.step(sch, *args, **kwargs)
        if self._ledger is None:  # a program that stopped following the schedule on its own
            self._ledger = dict(self._schedule_state(), missed=[])
        _record(self._ledger["missed"], i, args, kwargs)
        lrs = self._program_lrs(sch.optimizer)
        if lrs is not None:
            sch._last_lr = lrs

    def _program_lrs(self, optimizer):
        """For each group of a user optimizer: the learning rate the running program gives its parameters (the
        exposed group that holds the group's first parameter), or None when that cannot be read."""
        groups = list(self.param_groups)
        key = tuple(id(g) for g in groups)
        cache = getattr(self, "_lr_where", None)
        if cache is None or cache[0] != key:  # parameter -> exposed group, rebuilt only when the groups change
            where = {}
            for n, g in enumerate(groups):
                for p in g.get("params", ()):
                    where.setdefault(id(p), n)
            cache = self._lr_where = (key, where)
        out = []
        for g in optimizer.param_groups:
            n = cache[1].get(id(g["params"][0])) if g["params"] else None
            lr = groups[n].get("lr") if n is not None else None
            if not isinstance(lr, (int, float)) or isinstance(lr, bool) or not math.isfinite(lr):
                return None
            out.append(lr)
        return out


def _same(a, b):
    try:
        if torch.is_tensor(a) or torch.is_tensor(b):
            return torch.is_tensor(a) and torch.is_tensor(b) and torch.equal(a, b)
        return bool(a == b)
    except Exception:
        return a is b
