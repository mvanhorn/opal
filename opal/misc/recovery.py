"""When a new program fails: go back to the previous one, undoing what it did exactly where a copy was kept. A program
replaced while still in its trial window is not a fallback: its successor falls back to the program before it.

Before a new updater is built (the agent's answer is applied), the wrapper keeps a CPU copy of the parameters, the
gradients, the state of the training script's optimizers and the old updater's state, unless it would exceed
snapshot_limit_gb. With that copy, a build that fails, and a new updater that fails at its first update, are undone
exactly (also what build() changed in shared objects). A step exception at a later update is not undone: the parameters and
state keep whatever that update had done, and the agent and the run's log are told so. The user's learning-rate
schedule is put where the previous program would have it (misc/schedulers.py) on every path. DDP keeps the pre-trial
copy until the trial passes: rank divergence from a consistent starting point restores it without redoing an update."""

from __future__ import annotations

import copy
import json
import traceback

import torch

from .. import programs
from .nested import devices, nbytes, restore_devices, to_cpu
from .schedulers import follows
from .skip import takes_ctx


class RecoveryMixin:
    def _hyper_state(self):
        """Hyperparameters (everything but the tensors) of every group the running program or the user's optimizers
        expose: cheap to save, and restored when a new program is rejected or reverted, so a rejected build cannot leave
        a changed learning rate behind."""
        seen, out = set(), []
        for g in [g for o in self.user_optimizers for g in o.param_groups] + list(self.param_groups):
            if id(g) not in seen:
                seen.add(id(g))
                out.append((g, {k: copy.deepcopy(v) for k, v in g.items() if k != "params"}))
        return out

    @staticmethod
    def _restore_hypers(saved):
        for g, h in saved or []:
            for k in [k for k in g if k != "params" and k not in h]:
                del g[k]
            g.update(h)

    def _agent_exempt(self):
        """The starting program's state is the user's own objects, checkpointed as in the plain run: not checked."""
        return getattr(self, "_starting_files", None)

    def _take_snapshot(self):
        """Before a new updater is built: the objects (to restore them in place) and CPU copies of the parameters,
        gradients, the user's optimizers' state and the old updater's state. Otherwise why there is none (a string):
        above snapshot_limit_gb or the old updater's state_dict() raised, on any rank (every rank decides alike)."""
        live = dict(
            grads=[p.grad for _, p in self.named],
            # (parameter, its state dict, the values in it with their devices) at this moment
            opts=[
                [(p, st, {k: (v, v.device if torch.is_tensor(v) else None) for k, v in st.items()}) for p, st in o.state.items()]
                for o in self.user_optimizers
            ],
        )
        why = None
        try:
            old = self.updater.state_dict() if self.updater is not None else None
        except programs.AGENT_ERRORS:
            old, why = None, "not taken (the previous updater's state_dict() raised)"
            self.errors.append(
                dict(
                    update=self.update,
                    where="the running updater's state_dict(), before a new updater was built (the swap went ahead)",
                    error=traceback.format_exc(limit=6)[-3000:],
                )
            )
        values = dict(
            params=[p.detach() for _, p in self.named],
            grads=[None if g is None else g.detach() for g in live["grads"]],
            opts=[[{k: v for k, v in st.items()} for _, st, _ in o] for o in live["opts"]],
            old=old,
        )
        size = nbytes(values)
        if why is None and size > self.snapshot_limit:
            why = "above snapshot_limit_gb"
            self._warn(
                "snapshot-size",
                f"the copy that undoes a failing new program would take {size / 2**30:.2f} GiB of host memory, above "
                f"snapshot_limit_gb={self.snapshot_limit / 2**30:g}; it is skipped (raise the limit to keep it), so a "
                "failure is not undone",
            )
        if not self._agree(why is None):
            return why or "not taken on another rank"
        with programs.keep_param_group_identity():
            return dict(live, copies=to_cpu(values), old_devices=devices(old), size=size)

    def _restore_snapshot(self, snap, old_updater):
        """Put back the parameters, gradients and the user's optimizers' state in place (the same tensor objects, so
        programs that hold references keep them), and the state of `old_updater` (the updater running when the copy was
        taken)."""
        c = snap["copies"]
        for (_, p), w, d, dev in zip(self.named, c["params"], self._dtypes, self._devices):
            if p.dtype is not d or p.device != dev or p.shape != w.shape:
                p.data = w.to(dev, d)
            else:
                p.data.copy_(w)
        for (_, p), g, w, dev in zip(self.named, snap["grads"], c["grads"], self._devices):
            if g is None:
                p.grad = None
                continue
            if g.dtype is not w.dtype or g.shape != w.shape or g.device != dev:
                g.data = w.to(dev)
            else:
                g.copy_(w)
            if p.grad is not g:
                p.grad = g
        for o, entries, values in zip(self.user_optimizers, snap["opts"], c["opts"]):
            o.state.clear()
            for (p, st, orig), vals in zip(entries, values):
                st.clear()
                for k, w in vals.items():
                    v, dev = orig[k]
                    if not torch.is_tensor(w):
                        st[k] = copy.deepcopy(w)
                        continue
                    if v.dtype is w.dtype and v.shape == w.shape and v.device == dev:
                        v.copy_(w)
                    else:
                        v.data = w.to(dev)
                    st[k] = v
                o.state[p] = st
        if c["old"] is not None and old_updater is not None:
            with programs.keep_param_group_identity():
                old_updater.load_state_dict(restore_devices(c["old"], snap["old_devices"], self.device))

    def _put_back_placement(self):
        """Parameters and gradients go back to the dtype and device the training script gave them (a program that
        cast one and then raised); their values keep any rounding the cast caused."""
        for (_, p), d, gd, dev in zip(self.named, self._dtypes, self._grad_dtypes, self._devices):
            if p.dtype is not d or p.device != dev:
                p.data = p.data.to(dev, d)
            g = p.grad
            if g is not None and ((gd is not None and g.dtype is not gd) or g.device != dev):
                g.data = g.data.to(dev, gd or g.dtype)

    def _recover(self, u, grads, err, trial, first=False, rollback=False):
        """The running updater raised (or broke the precision contract). Within its trial window every rank goes back
        to the previous updater: on its first update with everything restored exactly from the copy taken before it
        was built, at later updates without undoing that update. Later, one process rebuilds the previous files; DDP
        writes a crash marker and re-raises (ranks cannot agree on a revert without a collective in every update); the
        resumed run then falls back to the previous program. rollback=True rejects rank divergence: restore the entire
        trial, including later updates, without redoing an update with gradients from the discarded trajectory."""
        if self._prev_files is None:
            raise RuntimeError(
                f"opal: the program raised at update {u} and there is no earlier program to go back to:\n{err}"
            )
        snapshot = self._rank_snapshot if rollback else self._snapshot
        if self.dist and (not trial or (rollback and not isinstance(snapshot, dict))):
            try:  # every rank that crashes leaves a marker; the run folder is shared on one node
                marker = self._run_path / f"crashed-{self.digests[0]}-r{self.rank}.json"
                marker.write_text(json.dumps(dict(version=self.version, update=u, rank=self.rank, error=err[-1500:])))
            except OSError:
                pass
            if self.rank0:
                self.run.event("crashed", update=u, version=self.version, error=err)
            why = "no exact pre-swap snapshot is available" if rollback else "after its trial window"
            raise RuntimeError(
                f"opal: the program v{self.version:03d} failed at update {u}: {why}. Restart from "
                f"your checkpoint: the resumed run will use the previous program instead.\n{err}"
            )
        bad_version = self.version
        snap = snapshot if (first or rollback) and isinstance(snapshot, dict) else None
        why = self._snapshot if isinstance(self._snapshot, str) else "not taken"
        self._put_back_placement()
        if snap is not None:
            self._restore_snapshot(snap, self._prev_updater)
        self._restore_hypers(self._hypers)
        rec, changes = self._install_rec, None
        failing = self.updater
        # the updater side goes back; the analysis side stays as it is now
        files = dict(programs.closure(self._prev_files, "updater.py"))
        files.update(programs.closure(self.files, "analysis.py"))
        old, old_tag = self._prev_updater, self._prev_tag
        self._prev_files, self._prev_updater, self._snapshot, self._hypers, self._trial_until = (
            None,
            None,
            None,
            None,
            0,
        )
        self._first_pending = False
        self._rank_snapshot, self._rank_trial_equal = None, False
        if old is not None:
            self.version += 1
            self.files, self.digests = files, programs.side_digests(files)
            self.updater, self._updater_tag = old, old_tag
            self._step_ctx, self._skips_in_a_row, self._ctx_any = takes_ctx(old), 0, None
            programs.write_dir(self._prog_root / f"v{self.version:03d}", files)
            if self.rank0:
                (self.run.dir / "FINAL").write_text(f"v{self.version:03d}\n")
        else:
            bad = self._install(files, previous=failing, handover=False)
            if bad:
                bad = self._install(files, previous=None, handover=False)
            if bad:
                raise RuntimeError(f"opal: could not restore the previous programs:\n{bad}")
            changes = self._build_changes
        self._schedule_revert(rec, follows(self.updater), changes)
        self._forget_retired()
        if self._last_applied is not None:
            self._last_applied = dict(self._last_applied, reverted_at=u)
        self._timing["program_ms"], self._events = None, None
        self._timing["audit"], self._audit_at = None, u + 1
        restored = snap is not None
        if rollback:
            what = "the pre-swap parameters, gradients and optimizer/updater state were restored exactly; no update was redone"
        elif restored:
            what = "the update was undone exactly and redone"
        elif first:
            what = f"this update was not undone, because the copy that undoes it was {why}; it may be partly applied"
        else:
            what = "this update was not undone and may be partly applied"
        self.errors.append(dict(update=u, where=f"updater.step of v{bad_version:03d} (reverted: {what})", error=err))
        if self.rank0:
            self.run.event("reverted", update=u, version=bad_version, restored=restored, error=err)
            reason = "made ranks differ" if rollback else "raised in step()"
            self._log(f"update {u}: v{bad_version:03d} {reason}; reverted ({what})")
            if rollback:
                self._live.refresh()
        if restored and not rollback:
            self._in_updater = True
            try:
                self._call_updater({n: p.grad for n, p in self.named}, u)
            finally:
                self._in_updater = False
