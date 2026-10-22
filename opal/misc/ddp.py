"""Collectives on the control group (gloo) and the periodic check that every rank holds the same parameters."""

from __future__ import annotations

import hashlib
import sys

import torch
import torch.distributed as dist
from torch.distributed.algorithms.join import JoinHook


UNEVEN_INPUTS_ERROR = (
    "opal: uneven inputs across DDP ranks (torch.distributed.algorithms.Join) are not supported; "
    "give every rank the same number of steps"
)


class _UnevenInputsHook(JoinHook):
    def __init__(self, optimizer):
        self.optimizer = optimizer

    def main_hook(self):
        self.optimizer._check_join_inputs(joined=True)




def _parameter_chunks(tensor, limit=1_048_576):
    """Yield logical row-major chunks without reshaping an entire noncontiguous parameter.

    Narrow before making a copy. Each resolved/contiguous/CPU temporary has at most `limit` elements,
    even for transposes, strided slices, conjugate/negative views and scalar parameters.
    """
    if tensor.numel() <= limit:
        yield tensor.reshape(-1)
    elif tensor.is_contiguous():
        yield from tensor.view(-1).split(limit)
    else:
        # Split the first nontrivial axis; completed leading axes remain in logical C order.
        axis = next(i for i, n in enumerate(tensor.shape) if n > 1)
        per_slice = tensor.numel() // tensor.shape[axis]
        width = max(1, limit // per_slice)
        for start in range(0, tensor.shape[axis], width):
            yield from _parameter_chunks(tensor.narrow(axis, start, min(width, tensor.shape[axis] - start)), limit)


class DDPMixin:
    def join_hook(self, **kwargs):
        return _UnevenInputsHook(self)

    @property
    def join_device(self):
        return self.device

    @property
    def join_process_group(self):
        # Join requires the model and optimizer to use the same process group.
        from ..context import _wrappers

        for model in _wrappers(self.model):
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                return model.process_group
        return dist.group.WORLD

    def _check_join_inputs(self, joined=False):
        if not self.dist or not self._join_config.enable:
            return
        active = torch.tensor([0 if joined else 1])
        dist.all_reduce(active, op=dist.ReduceOp.MIN, group=self.ctrl)
        if not active.item():
            raise RuntimeError(UNEVEN_INPUTS_ERROR)

    def _agree(self, ok: bool) -> bool:
        """True on every rank only if `ok` on every rank (one gloo all-reduce; no-op on one process)."""
        if not self.dist:
            return ok
        t = torch.tensor([1 if ok else 0])
        dist.all_reduce(t, op=dist.ReduceOp.MIN, group=self.ctrl)
        return bool(t.item())

    def _from_rank0(self, flag: bool) -> bool:
        """Rank 0's flag on every rank (one gloo broadcast; no-op on one process)."""
        if not self.dist:
            return flag
        t = torch.tensor([1 if (self.rank0 and flag) else 0])
        dist.broadcast(t, src=0, group=self.ctrl)
        return bool(t.item())

    def _rank_differences(self):
        """Compare SHA-256 digests of every tensor's dtype, shape and ordered bytes (two gloo all-reduces).

        These collision-resistant digests detect permutations that sum/sum-of-squares misses. Checks copy parameter
        bytes to CPU in bounded chunks and synchronize device work at check boundaries. Optimizer state is neither
        compared nor broadcast: a sharded optimizer keeps different state on each rank by design.
        """
        digests = []
        for _, p in self.named:
            digest = hashlib.sha256(f"{p.dtype}:{tuple(p.shape)}".encode())
            for chunk in _parameter_chunks(p.detach()):
                raw = chunk.resolve_conj().resolve_neg().contiguous().view(torch.uint8).cpu().numpy()
                digest.update(memoryview(raw))
            digests.append(list(digest.digest()))
        s = torch.tensor(digests, dtype=torch.uint8)
        lo, hi = s.clone(), s.clone()
        dist.all_reduce(lo, op=dist.ReduceOp.MIN, group=self.ctrl)
        dist.all_reduce(hi, op=dist.ReduceOp.MAX, group=self.ctrl)
        bad = torch.nonzero((lo != hi).any(dim=1)).flatten().tolist()
        return [self.named[i][0] for i in bad]

    def _check_ranks(self, u):
        """Reject a fresh updater's drift from a consistent starting point; otherwise repair and flag the run."""
        names = self._rank_differences()
        self._rank_checked_at = u
        if not names:
            return names
        shown = ", ".join(names[:20]) + (f" and {len(names) - 20} more" if len(names) > 20 else "")
        if u <= self._trial_until and self._rank_trial_equal:
            msg = (
                f"Your updater v{self.version:03d} made the ranks' parameters differ at update {u} "
                f"({len(names)} of {len(self.named)} tensors: {shown}). They agreed before this trial. "
                "Check rank-local state: for example, momentum copied from a sharded optimizer contains only "
                "this rank's shard; using it to update every matrix independently makes ranks diverge. "
                "The new updater failed its trial."
            )
            if self.rank0:
                self.run.event(
                    "rank_divergence", update=u, version=self.version, tensors=len(names), names=names,
                    snapshot_available=isinstance(self._rank_snapshot, dict), error=msg,
                )
            self._recover(u, None, msg, trial=True, rollback=True)
            return names
        for _, p in self.named:
            dist.broadcast(p.data, src=0)
        self._ranks_repaired_at.append(u)
        if self.rank0:
            msg = (
                f"parameters differed across ranks at update {u} ({len(names)} of {len(self.named)} tensors: {shown}); "
                "rank 0's parameters were broadcast, the updater's state was not. The updater must act identically "
                "on every rank: state it keeps per rank (for example momentum copied from a sharded optimizer) "
                "makes the ranks drift again."
            )
            self.errors.append(dict(update=u, where="rank consistency check", error=msg))
            self.run.event("ranks_resynced", update=u, version=self.version, tensors=len(names), names=names)
            self._live.refresh()
            print(f"[opal] WARNING: {msg}", file=sys.stderr, flush=True)
        return names
