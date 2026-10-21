"""Helpers for nested structures of tensors (updater state, snapshots)."""

from __future__ import annotations

import copy

import torch


def devices(obj):
    if torch.is_tensor(obj):
        return obj.device.type
    if isinstance(obj, dict):
        return {k: devices(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [devices(v) for v in obj]
    return None


def restore_devices(obj, dev, default, key=None):
    """Checkpoints are often loaded with map_location='cpu', and in DDP every rank loads rank 0's file. Put each
    tensor of the updater state back on this rank's device if it was on an accelerator, or on the CPU if it was
    there. Old checkpoints without device records: everything to the device except torch optimizers' step counters."""
    if torch.is_tensor(obj):
        if isinstance(dev, str):
            return obj.to("cpu") if dev == "cpu" else obj.to(default)
        return obj if key == "step" else obj.to(default)
    if isinstance(obj, dict):
        return {
            k: restore_devices(v, dev.get(k) if isinstance(dev, dict) else None, default, k) for k, v in obj.items()
        }
    if isinstance(obj, (list, tuple)):
        out = [
            restore_devices(v, dev[i] if isinstance(dev, list) and i < len(dev) else None, default)
            for i, v in enumerate(obj)
        ]
        return type(obj)(out) if isinstance(obj, tuple) else out
    return obj


def nbytes(obj, seen=None):
    """Bytes of the distinct tensors in a nested structure (a tensor reachable twice counts once)."""
    seen = set() if seen is None else seen
    if torch.is_tensor(obj):
        if id(obj) in seen:
            return 0
        seen.add(id(obj))
        return obj.numel() * obj.element_size()
    if isinstance(obj, dict):
        return sum(nbytes(v, seen) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(nbytes(v, seen) for v in obj)
    return 0


def to_cpu(obj, memo=None):
    """CPU copy of a nested structure; a tensor reachable twice is copied once."""
    memo = {} if memo is None else memo
    if torch.is_tensor(obj):
        if id(obj) not in memo:
            memo[id(obj)] = obj.detach().to("cpu", copy=True)
        return memo[id(obj)]
    if isinstance(obj, dict):
        return {k: to_cpu(v, memo) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_cpu(v, memo) for v in obj)
    return copy.deepcopy(obj)
