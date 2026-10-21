"""One audited update per new updater (rank 0, second update of each version): operator calls per update, and host
synchronisations per update on a GPU. These show when an updater is slow because of how it is written (values read
on the host, Python loops over tensors) rather than because of the algorithm it implements.

On a GPU the CUDA profiler is not used: synchronisations are counted with torch's sync debug mode, a flag checked at each sync, and
operators with the CPU-only profiler."""

from __future__ import annotations

import warnings

COUNT_OPS_ON_CUDA = True  # the CPU-only profiler on a GPU


def audited(fn, device):
    """Call fn() once and return its counts, or None if they cannot be taken (fn still runs; its exceptions
    propagate)."""
    if device.type != "cuda":
        return _profiled(fn)
    import torch

    try:
        prev = torch.cuda.get_sync_debug_mode()
    except Exception:
        fn()
        return None
    ops = {}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        torch.cuda.set_sync_debug_mode(1)
        try:
            if COUNT_OPS_ON_CUDA:
                ops = _profiled(fn) or {}
            else:
                fn()
        finally:
            torch.cuda.set_sync_debug_mode(prev)
    syncs = sum(1 for w in caught if "synchroniz" in str(w.message))
    return dict(ops or {}, host_syncs=int(syncs))


def _profiled(fn):
    """fn() under the CPU-only profiler: outermost operator calls (an operator called from Python, not one called by
    another operator)."""
    try:
        from torch.profiler import ProfilerActivity, profile

        try:
            prof = profile(activities=[ProfilerActivity.CPU], acc_events=True)  # no end-of-cycle warning
        except TypeError:  # torch < 2.4
            prof = profile(activities=[ProfilerActivity.CPU])
        prof.__enter__()
    except Exception:
        fn()
        return None
    try:
        fn()
    finally:
        try:
            prof.__exit__(None, None, None)
        except Exception:
            prof = None
    if prof is None:
        return None
    try:
        ops = sum(
            1
            for e in prof.events()
            if e.name.startswith("aten::") and (e.cpu_parent is None or not e.cpu_parent.name.startswith("aten::"))
        )
    except Exception:
        return None
    return dict(operator_calls=int(ops))


def describe(audit):
    if not audit:
        return "not measured"
    parts = []
    if "operator_calls" in audit:
        parts.append(f"{audit['operator_calls']} operator calls")
    if "host_syncs" in audit:
        parts.append(f"{audit['host_syncs']} host synchronisations")
    return " and ".join(parts) + " per update"
