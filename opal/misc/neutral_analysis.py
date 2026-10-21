"""The starting analysis.py of the draft neutral prompt (prompt="neutral"): raw measurements only. The default prompt keeps
programs.STARTING_ANALYSIS."""

from __future__ import annotations

TEXT = '''"""Analysis: reports what you want to see at your next decision. Runs on rank 0 only, every EVERY updates,
after the backward pass and before the update. It never changes the update; the updater computes its own signals.

obs.update, obs.total_updates, obs.losses (loss of each update since the previous call), obs.params and obs.grads
(name -> tensor), obs.updater (the running updater object), obs.metrics (values the training script chose to show
you), obs.program_ms_per_update, obs.activations (only if enabled). memory is a dict that persists across calls.
Return {{"report": <small JSON>}} to keep it for your next decision, {{"wake": "<reason>"}} to ask for a decision now.

Starting program: the mean loss of each of the last 4 calls' windows, the least-squares loss slope per 1,000 updates
and the standard deviation of per-update losses over the last window{epoch_doc}; per parameter role the
update-to-weight ratio (since the previous call, and per update) and the cosine between gradient and momentum of one
sampled tensor, and gradient and weight norms; the gradient norm of each block.
"""
import math
import re

import torch

EVERY = {every}
EPOCHS = {epochs}  # set when the run gave its epochs and the data repeat in the same order each epoch
WINDOWS = 4


def _role(name):
    return re.sub(r"\\.\\d+\\.", ".*.", name)


def _block(name):
    m = re.search(r"\\.(\\d+)\\.", name)
    return int(m.group(1)) if m else None


def _mean(xs):
    return sum(xs) / len(xs) if xs else None


def _momentum(updater):
    """param -> momentum tensor (exp_avg or momentum_buffer) of the torch optimizers the updater holds."""
    opts = list(getattr(updater, "opts", None) or [])
    opts += [v for v in vars(updater).values() if isinstance(v, torch.optim.Optimizer)] if hasattr(updater, "__dict__") else []
    out = {{}}
    for o in opts:
        for p, st in o.state.items():
            for k in ("exp_avg", "momentum_buffer"):
                if torch.is_tensor(st.get(k)):
                    out[p] = st[k]
                    break
    return out


def analyze(obs, memory):
    names = [n for n, g in obs.grads.items() if g is not None]
    if not names:
        return None
    report = dict(update=obs.update, program_ms_per_update=obs.program_ms_per_update)

    # loss: means of the last WINDOWS windows, slope and spread over the last one
    losses = [x for x in obs.losses if math.isfinite(x)]
    report["nonfinite_losses"] = len(obs.losses) - len(losses)
    means = memory.setdefault("window_means", [])
    if losses:
        means.append(_mean(losses))
        del means[:-WINDOWS]
    report["loss_window_means"] = [round(x, 5) for x in means]
    n = len(losses)
    if n >= 2:
        mx, my = (n - 1) / 2, _mean(losses)
        slope = sum((i - mx) * (y - my) for i, y in enumerate(losses)) / sum((i - mx) ** 2 for i in range(n))
        report["loss_slope_per_1000_updates"] = round(1000 * slope, 5)
        report["loss_std"] = round(math.sqrt(sum((y - my) ** 2 for y in losses) / (n - 1)), 5)
    if EPOCHS:  # the same updates one epoch earlier saw the same data
        per_epoch = round(obs.total_updates / EPOCHS)
        hist = memory.setdefault("epoch_losses", [])
        hist.extend(obs.losses)
        del hist[: max(0, len(hist) - per_epoch - len(obs.losses))]
        if len(hist) >= per_epoch + len(obs.losses) and obs.losses:
            now = [x for x in hist[-len(obs.losses) :] if math.isfinite(x)]
            then = [x for x in hist[-per_epoch - len(obs.losses) : -per_epoch] if math.isfinite(x)]
            if now and then:
                report["loss_minus_one_epoch_earlier"] = round(_mean(now) - _mean(then), 5)

    # per role: one sampled tensor (the middle one) for the update-to-weight ratio
    by_role = {{}}
    for n_ in names:
        by_role.setdefault(_role(n_), []).append(n_)
    sample = {{r: ns[len(ns) // 2] for r, ns in by_role.items()}}
    snap, since = memory.get("snapshot", {{}}), obs.update - memory.get("snapshot_update", obs.update)
    ratios = {{}}
    for r, n_ in sample.items():
        w = obs.params[n_].detach()
        if n_ in snap and since > 0:
            ratios[r] = (torch.linalg.vector_norm((w - snap[n_]).float()) / torch.linalg.vector_norm(w.float()).clamp_min(1e-12))
    memory["snapshot"] = {{n_: obs.params[n_].detach().clone() for n_ in sample.values()}}  # one tensor per role
    memory["snapshot_update"] = obs.update

    # norms per role over all tensors; the gradient-momentum cosine of the sampled tensor of each role
    gs = [obs.grads[n_].float() for n_ in names]
    gn = torch.stack(torch._foreach_norm(gs))
    wn = torch.stack(torch._foreach_norm([obs.params[n_].detach().float() for n_ in names]))
    mom = _momentum(obs.updater)
    picked = set(sample.values())
    ms = [mom.get(obs.params[n_]) if n_ in picked else None for n_ in names]
    have = [i for i, m in enumerate(ms) if m is not None]
    if have:
        dots = torch.stack([torch.dot(gs[i].reshape(-1), ms[i].float().reshape(-1)) for i in have])
        mn = torch.stack(torch._foreach_norm([ms[i].float() for i in have]))
    ratio_t = torch.stack(list(ratios.values())) if ratios else None
    gn, wn = gn.tolist(), wn.tolist()  # host reads, once per call
    dots, mn = (dots.tolist(), mn.tolist()) if have else ([], [])
    ratio_v = dict(zip(ratios, ratio_t.tolist())) if ratios else {{}}

    roles = {{}}
    for i, n_ in enumerate(names):
        acc = roles.setdefault(_role(n_), [0.0, 0.0, 0.0, 0.0, 0.0])
        acc[0] += gn[i] ** 2
        acc[1] += wn[i] ** 2
    for j, i in enumerate(have):
        acc = roles[_role(names[i])]
        acc[2] += dots[j]
        acc[3] += gn[i] ** 2
        acc[4] += mn[j] ** 2
    blocks = {{}}
    for i, n_ in enumerate(names):
        b = _block(n_)
        if b is not None:
            blocks[b] = blocks.get(b, 0.0) + gn[i] ** 2
    report.update(
        grad_norm_total=round(math.sqrt(sum(g * g for g in gn)), 6),
        grad_norm_by_role={{r: round(math.sqrt(a[0]), 6) for r, a in roles.items()}},
        weight_norm_by_role={{r: round(math.sqrt(a[1]), 4) for r, a in roles.items()}},
        grad_momentum_cosine_by_role={{
            r: round(a[2] / math.sqrt(a[3] * a[4]), 4) for r, a in roles.items() if a[3] > 0 and a[4] > 0
        }},
        update_to_weight_by_role={{r: round(v, 7) for r, v in ratio_v.items()}},
        update_to_weight_per_update_by_role={{r: round(v / since, 9) for r, v in ratio_v.items()}},
        grad_norm_by_block=[round(math.sqrt(blocks[b]), 6) for b in sorted(blocks)],
    )
    return {{"report": report}}
'''


def text(every, epochs=None) -> str:
    doc = (
        ", and the mean loss over the last window minus that over the same updates one epoch earlier" if epochs else ""
    )
    return TEXT.format(every=every, epochs=repr(epochs if epochs else None), epoch_doc=doc)


def note(epochs=None) -> str:
    """The prompt's sentences on the starting analysis."""
    epoch = (
        ", and the mean loss over the last window minus that over the same updates one epoch earlier" if epochs else ""
    )
    return (
        "The starting analysis.py reports, at each call, the mean training loss of each of the last 4 windows between "
        "calls, the loss slope per 1,000 updates and the standard deviation of per-update losses over the last window"
        + epoch
        + "; per parameter role, gradient and weight norms, and for one sampled tensor the update-to-weight ratio and the "
        "cosine between gradient and momentum; and the gradient norm of each block. These are starting measurements; "
        "replace or extend them."
    )
