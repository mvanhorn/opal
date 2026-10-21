"""Example updater: how an optimizer program is put together. It illustrates the interface only.

Every setting comes from the training script's own optimizer, scheduler and clipping (spec["user_optimizers"],
spec["user_schedulers"], spec["clip_grad_norm"]), so with the defaults below it trains as the starting program does
when the training script uses torch.optim.AdamW, with no scheduler or with one stepped once per update that is a
LambdaLR or has a closed form (cosine, linear, step); for anything else build() raises and says what to write. Nothing
here is a recommendation: the multipliers, knots and rule table are left at values that change nothing.

To use parts of it, copy them into updater.py.
"""

import copy
import re

import torch

# ----- your choices (the defaults change nothing) -------------------------------------------------------------------

# A learning-rate multiplier per parameter role and per block index (missing entries mean no change). Roles are the
# names returned by role_of() below.
ROLE_LR_MULT = {}  # role -> multiplier
BLOCK_LR_MULT = {}  # block index -> multiplier

# A piecewise-linear multiplier on top of the starting schedule: knots (fraction of training, multiplier), fractions in
# increasing order from 0.0 to 1.0, linear in between.
KNOTS = [(0.0, 1.0), (1.0, 1.0)]

# Which rule updates which tensor: parameter name -> rule name ("adamw" or "yours"). Names not listed use "adamw".
# Moving a name from one rule to another carries its state over by name (see build_rules).
RULE_OF = {}

# Feedback on this update's loss, computed on the device: when ctx.loss is above FEEDBACK_THRESHOLD, the gradients of
# this update are multiplied by FEEDBACK_SCALE. Both None: off. Set them from your own measurements.
FEEDBACK_THRESHOLD = None
FEEDBACK_SCALE = None


# ----- grouping by role and block -----------------------------------------------------------------------------------

ROLE_PATTERNS = (  # first match wins; adjust to this model's parameter names
    ("bias", r"\.bias$"),
    ("norm", r"(norm|ln_|\.ln\d*\.|ln_f)"),
    ("embedding", r"(embed|wte|wpe|tok_emb|pos_emb)"),
    ("output", r"(lm_head|output|head\.)"),
    ("attention", r"(attn|attention|q_proj|k_proj|v_proj|o_proj)"),
    ("mlp", r"(mlp|ffn|feed_forward|fc)"),
)


def role_of(name):
    for role, pattern in ROLE_PATTERNS:
        if re.search(pattern, name):
            return role
    return "other"


def block_of(name):
    m = re.search(r"\.(\d+)\.", name)
    return int(m.group(1)) if m else None


# ----- the schedule: the training script's own, times your knots ---------------------------------------------------


def knot_multiplier(update, total):
    x = update / total
    for (x0, y0), (x1, y1) in zip(KNOTS, KNOTS[1:]):
        if x <= x1:
            return y0 if x1 == x0 else y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return KNOTS[-1][1]


def starting_multipliers(scheduler, n_groups):
    """For each group of the training script's optimizer: update -> the multiplier its scheduler gives at that update.
    The scheduler has stepped (update - 1) times when update `update` is applied."""
    if scheduler is None:
        return [lambda update: 1.0 for _ in range(n_groups)]
    fns = getattr(scheduler, "lr_lambdas", None)
    if fns:  # LambdaLR: call the training script's own functions
        return [lambda update, f=f: f(update - 1) for f in fns]
    if hasattr(scheduler, "_get_closed_form_lr"):  # cosine, linear, step...: the scheduler's own closed form

        def at(update, g):
            saved = scheduler.last_epoch
            scheduler.last_epoch = update - 1
            try:
                return scheduler._get_closed_form_lr()[g] / scheduler.base_lrs[g]
            finally:
                scheduler.last_epoch = saved

        return [lambda update, g=g: at(update, g) for g in range(n_groups)]
    raise NotImplementedError(f"write the schedule of {type(scheduler).__name__} in starting_multipliers()")


def check_stepped_per_update(scheduler, update, previous):
    """starting_multipliers() reads the scheduler as stepped once per update. While the previous program follows the
    training script's schedule, such a scheduler has stepped update - 1 times when update `update` is applied."""
    if scheduler is None or update is None or not getattr(previous, "follow_user_scheduler", False):
        return
    if scheduler.last_epoch != update - 1:
        raise NotImplementedError(
            f"the training script's {type(scheduler).__name__} has stepped {scheduler.last_epoch} times by update "
            f"{update}, not once per update: write its schedule in starting_multipliers()"
        )


# ----- the rules ----------------------------------------------------------------------------------------------------


class AdamWRule:
    """torch.optim.AdamW with the training script's settings, one group per (role, block, starting group) so each group
    can get its own learning rate. torch keeps AdamW's state in the parameter's dtype."""

    def __init__(self, entries, carried):
        # entries: (name, param, settings of the training script's group, starting multiplier function, its group index)
        groups = {}
        for name, p, settings, start, g in entries:
            key = (role_of(name), block_of(name), g)
            groups.setdefault(key, dict(params=[], names=[], start=start, base=settings["lr"], key=key, **settings))
            groups[key]["params"].append(p)
            groups[key]["names"].append(name)
        self.groups = list(groups.values())
        self.opt = torch.optim.AdamW(
            [{k: v for k, v in g.items() if k not in ("names", "start", "key", "base")} for g in self.groups],
            foreach=True,
        )
        self.by_name = {n: p for g in self.groups for n, p in zip(g["names"], g["params"])}
        for name, p in self.by_name.items():  # state keyed by parameter name, never by position
            if carried.get(name):
                self.opt.state[p] = copy.deepcopy(carried[name])  # a copy: never share state with the old updater

    def set_lrs(self, update, total):
        k = knot_multiplier(update, total)
        for spec_group, g in zip(self.groups, self.opt.param_groups):
            role, block, i = spec_group["key"]
            mult = ROLE_LR_MULT.get(role, 1.0) * BLOCK_LR_MULT.get(block, 1.0)
            g["lr"] = spec_group["base"] * spec_group["start"](update) * k * mult

    def step(self):
        self.opt.step()  # foreach kernels; reads p.grad

    def state_by_name(self):
        return {n: self.opt.state[p] for n, p in self.by_name.items() if p in self.opt.state}

    def load_state_by_name(self, state):
        for n, s in state.items():
            if n in self.by_name:
                self.opt.state[self.by_name[n]] = s
        # If you load with torch's opt.load_state_dict instead, it also brings back the saved groups' lr, betas and
        # weight decay: set your own settings again right after it.


class YourRule:
    """A slot for your own update rule. It keeps per-parameter state by name and implements no algorithm: write the
    update in step(). Create new state with torch.zeros_like(p) (the parameter's dtype) or in float32, update all
    tensors with torch._foreach_* operations, and do not read device values on the host."""

    def __init__(self, entries, carried):
        self.by_name = {name: p for name, p, *_ in entries}
        self.state = {name: copy.deepcopy(carried.get(name) or {}) for name in self.by_name}  # carried over by name

    def set_lrs(self, update, total):
        pass

    def step(self):
        if self.by_name:
            raise NotImplementedError("write your update rule in YourRule.step()")

    def state_by_name(self):
        return self.state

    def load_state_by_name(self, state):
        self.state.update({n: s for n, s in state.items() if n in self.by_name})


RULES = {"adamw": AdamWRule, "yours": YourRule}


# ----- the updater --------------------------------------------------------------------------------------------------


class Updater:
    follow_user_scheduler = False  # this program sets its own learning rates; the training script's scheduler is idle

    def __init__(self, named_params, previous=None, spec=None):
        self.named, self.total = named_params, spec["total_updates"]
        self.clip = spec.get("clip_grad_norm")  # the starting program's gradient clipping (None: none)
        # The training script's optimizers keep their state, also where this program no longer steps them: if it
        # fails, the run goes back to the previous program, which may use that state.
        self.user_optimizers = spec["user_optimizers"]
        # each parameter's settings from the training script's group that holds it
        settings, starts = {}, {}
        for i, o in enumerate(self.user_optimizers):
            if not isinstance(o, torch.optim.AdamW):
                raise NotImplementedError(
                    "this example rebuilds torch.optim.AdamW from the training script's settings, and the training "
                    f"script uses {type(o).__name__}: write a rule for it (see YourRule)"
                )
            scheduler = next((s for s in spec["user_schedulers"] if getattr(s, "optimizer", None) is o), None)
            check_stepped_per_update(scheduler, spec.get("update"), previous)
            mults = starting_multipliers(scheduler, len(o.param_groups))
            for g_index, g in enumerate(o.param_groups):
                s = {k: g[k] for k in ("betas", "eps", "weight_decay", "amsgrad", "maximize") if k in g}
                s["lr"] = g.get("initial_lr", g["lr"])
                for p in g["params"]:
                    settings[id(p)], starts[id(p)] = s, (mults[g_index], (i, g_index))
        carried = self.carry_from(previous, named_params)
        entries = {}
        for name, p in named_params:
            start, g_index = starts[id(p)]
            entries.setdefault(RULE_OF.get(name, "adamw"), []).append((name, p, settings[id(p)], start, g_index))
        self.rules = {r: RULES[r](entries.get(r, []), carried) for r in RULES}

    def carry_from(self, previous, named_params):
        """State by parameter name from the previous updater, or from the training script's optimizers when the previous
        updater was the starting program."""
        if previous is None:
            return {}
        rules = getattr(previous, "rules", None)
        if rules:
            return {n: s for rule in rules.values() for n, s in rule.state_by_name().items()}
        by_param = {id(p): s for o in self.user_optimizers for p, s in o.state.items()}
        return {n: by_param[id(p)] for n, p in named_params if id(p) in by_param}

    @property
    def param_groups(self):
        return self.rules["adamw"].opt.param_groups  # each parameter in at most one group

    def step(self, grads, update, ctx):
        # No host reads in here (.item(), float(tensor), if on a tensor): everything below stays on the device.
        if self.clip is not None:  # as the starting program: the global norm of all gradients, before the update
            torch.nn.utils.clip_grad_norm_([p for _, p in self.named], self.clip)
        if FEEDBACK_THRESHOLD is not None and FEEDBACK_SCALE is not None and ctx.loss is not None:
            live = [g for g in grads.values() if g is not None]
            factor = torch.where(
                ctx.loss > FEEDBACK_THRESHOLD, torch.full_like(ctx.loss, FEEDBACK_SCALE), torch.ones_like(ctx.loss)
            )
            torch._foreach_mul_(live, factor)
        for rule in self.rules.values():
            rule.set_lrs(update, self.total)
            rule.step()

    def state_dict(self):
        return {r: rule.state_by_name() for r, rule in self.rules.items()}  # keyed by parameter name

    def load_state_dict(self, state):
        for r, rule in self.rules.items():
            rule.load_state_by_name(state.get(r, {}))


def build(named_params, previous=None, spec=None):
    return Updater(named_params, previous, spec)
