"""OPAL 0.1.2-aggressive (prompt="aggressive"; the authors' wording of 2026-10-18): the neutral prompt, except that each
decision starts with a review of every parameter group's hyperparameters, the rationale has five parts (observed,
review, changed, prediction, rules), the decision message no longer says that keeping the programs is a valid decision,
and reference/example_updater.py has a feedback controller of the learning rates (off by default). Everything else is
neutral.md word for word; this module holds only the differences, applied to neutral.md and example_updater.py."""

from __future__ import annotations

# ----- the system prompt: neutral.md's DECISIONS section, old -> new ------------------------------------------------

PROMPT_EDITS = [
    (
        "At each decision you may edit your programs or keep them; keeping them is a valid decision. Before changing "
        "anything,\ncheck what you predicted at your previous decision against what happened. End every decision with a "
        "short rationale:\n",
        "At each decision, first check what you predicted at your previous decision against what happened. Then review, "
        "for\neach parameter group, whether its current hyperparameters (the update rule, the learning rate and its "
        "schedule,\nmomentum or betas, epsilon, weight decay, clipping) suit the next stretch of training, and which "
        "adjustments, in which\ndirection and by how much, would improve performance. Rules in your programs can also "
        "make such adjustments between\ndecisions, at every update, from signals they measure. End every decision with a "
        "short rationale:\n",
    ),
    (
        "2. Changed or kept: what you edited and why; or why you kept the programs and what evidence you are waiting "
        "for.\n3. Prediction: the effect you expect, by roughly when, and what result would show you were wrong.\n4. "
        "Rules: for each rule in your programs, whether it is a fixed schedule or reacts to a signal.\n",
        "2. Review: for each parameter group, whether its hyperparameters suit the next stretch, and which adjustments\n"
        "   (direction and size) you expect to help.\n3. Changed: what you edited and why.\n4. Prediction: the effect you "
        "expect, by roughly when, and what result would show you were wrong.\n5. Rules: for each rule in your programs, "
        "whether it is a fixed schedule or reacts to a signal.\n",
    ),
]

# ----- the decision message (agent.format_message) ------------------------------------------------------------------

EDIT_SENTENCE = "Edit the programs in place. "
RATIONALE_SENTENCE = (
    "End with your rationale in the five parts of DECISIONS: observed, review, changed, prediction, rules."
)


def _apply(text, edits, what):
    for old, new in edits:
        if text.count(old) != 1:
            raise RuntimeError(f"opal: the aggressive variant no longer matches {what}: {old[:80]!r}")
        text = text.replace(old, new)
    return text


def template(neutral: str) -> str:
    return _apply(neutral, PROMPT_EDITS, "prompts/neutral.md")


# ----- reference/example_updater.py: a feedback controller of the learning rates, off by default ---------------------

CONSTANTS = """

# A feedback controller of the learning rates, off by default: at every update it measures, on the device, a running
# mean of each AdamW group's update-to-weight ratio, and lr_rule() below turns that signal into a multiplier of the
# group's learning rate. To use it, set LR_CONTROLLER = True and SIGNAL_DECAY (the weight of the newest measurement in
# the running mean), and write lr_rule() from your own measurements.
LR_CONTROLLER = False
SIGNAL_DECAY = None
"""

CONTROLLER = '''

# ----- a feedback controller of the learning rates (off by default) -------------------------------------------------


def lr_rule(signal, multiplier, update, total):
    """Your rule: the new learning-rate multiplier of one group, from its signal (a 0-d tensor on the device: the
    running mean of the group's update-to-weight ratio) and its current multiplier (a 0-d tensor). Keep it on the
    device (torch.where, torch.clamp, arithmetic; no .item(), no if on a tensor). As written it changes nothing."""
    return multiplier


class LRController:
    """Per AdamW group (keyed like the group): the running mean of its update-to-weight ratio and its learning-rate
    multiplier, 0-d float32 tensors on the device. The multiplier scales the group's whole update after the rule's
    step, which for AdamW is the same as scaling its learning rate, so nothing is read on the host."""

    def __init__(self, rule, carried):
        if SIGNAL_DECAY is None:
            raise ValueError("set SIGNAL_DECAY to use the learning-rate controller")
        self.rule = rule
        self.keys = [str(g["key"]) for g in rule.groups]
        self.signal, self.mult = {}, {}
        for key, g in zip(self.keys, rule.groups):
            device = g["params"][0].device
            self.signal[key] = carried.get("signal", {}).get(key, torch.zeros((), device=device)).clone()
            self.mult[key] = carried.get("mult", {}).get(key, torch.ones((), device=device)).clone()

    def before(self):
        return [torch._foreach_mul([p.detach() for p in g["params"]], 1) for g in self.rule.groups]  # copies

    def after(self, before, update, total):
        for key, g, old in zip(self.keys, self.rule.groups, before):
            params = [p.detach() for p in g["params"]]
            delta = torch._foreach_sub(params, old)
            moved = torch.linalg.vector_norm(torch.stack(torch._foreach_norm(delta)).float())
            size = torch.linalg.vector_norm(torch.stack(torch._foreach_norm(old)).float())
            ratio = moved / size.clamp_min(torch.finfo(size.dtype).tiny)
            self.signal[key].mul_(1 - SIGNAL_DECAY).add_(ratio * SIGNAL_DECAY)
            self.mult[key] = lr_rule(self.signal[key], self.mult[key], update, total)
            torch._foreach_mul_(delta, (self.mult[key] - 1).to(delta[0].dtype))
            torch._foreach_add_(params, delta)

    def state_dict(self):
        return {"signal": dict(self.signal), "mult": dict(self.mult)}

    def load_state_dict(self, state):
        for name in ("signal", "mult"):
            mine = getattr(self, name)
            mine.update({k: v for k, v in state.get(name, {}).items() if k in mine})
'''

EXAMPLE_EDITS = [
    (
        "Nothing\nhere is a recommendation: the multipliers, knots and rule table are left at values that change "
        "nothing.",
        "Nothing\nhere is a recommendation: the multipliers, knots and rule table are left at values that change "
        "nothing, and the\nfeedback controller of the learning rates is off.",
    ),
    ("\n\n# ----- grouping by role and block", CONSTANTS + "\n\n# ----- grouping by role and block"),
    ("\n\n# ----- the updater -----", CONTROLLER + "\n\n# ----- the updater -----"),
    (
        "        self.rules = {r: RULES[r](entries.get(r, []), carried) for r in RULES}\n",
        "        self.rules = {r: RULES[r](entries.get(r, []), carried) for r in RULES}\n"
        '        old = getattr(previous, "controller", None)  # its signals and multipliers carry over\n'
        "        self.controller = (\n"
        '            LRController(self.rules["adamw"], old.state_dict() if old else {}) if LR_CONTROLLER else None\n'
        "        )\n",
    ),
    (
        "        for rule in self.rules.values():\n            rule.set_lrs(update, self.total)\n            rule.step()\n",
        "        before = self.controller.before() if self.controller else None\n"
        "        for rule in self.rules.values():\n            rule.set_lrs(update, self.total)\n            rule.step()\n"
        "        if self.controller:\n            self.controller.after(before, update, self.total)\n",
    ),
    (
        "    def state_dict(self):\n        return {r: rule.state_by_name() for r, rule in self.rules.items()}  # keyed by "
        "parameter name\n\n    def load_state_dict(self, state):\n        for r, rule in self.rules.items():\n"
        "            rule.load_state_by_name(state.get(r, {}))\n",
        "    def state_dict(self):\n        out = {r: rule.state_by_name() for r, rule in self.rules.items()}  # keyed by "
        'parameter name\n        if self.controller:\n            out["lr_controller"] = self.controller.state_dict()\n'
        "        return out\n\n    def load_state_dict(self, state):\n        for r, rule in self.rules.items():\n"
        '            rule.load_state_by_name(state.get(r, {}))\n        if self.controller and "lr_controller" in state:\n'
        '            self.controller.load_state_dict(state["lr_controller"])\n',
    ),
]


def example(base: str) -> str:
    return _apply(base, EXAMPLE_EDITS, "prompts/example_updater.py")
