"""OPAL in a Hugging Face Trainer: OpalTrainer is the Trainer with its optimizer wrapped by opal.wrap.

    from opal.hf import OpalTrainer

    trainer = OpalTrainer(model=model, args=args, train_dataset=train, eval_dataset=dev,
                          opal=dict(goal="...", max_cost_usd=20))   # opal.wrap's keyword arguments
    trainer.train()

The Trainer builds its optimizer and learning-rate scheduler as usual; OpalTrainer builds the scheduler on the Trainer's
own torch optimizer and then wraps that optimizer with it, so the scheduler stays on the Trainer's own optimizer (a scheduler built on the wrapper would overwrite the
rates the agent's program sets) and total_steps is the Trainer's number of updates. The Trainer calls optimizer.step()
without the loss, so OpalTrainer adds up the loss training_step returns for each micro-batch of an update and hands it
to OPAL at that update's step(): the update's mean loss (also at the end of an epoch, where an update can have fewer or
more micro-batches than gradient_accumulation_steps). The Trainer's evaluation values (eval_loss, ...) go into OPAL's
report and are not shown to the agent.

Clipping: the Trainer clips the gradients (max_grad_norm, 1.0 by default) before the optimizer's step; that clipping
stays the Trainer's, outside the optimizer, and the agent's program gets clipped gradients and cannot change it. This
keeps the run bit for bit the plain Trainer's. To hand the clipping to the agent, turn the Trainer's off and give it to
OPAL: TrainingArguments(max_grad_norm=0) and opal=dict(clip_grad_norm=1.0). The run then starts the same up to rounding
(accelerate computes the Trainer's gradient norm its own way, so not bit for bit: about 3e-4 apart after 6 updates of
the tiny test model). clip_grad_norm in opal= while max_grad_norm is set raises: the gradients would be clipped twice.

Needs transformers (pip install "opal-optimizer[hf]"). Tested on CPU with transformers 5.20 and accelerate 1.15.
"""

from __future__ import annotations

import torch

try:
    from transformers import Trainer
except ImportError as exc:  # pragma: no cover - the optional extra
    raise ImportError('opal.hf needs transformers: pip install "opal-optimizer[hf]"') from exc

from .wrapper import AgentOptimizer, wrap


class OpalTrainer(Trainer):
    def __init__(self, *args, opal=None, **kwargs):
        opal = dict(opal or {})
        targs = kwargs.get("args") if "args" in kwargs else (args[1] if len(args) > 1 else None)
        if opal.get("clip_grad_norm") is not None and (targs is None or getattr(targs, "max_grad_norm", None)):
            raise ValueError(
                "OpalTrainer: opal= sets clip_grad_norm and args.max_grad_norm is set too, so the gradients would be "
                "clipped twice; to hand the clipping to the agent, also pass TrainingArguments(max_grad_norm=0)"
            )
        super().__init__(*args, **kwargs)
        self.opal_kwargs = opal
        self._opal_losses = []  # the losses training_step returned for this update's micro-batches
        self._opal_loss_step = None

    def create_scheduler(self, num_training_steps, optimizer=None):
        """Build the Trainer's schedule on its own torch optimizer, then wrap that optimizer with it. Transformers 4
        calls this from create_optimizer_and_scheduler; 5 calls it after accelerate has wrapped the optimizer, so the
        AgentOptimizer takes the torch optimizer's place inside accelerate's wrapper."""
        holder = optimizer if optimizer is not None else self.optimizer
        raw = holder
        while isinstance(getattr(raw, "optimizer", None), torch.optim.Optimizer):  # accelerate's wrapper
            raw = raw.optimizer
        if isinstance(raw, AgentOptimizer):  # wrapped already
            return super().create_scheduler(num_training_steps, optimizer=optimizer)
        scheduler = super().create_scheduler(num_training_steps, optimizer=raw)
        agent = wrap(raw, self.model, scheduler=scheduler, total_steps=num_training_steps, **self.opal_kwargs)
        agent._loss_hook = self._update_loss  # step() takes the update's loss from here
        if self.optimizer is raw:
            self.optimizer = agent
        else:
            outer = self.optimizer
            while getattr(outer, "optimizer", None) is not raw:
                outer = outer.optimizer
            outer.optimizer = agent
        return scheduler

    def training_step(self, *args, **kwargs):
        if self._opal_loss_step != self.state.global_step:
            self._opal_losses = []  # a GradScaler-skipped update never called the optimizer's loss hook
            self._opal_loss_step = self.state.global_step
        loss = super().training_step(*args, **kwargs)
        self._opal_losses.append(loss.detach())
        return loss

    def _update_loss(self):
        """The update's mean loss from its micro-batches' losses as training_step returns them: each is already scaled
        for the update in recent Transformers (by current_gradient_accumulation_steps, or by the number of items), so
        they add up to it; older ones divide by gradient_accumulation_steps whatever the update's micro-batches are."""
        losses, self._opal_losses = self._opal_losses, []
        if not losses:
            return None
        total = torch.stack([x.float().reshape(()) for x in losses]).sum()
        if not hasattr(self, "current_gradient_accumulation_steps"):
            total = total * self.args.gradient_accumulation_steps / len(losses)
        return total

    def log(self, logs, *args, **kwargs):
        super().log(logs, *args, **kwargs)
        opt = agent_optimizer(self)
        if opt is not None:
            shown = {k: v for k, v in logs.items() if k.startswith("eval_") and isinstance(v, (int, float))}
            if shown:
                opt.log(**shown)  # in OPAL's report; not shown to the agent


def agent_optimizer(trainer):
    """The AgentOptimizer inside a Trainer's optimizer (accelerate wraps it), or None before it is built."""
    opt = trainer.optimizer
    while opt is not None and not isinstance(opt, AgentOptimizer):
        opt = getattr(opt, "optimizer", None)
    return opt
