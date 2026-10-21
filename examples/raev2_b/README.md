# RAEv2-B on ImageNet 256, 20 epochs

A diffusion transformer with a wide DDT head (DiT^DH-B) trained on representation-autoencoder latents (DINOv3
ViT-L/16 encoder, ViT-XL decoder) from scratch on ImageNet 256×256 for 25,020 updates of batch 1,024 (20 epochs),
judged by FID-50K of the EMA weights' samples, without guidance.

| FID-50K ↓ (one run each) | Official recipe (our run) | Reactive optimizer, asked for efficiency | Reactive optimizer, neutral prompt |
|---|---|---|---|
| after 25,000 updates | 3.385 | **3.101** | 3.455 (worse than official) |

The paper's figure for this setting is 3.37 (its Table 6, B at 20 epochs).

**Reproduced on OPAL 0.3.x: pending.** These runs were not made with this package. They come from the harness
OPAL grew from (October 2026): Claude Code (Opus 5.5, effort `high`), 100 decisions every 250 updates, training paused
at each decision. The better agent had a prompt that asked for efficiency ("Make this diffusion-model training run more
efficient by designing its optimization algorithm and controlling it online ... The aim is a better model from the same
25,020 updates."); with that harness's neutral prompt the agent ended worse than the official recipe. OPAL's
default is a neutral prompt.

## Official code

[nanovisionx/RAEv2](https://github.com/nanovisionx/RAEv2) at commit `8a0d238f8dc3b261aba98b217f6c79c0182e8e94`. Its
license is **CC BY-NC 4.0** (non-commercial), so this folder ships no RAEv2 code and no patch: the changes are described
below in words. The recipe is upstream's `configs/stage2/training/imagenet-dinov3l-k1.yaml` with the DiT^DH-B stage-2
block the authors gave in the repository's issue #3 and 20 epochs: GMuon (learning rate 2e-4, momentum 0.95,
Nesterov) on the matrices and AdamW (betas 0.9, 0.95) on the rest, gradient clipping at norm 1, EMA 0.9995. Upstream's
schedule keeps the learning rate at 2e-4 until epoch 25 (its warmup does not start from zero) and then decays it
linearly to 2e-5 by epoch 50, so in 20 epochs it stays constant.

## Wiring it to OPAL (instructions; untested on 0.3.x)

1. Copy `configs/stage2/training/imagenet-dinov3l-k1.yaml` and in the copy set `hidden_size: [768, 2048]`,
   `depth: [12, 2]`, `num_heads: [12, 16]`, `base_model_depth: 4` under `stage_2.params`, and `epochs: 20` under
   `training`.
2. In `src/train.py`, right after the block that calls `build_scheduler(...)` and before any checkpoint is loaded,
   wrap the two optimizers inside RAEv2's composite optimizer (GMuon and AdamW), not the composite itself: its
   `MuonAdamW` subclasses `torch.optim.Optimizer` without calling `Optimizer.__init__`, so `opal.wrap(optimizer, ...)`
   fails with `AttributeError: 'MuonAdamW' object has no attribute '_optimizer_step_pre_hooks'`. The scheduler stays
   the one `build_scheduler` made, which sets the learning rates of the same parameter groups:

   ```python
   import os
   import opal

   optimizer = opal.wrap(
       [optimizer._muon, optimizer._adamw], model, scheduler=scheduler,
       total_steps=config.training.epochs * steps_per_epoch,
       goal="DiT^DH-B trained from scratch on ImageNet 256x256 (DINOv3 representation-autoencoder latents) for "
            "20 epochs; judged by the FID of samples from the EMA weights after the last update (lower is better)",
       clip_grad_norm=config.training.clip_grad,   # the starting program clips; the agent sees unclipped gradients
       backend=os.environ.get("OPAL_BACKEND", "claude"),   # "keep": no agent
       max_cost_usd=float(os.environ["OPAL_MAX_COST_USD"]) if "OPAL_MAX_COST_USD" in os.environ else None,
   )
   ```

   Checked on CPU only, with a copy of `MuonAdamW` (SGD standing in for GMuon), RAEv2's linear `LambdaLR` and
   `backend="keep"`: the wrapped run ended with the same parameters as the plain one, byte for byte. PyTorch warns once
   that `lr_scheduler.step()` was called before `optimizer.step()`, since the composite's own `step` is no longer
   called; the schedule is unaffected.

3. In `src/stage2/engine.py`, delete the `clip_grad_norm_` call before the optimizer step (step 2 gave the clipping to
   OPAL) and pass the update's mean training loss to the step: `optimizer.step(loss=loss)`, called once per update.
4. Run the dry run first for a few hundred updates and check that the loss matches the unwrapped run.

```bash
CFG=<your 20-epoch config>
# Each run needs its own EXPERIMENT_NAME: RAEv2's train.py requires it and resumes from the newest checkpoint in
# ckpts/$EXPERIMENT_NAME, so a shared name would start the dry run or the agent run from another run's checkpoint.
EXPERIMENT_NAME=raev2-b20-official torchrun --nproc_per_node 8 src/train.py --config $CFG --precision bf16 --compile   # official: upstream code without steps 2-3
EXPERIMENT_NAME=raev2-b20-keep OPAL_BACKEND=keep \
    torchrun --nproc_per_node 8 src/train.py --config $CFG --precision bf16 --compile                                  # dry run
EXPERIMENT_NAME=raev2-b20-agent OPAL_MAX_COST_USD=20 \
    torchrun --nproc_per_node 8 src/train.py --config $CFG --precision bf16 --compile                                  # agent
```

## Our runs

| | Official recipe | Asked for efficiency | Neutral prompt |
|---|---|---|---|
| Hardware | 8× H100 80GB | 8× H100 80GB | 8× H100 80GB |
| Training time | 3.1 h | 4.6 h, of which 1.4 h waiting for the agent | – |
| Agent cost | – | $15.99 | $9.94 |
| Decisions answered / that changed code | – | 100 / 16 (the last at decision 45) | 100 / 6 |
| FID-50K at 25,000 | 3.385 | 3.101 | 3.455 |

- **Code.** Our runs used upstream's code with an optimizer hook for the harness and our own logging; launched as above. That code and the harness are not in this repository.
- **Evaluation.** Online FID-50K during training (every 5,000 updates) on the ImageNet validation labels, 50 Euler
  steps, no guidance, EMA weights; torch-fidelity against the default ImageNet 256 reference statistics
  (`imagenet_256_fid_stats`) of upstream's pinned evaluator, `nanovisionx/nanogen-evals` at `19dfb4c`.
