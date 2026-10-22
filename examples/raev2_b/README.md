# RAEv2-B on ImageNet 256, 20 epochs

We trained a diffusion transformer with a wide DDT head, DiT^DH-B, from scratch on ImageNet 256×256.
It uses representation-autoencoder latents from a DINOv3 ViT-L/16 encoder and ViT-XL decoder.
Training ran for 25,020 updates with batch 1,024, or 20 epochs.
The score is FID-50K of samples from the EMA weights, without guidance.

| FID-50K ↓ (one run each) | Official recipe (our run) | Reactive optimizer |
|---|---|---|
| after 25,000 updates | 3.385 | **3.192** |

The paper's figure for this setting is 3.37 (its Table 6, B at 20 epochs).

The agent run used OPAL 0.3 with the neutral prompt: 50 decisions of which 6 changed the program,
$13.30, 3 h 45 min on 8× H100. This wall time includes evaluations and waiting for the agent.

## Official code

[nanovisionx/RAEv2](https://github.com/nanovisionx/RAEv2) at commit `8a0d238f8dc3b261aba98b217f6c79c0182e8e94`.
The license is CC BY-NC 4.0, for non-commercial use. This folder includes no RAEv2 code or patch.
The changes are described below.

The recipe uses upstream's `configs/stage2/training/imagenet-dinov3l-k1.yaml`, the DiT^DH-B stage-2 block from the
authors' issue #3, and 20 epochs. It uses GMuon on matrices with learning rate 2e-4, momentum 0.95 and Nesterov.
AdamW handles the rest with betas (0.9, 0.95). Gradient clipping is at norm 1 and EMA is 0.9995.
The upstream schedule holds the rate at 2e-4 until epoch 25, then decays linearly to 2e-5 by epoch 50.
Its warmup does not start at zero. For 20 epochs, the rate stays constant.

## Wiring it to OPAL

1. Copy `configs/stage2/training/imagenet-dinov3l-k1.yaml` and in the copy set `hidden_size: [768, 2048]`,
   `depth: [12, 2]`, `num_heads: [12, 16]`, `base_model_depth: 4` under `stage_2.params`, and `epochs: 20` under
   `training`.
2. In `src/train.py`, right after the block that calls `build_scheduler(...)` and before any checkpoint is loaded,
   wrap GMuon and AdamW inside RAEv2's composite optimizer. `MuonAdamW` subclasses `torch.optim.Optimizer` without
   calling `Optimizer.__init__`. Wrapping the composite with `opal.wrap(optimizer, ...)` fails with
   `AttributeError: 'MuonAdamW' object has no attribute '_optimizer_step_pre_hooks'`.
   Keep the scheduler from `build_scheduler`. It sets rates on the same parameter groups:

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

   A CPU check used a copy of `MuonAdamW` with SGD in place of GMuon, RAEv2's linear `LambdaLR` and
   `backend="keep"`. The wrapped and plain runs ended with identical parameter bytes.
   PyTorch warns once that `lr_scheduler.step()` preceded `optimizer.step()` because the composite's `step` is no
   longer called. The schedule is unaffected.

3. In `src/stage2/engine.py`, delete the `clip_grad_norm_` call before the optimizer step. OPAL now handles clipping.
   Pass the update's mean training loss to `optimizer.step(loss=loss)`, once per update.
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

| | Official recipe | Reactive optimizer |
|---|---|---|
| Hardware | 8× H100 80GB | 8× H100 80GB |
| Reported run time, including agent waits | 3.21 h | 3.75 h, of which 0.49 h waiting for the agent |
| Agent cost | – | $13.30 |
| Decisions answered / that changed code | – | 50 / 6 |
| FID-50K at 25,000 | 3.385 | 3.192 |

The official run's log records 3.21 h including evaluations and 2.65 h without them.

Both runs used upstream code with our logging, launched as above.
That code is not in this repository.

We evaluated FID-50K every 5,000 updates during training, using ImageNet validation labels, 50 Euler steps,
no guidance and EMA weights. Scoring used torch-fidelity against `imagenet_256_fid_stats`, the default ImageNet 256
reference statistics of upstream's pinned `nanovisionx/nanogen-evals` evaluator at `19dfb4c`.
