# SiT-B/2 on ImageNet 256, 400K updates

SiT-B/2 (a flow-matching transformer, base size, patch 2, on SD-VAE latents) trained from scratch on ImageNet 256×256 for
400,000 updates of batch 256, judged by FID-50K of the EMA weights' samples (no guidance).

| FID-50K ↓ (one run each) | Official AdamW | Muon | Reactive optimizer |
|---|---|---|---|
| at 400K updates | 35.68 | 30.29 | **23.64** |

**Reproduced on OPAL 0.3.x: pending.** The numbers above were not made with this package. They come from the
harness OPAL grew from (October 2026): a training-side host and a broker that ran Claude Code (Opus 5.5, effort
`high`) with that harness's own prompt, 94 decisions (19 every 1,251 updates, then every 5,004), training paused at each
decision. The wiring below is for OPAL 0.3.x and has not been run on a GPU yet.

## Official code

[willisma/SiT](https://github.com/willisma/SiT) at commit `cbde832a40b153ccc79603412409da9c9b0c568c`, MIT license
(Copyright (c) Meta Platforms, Inc. and affiliates). The recipe is upstream's `train.py`: AdamW, learning rate 1e-4,
betas (0.9, 0.999), weight decay 0, constant learning rate, batch 256, EMA 0.9999, the linear path with velocity
prediction.

## Wiring it to OPAL

[`opal.patch`](opal.patch) changes upstream `train.py` in five places: `--max-steps` (to stop at 400K), the
`opal.wrap(...)` call behind `--opal`, `opt.step(loss=loss)`, `opt.close()` at the end, and the flags
`--opal-max-cost-usd` and `--opal-backend`. Without `--opal` the script runs as upstream does. The patch applies
cleanly to that commit and the patched file compiles; nothing else has been checked.

```bash
git clone https://github.com/willisma/SiT && cd SiT && git checkout cbde832a40b153ccc79603412409da9c9b0c568c
git apply /path/to/opal/examples/sit_b2/opal.patch
pip install -e /path/to/opal               # into SiT's environment (environment.yml)
export ENTITY=none PROJECT=none            # upstream train.py reads these even without --wandb

# official recipe
torchrun --nproc_per_node=8 train.py --model SiT-B/2 --data-path /data/imagenet/train --max-steps 400000
# dry run: wrapped, no agent, no cost; must match the official run (check the first few hundred updates)
torchrun --nproc_per_node=8 train.py --model SiT-B/2 --data-path /data/imagenet/train --max-steps 400000 \
    --opal --opal-backend keep
# reactive optimizer, OPAL defaults (neutral prompt, 50 decisions, training waits for each)
torchrun --nproc_per_node=8 train.py --model SiT-B/2 --data-path /data/imagenet/train --max-steps 400000 \
    --opal --opal-max-cost-usd 20

# 50,000 samples of the EMA weights: upstream's SDE sampler (Euler, 250 steps), no guidance, seed 7 as in our runs
torchrun --nproc_per_node=8 sample_ddp.py SDE --model SiT-B/2 --ckpt results/<run>/checkpoints/0400000.pt \
    --num-fid-samples 50000 --num-sampling-steps 250 --cfg-scale 1.0 --global-seed 7
```

**How the FID-50K numbers were computed.** torch-fidelity's Inception-v3 in compat mode (`inception-v3-compat`) gives
the 2048-dimensional pool features of the 50,000 uint8 PNG samples; the FID is the Fréchet distance between their
mean and covariance and the `mu` and `sigma` of `jit_in256_stats.npz`, the precomputed ImageNet 256 reference
statistics that [LTH14/JiT](https://github.com/LTH14/JiT) ships in `fid_stats/` (its `prepare_ref.py` builds the
reference from the training set; the file we used has the same sha256 as the one at JiT commit `cbc743a`:
`412046720c0d496dc6d72a30eac3e22b9ef6ddac3501cce7f996674ad227ba4c`). This repository has no script for
that step. Our samples came from our own sampling script with upstream's sampler settings and seed 7, which draws its
noise per GPU differently from `sample_ddp.py` (and ran on 2 GPUs), so the command above does not give the same
50,000 images; expect the score to differ by sampling noise.

Claude Code must be installed and logged in on the machine that runs rank 0, or use the mailbox broker
([guide](../../docs/GUIDE.md#gpu-machines-without-an-agent)). The run's records land in `opal_runs/<time>-...` in the
working directory; `opal report` turns them into one page.

## Our runs

| | Official AdamW | Muon | Reactive optimizer |
|---|---|---|---|
| Hardware | 8× H100 80GB | 8× H100 80GB | 8× H100 80GB |
| Training loop time | 7.1 h | 13.0 h | 11.2 h, of which 1.05 h waiting for the agent |
| Agent cost | – | – | $11.28 |
| FID-50K at 400K | 35.684 | 30.290 | 23.643 |

- **Code.** Our runs used a modified copy of upstream `train.py`: latents precomputed (the posterior mean and standard
  deviation of the centre crop and its flip, a fresh sample at every update, as upstream's on-the-fly encoding), our own
  logging, checkpoints and in-run FID-10K. That copy, the harness and its prompt are not in this repository.
- **Evaluation.** 50,000 samples of the EMA weights, upstream's SDE sampler with Euler–Maruyama and 250 steps, no
  guidance, seed 7; FID by torch-fidelity (Inception-v3, compat mode) against JiT's ImageNet-256 reference statistics
  (`jit_in256_stats.npz`, above). This is
  not the ADM evaluator of the SiT paper, which reports 33.0 for SiT-B/2 at 400K; our official run was not checked
  against that figure.
- **Muon** is our baseline: Muon (momentum 0.95, Nesterov, 5 Newton–Schulz steps, update scale 0.2·√max(m, n)) on the
  matrices, AdamW on the rest, learning rate 1e-4, weight decay 0. Neither baseline was tuned.
- **What the agent did.** 15 of its 94 decisions changed a file, 10 of them the update rule; the last change came at
  update 190,152.
