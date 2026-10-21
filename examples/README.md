# Examples

Everything we ran with OPAL, with the numbers. Each large example has its own folder with the official repository and
commit, the lines that wrap the optimizer, the commands for both runs, the hardware, time and cost, and where each
number comes from.

## Reproduce our results

We ran each setting once, and each agent started from the task's own recipe. The "Reactive optimizer" column comes
from the harness this package grew out of (for modded-nanogpt, from OPAL 0.1.0). The last column is a rerun on OPAL
0.3 with the neutral prompt and Opus 5.5 at effort high. In all of these runs, training paused at each decision while
the agent answered, which is OPAL's default.

| Example | Measure | Official recipe | Reactive optimizer | Made with | On OPAL 0.3 |
|---|---|---|---|---|---|
| [SiT-B/2](sit_b2/), ImageNet 256, 400K updates, 8× H100 | FID-50K ↓ | 35.7 with our evaluator (the SiT paper reports 33.0 with the ADM evaluator; Muon: 30.3) | 23.6 | earlier harness | not rerun |
| [RAEv2-B](raev2_b/), ImageNet 256, 20 epochs, 8× H100 | FID-50K ↓ | 3.39 | 3.10 when asked for efficiency; 3.46 with the neutral prompt | earlier harness | 3.19 |
| [Goldfish 125M GPT](goldfish_swahili/), 1 GB of Swahili, 1× H100 | FLORES-200 bits ↓ | 79.37 | 77.60 | earlier harness | 77.25 |
| [modded-nanogpt record #36](modded_nanogpt/), 162M GPT, 8× H100 | validation loss ↓ | 3.2781 | 3.2749 (a small gain: the track assumes σ = 0.0013 per run) | OPAL 0.1.0 | not rerun |

It does not always help. On modded-nanogpt, a heavily tuned benchmark, a run on OPAL 0.2.0 with the neutral prompt
ended at 3.2878, worse than record #36. Your numbers may also differ from ours, because the agent can make different
choices from one run to the next and hosted models change between versions. Still, most of our reruns on OPAL 0.3
ended ahead of the recipe they started from. The [blog post](https://xingjianbai.com/reactive-optimizers/) has more.

## Small examples (CPU)

`mnist_mlp.py` trains an MLP on MNIST with AdamW and a cosine schedule for 5 epochs. It doubles as the install
check. From a clone, after `pip install -e ".[examples]"`:

```bash
python examples/mnist_mlp.py                            # plain PyTorch: final test accuracy 0.9850
python examples/mnist_mlp.py --opal --backend keep      # wrapped, no agent, free: the same run, bit for bit
python examples/mnist_mlp.py --opal                     # Claude Code rewrites the optimizer (capped at $5)
```

On an M1 Max CPU the first two take about 15 s each and print the same `parameter checksum` (the first 12 hex digits
of the sha256 of the final parameters). Our agent run ended at 0.9863. It took 862 s, 532 s of which was waiting for
the agent; the agent was asked 42 times, changed the program 6 times and cost $2.70.

## More scripts (one GPU)

| Script | Task | Loss ↓ (plain → OPAL) | Task metric ↑ (plain → OPAL) | Time: plain → OPAL |
|---|---|---|---|---|
| `cifar10_resnet.py` | ResNet-18 on CIFAR-10, SGD-Nesterov + one-cycle | test cross-entropy 0.2576 → 0.2512 | test accuracy 94.49% → 94.56% | 66 s → 27 min |
| `diffusion_mnist.py` | small UNet flow-matching diffusion on MNIST, EMA | EMA validation MSE 0.1696 → 0.1623 | none | 71 s → 29 min |
| `gpt_small.py` | 34M GPT on FineWeb tokens, AdamW + warmup-cosine | validation cross-entropy 4.4763 → 4.3419 | none | 186 s → 31 min |
| `overfit/train.py` | ResNet-18 with centred LogitNorm on CIFAR-10, AdamW, built to overfit (10% wrong training labels); the agent also sees a development set | test cross-entropy 2.4935 → 0.6507 | test accuracy 73.53% → 80.42% | 8 min → 39 min |

Each row is one plain run and one OPAL run on one H100. The first three scripts use recipes we wrote from common
settings; we did not tune their learning rates or other hyperparameters. On CIFAR-10 the gain is within run-to-run
noise: a second run in which the agent changed nothing ended at 94.63%. The overfit recipe is tuned: its learning
rate of 3e-3 and its 120 epochs came from a pilot on an earlier version of the model. With the current model, plain
AdamW at 3e-4 and 1e-3 ends at 75.11% and 75.31%, both below the OPAL run.

Each script runs plain by default. `--opal` wraps the optimizer and `--backend keep` makes it a free dry run. The
overfit row used `--setting label10 --lr 3e-3 --epochs 120`, plus `--opal --validation --decisions 30
--max-cost-usd 40 --test-at-end` for the agent run, with this goal:

```text
ResNet-18 (11.2M parameters) trained from scratch on 45,000 CIFAR-10 training images for 120 epochs; judged by the cross-entropy loss on the 10,000 CIFAR-10 test images after the last update (lower is better). Do not change the parameters for an evaluation and change them back afterwards.
```

These runs are short, so most of the OPAL time is training waiting for the agent. Keep the default `blocking=True`
for short runs.

## Codex backend

Our single runs with the experimental Codex backend (gpt-6.1-sol, effort high) on one H100, against plain:
`gpt_small` validation loss 4.3125 vs 4.4763, `diffusion_mnist` EMA validation loss 0.1687 vs 0.1696.
