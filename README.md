
<h1 align="center">OPAL: a reactive optimizer</h1>

<p align="center"><b>A PyTorch optimizer wrapper that lets a coding agent change the optimizer while your model trains.</b></p>

<p align="center">
  <a href="https://xingjianbai.com/reactive-optimizers/">Blog post</a> ·
  <a href="docs/GUIDE.md">Guide</a> ·
  <a href="#start-with-your-coding-agent">Start with your agent</a> ·
  <a href="#examples">Examples</a>
</p>

Most networks are trained with an optimizer that is fixed before training starts. OPAL (OPtimizer with an Agent in
the Loop) turns the optimizer into a program that an agent keeps rewriting during the run. You wrap your PyTorch
optimizer in one line. Every so often the agent (Claude Code) reads what the run reports and decides whether to
rewrite two small programs: the optimizer program, which turns gradients into updates at every step, and an analysis
program, which measures whatever the agent wants to look at.

The agent can change anything that can be written as optimizer code: the update rule of each parameter group, the
learning rates and their schedules, momentum and β₂, preconditioning, clipping, guards against loss spikes. It cannot
touch your model, data, loss, batch size, number of steps or evaluation. Until the agent changes something, the
wrapped run is bit-for-bit the same as the plain one.

## Examples

### Reproduce our results

We ran each setting once, and each agent started from the task's own recipe. Each folder has the official repository
and commit, the lines that wrap the optimizer, the commands for both runs, the hardware, time and cost, and where each
number comes from. The "Reactive optimizer" column comes from the harness this package grew out of (for
modded-nanogpt, from OPAL 0.1.0). The last column is a rerun on OPAL 0.3 with the neutral prompt and Opus 5.5 at
effort high. In all of these runs, training paused at each decision while the agent answered, which is OPAL's default.

| Example | Measure | Official recipe | Reactive optimizer | Made with | On OPAL 0.3 |
|---|---|---|---|---|---|
| [SiT-B/2](examples/sit_b2/), ImageNet 256, 400K updates, 8× H100 | FID-50K ↓ | 35.7 with our evaluator (the SiT paper reports 33.0 with the ADM evaluator; Muon: 30.3) | 23.6 | earlier harness | not rerun |
| [RAEv2-B](examples/raev2_b/), ImageNet 256, 20 epochs, 8× H100 | FID-50K ↓ | 3.39 | 3.10 when asked for efficiency; 3.46 with the neutral prompt | earlier harness | 3.19 |
| [Goldfish 125M GPT](examples/goldfish_swahili/), 1 GB of Swahili, 1× H100 | FLORES-200 bits ↓ | 79.37 | 77.60 | earlier harness | 77.25 |
| [modded-nanogpt record #36](examples/modded_nanogpt/), 162M GPT, 8× H100 | validation loss ↓ | 3.2781 | 3.2749 (a small gain: the track assumes σ = 0.0013 per run) | OPAL 0.1.0 | not rerun |

It does not always help. On modded-nanogpt, a heavily tuned benchmark, a run on OPAL 0.2.0 with the neutral prompt
ended at 3.2878, worse than record #36. Your numbers may also differ from ours, because the agent can make different
choices from one run to the next and hosted models change between versions. Still, most of our reruns on OPAL 0.3
ended ahead of the recipe they started from. The [blog post](https://xingjianbai.com/reactive-optimizers/) has more.

### Small examples (CPU)

`examples/mnist_mlp.py` trains an MLP on MNIST with AdamW and a cosine schedule for 5 epochs. It doubles as the
install check. From a clone, after `pip install -e ".[examples]"`:

```bash
python examples/mnist_mlp.py                            # plain PyTorch: final test accuracy 0.9850
python examples/mnist_mlp.py --opal --backend keep      # wrapped, no agent, free: the same run, bit for bit
python examples/mnist_mlp.py --opal                     # Claude Code rewrites the optimizer (capped at $5)
```

On an M1 Max CPU the first two take about 15 s each and print the same `parameter checksum` (the first 12 hex digits
of the sha256 of the final parameters). Our agent run ended at 0.9863. It took 862 s, 532 s of which was waiting for
the agent; the agent was asked 42 times, changed the program 6 times and cost $2.70.

The whole test suite runs on CPU with no agent and no cost: `pip install -e ".[examples,test,hf]"`, then `pytest -q`. It
takes a few minutes on a laptop. Three tests are skipped unless you set `OPAL_REAL_CLI=1`; they start the real Claude
Code CLI against a local mock.

### More scripts (one GPU)

| Script | Task | Plain → `--opal` (1× H100) | Time: plain → `--opal` |
|---|---|---|---|
| `examples/cifar10_resnet.py` | ResNet-18 on CIFAR-10, SGD-Nesterov + one-cycle | test accuracy 0.9449 → 0.9456 | 66 s → 27 min |
| `examples/diffusion_mnist.py` | small UNet flow-matching diffusion on MNIST, EMA | EMA validation loss 0.1696 → 0.1623 | 71 s → 29 min |
| `examples/gpt_small.py` | 34M GPT on FineWeb tokens, AdamW + warmup-cosine | validation loss 4.4763 → 4.3419 | 186 s → 31 min |
| `examples/overfit/train.py` | ResNet-18 on CIFAR-10 with AdamW, built to overfit (10% wrong labels); the agent also sees a development set | test accuracy 73.53% → 80.42% | → 39 min |

Each script runs plain by default. `--opal` wraps the optimizer and `--backend keep` makes it a free dry run. The
overfit row used `--setting label10 --lr 3e-3 --epochs 120`, plus `--opal --validation --decisions 30
--max-cost-usd 40 --test-at-end` for the agent run. These runs are short, so most of the `--opal` time is training
waiting for the agent. Keep the default `blocking=True` for short runs.

## Prerequisites

- **A PyTorch training script** that uses `torch.optim` optimizers, on CPU, one GPU or several GPUs of one node (DDP).
  Python 3.9 or newer, PyTorch 2.3 or newer.
- **For an agent run, Claude Code, logged in on the machine that runs rank 0.** Install
  [Claude Code](https://docs.claude.com/en/docs/claude-code), run `claude` once and log in with a Claude subscription
  or an API key. The agent's usage is billed to that account. `max_cost_usd` caps it; there is no cap by default.
  The short summaries of code changes in the report are not counted (`summarize=False` turns them off). A dry run
  with `backend="keep"` needs neither and makes no model calls.
- **No internet or no login on the GPU machine?** Run the agent on your laptop or a login node and connect it to the
  training job through a shared folder (see [the mailbox broker](docs/GUIDE.md#gpu-machines-without-an-agent)).

## Start with your coding agent

Clone this repository, open your own project in a coding agent (Claude Code, Codex, Cursor or similar) and paste this
prompt. The agent installs OPAL, wires it into your training script, asks you what it cannot know, and checks that
nothing changed before anything costs money.

```text
I want to train my model with OPAL, a reactive optimizer: a coding agent that rewrites the optimizer while the
model trains. OPAL is cloned at <PATH TO OPAL>. Please set it up for this project:

1. Read <PATH TO OPAL>/README.md and <PATH TO OPAL>/skills/train-with-agent/SKILL.md, and follow the skill. Do not
   edit any file inside <PATH TO OPAL>; configure everything through opal.wrap arguments.
2. Install OPAL into the Python environment this project trains with (pip install -e <PATH TO OPAL>), and
   run `opal doctor` (free, offline) to check that torch and Claude Code are ready on the machine that will run
   training; run `opal doctor --live` (one tiny paid agent call) only once I approve an agent run. If Claude Code is
   missing or not logged in there, tell me what to run; do not log in for me.
3. Find my training script and loop. Tell me what you found (optimizer, scheduler, number of steps, single GPU or
   DDP, gradient clipping, mixed precision, checkpoints) and anything OPAL does not support.
4. Ask me for the goal sentence: what the run trains and what it is judged by at the end. Do not invent the metric.
5. Wrap the optimizer and show me the diff before running anything.
6. Smoke test: about 100 steps unwrapped and 100 steps wrapped with backend="keep" (no agent, no cost); the losses
   must match step for step. Show me both. If I only asked for a dry run, stop here.
7. Before the real run, time an update in the smoke test and run `opal estimate --total-steps N --sec-per-step S`;
   show me what it prints (how long training will wait for the agent, also as a multiple of the run's own time, and
   the cost). Ask me for a cost cap (max_cost_usd) and how many times the agent should decide, and wait for my
   go-ahead; then give me the exact command to launch it, and how to follow it (opal show, opal report).
```

## Start by hand

```bash
pip install "git+https://github.com/xingjian-bai/opal"   # or: git clone, then pip install -e ".[examples]"
opal doctor                          # torch and Claude Code ready?
```

The package is called `opal-optimizer`, because `opal` on PyPI is an unrelated web framework. Both install as
`import opal`, so do not install both.

```python
import torch, opal

opt = torch.optim.AdamW(model.parameters(), lr=3e-4)                  # your optimizer, unchanged
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)  # your scheduler, unchanged
opt = opal.wrap(opt, model, scheduler=sched, total_steps=steps,
                goal="ResNet-18 on CIFAR-10; judged by test accuracy after the last update",
                backend="keep")    # a dry run: no agent, no cost. For the agent run, use max_cost_usd=20 instead.

for step in range(steps):
    loss = criterion(model(x), y)
    loss.backward()
    opt.step(loss=loss)                     # pass the loss so the agent sees it
    opt.zero_grad()
    sched.step()
    if step % 500 == 0:
        opt.log(test_acc=evaluate(model))   # recorded for you, not shown to the agent
opt.close()
```

`goal` is one sentence: what you train and how it is judged. Then:

```bash
python train.py                                  # backend="keep": must match the plain run exactly
opal doctor --live                               # one tiny agent call to check the setup
opal estimate --total-steps 10000 --sec-per-step 0.05   # waiting time and cost, before you launch
python train.py                                  # with max_cost_usd=20: the agent rewrites the optimizer
opal show   opal_runs/<folder>                   # one line per decision
opal report opal_runs/<folder>                   # one HTML page: loss, learning rates, rationales, code diffs
```

The guide shows how to check the keep run against your plain run byte for byte, and how to use the Hugging Face
Trainer: [Try it without an agent](docs/GUIDE.md#try-it-without-an-agent).

## How it works

```
 training script ──grads──▶ updater.py ──▶ parameters        (every step, every rank)
        │                      ▲
        └──observations──▶ analysis.py ──reports──▶ agent    (rank 0, every `every` steps / at decisions)
                               ▲                      │
                               └──── new programs ◀───┘      (at decision points)
```

- **Two programs.** `updater.py` starts as a call to your own optimizer and scheduler. `analysis.py` starts with a few
  statistics of the loss and the gradients. The agent edits both in `opal_runs/<folder>/workspace/`.
- **One conversation for the whole run.** The agent remembers what it tried, what it predicted and what happened. Each
  decision ends with a short rationale: what it saw, what it changed and what it expects.
- **Safe swaps.** A new optimizer program takes over at the next step, on every GPU at once, and runs on trial for 20
  steps. If it raises an error, the run goes back to the previous program. A failure at its first step is undone
  exactly when a copy of the parameters and optimizer state fits in `snapshot_limit_gb` (default 2); a later failure
  is not undone, and the agent is told. Every version is kept in `programs/`.
- **Training waits for the agent by default.** With `blocking=False` training keeps going while the agent thinks, and
  `"auto"` waits only when an edit would land too late. Both are faster, and noisier.
- **What the agent sees.** Your goal sentence, the names and shapes of the parameters, the training losses, its own
  analysis reports, and read-only copies of your training script and model code (`share_code=False` keeps them out).
  Values you log are only recorded, unless you pass `show_agent=True`. With `validation=N` it also sees the
  development-set values you report with `opt.report_dev(...)`.

## Options

| Option | Default | Meaning |
|---|---|---|
| `decisions` | see right | How many times the agent is asked. Each time it may also keep everything as it is. If you do not set it, a run that waits for the agent asks at most as many times as keep the waiting under 4× the run's own training time (`opal estimate` shows the number), and at most 50. If you set it, that cap is off. A fifth of the decisions fall in the first 5% of the run; `schedule="uniform"` spaces them evenly. More in the [guide](docs/GUIDE.md#cost-and-limits). |
| `every` | 50 | Steps between checks: losses are read, a non-finite loss is caught, the analysis program runs. |
| `blocking` | `True` | `False` or `"auto"` to keep training while the agent thinks. |
| `max_cost_usd` | none | Stop asking the agent once it has spent this much. |
| `backend_options` | `{}` | `model=` (default `claude-opus-5-5`), `effort=` (default `high`) and `timeout=`. A failed agent turn is retried with growing pauses; an agent that is unavailable (login, usage limit, outage) is waited for, with no deadline unless you set `unavailable_timeout=`. Details in the [guide](docs/GUIDE.md#cost-and-limits). |
| `prompt` | `"neutral"` | The agent's instructions: the facts of the run and the protocol, with no recommended values. `"aggressive"` adds a review of every group's hyperparameters at each decision. You can also pass the path of your own template. |
| `validation` | none | `N`: your script evaluates a held-out development set about every N updates, at times it does not announce, and reports it with `opt.report_dev(loss=..., accuracy=...)`. The agent sees each result. See the [guide](docs/GUIDE.md#what-it-can-and-cannot-change). |
| `eval_integrity` | on with `validation` | Flags a run whose program moves the parameters somewhere else for an evaluation and back afterwards. The agent is not told. |

Gradient accumulation, GradScaler, checkpoints and resume, DDP, GPU machines without internet, privacy and cost are
covered in the [guide](docs/GUIDE.md).

### Experimental Codex backend

Claude is the default. To use Codex, pass `backend="codex", backend_options=dict(model="gpt-6.1-sol", effort="high")`
(`xhigh` works too), or run `opal broker <box> --backend codex --model gpt-6.1-sol --effort high`. Log in separately
with `CODEX_HOME=~/.codex-opal codex login --device-auth`, and do not copy or link the `auth.json` of your default
Codex home. `max_cost_usd` does not cap Codex; `backend_options=dict(max_tokens=N)` is a soft token cap that a turn
in progress can exceed. The run stops with an error if Codex compacts the conversation (around 450k tokens of context
with Codex CLI 0.160.1). Our single runs on one H100, Codex against plain: `gpt_small` validation loss 4.3125 vs
4.4763, `diffusion_mnist` EMA validation loss 0.1687 vs 0.1696.

## Status and limitations

v0.3.4, under active development. Tested on one GPU and on DDP within one node (NCCL, eager and `torch.compile`).
Multi-node, FSDP and DeepSpeed are not supported. Accelerate works if you wrap after `accelerator.prepare`, without
fp16 GradScaler or Accelerate's gradient accumulation; the HF Trainer works through `opal.hf.OpalTrainer`. Both are
tested on CPU. Lightning should work through `opt.observe(loss)` but we have not tried it. The agent can make a run
worse: it cannot restart or roll back training, it can only fix its programs at the next decision.

## Citation

```bibtex
@misc{bai2026reactive,
  title  = {Reactive Optimizers},
  author = {Bai, Xingjian and Liang, Jia-Hao and Teoh, Jayden and Sitzmann, Vincent},
  year   = {2026},
  month  = {oct},
  note   = {Blog post}
}
```

## License

MIT, see [LICENSE](LICENSE).

*The name:* OPAL stands for OPtimizer with an Agent in the Loop.
