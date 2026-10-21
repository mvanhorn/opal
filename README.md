
<h1 align="center">OPAL: a reactive optimizer</h1>

<p align="center"><b>A PyTorch optimizer wrapper that lets a coding agent adjust the optimizer while your model trains.</b></p>

<p align="center">
  <a href="https://xingjianbai.com/reactive-optimizers/">Blog post</a> ·
  <a href="docs/GUIDE.md">Guide</a> ·
  <a href="#start-with-your-coding-agent">Start with your agent</a> ·
  <a href="#examples">Examples</a>
</p>

<!-- teaser, once docs/assets/reel.gif exists:
<p align="center"><img src="docs/assets/reel.gif" width="760" alt="Training loss and task measure of agent runs against the official recipes"></p>
-->

A network is usually trained with an optimizer chosen before training starts. OPAL (Optimizer with an Agent in the Loop)
makes the optimizer a program that an agent keeps rewriting while the run goes on. You wrap your PyTorch optimizer
with one line. At decision points the agent (Claude Code) reads what the run reports, then keeps or rewrites two small
programs: the optimizer program, which turns gradients into updates at every step, and an analysis program, which
measures what it wants to see.

Anything that can be written as optimizer code is the agent's to change: which update rule each parameter group
follows, per-group learning rates and their schedules, momentum and β₂, preconditioning, clipping, guards against loss
spikes. Your model, data, loss, batch size, number of steps and evaluation stay as they are. Until the agent changes
something, the wrapped run is bitwise identical to the plain one.

## Examples

### Reproduce our results

One run per setting, each agent starting from the task's own recipe. Each folder gives the official repository and
commit, the lines that wrap its optimizer, the commands for the official run and the reactive-optimizer run, the hardware,
time and cost, and where each number comes from. The "Reactive optimizer" column comes from the harness this package
grew from, or, for modded-nanogpt, from a pre-release of OPAL (0.1.0); the last column is one run on this release (OPAL
0.3.x, neutral prompt, Opus 5.5 at effort high) where we reran the setting. Training waited for the agent at each
decision in all of them, as OPAL does by default.

| Example | Measure | Official recipe | Reactive optimizer (one run) | Made with | On OPAL 0.3.x |
|---|---|---|---|---|---|
| [SiT-B/2](examples/sit_b2/), ImageNet 256, 400K updates, 8× H100 | FID-50K ↓ | 35.7 with our evaluator (the SiT paper reports 33.0 with the ADM evaluator; Muon: 30.3) | 23.6 | earlier harness | not rerun |
| [RAEv2-B](examples/raev2_b/), ImageNet 256, 20 epochs, 8× H100 | FID-50K ↓ | 3.39 | 3.10 asked for efficiency; 3.45 with the neutral prompt | earlier harness | 3.19 |
| [Goldfish 125M GPT](examples/goldfish_swahili/), 1 GB of Swahili, 1× H100 | FLORES-200 bits ↓ | 79.37 | 77.60 | earlier harness | 77.25 |
| [modded-nanogpt record #36](examples/modded_nanogpt/), 162M GPT, 8× H100 | validation loss ↓ | 3.2781 | 3.2749 (small: the track assumes σ = 0.0013 per run) | OPAL 0.1.0 | not rerun |

Where it did not help, the folder says so: on modded-nanogpt, a heavily tuned
benchmark, one run on OPAL 0.2.0 with the neutral prompt ended at 3.2878, worse than record #36.
More in the [blog post](https://xingjianbai.com/reactive-optimizers/).
[What we checked](docs/checks.md) · [Run times and agent costs](docs/times.md)

### Small examples (CPU)

`examples/mnist_mlp.py` (an MLP on MNIST, AdamW with a cosine schedule, 5 epochs) is the install check, run from a
clone with `pip install -e ".[examples]"`:

```bash
python examples/mnist_mlp.py                            # plain PyTorch: final test accuracy 0.9850
python examples/mnist_mlp.py --opal --backend keep      # wrapped, no agent, no cost: the same run, bit for bit
python examples/mnist_mlp.py --opal                     # Claude Code rewrites the optimizer (capped at $5)
```

On an Apple M1 Max CPU (PyTorch 2.14.1) the first two took about 15 s each (8 s of training) and ended with the same
parameters, byte for byte: both print the same `parameter checksum` (the first 12 hex digits of the sha256 of the final
parameters). The agent run (`--opal`, on OPAL 0.3.x) ended at 0.9863 after 862 s, 532 s of it waiting for the agent;
it asked 42 decisions, changed the program at 6, and cost $2.70. `tests/test_cpu_examples.py` runs the first two on synthetic data in a
few seconds.

The whole test suite runs on CPU, with no agent and no cost: `pip install -e ".[examples,test,hf]"`, then `pytest -q` in
this folder (about 3 minutes on that laptop; 3 tests are skipped: the opt-in ones that start the real Claude Code
CLI against a local mock, `OPAL_REAL_CLI=1`).

### More scripts (one GPU)

| Script | Task | Plain → `--opal` (one run, 1× H100) | Time: plain → `--opal` |
|---|---|---|---|
| `examples/cifar10_resnet.py` | ResNet-18 on CIFAR-10, SGD-Nesterov + one-cycle | test accuracy 0.9449 → 0.9456 | 66 s → 27 min |
| `examples/diffusion_mnist.py` | small UNet flow-matching diffusion on MNIST, EMA | EMA validation loss 0.1696 → 0.1623 | 71 s → 29 min |
| `examples/gpt_small.py` | 34M GPT on FineWeb tokens, AdamW + warmup-cosine | validation loss 4.4763 → 4.3419 | 186 s → 31 min |
| `examples/overfit/train.py` | ResNet-18 on CIFAR-10 with AdamW, designed to test overfitting (10% wrong labels, or 20% of the data); `--validation` shows the agent a development set | `--setting label10 --validation --test-at-end`: test accuracy 73.53% → 80.42% | → 39 min |

With `--opal` these scripts schedule up to 50 decisions (34–46 were asked in these runs), so most of the time is training waiting for the agent.

Each runs plain by default; `--opal` wraps the optimizer and `--backend keep` is a dry run with no agent.

## Prerequisites

- **A PyTorch training script** that uses `torch.optim` optimizers, on one GPU or on several GPUs of one node (DDP).
  Python ≥ 3.9, PyTorch ≥ 2.3.
- **Claude Code, logged in, on the machine that runs your training's rank 0.** OPAL's agent is a Claude Code
  session: install [Claude Code](https://docs.claude.com/en/docs/claude-code), run `claude` once and log in with a
  Claude subscription or an API key. The agent's usage is billed to that account. `max_cost_usd` caps it: it is off
  by default, a run can overshoot it by one call, and the summaries of code changes for the report (claude-opus-5-5
  at low effort; `summarize=False` turns them off) are not counted.
- **No internet or no login on the GPU machine?** Run the agent on your laptop or a login node instead and connect it
  to the training job through a shared folder (the mailbox broker, see the [guide](docs/GUIDE.md#gpu-machines-without-an-agent)).

## Start with your coding agent

Clone this repository, open your own project in a coding agent (Claude Code, Codex, Cursor or similar), and paste
this prompt. The agent installs OPAL, wires it into your training script, asks you what it cannot know, and checks
that nothing changed before anything costs money.

```text
I want to train my model with OPAL, a reactive optimizer: a coding agent that rewrites the optimizer while the
model trains. OPAL is cloned at <PATH TO OPAL>. Please set it up for this project:

1. Read <PATH TO OPAL>/README.md and <PATH TO OPAL>/skills/train-with-agent/SKILL.md, and follow the skill. Do not
   edit any file inside <PATH TO OPAL>; configure everything through opal.wrap arguments.
2. Install OPAL into the Python environment this project trains with (pip install -e <PATH TO OPAL>), and
   run `opal doctor --live` to check that torch and Claude Code are ready on the machine that will run training.
   If Claude Code is missing or not logged in there, tell me what to run; do not log in for me.
3. Find my training script and loop. Tell me what you found (optimizer, scheduler, number of steps, single GPU or
   DDP, gradient clipping, mixed precision, checkpoints) and anything OPAL does not support.
4. Ask me for the goal sentence: what the run trains and what it is judged by at the end. Do not invent the metric.
5. Wrap the optimizer and show me the diff before running anything.
6. Smoke test: about 100 steps unwrapped and 100 steps wrapped with backend="keep" (no agent, no cost); the losses
   must match step for step. Show me both.
7. Before the real run, time an update in the smoke test and run `opal estimate --total-steps N --sec-per-step S`;
   show me what it prints (how long training will wait for the agent, also as a multiple of the run's own time, and
   the cost). Ask me for a cost cap (max_cost_usd) and how many times the agent should decide, and wait for my
   go-ahead; then give me the exact command to launch it, and how to follow it (opal show, opal report).
```

## Start by hand

```bash
pip install "git+https://github.com/xingjian-bai/opal"     # or: git clone, then pip install -e ".[examples]"
opal doctor                          # torch and Claude Code ready?
```

The distribution is `opal-optimizer` (the PyPI name `opal` is an unrelated web framework; installing both would
collide on `import opal`).

```python
import opal

opt = torch.optim.AdamW(model.parameters(), lr=3e-4)                  # your optimizer, unchanged
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)  # your scheduler, unchanged
opt = opal.wrap(opt, model, scheduler=sched, total_steps=steps,
                   goal="ResNet-18 on CIFAR-10; judged by test accuracy after the last update",
                   decisions=50,    # how many times the agent is asked; each time it may also keep everything
                   backend="keep")  # the dry run: no agent, no cost (for the agent run: max_cost_usd=20 instead)

for step in range(steps):
    loss = criterion(model(x), y)
    loss.backward()
    opt.step(loss=loss)                     # pass the loss so the agent sees it
    opt.zero_grad()
    sched.step()
    if step % 500 == 0:
        opt.log(test_acc=evaluate(model))   # recorded for you; not shown to the agent
opt.close()
```

`goal` is one sentence: what you train and what it is judged by. Then:

```bash
python train.py                                  # backend="keep": no agent, must match the plain run exactly
opal doctor --live                               # one tiny agent call to check the setup
opal estimate --total-steps 10000 --sec-per-step 0.05   # the waiting for the agent and its cost, before launch
python train.py                                  # max_cost_usd=20 instead: the agent rewrites the optimizer
opal show   opal_runs/<folder>                   # one line per decision
opal report opal_runs/<folder>                   # one HTML page: loss, learning rates, rationales, code diffs
```

## How it works

```
 training script ──grads──▶ updater.py ──▶ parameters        (every step, every rank)
        │                      ▲
        └──observations──▶ analysis.py ──reports──▶ agent    (rank 0, every `every` steps / at decisions)
                               ▲                      │
                               └──── new programs ◀───┘      (at decision points)
```

- **Two programs.** `updater.py` starts as a call to your own optimizer and scheduler; `analysis.py` starts with a
  few statistics of the loss and the gradients. The agent edits both in
  `opal_runs/<folder>/workspace/`.
- **One conversation for the whole run.** The agent remembers what it tried, what it predicted and what happened, and
  ends each decision with a short rationale: what it observed, what it changed, what it expects.
- **Safe swaps.** A new optimizer program takes over at the next step on every GPU at once. It runs on trial for 20
  steps; if it raises, the run goes back to the previous program. A failure at its first step is undone exactly (when
  a copy of the parameters and optimizer state fits in `snapshot_limit_gb`, default 2); a later one is not, and
  the agent is told. Every version is kept in `programs/`.
- **Training waits for the agent by default.** `blocking=False` keeps training while the agent thinks, and `"auto"`
  waits only when an edit would land too late: faster, but noisier.
- **What the agent sees.** Your goal sentence, the parameters' names and shapes, the training losses, its own analysis
  reports, and read-only copies of your training script and model code (`share_code=False` keeps them out). Values
  you log are record-only unless you pass `show_agent=True`; with `validation=N` it also sees the development-set
  values you report with `opt.report_dev(...)`.

## Options

| Option | Default | Meaning |
|---|---|---|
| `decisions` | 50 | times the agent is asked over the run, that is, chances to change the optimizer: at each it may also keep everything as it is (a fifth of them in the first 5%; `schedule="uniform"` spaces them evenly, and `schedule` also takes segments or update numbers, see the [guide](docs/GUIDE.md#cost-and-limits)); its analysis can ask for more. A short run gets fewer: decisions fall on checks (every `every` updates), two that land on the same check merge, and none falls in the last `every` updates (the MNIST example has 42). Not set, a run whose training waits for the agent asks at most as many as keep the expected waiting within 4x its own training time (`opal estimate` shows it before launch) |
| `every` | 50 | steps between checks: losses are read, a non-finite loss is caught, the analysis runs |
| `blocking` | `True` | `False` or `"auto"` to keep training while the agent thinks |
| `max_cost_usd` | none | stop asking once the agent has spent this much |
| `backend_options` | `{}` | `model=` (default `claude-opus-5-5`), `effort=` (default `high`; the blog's agent: another prints a warning), `timeout=` for the agent; a failed agent turn is made again after 15 s, 30 s, 60 s, ... (at most 10 min apart) while `timeout` allows (`retries=0`: never). An unavailable agent (login, usage limit, API/network outage) is waited for separately, with no deadline unless `unavailable_timeout=` is set ([guide](docs/GUIDE.md#cost-and-limits)) |
| `prompt` | `"neutral"` | the agent's instructions: the run's facts and the protocol, with no recommended values (`epochs=`, `train_size=`, `batch_size=`, `task_facts=` add facts); `"aggressive"` adds a review of every group's hyperparameters at each decision; or the path of a template file |
| `validation` | none | `N`: your script evaluates a development set held out from training about every N updates, at times it does not announce (draw them at random), and reports it with `opt.report_dev(loss=..., accuracy=...)`; the agent sees each measurement and the prompt says so ([guide](docs/GUIDE.md#what-it-can-and-cannot-change)) |
| `eval_integrity` | on with `validation` | flags a run whose program moves the parameters to another point for an evaluation and back after it (an `eval_integrity` event, the report, stderr); the agent is not told; `eval_integrity_threshold=5` |

Gradient accumulation, GradScaler, checkpoints and resume, DDP, GPU machines without internet, privacy and cost are in
the [guide](docs/GUIDE.md).

### Experimental Codex backend

Claude remains the default. Select `backend="codex", backend_options=dict(model="gpt-6.1-sol", effort="high")` (`xhigh` also supported), or `opal broker <box> --backend codex --model gpt-6.1-sol --effort high`.
Log in separately with `CODEX_HOME=~/.codex-opal codex login --device-auth`; never copy or link the default home's `auth.json`.
`max_cost_usd` does not cap Codex; optional `backend_options=dict(max_tokens=N)` (broker: `--max-tokens N`) is a soft input + output token cap that an in-flight turn can exceed.
It fails closed when Codex compacts conversation history (about 450k context tokens with Codex CLI 0.160.1 defaults).
Recorded single runs on one H100, Codex vs plain: `gpt_small` validation loss 4.3125 vs 4.4763; `diffusion_mnist` EMA validation loss 0.1687 vs 0.1696.

## Status and limitations

v0.3.3, under active development. Tested on one GPU and on DDP within one node (NCCL; eager and `torch.compile`).
Multi-node, FSDP and DeepSpeed are not supported. Accelerate works when you wrap after `accelerator.prepare` (without
fp16 GradScaler or Accelerate's gradient accumulation), and the HF Trainer through `opal.hf.OpalTrainer`, both tested on
CPU; Lightning should work through `opt.observe(loss)` but is untested. The agent can make a run worse: it cannot restart or roll back training, only
fix its programs at the next decision.

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

*The name:* OPAL stands for Optimizer with an Agent in the Loop.
