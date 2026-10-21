# OPAL guide

Everything the [README](../README.md) leaves out: how it runs, what the agent edits, fitting it into your script, cost,
privacy, safety, clusters.

## What you get

Each run writes `report.html` in its folder (`opal_runs/<folder>/`): the training loss with a tick at each decision,
when each decision was asked and when its edit took effect, a table of the code changes (files, changed constants, a
short description and tags), what the agent said at each decision, and the code diff of each change. It is rewritten
after every decision and at the end, so you can watch it while the run goes. `opal show <folder>` prints one line per
decision (when it was asked, what became of it, its time and cost, the start of the rationale); `opal report <folder>`
rewrites the page at any time. Every program version is in `programs/`, every decision's message and answer in
`decisions/`, and the events in `events.jsonl`.

## How it works

**Only rank 0 talks to the agent; by default training waits for each of its answers.**

```
rank 0, main thread:   step → step → step → … → swap → step …
rank 0, background:       └─ claude process (one conversation for the whole run) ─┘ answer ready
other ranks:           step → step → step → … → swap → step …
```

(The picture is `blocking=False`; with the default `blocking=True` every rank waits at the decision instead of
stepping on.)

- Every `every` updates (default 50) rank 0 reads its losses, runs the agent's analysis when due, and, at scheduled
  points or when the agent's analysis asks, hands a decision to a background thread, which runs the agent CLI.
  By default training waits for the answer; with `blocking=False` it continues on the current program meanwhile.
- When the answer is ready, the new program is swapped in at the next update boundary: at once on one GPU; in DDP,
  rank 0 announces it in an asynchronous broadcast every `sync_every` updates (default 20) on a separate gloo group,
  and every rank swaps at the same update.
- All ranks wait at every decision by default (`blocking=True`, as in our published runs); with `blocking="auto"`
  only at the decisions whose answer would otherwise land late (below), and with `blocking=False` only once when the
  loss turns non-finite. Waiting happens on the gloo group, so a long agent turn never trips an NCCL timeout.
- A new updater runs on trial for `trial_steps` updates (default 20): if it raises on any rank, every rank goes back to
  the previous updater together. An updater replaced before its own trial ended is not gone back to: the fallback
  stays the last updater that passed its trial (or the starting one). Before a new updater is built, the wrapper keeps
  a CPU copy of the parameters, gradients, your optimizers' state and the old updater's state (if it fits in
  `snapshot_limit_gb`, default 2, on every rank): a build that fails, or a failure at the new updater's first update,
  is then undone exactly, including what `build()` did to shared objects. A failure at a later update of the window
  is not undone (that update may be partly applied), nor is a failure without a copy; the event log, the run's log
  line and the agent's next message say which. After the trial window, one process rebuilds the previous program and
  goes on, and an exception stops a DDP run; restarted from your checkpoint, the run replaces the crashed program by
  its predecessor. On every path your learning-rate schedule is put where the previous program would have it (see
  learning-rate logging below).

DDP compares parameter fingerprints before a new updater, after its first update and at the end of its trial: new divergence from equal ranks restores the pre-trial snapshot and reverts the updater (or stops for checkpoint recovery if no snapshot is available); the agent is told which tensors differed.
Later divergence broadcasts rank 0's parameters and is recorded in the event log and stderr; the report and `opal show` mark “ranks repaired at updates …” (optimizer state is not repaired).

## The two programs the agent edits

The agent works in `opal_runs/<folder>/workspace/`: the programs, `inbox/` (every decision's inputs), `reference/`
(read-only copies of your training script and model source) and `scratch/`. The program is `updater.py`,
`analysis.py` and the modules they import; nothing else in the folder counts.

```python
# updater.py: turns gradients into updates, on every rank
def build(named_params, previous=None, spec=None):   # previous = the updater it replaces (carry its state over)
    return Updater(...)                                # spec: total_updates, update, world_size, rank,
                                                       #       user_optimizers, user_schedulers, clip_grad_norm
class Updater:
    follow_user_scheduler = False                      # True: your loop's scheduler.step() keeps stepping your schedulers
    def step(self, grads, update): ...                 # grads[name] is param.grad; update counts from 1
    def state_dict(self): ...                          # plain tensors, numbers, strings, lists, tuples, dicts only
    def load_state_dict(self, state): ...
    param_groups = [...]                               # recommended: torch.optim-style groups with "lr", each param once

# analysis.py: what the agent wants to see, on rank 0
EVERY = 50
def analyze(obs, memory):                              # obs.losses, obs.params, obs.grads, obs.metrics, obs.updater, ...
    return {"report": {...}, "wake": "reason"}         # report: kept for the next decision; wake: ask for a decision now
```

The starting `updater.py` simply calls your own optimizer and scheduler. The agent's full instructions are in
`opal/prompts/neutral.md`, the default prompt; `prompt="aggressive"` and the path of a template file of your own are
the other choices (see `prompt` under cost and limits).

## Skipping an update

The agent's updater can discard an update it does not trust. With `can_skip = True` on the updater, the wrapper keeps
a copy of the parameters and of the updater's `state_dict()` before each update (and of your optimizers' state just
before the program steps them, unless its `state_dict()` holds it); when `step` returns `True` (a bool or a 0-d bool
tensor), that update is undone and training goes on from the state before it. `step(grads, update, ctx)`
also receives `ctx.loss`, this update's training loss on the device (the mean over ranks, so every rank decides the
same). After 10 skips in a row the next update is applied anyway and the agent is asked for a decision. Cost, only
for programs that set it: memory for one copy of the parameters and optimizer state, one device copy per update and
one host synchronisation per update; on a small test model (RTX 3090) that was about 10% more time per update, on one
GPU and on two. Things outside the optimizer are not undone: your loop's EMA of the weights,
GradScaler, BatchNorm statistics from that forward pass, and the batch itself (it is simply not learned from).

## Try it without an agent

`backend="keep"` answers every decision with "keep the programs": the run must match the unwrapped one. To test what
happens when the code changes, pass a function as the agent. It gets the agent's folder, the decision message and the
system prompt, may edit the program files there, and returns its rationale:

```python
def fake_agent(workspace, message, system):
    (workspace / "updater.py").write_text(MY_UPDATER_SOURCE)
    return "switching to SGD with momentum"

opt = opal.wrap(opt, model, total_steps=steps, goal="...", backend=fake_agent)
```

## Experimental Codex backend

Claude remains the default. Select `backend="codex", backend_options=dict(model="gpt-6.1-sol", effort="high")` (`xhigh` also supported), or `opal broker <box> --backend codex --model gpt-6.1-sol --effort high`.
Log in separately with `CODEX_HOME=~/.codex-opal codex login --device-auth`; never copy or link the default home's `auth.json`.
`max_cost_usd` does not cap Codex; optional `backend_options=dict(max_tokens=N)` (broker: `--max-tokens N`) is a soft input + output token cap that an in-flight turn can exceed.
It fails closed when Codex compacts conversation history (about 450k context tokens with Codex CLI 0.160.1 defaults).
Recorded single runs on one H100, Codex vs plain: `gpt_small` validation loss 4.3125 vs 4.4763; `diffusion_mnist` EMA validation loss 0.1687 vs 0.1696.

## What the agent sees, and what leaves your machine

The agent sees your goal sentence, every parameter's name and shape, the training losses, its own analysis reports,
the `opt.log` values you pass with `show_agent=True` (none by default), errors and timings, and `reference/` copies
of your training script and of the user code your model, optimizer and scheduler come from. **All of this is sent to the model provider of the backend you choose**
(Anthropic for Claude Code), as with any coding agent. `share_code=False` keeps the script and source
files out. The agent's CLI gets an allowlist of your environment variables (PATH, HOME, user and locale, proxies
and CA files, and what Python needs: `LD_LIBRARY_PATH`, `PYTHONPATH`, `VIRTUAL_ENV`, `CONDA_PREFIX`, thread counts)
and the backend's own login variables (and the AWS/Google ones when Claude Code runs on Bedrock or Vertex); the login
variables are unset again for the agent's shell (Claude Code's `sandbox.credentials`). Anything else, such as a
database URL, an ssh agent or a token in an oddly named variable, does not reach the agent; a variable your Python needs
that is not on the list is missing in the agent's CPU checks too. The agent gets no CLAUDE.md, memory, skills, plugins or MCP servers of yours, and no git status
of the folders around its workspace (no branch, user name, file names or commits); Claude Code still tells it your
account's e-mail address and whether its folder lies inside a git repository.

## Cost and limits

| Option | Default | Meaning |
|---|---|---|
| `decisions` | 50 | how many times the agent is asked over the run: chances to change the programs, not changes (at each it may keep everything as it is); a fifth in the first 5% of training, the rest spread evenly (fewer on short runs). Not set, a run whose training waits for the agent asks at most as many as keep the expected waiting within 4x its own training time (below) |
| `max_decisions` | 2 x the scheduled decisions | cap including the agent's own wake-up calls; wake-ups alone never take the decisions the remaining scheduled ones need, and the message of the last decision says it is the last |
| `max_cost_usd` | none | stop asking once Claude Code reports this much spent (checked before each decision; the Claude backend also caps each decision at what remains, with `--max-budget-usd`, which can overshoot by one model call; `backend_options=dict(max_budget_usd=...)` caps each one) |
| `every` | 50 | updates between checks; decisions and the non-finite check happen at checks |
| `backend_options` | `{}` | `model=` (default `claude-opus-5-5`) and `effort=` (default `high`): the agent of the blog's runs; with another model or effort the run prints one line at the start saying that the blog's results may not be reproduced with it. `timeout=` (seconds per decision, default 3600), `shell_timeout=` (seconds per shell command of the agent), `max_budget_usd=` (dollars per decision, passed to Claude Code as `--max-budget-usd`, which stops the turn once it is spent, so it can overshoot by one model call), `retries=` (a failed agent turn, such as a tool crash, is made again after `backoff=15` s, then twice as long, ... at most `backoff_max=600` s apart, from the programs the decision started with, while `timeout` allows; `None`, the default: as often as that allows; `0`: never; each failure's error text goes to the run's log, `decisions/NNN/attempts.jsonl` and a broker's log; `opal broker --retries N`). An unavailable agent is waited for separately, as described below (`unavailable_timeout=` sets that wait's deadline) |
| `blocking` | True | training waits for each decision (the protocol of our published runs); False: it keeps going; "auto": only when an answer would land more than `max_lag_updates` updates late, and at the first decision |
| `max_lag_updates` | 50 | with `blocking="auto"`: the predicted lag, in updates, above which training waits |
| `share_code` | True | False: the training script and source files are not copied for the agent |
| `schedule` | "front" | when the `decisions` are asked (not the learning-rate schedule, which is `scheduler`). "front": a fifth of them evenly over the first 5% of training, the rest evenly over the remainder. "uniform": every total_steps // decisions updates. A list of segments, `[(0.05, 20), (1.0, 30)]`: each is (end as a fraction of training, count), here 20 decisions evenly over the first 5%, then 30 evenly over the rest. A list of update numbers, `[500, 1000, 5000]`: these updates. A list gives the count itself: leave `decisions` out, or make it agree (otherwise `wrap` raises). Every form is moved up to the next check (every `every` updates), two on one check become one, and none falls in the last `every` updates; the run's first log line says how many are left |
| `show_next_decision` | on with the neutral and aggressive prompts | True: each decision message gives the update of the next scheduled decision; False turns it off |
| `prompt` | "neutral" | the run's facts, the protocol and short directions without recommended values, with `task_facts="..."` for facts such as the data, repetition, batch and what the loss is computed on (and `epochs=`, `train_size=`, `batch_size=`, `same_order_each_epoch=`); "aggressive" (OPAL 0.1.2-aggressive): the neutral prompt, except that each decision starts with a review of every parameter group's hyperparameters, the rationale has five parts (observed, review, changed, prediction, rules), the message no longer calls keeping the programs a valid decision, and `reference/example_updater.py` has a feedback controller of the learning rates, off by default (`opal/misc/aggressive.py` holds every difference); or the path of a template file, whose `{fields}` are filled as in `opal/prompts/neutral.md` |

**Before you launch: `opal estimate`.** `opal estimate --total-steps N --sec-per-step S [--decisions K] [--every E]
[--no-blocking] [--mailbox]` prints the run's time without the agent, the expected time training waits for the agent
(a range, and as a multiple of the run's own time) and the expected cost. The numbers per decision come from our runs
of claude-opus-5-5 at effort high (`opal/estimate.py` lists them and their sources): the first decision, which reads
your code and writes the first program, took 100 to 430 s ($0.30 to $1.60), and each later one 10 to 50 s on average
($0.05 to $0.38). A run prints the same estimate at its first check, from its own time per update. When `decisions` is
not set, a run whose training waits for the agent schedules at most as many decisions as keep the expected waiting
within 4x its own training time (at least one, in the same schedule; the line says so, and `events.jsonl` has an
`estimate` event); a `decisions` you set is kept, and the line says when its waiting is above 4x. A 66-second CIFAR-10
run gets one decision this way, a five-hour run its 50. A run resumed from a checkpoint keeps what its first check
chose.

From our runs: SiT-B/2, 94 decisions with Claude Opus, $11.28; a run whose agent woke on its own forecast, 16
decisions, $3.23.

**When the agent is unavailable.** A decision whose agent cannot work at all is kept and waited for, not failed:
Claude Code not logged in (or its key invalid), a usage, spend or rate limit, the API overloaded or the network
down, the CLI missing, and with a mailbox no broker answering for `broker_wait` seconds (120) or a broker whose own
Claude Code is unavailable. The run prints one line when the wait starts, with the CLI's words and what to do ("decision
7: Claude Code is unavailable (Not logged in · Please run /login); training waits for the agent and tries again in 60
s, then at most every 5 min. To go on: log in on this machine ..."), a reminder every 10 minutes, and one line when the
agent is back; the broker's log says the same, and a broker never answers such a request as failed. The agent tries
again after 60 s, then at most every 5 minutes (sooner when the message names a reset time that comes sooner). Once it
is back, the same decision goes on with the same message, so the run makes the decisions it would have made (if a call
had begun to edit before it stopped, the folder is set back and the message says so, as for any failed call). The
waiting has no deadline and does not count against the decision's `timeout`; `events.jsonl` (`unavailable_seconds`) and
the report show it apart from the agent's own time. `backend_options=dict(unavailable_timeout=s)` (and `opal broker
--unavailable-timeout s`) fails the decision after s seconds instead; nothing ever stops the run from asking again. With
`blocking=False`, training goes on with the current programs meanwhile and no new decision is asked while this one
waits. A failure of the agent's own turn (a crash, a time limit, a budget it spent) is not waited for: it keeps
`retries`.

**Choosing when training waits.** By default (`blocking=True`) training waits for the agent's answer at each decision,
and the edit takes effect from the update at which the decision was asked; the agent's thinking time adds to the wall
time. `blocking=False` and `blocking="auto"` are there to go faster. Without waiting, an answer takes effect (answer
time / time per update) updates after the state the agent saw, and no new decision is asked while one is in flight, so
with fast steps the agent may be asked far fewer times. In one finished non-blocking run, one seed, of a
162M-parameter GPT over 3,250 updates (OPAL 0.1.0), 17 of the 100 scheduled decisions were asked and 2 applied;
answers took effect a median of 140 updates after their request (120 to 890), and the final result, 3.2781, was level
with the baseline's. `blocking="auto"` waits at a decision only when the answer would land more than `max_lag_updates`
updates late (default 50): the median of the agent's last 5 answer times divided by the time per update (the median
over the last 5 check intervals, pauses left out); it always waits at the first decision. Each decision's choice, its
reason and the lag it had are in `events.jsonl` and the report.

## The report

Each run keeps `report.html` in its run folder: the loss with a tick per decision, your `opt.log` values, up to 12
numbers the agent's analysis reports (numbers in its `report` dict, one level of nesting such as `grad_norm.embed`,
present in at least three reports), a table of the code changes, and every decision with its rationale and diff. Rank 0
rewrites it in a background thread after each decision and at the end; until the run ends the page reloads every
minute. `opal report <run>` rewrites it at any time. Its first line, and `opal show`, count the decisions
scheduled, asked, answered (the agent's answer came) and applied (it changed the programs), with the median lag
from request to effect, so a run whose agent was rarely asked shows at once.

The report opens with a short overview built from the records, one loss plot with numbered applied decisions
(hover, focus or tap for the change and the agent's reason; kept decisions are faint ticks), and a compact table:
decision number, requested → applied update, one-line change, a shortened quotation of the agent's reason, and tags.
Analysis plots, full decision messages and answers, code diffs (including file and constant changes), and run facts
start folded. Recorded costs and the first-to-last timestamp span are shown; missing timing, commit, model or effort
fields say "not recorded" rather than borrowing the report generator's settings. The single HTML file works offline,
on phones, in light or dark mode; displayed run paths are relative and private locations and recorded identities are
omitted from messages, code and plot data too. Rendering itself makes no model calls. Cached descriptions and tags
come from a language model (Claude Opus at low effort through the Claude Code CLI, no
tools, one call per code change; about a cent each, recorded in `decisions/NNN/summary.json` and not counted in `max_cost_usd`). Tags, any number of them:
`common practice` (standard in current recipes: Muon, warmup, cosine or linear decay, clipping, AdamW betas),
`niche practice` (published but uncommon: SOAP, Shampoo, per-tensor LR controllers, NorMuon), `block-specific` (some
groups, layers or blocks treated differently), `observation-originated` (triggered by something the agent saw in this
run and cites), `time-step-specific` (the program treats a particular update or phase differently), and
`fix or revert` (repairs or undoes the agent's own earlier change). They describe a change, not its quality, and are a
first pass: read the rationale and diff before relying on one (on 95 decisions of earlier runs, two labellers agreed
on the full set of tags for a third of them). With `backend="claude"` they are
made during the run (`summarize=False` turns this off, `summarize="<model>"` picks another model); for other backends,
or to fill gaps, run `opal report <run> --summarize` on a machine with Claude Code.

## Comparing with our runs

Every example script with `--opal` writes, with `--out <file>`, a result.json whose first fields are the same in all of
them: the metric (name, value, and whether higher or lower is better), the training time and the time training waited
for the agent, the agent's cost, the decisions asked, applied and failed, the hardware, the OPAL commit, the full
command and the OPAL run folder (a plain run has the same fields, the OPAL ones empty). `--save <folder>` writes the
final weights (`model.pt`, and `ema.pt` where the script keeps an EMA). Each example's records of ours, where its folder
under `examples/` links them, hold the same: `report.html` and the run folder, result.json and the final weights. Set
two result.json files side by side by their metric and command, and read the decisions with `opal show <run folder>`.
To train with our final program and no agent, use `opal.replay("<our run folder>", optimizer, model, total_steps=...)`
in place of `opal.wrap(...)`: it runs the version the folder's `FINAL` names, deterministically. The example scripts
take `--decisions`, `--schedule`, `--blocking`, `--model` and `--effort`, so a command can state the settings of a run
explicitly (they default to `decisions=50`, `schedule="front"`, blocking, claude-opus-5-5 at effort high).

## Fitting it into your training script

- **Gradient accumulation:** one `opt.step(loss=...)` per update, with the mean loss of that update.
- **The schedule's numbers:** the agent reads your scheduler's settings and, for a `LambdaLR`, the function's source,
  but not the values its closure holds. If the warmup and decay lengths are not visible in the code it reads (they come
  from variables or command-line flags), state them in `task_facts`, e.g. `task_facts="linear warmup over 7,935
  updates, then linear decay to 0 at update 79,355"`.
- **An older torch pinned by the repository:** OPAL needs torch 2.3 or newer (and Python 3.9). A repository that pins
  an older torch or Transformers then runs on a newer torch, which may need small fixes; for instance, older
  Transformers with a recent torch need `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` to resume from a checkpoint.
- **Gradient clipping is part of the optimizer.** If your loop clips gradients, hand the clipping to OPAL: delete
  `clip_grad_norm_(model.parameters(), x)` from the loop and pass `clip_grad_norm=x` to `wrap`. The run starts
  identical (the starting program clips first, bitwise like the loop when the optimizer holds all of the model's
  trainable parameters), and the agent, which then sees the unclipped gradients, may change the clipping like any
  other part of the update. Clipping left in the loop happens outside the optimizer: the agent's program gets clipped
  gradients and cannot change the threshold. (With the HF Trainer and Accelerate, see below.)
- **Learning-rate logging:** log `opt.param_groups[i]["lr"]` after `opt.step()` and before `scheduler.step()`; a
  program that sets its rates inside `step` has not set this update's rate before it. While the agent's program owns the schedule, your
  loop's `scheduler.step()` does nothing and `get_last_lr()` reports, for each of your optimizer's groups, the rate the
  program gives its parameters, where it can. When control returns to a program that follows your scheduler (the agent
  restores the starting updater, or a failing program is reverted to one), your schedule continues as if it had never
  stopped: your optimizers' group settings go back to the moment the agent's program took over and the
  `scheduler.step()` calls made since are replayed (also across a checkpoint). A `ReduceLROnPlateau` keeps the metric
  you pass to its `step()` (as `best`), and the agent's programs can read it through `spec["user_schedulers"]`: pass it
  only a value you would show the agent.
- **Several optimizers or schedulers:** pass lists, `wrap([muon, adamw], model, scheduler=[s1, s2], ...)`.
- **fp16 with GradScaler:** `scaler.step(opt, loss=loss)`. Updates the scaler skips are not counted, so call
  `opt.close()` when your loop ends. With a fused optimizer the scaler takes its generic unscale path: the starting
  program then matches the plain run numerically but not bit for bit.
- **DDP (one node):** wrap the bare or the DDP-wrapped model, on every rank, with the same arguments; call
  `opt.step(loss=...)` and `opt.close()` on every rank. The agent sees rank 0's loss; `opt.log` values from rank 0 are
  kept. Save `opt.state_dict()` from rank 0 as you save your model, and load it on every rank after `wrap(...)`. If a
  program crashes after its trial window, the job stops; restarted from the checkpoint (same run folder, or a new one:
  the checkpoint remembers the old), the crashed program is replaced by its predecessor, or by the starting program
  if that crashed too.
- **Checkpoints and resume:** save `opt.state_dict()` (programs, their state, the agent's session, and your optimizers'
  and schedulers' own state whatever the program's `state_dict()` holds) and your scheduler's `state_dict()`. The
  checkpoint grows by OPAL's state: the agent's programs' state, and the analysis memory, which for the starting
  analysis holds a copy of one tensor per parameter role (for GPT-2 that includes the whole token embedding; a small
  Goldfish model's optimizer file grew from 27 MB to 47 MB). To resume, rebuild the optimizer and scheduler, `wrap(...)` with the same arguments,
  then `load_state_dict`. With the default run folder the run continues in the checkpoint's folder and conversation
  when that folder is the same run, interrupted (in this run's `opal_runs/`, and its run did not finish); records the
  interrupted attempt wrote after the checkpoint move to `attempts/`, and the agent is told that its later decisions
  were lost. A checkpoint of a run that finished, or one copied from another place, starts the resumed run in its own
  folder and leaves the original as it was.
- **Early stopping or an exception:** call `opt.close()`; it runs at exit too, and a run that stopped before
  `total_steps` without `close()` counts as interrupted (a mailbox broker then waits for its resumption). For a planned
  stop that you will resume (a job's time limit), call `opt.close(interrupted=True)`: the broker keeps waiting. When
  `close()` returns, the final `report.html` is written (with Claude summaries pending, it waits for them up to 30 s).
- **Reproducibility:** the default `blocking=True` is deterministic given the agent's answers, and `opal.replay` of
  a final program is deterministic. With `blocking=False` or `"auto"`, the update at which an edit lands depends on
  timing, so two runs with the same answers differ.
- **Turning the agent off mid-run:** resume with `backend="keep"`, or start a fresh run with `opal.replay(...)`.
- **Hugging Face Trainer:** use `opal.hf.OpalTrainer` (`pip install "opal-optimizer[hf]"`) in place of `Trainer`:

  ```python
  from opal.hf import OpalTrainer

  trainer = OpalTrainer(model=model, args=training_args, train_dataset=train, eval_dataset=dev,
                        opal=dict(goal="GPT-2 small on Swahili; judged by FLORES-200 bits", max_cost_usd=20))
  trainer.train()
  ```

  It wraps the optimizer once the Trainer has built it and its schedule (so the schedule stays on the Trainer's own
  optimizer), gives OPAL each update's mean loss (the Trainer calls `step()` without it), and records `eval_*` values
  in the report (not shown to the agent). With `backend="keep"` it is bitwise the plain Trainer (tested on CPU,
  transformers 5.20). The Trainer's own clipping (`max_grad_norm`) stays outside the optimizer; to hand it to the
  agent, pass `TrainingArguments(max_grad_norm=0)` and `opal=dict(clip_grad_norm=1.0)`: the run then starts the same
  up to rounding (accelerate computes the norm its own way). Older Transformers with a recent torch need `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` to resume
  from a checkpoint (their RNG state does not load with `weights_only=True`).
- **Accelerate:** wrap after `accelerator.prepare(...)`: `opt = opal.wrap(opt, model, scheduler=sched, ...)` with the
  prepared optimizer and scheduler; OPAL wraps the torch optimizer inside Accelerate's, and your loop calls
  `opt.step(loss=loss)` once per update. That is bitwise the plain loop (tested on CPU) when Accelerate neither scales
  fp16 gradients (a GradScaler) nor accumulates them (`gradient_accumulation_steps`); `wrap` refuses those. Wrapping
  before `prepare` does not work: `prepare` saves and reloads the wrapper's state. The model may be wrapped any number
  of times (DDP, Accelerate, `torch.compile`).
- **Another framework calls `step()` for you** (Lightning; untested): keep a handle to the wrapped optimizer, call
  `opt.observe(loss)` once per update and `opt.log(...)` for metrics.
- **Closure-only optimizers** (LBFGS) are not supported as the starting program.

## What it costs your training

Besides the agent's own update rule, the wrapper adds: no host synchronisation per update (the loss is copied into a
device buffer and read every `every` updates); in DDP, a few small gloo broadcasts per check (one, or three in
blocking mode), an asynchronous one every `sync_every` updates, and one gloo all-reduce per update during the
`trial_steps` updates after each new updater (an `EVERY` below `every` adds a check at each analysis step); a
CPU copy of parameters, gradients and your optimizer's state before a new updater is built (skipped, with a
warning, above `snapshot_limit_gb`, default 2; a 125M-parameter model with AdamW needs about 1.9 GiB, more once the
previous updater has state of its own); for an agent's updater, a read of each parameter's and gradient's dtype and
device after every update (tens of microseconds for a few hundred tensors) and a walk over its state at check steps;
no hooks in your model; no recompilation of a `torch.compile`d model when the program
changes (tested); nothing at all when the agent keeps the programs. Each decision shows the agent its program's time
per update next to the starting program's.

The agent's own update rule usually costs more than the wrapper. We timed the final programs of 16 runs of the earlier
harness on one A100, each against a standard implementation of the same algorithm on the same parameters
(`torch.optim.Muon`, or the official GMuon): 9 were within 4 ms per update of it; 6 added 7 to 26 ms through
Newton–Schulz in float32 instead of bfloat16 (10 to 15 ms of it on a 125M model), Python loops over matrices or
parameters, statistics recomputed for every tensor at every update, and host reads inside the update; one ran a larger
Muon. The prompt's EFFICIENCY rules cover all of these, and its precision contract the float32 Newton–Schulz. Our own hand-written Muon baselines had
the loops too and were twice as slow as `torch.optim.Muon`.

## GPU machines without an agent

Usually not needed: Claude Code runs on the GPU machine itself if that machine reaches the internet and you have run
`claude` there once to log in (on a cluster, the login usually carries over to the compute nodes through the shared
home directory). The mailbox is for machines that cannot do that (no outbound internet, or you would rather not keep
your agent login there): point the run at a folder and serve it from a laptop or login node over ssh (or a shared
filesystem):

```python
opt = opal.wrap(opt, model, total_steps=steps, goal="...", backend="mailbox:/scratch/me/opal-box")
```

```bash
opal broker /scratch/me/opal-box --ssh me@gpu-host --backend claude --keep-serving
```

`opal doctor --backend mailbox:/scratch/me/opal-box` on the GPU machine checks that the folder is writable and whether a
broker serves it (no Claude Code is needed there). With a mailbox, the agent's model and effort are set on the broker
(`opal broker ... --model claude-opus-5-5 --effort high`, the defaults), not in the run's `backend_options`. The
broker's machine must stay awake for the whole run (on a Mac: `caffeinate -i opal broker ...`): while no broker answers,
the run waits for one (see "When the agent is unavailable"). The report's summaries of code changes are not made during
a mailbox run; run `opal report <run> --summarize` where Claude Code is logged in.

The run prints its run id and the exact broker command at its first update. Each run gets its own subfolder of the
box (mode 0700) and every request and answer carries the run's id and an attempt id, so one box can serve many runs
and a resumed run never picks up an old answer. The broker refreshes a heartbeat every 30 s, also while the agent
thinks; if no broker has been seen for 2 minutes, the run keeps the request and waits, and says how to start one.
With `unavailable_timeout` set, it stops waiting after that limit. A broker that comes back still answers it: the answer is applied like any other if it arrives before the next request, and
discarded after that, which the next message tells the agent (every message says what became of the previous answer:
the update its edit took effect at, or why it did not). Without
`--keep-serving` the broker serves one run (the newest with a pending request, or `--run <id>`) until it is done, and
waits through an interrupted run for its resumption; while its run is idle it moves to another run of the box that is
waiting, so a crashed run does not hold it (unless it is pinned with `--run`). With `--keep-serving` it serves every
run in the box, taking the runs with pending requests in turn (the longest waiting first, so a run that asks often
cannot starve another), and keeps a box-level heartbeat that every run accepts. When a run ends with a decision in
flight, the broker stops that agent turn.

Each request carries a manifest of the workspace, and every file besides the programs is also kept in
`<box>/<run>/files/`: a broker started on another machine or with a new `--workdir` fetches what it lacks (the earlier
inputs in `inbox/`, `reference/`) and drops inbox files the run no longer has. The agent's own files (`NOTES.md`,
`scratch/`; small text files only, 256 KB each, 2 MB per answer) come back with each answer and are kept in the run's
workspace, so its notes survive a change of broker; a broker that has no conversation of the run's agent tells the agent
so in its first message. A broker claims a request (`claim-NNNN.json`, created exclusively) before it starts the agent,
so two brokers on one box never both answer (and bill) it; a claim its broker has not refreshed for 2 minutes is taken
over. The broker runs the agent for at most the time the agent was told it has (the mailbox `timeout`, 6 hours by
default), less the time the request waited and a minute to write the answer. A broker `--timeout` shorter than that is
kept, and the broker's log says so; whenever the broker stops the agent sooner than the run told it (that `--timeout`,
or a request that waited long for a broker), the message the agent gets starts with the limit that holds ("This
decision may take up to 15 minutes, less than your instructions state."). To tell the agent the same limit from the
start, give the run that limit too (`backend_options=dict(timeout=...)`).
`--backend keep` tests the whole path at no cost. The broker's log lines carry the time and are flushed at once (fine under `nohup` or a file); when reads of the mailbox keep failing it says so every ten failures, and `--give-up-after-min M` makes it exit with an error after M minutes of failures. The mailbox backend takes `backend_options=dict(poll=2,
broker_wait=120, timeout=21600)`. Run `opal show`, `report` and `say` on the training side's run folder.

## Safety

For Claude Code, the agent's file tools are confined to its folder and its shell runs in the OS sandbox with no
network, reading only its folder and your Python installation. On Linux Claude Code's sandbox needs `bubblewrap` and
`socat`, and bubblewrap must be allowed to create namespaces (some distributions forbid that to unprivileged users):
OPAL starts bubblewrap once when the run starts and, if the sandbox cannot start, gives the agent file tools only
and says why (with `backend_options=dict(sandbox=True)` it stops instead). A watchdog stops the agent if the training
process dies, and after each decision whatever the agent's CLI left running (a background job of its shell) is
stopped. On macOS the shell cannot reach the GPU through Metal either (checked with Claude Code 2.1.302: torch
imports, and `torch.backends.mps.is_available()` is False in the sandbox). **The code the agent writes is not sandboxed: it
runs inside your training process, with your permissions**, like any optimizer you import. Every version is kept in
`programs/vNNN`; the final program is plain Python you can read before reusing it.

The agent's shell can run one Python: the interpreter of the process that runs the Claude backend, which is your
training process with `backend="claude"` and the `opal broker` process with a mailbox (so for the agent's CPU checks
the broker's environment needs torch). It is first on the shell's PATH as `python`, and the sandbox may read its venv,
its site-packages, the base installation (as given and with links resolved) and the folder of each link on the way to
the real binary, but never a folder that contains your home folder or the run folder: an interpreter started as
`~/bin/python` (a link to a conda or uv Python) opens `~/bin`, not `~`. The macOS sandbox checks real paths; up to
OPAL 0.1.2 the resolved base installation was not readable, so a venv on Homebrew's Python
(`/opt/homebrew/opt/python@3.12` links into `Cellar/`) failed with `operation not permitted: .../python3.12` (a Goldfish
broker, 2026-10-17). Now such a venv starts (checked on macOS with Claude Code 2.1.302 and a `python -m venv` of
Homebrew's 3.12: the old settings failed, the new ones ran `python`); a venv on a uv-managed Python works too
(`uv venv --managed-python`). If an essential folder had to be left out (an interpreter installed into your home
folder itself), OPAL says so when the run starts.

## What OPAL does not protect against

The agent's programs (`updater.py`, `analysis.py` and their imported helpers) run inside your training process,
with its permissions. They are **not sandboxed**, even when the agent's shell is sandboxed. A program can read
files the process can read, inspect the training script's globals, or change objects outside the intended optimizer
interface. Its analysis report is returned to the agent, so data read by a program can reach the agent and its provider.
`share_code=False` prevents copying source into the agent's reference folder; it does not prevent these accesses.

OPAL detects some evaluation-time parameter moves when `eval_integrity` is enabled. Separately, the static
`boundary` check flags common file/network I/O, dynamic imports and builtins access, introspection, hook registration
and model/loss method assignments in program source before loading each candidate version, on rank 0. Each finding
is recorded in `events.jsonl` with its version, file, line, category and source snippet; flagged versions get one
stderr line and a short **Boundary flags** note in `report.html`. Candidates that later fail to build may have flags
too. The boundary check only records: it does not block a program or change its execution, and a check failure logs
one line and lets the existing program-loading checks proceed. These are incomplete heuristics, with possible false
positives and missed indirect or deliberately hidden accesses. No flags is not evidence of isolation.

Keep test data and labels out of the training process and evaluate the final checkpoint in a separate process.
If test files are still readable with the training process's permissions, separate evaluation alone does not make
them inaccessible. Review the boundary flags in the report, their details in `events.jsonl`, and the saved programs
before trusting a result. Prompt rules describe intended behavior; they do not enforce a security boundary.

## What it can and cannot change

| Works | Not supported |
|---|---|
| Plain PyTorch loops on one GPU or several GPUs of one node (DDP) | Multi-node training (untested), FSDP, ZeRO, DeepSpeed |
| `torch.compile`d models, bf16 autocast, fp16 GradScaler, gradient accumulation | Tensor/pipeline parallelism, JAX, whole-step CUDA graphs |
| One or several optimizers (e.g. Muon + AdamW), LR schedulers | Closure-only optimizers as the starting program |
| Anything an update can express: algorithm per tensor, per-group and per-block LR, schedules, momentum and betas, preconditioners, clipping inside the optimizer, weight decay | Anything outside the optimizer: model, data, batch size, loss, precision, number of steps |

Precision is fixed: the agent's program may not change the dtype or device of parameters or gradients, keeps the state
it creates in the parameters' dtype or in float32, and computes each update at the algorithm's standard precision
(Muon's Newton–Schulz in bfloat16, as in `torch.optim.Muon`). A program that changes a dtype or device, or keeps
parameter-shaped state in a dtype that is neither the parameter's nor float32 (bfloat16 moments of float32 weights),
or casts the state of your optimizers, is reverted like one that raises. The check reads tensor attributes only, no
device work: around `build()`; the dtype and device of every parameter and `p.grad` after every update of an agent's
updater (not the `grads` dict it is handed, which it may use as scratch); its state, matched to parameters by name where
its keys are parameter names and by shape otherwise, and your optimizers' state after each update of the trial window
and at check steps. State your own optimizer creates in its own dtype is accepted. The precision of the update's own
arithmetic is set by the prompt and not checked. The agent's `analysis.py` may not change parameters or gradients at
all: their version counters, dtypes and devices are compared around each call, and an analysis that changed them is
not run again until the agent edits it (the change itself cannot be undone; the agent is told).

Evaluation results stay yours: `opt.log(...)` (keyword values, or a dict) records values in `metrics.jsonl` and the
report only, and they are shown to the agent (in `obs.metrics` and its next decision message) only if you pass
`show_agent=True`. An agent that sees a value can tune the run to it, so show only a split you accept it tuning on,
never the test set the run is judged by. "Not shown" is not "out of reach": the agent's programs run inside your
training process, which can read the run folder (`metrics.jsonl` included) and anything else the process can.
The prompt forbids this and the boundary check flags some patterns, but neither prevents it. The agent's
shell cannot read the run folder (sandbox), and program files are never read or written through symbolic links, so a
link in the workspace cannot carry a file into the agent's folder. Read `programs/` before you trust a result.

**Showing a development set (OPAL 0.1.1-validation).** `wrap(..., validation=N)` says that the training script
evaluates the model on a development set held out from training about every N updates, at times it does not announce,
and reports it with `opt.report_dev(loss=..., accuracy=...)` (`validation_size="5,000 images"` optionally names the
set's size). The agent
then sees each measurement in `obs.metrics` (as `dev_loss`, `dev_accuracy`) and, with the history of earlier ones, in a
line of its own in each decision message ("Development set (held out from training; not the test set in the goal),
..."); every measurement is in the inbox. With the neutral prompt, two sentences in TASK state the evaluation and that
the goal's test set is separate and never shown, and the sentence on evaluation data says the development values are
shown; nothing else changes. With `validation=None` (default) the prompt has neither sentence, and `report_dev`
only records. `validation=N` is your promise of how often the script evaluates, and of irregular times: OPAL does
not run the evaluation, and warns you (not the agent) once if the last three evaluations were equally spaced, since the
agent sees each measurement's update and could predict the next, or if they come on average much more or less often
than every N updates. Draw each time at random, as `examples/overfit/records.py` does (one uniformly random update in
each epoch). It needs a prompt that states it (the neutral prompt, or a template with a `{validation_note}` field);
with a template that has no `{validation_note}` field, `wrap` raises. `examples/overfit/train.py` uses it.

**Evaluation integrity (0.2.2).** Two overfit runs read the evaluation times (from the prompt, which then stated them,
and from the training script) and timed changes of the parameters to them: one scaled the classifier for each
evaluation and restored it at the next update, one copied a running average into the model 49 updates before each
evaluation and put the iterate back after it; both planned a final classifier scale chosen on these probes. Besides
not announcing the times, `eval_integrity` (on by default with `validation=N`; `True` or `False` to choose) checks
every evaluation you report: rank 0 copies a fixed sample of each parameter tensor's entries after every update (at
most 32K entries in all, one `torch.cat` into a ring of 138 updates on the device, no host synchronisation: about
0.04 ms per update on a CPU and 0.16 ms on Apple MPS for a ResNet-18), and 8 updates after each `report_dev` reads that
window once. A tensor is flagged when the last unusually large change at or before the evaluation (more than
`eval_integrity_threshold`, 5, times its median change per update in the window, and more than 0.1% of its norm) is
undone by the first unusually large change after it (the parameters come back to within a quarter of the excursion
of where they were), with ordinary updates on either side. The run then gets an `eval_integrity` event in
`events.jsonl` naming the tensors, the updates of the change and of the return and the norm factor, a banner in
`report.html`, and a message on stderr; a lone change that stays (a weight average swapped in for good) is recorded as
`eval_jump` and does not flag the run. The agent is told neither the check nor its results. The check covers changes
made by the program between 128 updates before and 7 after the evaluation; it does not see a change that is no larger
than about five of the tensor's ordinary updates (early in training, a small probe hides in the training's own
motion), and a program that undoes one of its own updates at the next one looks the same when an evaluation happens to
fall in between: read `programs/` before you trust a flag or its absence. Call `report_dev` right after evaluating the
parameters as the latest update left them; a swap of EMA weights your script does around its own evaluation happens
outside the updater and is not attributed to the program.

Accelerate and HF Trainer have CPU tests, as described above. Lightning remains untested (the wrapper keeps the
`torch.optim.Optimizer` interface, accepts `step(closure)` and `observe(loss)`).
