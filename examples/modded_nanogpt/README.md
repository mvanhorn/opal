# modded-nanogpt optimization track, record #36 (162M GPT)

The optimization track of modded-nanogpt: a 162M-parameter GPT (12 layers, width 768) trained from scratch on FineWeb
(GPT-2 tokens) for 3,250 updates of 524,288 tokens, judged by the validation loss after the last update. The starting
recipe is record #36, the track's tuned baseline: Muon on the blocks' matrices and AdamW on the rest, a stable-then-decay
learning rate.

| Validation loss ↓ (one run each, 8× H100) | Record #36 (our run) | Reactive optimizer |
|---|---|---|
| after 3,250 updates | 3.2781 | **3.2749** |

**A small effect.** The gap is 0.0032. The track's own rules assume a run-to-run standard deviation of 0.0013 per run,
so this is about 2.5 of those from one run on each side; record #36's published mean over ten runs is 3.2787 (row 36
of the record table in the
[track README on master](https://github.com/KellerJordan/modded-nanogpt/blob/master/records/track_3_optimization/README.md);
the README at the pinned commit below predates that row and lists only records 1 to 12).

**Not yet rerun on OPAL 0.3.x.** The 3.2749 run used a pre-release of this package (OPAL 0.1.0) with its default prompt of that time, which this release no longer ships: Claude Code (Opus 5.5, effort
`high`), 100 decisions requested with checks every 10 updates (97 scheduled after snapping, all answered), training
paused at each decision. The one run of this task on a released version, OPAL 0.2.0 with the neutral
prompt, 50 evenly spaced decisions, 2 GPUs and a goal worded as reaching the target loss, ended at **3.2878, worse than
record #36**.

## Other settings, on OPAL 0.3.x

The same script in settings with no recipe tuned for them, one agent run each (Opus 5.5, effort `high`, neutral prompt,
training paused at each decision), each starting from a record's recipe as it is. Validation loss after the last
update, raw weights.

| Setting | Start | Start's result | Reactive optimizer |
|---|---|---|---|
| ¼ batch: 13,000 updates of 131,072 tokens, 2× H100 | record #36 | 3.3104 | 3.3078 |
| Chinese: 1 GB Goldfish set, 3,000 updates of 524,288 tokens, 4× H100 | record #36 | 4.1845 | 4.1286 |
| Thai, the same | record #36 | 4.7200 | 4.6608 |
| Kyrgyz, the same | record #36 | 3.9123 | 3.8533 |
| Chinese, the same | record #2 (AdamW) | 4.2570 | 4.2140 |
| Thai, the same | record #2 (AdamW) | 4.7744 | 4.7208 |
| Kyrgyz, the same | record #2 (AdamW) | 4.0045 | 3.9482 |

Runs: OPAL 0.3 pre-releases.

## Official code

[KellerJordan/modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt),
`records/track_3_optimization/train_gpt_simple.py` at commit `452cc392f6af4f29834d1af64edeac55d1315397` ("Tune
train_gpt_simple baseline hyperparameters", record #36), MIT license. Our copy of the script is byte-identical to that
file.

## Wiring it to OPAL

[`opal.patch`](opal.patch) changes that file in five places, all behind `OPAL=1`: it turns the script's
`set_hparams` schedule into `LambdaLR` schedulers with the same values (which the starting program follows and the agent
can take over), wraps both optimizers with `opal.wrap`, passes each update's mean token loss to `step`, records the
validation loss in OPAL's report (the agent does not see it) and closes the run at the end. Without `OPAL=1` the
script trains as upstream does. The patch applies cleanly to that commit and the patched file compiles; that the
`LambdaLR` schedulers reproduce `set_hparams` bit for bit under `backend="keep"` was checked on CPU with a small model
(SGD standing in for Muon). It has not been run on GPUs.

```bash
git clone https://github.com/KellerJordan/modded-nanogpt && cd modded-nanogpt
git checkout 452cc392f6af4f29834d1af64edeac55d1315397
git apply /path/to/opal/examples/modded_nanogpt/opal.patch
pip install torch==2.11 huggingface_hub && pip install -e /path/to/opal    # upstream's track README pins torch 2.11
python data/cached_fineweb10B.py 40        # 4B training tokens, as upstream's track README says
S=records/track_3_optimization/train_gpt_simple.py

torchrun --standalone --nproc_per_node=8 $S                                     # record #36
OPAL=1 OPAL_BACKEND=keep torchrun --standalone --nproc_per_node=8 $S            # dry run: no agent, no cost
OPAL=1 OPAL_MAX_COST_USD=20 torchrun --standalone --nproc_per_node=8 $S         # reactive optimizer, OPAL defaults
OPAL=1 OPAL_DECISIONS=100 OPAL_EVERY=10 OPAL_MAX_COST_USD=20 \
    torchrun --standalone --nproc_per_node=8 $S                                 # the decision count of our run
```

Each line prints `val_loss` as it trains; the last one is the score. Claude Code must be installed and logged in on the
machine that runs rank 0. The run's records land in `opal_runs/<time>-...` in the working directory.

## Our runs

| | Record #36 | Reactive optimizer |
|---|---|---|
| Run | `t3-muon36-plain` | `t3-agent-opus-muon36-d` |
| Hardware | 8× H100 | 8× H100 |
| Training time | 508 s | 2,230 s, 73% of it waiting for the agent |
| Agent cost | – | $8.90 |
| Validation loss | 3.2781 | 3.2749 |

- **Code.** Our script was this file with the same schedule written as `LambdaLR`, the same OPAL wrap, a seed, our
  logging, and scenario hooks that were off in these runs.
- **What the agent changed.** A Muon momentum warmup from 0.85 to 0.95 over the first 300 updates, a technique from
  earlier modded-nanogpt records, at its first decision; a clip of outlier updates, which fired once.
- **Same recipe through the wrapper with no agent:** 3.2782 (training time 519 s against 508 s).
- **Other agent runs on this task, OPAL 0.1.0, one run each:** 10 decisions 3.2753 and 30 decisions 3.2777 (4 GPUs,
  where record #36 gave 3.2786; in both, the agent's analysis program read the training script's own Muon source and
  schedule constants through its globals, outside its workspace, but no evaluation data); training not waiting for the agent 3.2781 (17 of 97 decisions asked, 2 edits, $1.68);
  that version's defaults 3.2789; Claude Haiku 3.2757 and Claude Sonnet 3.2779 (4 GPUs, 100 decisions).
- **Where the numbers come from:** the experiment log of 2026-10-18 (Track 3 entries of 03:42 and 06:55 EDT) and the
  broker's `broker-state.json` (cost) in our experiment records; the runs' raw metrics are in our cloud storage only.
