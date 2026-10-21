# modded-nanogpt optimization track, record #36 (162M GPT)

The modded-nanogpt optimization track trains a 162M-parameter GPT from scratch on FineWeb with GPT-2 tokens.
It has 12 layers and width 768. Training runs for 3,250 updates of 524,288 tokens.
The score is validation loss after the last update. We start from record #36, the track's tuned baseline:
Muon on the blocks' matrices, AdamW on the rest, and a stable-then-decay learning rate.

| Validation loss ↓ (one run each, 8× H100) | Record #36 (our run) | Reactive optimizer |
|---|---|---|
| after 3,250 updates | 3.2781 | **3.2749** |

The gap is small: 0.0032. The track's rules assume a run-to-run standard deviation of 0.0013.
The gap is about 2.5 times that, from one run on each side. Record #36's published mean over ten runs is 3.2787.
It appears in row 36 of the [track README on master](https://github.com/KellerJordan/modded-nanogpt/blob/master/records/track_3_optimization/README.md).
The README at the pinned commit below predates that row and lists only records 1 to 12.

**Not rerun on OPAL 0.3 yet.** The 3.2749 run used OPAL 0.1.0 with that version's default prompt, which this release
no longer ships: Claude Code (Opus 5.5, effort `high`), 100 decisions requested with checks every 10 updates (97 after
snapping, all answered), training paused at each decision. A later run on OPAL 0.2.0 with the neutral prompt (50 evenly
spaced decisions, 2 GPUs, a goal worded as reaching the target loss) ended at **3.2878, worse than record #36**.

## Other settings, on OPAL 0.3.x

We also ran the same script in settings with no recipe tuned for them.
Each started from a record's recipe as it is, with one agent run, Opus 5.5 at effort `high` and the neutral prompt.
Each used 50 uniformly spaced decisions and paused training at every decision.
The scores are validation loss after the last update, using raw weights.

| Setting | Start | Start's result | Reactive optimizer |
|---|---|---|---|
| ¼ batch: 13,000 updates of 131,072 tokens, 2× H100 | record #36 | 3.3104 | 3.3078 |
| Chinese: 1 GB Goldfish set, 3,000 updates of 524,288 tokens, 4× H100 | record #36 | 4.1845 | 4.1286 |
| Thai, the same | record #36 | 4.7200 | 4.6608 |
| Kyrgyz, the same | record #36 | 3.9123 | 3.8533 |
| Chinese, the same | record #2 (AdamW) | 4.2570 | 4.2140 |
| Thai, the same | record #2 (AdamW) | 4.7744 | 4.7208 |
| Kyrgyz, the same | record #2 (AdamW) | 4.0045 | 3.9482 |

## Official code

[KellerJordan/modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt),
`records/track_3_optimization/train_gpt_simple.py` at commit `452cc392f6af4f29834d1af64edeac55d1315397` ("Tune
train_gpt_simple baseline hyperparameters", record #36), MIT license. Our copy of the script is byte-identical to that
file.

## Wiring it to OPAL

[`opal.patch`](opal.patch) changes the file in five places, all behind `OPAL=1`.
It writes `set_hparams` as `LambdaLR` schedulers with the same values, which the starting program follows and the agent
can take over. It wraps both optimizers with `opal.wrap`, passes each update's mean token loss to `step`, records
validation loss in the report without showing it to the agent, and closes the run at the end.
Without `OPAL=1`, the script trains as upstream does.

The patch applies to that commit and the patched file compiles.
On CPU, its `LambdaLR` schedulers matched `set_hparams` byte for byte with `backend="keep"`, a small model and SGD in
place of Muon. The patch has CPU checks only.

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
machine that runs rank 0. The run's records go in `opal_runs/<time>-...` in the working directory.

## Our runs

| | Record #36 | Reactive optimizer |
|---|---|---|
| Hardware | 8× H100 | 8× H100 |
| Training time | 508 s | 2,230 s, 73% of it waiting for the agent |
| Agent cost | – | $8.90 |
| Validation loss | 3.2781 | 3.2749 |

Our script used this file with the same schedule written as `LambdaLR`, the same OPAL wrap, a seed and our logging.
Its scenario hooks were off in these runs.

At its first decision, the agent added a Muon momentum warmup from 0.85 to 0.95 over the first 300 updates.
This technique comes from earlier modded-nanogpt records. It also added outlier-update clipping, which fired once.
