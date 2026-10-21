# Run times and agent costs

These are the completed OPAL 0.3.x example runs recorded overnight on October 20–21, 2026, plus the checks of decision schedules and non-blocking execution. Attempts marked “not a result” in the source records are omitted. Runs with known problems are labelled below. Each table pairs an OPAL run with its static starting recipe; record #36 settings use #36, and Goldfish uses its starting recipe.

Most runs used Opus 5.5 at high effort through a broker on the Mac. The CPU MNIST run left effort at the CLI default. OPAL Sol used the experimental Codex backend with gpt-6.1-sol at high effort. GPU runs used H100 80 GB GPUs; hardware counts are stated in each table. GPU-hours count the GPUs used by the run, not unused GPUs on the node.

## How to read the times

- **Training / loop clock** is the source's training timer. OPAL clocks include agent waiting unless stated otherwise. Evaluations are included where noted. It is not a measurement of GPU kernel execution alone.
- **Loop less waiting**, where given, is arithmetic on the recorded loop clock and recorded wait. It still includes loop overhead and any evaluations inside that clock. It is labelled as derived, not as an independently timed training measurement.
- **Waiting** is the sum of decision durations recorded in OPAL's events. In blocking mode training waits for those answers. It is not an extra term to add to a loop or wall clock that already includes it.
- **Wall time** uses the source's start/end timestamps or command timer and includes waiting and evaluations. Setup/data download is included only when the source timer includes it. The recorded endpoints are retained when there is no single total.
- **GPU-hours** are wall hours multiplied by GPU count. Values calculated here from recorded endpoints are labelled derived. A missing wall time is not replaced by the training timer.
- **Agent cost** is the recorded cost_usd for Claude, excluding GPU cost. For Codex it is token usage; cached input is part of input, and reasoning output is part of output. These columns do not reconcile subscription bills. **Decisions** are asked / applied / failed; missing failure counts, cancellations, and rejected trials are stated explicitly.
- **Not reported separately** means the named record does not provide that measurement. It does not mean zero. Static runs have no agent decisions or agent cost.

The small-script batch used the scripts' `--decisions 50` default, which passes `decisions=50` explicitly to OPAL. OPAL's 4× waiting cap applies only when `decisions` is not set, so it did not apply: blocking waits were 9–25× the plain training loop. The overfit runs instead passed `--decisions 30` and are listed with their actual asked counts, including extra requests.

Image timings come from log/metrics timestamps; RAEv2 wall time includes its in-run FID evaluations. Goldfish static loop times sum logged update windows; its static wall ends at the final checkpoint's upload, while OPAL wall uses start/end events. Goldfish's evaluations are inside the update windows, and OPAL training-only times were not computed in the record.

Track 3 static training clocks stop during validation and include compilation/warm-up. Static wall estimates use the end stdout upload and pre-run patch upload with the source's sync-loop correction; they exclude earlier machine setup and data download. The source estimates accuracy at about 10 s. OPAL clocks and waiting come from the harness and events.

For the small examples, loop times come from each script's timer or final logged elapsed time. The overfit and MNIST loop timers include evaluations; GPT and CIFAR loop timers exclude evaluation. Command wall times include the following sync where present. The static small scripts downloaded data on first use; the following OPAL runs could reuse it.

## RAEv2-B

Record [I]. OPAL used a 0.3 pre-release. Its wall time includes the FID evaluations; the record does not separate their duration from training.

| Measurement | Static | OPAL |
|---|---|---|
| Run | dopt-raev2-b20-official-v1 | dopt-raev2-b20-opal-rs2 |
| Hardware | 8 × H100 80 GB | 8 × H100 80 GB |
| Training / loop clock | 2.65 h without evaluations | Not reported separately |
| Waiting for the agent | Not applicable | 1,776 s |
| Wall time | 3.21 h, including in-run evaluations | 3 h 45 min, including in-run evaluations |
| GPU-hours | 25.7 | 30.0 |
| Agent cost | Not applicable | $13.30 |
| Decisions asked / applied / failed | Not applicable | 50 / 6 / 0 |

## Goldfish: Amharic

Record [G]. Starting-recipe baseline only. OPAL used a 0.3 pre-release. Training-only time was not computed in the record; do not read wall time minus waiting as pure training. The static wall includes an interruption and resume.

| Measurement | Static | OPAL |
|---|---|---|
| Run | dopt-gf-amh-official-v1 | dopt-gf-amh-opal-official-a2 |
| Hardware | 1 × H100 80 GB | 1 × H100 80 GB |
| Training / loop clock | 3.95 h of logged update windows, including evaluations | Not reported separately |
| Waiting for the agent | Not applicable | 2,929 s |
| Wall time | 4.29 h to the final checkpoint upload | 4 h 53 min, including evaluations |
| GPU-hours | ≈4.3 | 4.88 |
| Agent cost | Not applicable | $12.62 |
| Decisions asked / applied / failed | Not applicable | 51 / 6 / 0 |

## Goldfish: Swahili

Record [G]. Starting-recipe baseline only. OPAL used a 0.3 pre-release. Training-only time was not computed in the record; do not read wall time minus waiting as pure training.

| Measurement | Static | OPAL |
|---|---|---|
| Run | dopt-gf-swa-official-v1 | dopt-gf-swa-opal-official-a2 |
| Hardware | 1 × H100 80 GB | 1 × H100 80 GB |
| Training / loop clock | 4.83 h of logged update windows, including evaluations | Not reported separately |
| Waiting for the agent | Not applicable | 2,418 s |
| Wall time | 4.84 h to the final checkpoint upload | 5 h 39 min, including evaluations |
| GPU-hours | ≈4.8 | 5.65 |
| Agent cost | Not applicable | $17.41 |
| Decisions asked / applied / failed | Not applicable | 51 / 11 / 0 |

## Goldfish: Kyrgyz

Record [G]. Starting-recipe baseline only. OPAL used a 0.3 pre-release. Training-only time was not computed in the record; do not read wall time minus waiting as pure training.

| Measurement | Static | OPAL |
|---|---|---|
| Run | dopt-gf-kir-official-v1 | dopt-gf-kir-opal-official-a2 |
| Hardware | 1 × H100 80 GB | 1 × H100 80 GB |
| Training / loop clock | 4.16 h of logged update windows, including evaluations | Not reported separately |
| Waiting for the agent | Not applicable | 2,234 s |
| Wall time | 4.17 h to the final checkpoint upload | 4 h 54 min, including evaluations |
| GPU-hours | ≈4.2 | 4.91 |
| Agent cost | Not applicable | $12.50 |
| Decisions asked / applied / failed | Not applicable | 50 / 6 / 0 |

## Track 3: Chinese from #2

Record [T]. OPAL 0.3 (pre-release); ranks equal at the recorded checks.

| Measurement | Static | OPAL |
|---|---|---|
| Run | s3000-f3d-r02 | t3-o02d-3k-499-rs-p0 |
| Hardware | 4 × H100 80 GB | 4 × H100 80 GB |
| Training / loop clock | 888 s | 3,402 s (record clock; includes agent waiting) |
| Loop less waiting | Not applicable | 946 s (derived; record clock minus waiting) |
| Waiting for the agent | Not applicable | 2,456 s |
| Wall time | ≈918 s (15.3 min) | 57.1 min |
| GPU-hours | 1.02 | 3.80 |
| Agent cost | Not applicable | $9.67 |
| Decisions asked / applied / failed | Not applicable | 49 / 5 / 0 |

## Track 3: Thai from #2

Record [T]. OPAL 0.3 (pre-release); ranks equal at the recorded checks.

| Measurement | Static | OPAL |
|---|---|---|
| Run | s3000-f3f-r02 | t3-o02f-3k-499-rs-p0 |
| Hardware | 4 × H100 80 GB | 4 × H100 80 GB |
| Training / loop clock | 882 s | 2,995 s (record clock; includes agent waiting) |
| Loop less waiting | Not applicable | 943 s (derived; record clock minus waiting) |
| Waiting for the agent | Not applicable | 2,052 s |
| Wall time | ≈903 s (15.1 min) | 50.3 min |
| GPU-hours | 1.00 | 3.35 |
| Agent cost | Not applicable | $13.08 |
| Decisions asked / applied / failed | Not applicable | 49 / 6 / 0 |

## Track 3: Kyrgyz from #2

Record [T]. OPAL 0.3 (pre-release); ranks equal at the recorded checks.

| Measurement | Static | OPAL |
|---|---|---|
| Run | s3000-f3c-r02 | t3-o02c-3k-499-rs-p0 |
| Hardware | 4 × H100 80 GB | 4 × H100 80 GB |
| Training / loop clock | 886 s | 3,979 s (record clock; includes agent waiting) |
| Loop less waiting | Not applicable | 951 s (derived; record clock minus waiting) |
| Waiting for the agent | Not applicable | 3,028 s |
| Wall time | ≈913 s (15.2 min) | 66.7 min |
| GPU-hours | 1.01 | 4.44 |
| Agent cost | Not applicable | $11.73 |
| Decisions asked / applied / failed | Not applicable | 49 / 8 / 0 |

## Track 3: Thai from #36

Record [T]. OPAL 0.3 (pre-release); ranks equal at the recorded checks.

| Measurement | Static | OPAL |
|---|---|---|
| Run | s3000-f3f-r36 | t3-o36f-3k-499-p0 |
| Hardware | 4 × H100 80 GB | 4 × H100 80 GB |
| Training / loop clock | 937 s | 2,765 s (record clock; includes agent waiting) |
| Loop less waiting | Not applicable | 977 s (derived; record clock minus waiting) |
| Waiting for the agent | Not applicable | 1,788 s |
| Wall time | ≈966 s (16.1 min) | 46.4 min |
| GPU-hours | 1.07 | 3.09 |
| Agent cost | Not applicable | $9.89 |
| Decisions asked / applied / failed | Not applicable | 49 / 6 / 0 |

## Track 3: Kyrgyz from #36

Record [T]. OPAL 0.3 (pre-release); ranks equal at the recorded checks.

| Measurement | Static | OPAL |
|---|---|---|
| Run | s3000-f3c-r36 | t3-o36c-3k-499-p0 |
| Hardware | 4 × H100 80 GB | 4 × H100 80 GB |
| Training / loop clock | 942 s | 3,032 s (record clock; includes agent waiting) |
| Loop less waiting | Not applicable | 980 s (derived; record clock minus waiting) |
| Waiting for the agent | Not applicable | 2,052 s |
| Wall time | ≈980 s (16.3 min) | 50.9 min |
| GPU-hours | 1.09 | 3.39 |
| Agent cost | Not applicable | $12.53 |
| Decisions asked / applied / failed | Not applicable | 49 / 13 / 0 |

## Track 3: Chinese from #36

Record [T]. OPAL 0.3 (pre-release); ranks equal at every check.

| Measurement | Static | OPAL |
|---|---|---|
| Run | s3000-f3d-r36 | t3-o36d-3k-447-p0 |
| Hardware | 4 × H100 80 GB | 4 × H100 80 GB |
| Training / loop clock | 966 s | 2,586 s (record clock; includes agent waiting) |
| Loop less waiting | Not applicable | 982 s (derived; record clock minus waiting) |
| Waiting for the agent | Not applicable | 1,604 s |
| Wall time | ≈1,004 s (16.7 min) | 43.4 min |
| GPU-hours | 1.12 | 2.89 |
| Agent cost | Not applicable | $11.65 |
| Decisions asked / applied / failed | Not applicable | 49 / 3 / 0 |

## Track 3: Quarter batch from #36

Record [T]. Compared with #36 at the agent's starting learning rate. The static timing record is f2e-r36-lr1-loss, a repeat of that starting recipe; it is not the learning-rate-scaled static run.

| Measurement | Static | OPAL |
|---|---|---|
| Run | f2e-r36-lr1-loss | t3-og-499-rs2-p0 |
| Hardware | 2 × H100 80 GB | 2 × H100 80 GB |
| Training / loop clock | 2,458 s | 4,694 s (record clock; includes agent waiting) |
| Loop less waiting | Not applicable | 2,242 s (derived; record clock minus waiting) |
| Waiting for the agent | Not applicable | 2,452 s |
| Wall time | ≈3,063 s (51.0 min) | 87.7 min |
| GPU-hours | 1.70 | 2.92 |
| Agent cost | Not applicable | $14.23 |
| Decisions asked / applied / failed | Not applicable | 49 / 8 / 0 |

## Track 3 Thai from #36: A2: front-loaded, 50 decisions, every 10

Record [T]. OPAL 0.3 (pre-release), single-run schedule/check-interval check. The record gives the training clock, not separate start/end wall timestamps for this arm.

| Measurement | Static | OPAL |
|---|---|---|
| Run | s3000-f3f-r36-1g | t3-b3a-3k-r2-p0 |
| Hardware | 1 × H100 80 GB | 1 × H100 80 GB |
| Training / loop clock | 3,709.6 s | 5,400 s (record clock; includes agent waiting) |
| Loop less waiting | Not applicable | 3,528 s (derived; record clock minus waiting) |
| Waiting for the agent | Not applicable | 1,872 s |
| Wall time | Exact total not recorded; 21:46–22:56 EDT interval | Not reported separately |
| GPU-hours | Not reported separately | Not reported separately |
| Agent cost | Not applicable | $8.12 |
| Decisions asked / applied / failed | Not applicable | 50 / 8 / not reported |

## Track 3 Thai from #36: B2: uniform, 50 decisions, every 10

Record [T]. OPAL 0.3 (pre-release), single-run schedule/check-interval check. The record gives the training clock, not separate start/end wall timestamps for this arm.

| Measurement | Static | OPAL |
|---|---|---|
| Run | s3000-f3f-r36-1g | t3-b3b-3k-r2-p0 |
| Hardware | 1 × H100 80 GB | 1 × H100 80 GB |
| Training / loop clock | 3,709.6 s | 5,354 s (record clock; includes agent waiting) |
| Loop less waiting | Not applicable | 3,474 s (derived; record clock minus waiting) |
| Waiting for the agent | Not applicable | 1,880 s |
| Wall time | Exact total not recorded; 21:46–22:56 EDT interval | Not reported separately |
| GPU-hours | Not reported separately | Not reported separately |
| Agent cost | Not applicable | $11.96 |
| Decisions asked / applied / failed | Not applicable | 49 / 4 / not reported |

## Track 3 Thai from #36: C2: early-dense, every 10

Record [T]. OPAL 0.3 (pre-release), single-run schedule/check-interval check. The record gives the training clock, not separate start/end wall timestamps for this arm.

| Measurement | Static | OPAL |
|---|---|---|
| Run | s3000-f3f-r36-1g | t3-b3c-3k-r2-p0 |
| Hardware | 1 × H100 80 GB | 1 × H100 80 GB |
| Training / loop clock | 3,709.6 s | 5,737 s (record clock; includes agent waiting) |
| Loop less waiting | Not applicable | 3,491 s (derived; record clock minus waiting) |
| Waiting for the agent | Not applicable | 2,246 s |
| Wall time | Exact total not recorded; 21:46–22:56 EDT interval | Not reported separately |
| GPU-hours | Not reported separately | Not reported separately |
| Agent cost | Not applicable | $8.75 |
| Decisions asked / applied / failed | Not applicable | 50 / 6 / not reported |

## Track 3 Thai from #36: D2: middle burst, every 10

Record [T]. OPAL 0.3 (pre-release), single-run schedule/check-interval check. The record gives the training clock, not separate start/end wall timestamps for this arm.

| Measurement | Static | OPAL |
|---|---|---|
| Run | s3000-f3f-r36-1g | t3-b3d-3k-r2-p0 |
| Hardware | 1 × H100 80 GB | 1 × H100 80 GB |
| Training / loop clock | 3,709.6 s | 5,479 s (record clock; includes agent waiting) |
| Loop less waiting | Not applicable | 3,479 s (derived; record clock minus waiting) |
| Waiting for the agent | Not applicable | 2,000 s |
| Wall time | Exact total not recorded; 21:46–22:56 EDT interval | Not reported separately |
| GPU-hours | Not reported separately | Not reported separately |
| Agent cost | Not applicable | $12.15 |
| Decisions asked / applied / failed | Not applicable | 50 / 7 / not reported |

## Track 3 Thai from #36: E2: front-loaded, 10 decisions, every 10

Record [T]. OPAL 0.3 (pre-release), single-run schedule/check-interval check. The record gives the training clock, not separate start/end wall timestamps for this arm.

| Measurement | Static | OPAL |
|---|---|---|
| Run | s3000-f3f-r36-1g | t3-b3e-3k-r2-p0 |
| Hardware | 1 × H100 80 GB | 1 × H100 80 GB |
| Training / loop clock | 3,709.6 s | 4,354 s (record clock; includes agent waiting) |
| Loop less waiting | Not applicable | 3,550 s (derived; record clock minus waiting) |
| Waiting for the agent | Not applicable | 804 s |
| Wall time | Exact total not recorded; 21:46–22:56 EDT interval | Not reported separately |
| GPU-hours | Not reported separately | Not reported separately |
| Agent cost | Not applicable | $3.28 |
| Decisions asked / applied / failed | Not applicable | 10 / 4 / not reported |

## Track 3 Thai from #36: Uniform 25 decisions, every 10

Record [T]. OPAL 0.3 (pre-release), single-run schedule/check-interval check. The record gives the training clock, not separate start/end wall timestamps for this arm.

| Measurement | Static | OPAL |
|---|---|---|
| Run | s3000-f3f-r36-1g | t3-ev10-3k-p0 |
| Hardware | 1 × H100 80 GB | 1 × H100 80 GB |
| Training / loop clock | 3,709.6 s | 4,636 s (record clock; includes agent waiting) |
| Loop less waiting | Not applicable | 3,516 s (derived; record clock minus waiting) |
| Waiting for the agent | Not applicable | 1,120 s |
| Wall time | Exact total not recorded; 21:46–22:56 EDT interval | Not reported separately |
| GPU-hours | Not reported separately | Not reported separately |
| Agent cost | Not applicable | $5.41 |
| Decisions asked / applied / failed | Not applicable | 24 / 4 / not reported |

## Track 3 Thai from #36: Uniform 25 decisions, every 30

Record [T]. OPAL 0.3 (pre-release), single-run schedule/check-interval check. The record gives the training clock, not separate start/end wall timestamps for this arm.

| Measurement | Static | OPAL |
|---|---|---|
| Run | s3000-f3f-r36-1g | t3-ev30-3k-p0 |
| Hardware | 1 × H100 80 GB | 1 × H100 80 GB |
| Training / loop clock | 3,709.6 s | 4,584 s (record clock; includes agent waiting) |
| Loop less waiting | Not applicable | 3,448 s (derived; record clock minus waiting) |
| Waiting for the agent | Not applicable | 1,136 s |
| Wall time | Exact total not recorded; 21:46–22:56 EDT interval | Not reported separately |
| GPU-hours | Not reported separately | Not reported separately |
| Agent cost | Not applicable | $5.62 |
| Decisions asked / applied / failed | Not applicable | 24 / 5 / not reported |

## Track 3 Thai from #36: Uniform 25 decisions, every 120

Record [T]. OPAL 0.3 (pre-release), single-run schedule/check-interval check. The record gives the training clock, not separate start/end wall timestamps for this arm.

| Measurement | Static | OPAL |
|---|---|---|
| Run | s3000-f3f-r36-1g | t3-ev120-3k-p0 |
| Hardware | 1 × H100 80 GB | 1 × H100 80 GB |
| Training / loop clock | 3,709.6 s | 4,897 s (record clock; includes agent waiting) |
| Loop less waiting | Not applicable | 3,541 s (derived; record clock minus waiting) |
| Waiting for the agent | Not applicable | 1,356 s |
| Wall time | Exact total not recorded; 21:46–22:56 EDT interval | Not reported separately |
| GPU-hours | Not reported separately | Not reported separately |
| Agent cost | Not applicable | $6.31 |
| Decisions asked / applied / failed | Not applicable | 24 / 9 / not reported |

## CIFAR-10 ResNet, blocking

Record [S]. OPAL 0.3 (pre-release); plain and OPAL ran on the same node.

| Measurement | Static | OPAL |
|---|---|---|
| Run | op-cifar-499-plain | op-cifar-499-b |
| Hardware | 1 × H100 80 GB | 1 × H100 80 GB |
| Training / loop clock | 66.5 s | 1,648 s (loop; includes waiting) |
| Loop less waiting | Not applicable | 82 s (derived; loop minus waiting) |
| Waiting for the agent | Not applicable | 1,566 s |
| Wall time | 82 s (command) | Not reported separately |
| GPU-hours | 0.023 | Not reported separately |
| Agent cost | Not applicable | $8.13 |
| Decisions asked / applied / failed | Not applicable | 46 / 3 / not reported |

## GPT small, blocking

Record [S]. OPAL 0.3 (pre-release); plain and OPAL ran on the same node.

| Measurement | Static | OPAL |
|---|---|---|
| Run | op-gpt-499-plain | op-gpt-499-b |
| Hardware | 1 × H100 80 GB | 1 × H100 80 GB |
| Training / loop clock | 186 s; script 190 s | 1,862 s (loop; includes waiting) |
| Loop less waiting | Not applicable | 214 s (derived; loop minus waiting) |
| Waiting for the agent | Not applicable | 1,648 s |
| Wall time | Not reported separately | Not reported separately |
| GPU-hours | Not reported separately | Not reported separately |
| Agent cost | Not applicable | $6.97 |
| Decisions asked / applied / failed | Not applicable | 46 / 8 / not reported |

## MNIST diffusion, blocking

Record [S]. OPAL 0.3 (pre-release); plain and OPAL ran on the same node. The source gives wall endpoints, not a separate loop timer.

| Measurement | Static | OPAL |
|---|---|---|
| Run | op-diff-499-plain | op-diff-499-b |
| Hardware | 1 × H100 80 GB | 1 × H100 80 GB |
| Training / loop clock | 71 s (includes periodic EMA validation) | Not reported separately |
| Waiting for the agent | Not applicable | 1,582 s |
| Wall time | Not reported separately | 06:07:08–06:35:43 UTC; 1,715 s (derived) |
| GPU-hours | Not reported separately | 0.476 (derived from wall) |
| Agent cost | Not applicable | $10.10 |
| Decisions asked / applied / failed | Not applicable | 38 / 14 / not reported |

## Overfit label10, test arrays in the process

Record [S]. OPAL 0.3 (pre-release). The agent saw development metrics only. The comparison uses the final static AdamW recipe; early-stopped and EMA selections are not substituted for that run.

| Measurement | Static | OPAL |
|---|---|---|
| Run | ssln-s-label10-3e3 | op-ofit-499-b |
| Hardware | 1 × H100 80 GB | 1 × H100 80 GB |
| Training / loop clock | 497.8 s (includes evaluations); script 499.0 s | Loop not separated; script 2,723 s (includes evaluations and waiting) |
| Waiting for the agent | Not applicable | 1,988 s |
| Wall time | 504 s (command) | 45.4 min (02:02:51–02:48:12 UTC) |
| GPU-hours | 0.140 | 0.76 |
| Agent cost | Not applicable | $8.90 |
| Decisions asked / applied / failed | Not applicable | 37 / 24 / 0 |

## Overfit label10, test arrays loaded at the end

Record [S]. OPAL 0.3 (pre-release) with the test-at-end patch. The source's script timer and timestamp span differ slightly; both are retained.

| Measurement | Static | OPAL |
|---|---|---|
| Run | ssln-s-label10-3e3 | op-ofit-499-te2 |
| Hardware | 1 × H100 80 GB | 1 × H100 80 GB |
| Training / loop clock | 497.8 s (includes evaluations); script 499.0 s | Loop not separated; script 2,328 s (includes evaluations and waiting) |
| Waiting for the agent | Not applicable | 1,513 s |
| Wall time | 504 s (command) | 02:42:08–03:20:53 EDT; 2,325 s (derived) |
| GPU-hours | 0.140 | 0.646 (derived from wall) |
| Agent cost | Not applicable | $7.22 |
| Decisions asked / applied / failed | Not applicable | 34 / 13 / 0 |

## MNIST MLP on CPU

Record [S]. OPAL 0.3 (pre-release). The local run selected Opus 5.5 but left effort at the CLI default.

| Measurement | Static | OPAL |
|---|---|---|
| Run | mnist_mlp.py plain | mnist_mlp.py --opal |
| Hardware | Mac CPU; no GPU | Same Mac CPU; no GPU |
| Training / loop clock | 7 s (includes epoch evaluations) | 862 s (includes epoch evaluations and waiting) |
| Loop less waiting | Not applicable | 330.1 s (derived; loop minus waiting) |
| Waiting for the agent | Not applicable | 531.9 s |
| Wall time | 16 s (command) | 868 s (command) |
| GPU-hours | 0 | 0 |
| Agent cost | Not applicable | $2.70 |
| Decisions asked / applied / failed | Not applicable | 42 / 6 / 0 |

## CIFAR-10 ResNet, non-blocking

Record [S]. OPAL 0.3 (pre-release). No change to training arithmetic was applied. The record does not give this run's separate times or cost.

| Measurement | Static | OPAL |
|---|---|---|
| Run | op-cifar-499-plain | Non-blocking CIFAR-10 ResNet row in the small-example record |
| Hardware | 1 × H100 80 GB | 1 × H100 80 GB |
| Training / loop clock | 66.5 s | Not reported separately |
| Waiting for the agent | Not applicable | Not reported separately |
| Wall time | 82 s (command) | Not reported separately |
| GPU-hours | 0.023 | Not reported separately |
| Agent cost | Not applicable | Not reported separately |
| Decisions asked / applied / failed | Not applicable | 1 asked / 0 applied / failed not reported; 0 answered, request cancelled at end |

## GPT small, non-blocking

Record [S]. OPAL 0.3 (pre-release). No change to training arithmetic was applied. The record does not give this run's separate times or cost.

| Measurement | Static | OPAL |
|---|---|---|
| Run | op-gpt-499-plain | Non-blocking GPT small row in the small-example record |
| Hardware | 1 × H100 80 GB | 1 × H100 80 GB |
| Training / loop clock | 186 s; script 190 s | Not reported separately |
| Waiting for the agent | Not applicable | Not reported separately |
| Wall time | Not reported separately | Not reported separately |
| GPU-hours | Not reported separately | Not reported separately |
| Agent cost | Not applicable | Not reported separately |
| Decisions asked / applied / failed | Not applicable | 1 asked / 0 applied / failed not reported; 0 answered |

## MNIST diffusion, non-blocking

Record [S]. OPAL 0.3 (pre-release). No change to training arithmetic was applied. The record does not give this run's separate times or cost.

| Measurement | Static | OPAL |
|---|---|---|
| Run | op-diff-499-plain | Non-blocking MNIST diffusion row in the small-example record |
| Hardware | 1 × H100 80 GB | 1 × H100 80 GB |
| Training / loop clock | 71 s (includes periodic EMA validation) | Not reported separately |
| Waiting for the agent | Not applicable | Not reported separately |
| Wall time | Not reported separately | Not reported separately |
| GPU-hours | Not reported separately | Not reported separately |
| Agent cost | Not applicable | Not reported separately |
| Decisions asked / applied / failed | Not applicable | 1 asked / 0 applied / failed not reported; 0 answered |

## Overfit label10, non-blocking

Record [S]. OPAL 0.3 (pre-release). This run did apply agent programs; it is not one of the unchanged runs used to measure variation.

| Measurement | Static | OPAL |
|---|---|---|
| Run | ssln-s-label10-3e3 | op-ofit-499-nb |
| Hardware | 1 × H100 80 GB | 1 × H100 80 GB |
| Training / loop clock | 497.8 s (includes evaluations); script 499.0 s | 637 s (timer scope not specified in the result row) |
| Waiting for the agent | Not applicable | Not reported separately |
| Wall time | 504 s (command) | Not reported separately |
| GPU-hours | 0.140 | Not reported separately |
| Agent cost | Not applicable | $2.40 |
| Decisions asked / applied / failed | Not applicable | 12 asked / 5 applied / failed not reported; 6 kept, 1 cancelled |

## OPAL Sol: overfit label10, test at end

Record [S]. NOT A CLEAN RESULT. Codex gpt-6.1-sol, high effort, the experimental Codex backend plus the test-at-end patch. From request 22, answers failed closed after persistent-history compaction, but OPAL logged them as kept. The kept count must not be read as successful agent decisions.

| Measurement | Static | OPAL |
|---|---|---|
| Run | ssln-s-label10-3e3 | op-ofit-sol-te |
| Hardware | 1 × H100 80 GB | 1 × H100 80 GB |
| Training / loop clock | 497.8 s (includes evaluations); script 499.0 s | Loop not separated; script 6,042 s (includes waiting) |
| Waiting for the agent | Not applicable | 4,188 s |
| Wall time | 504 s (command) | Not reported separately |
| GPU-hours | 0.140 | Not reported separately |
| Agent cost | Not applicable | 11.03M input (10.51M cached); 123k output (81k reasoning) tokens |
| Decisions asked / applied / failed | Not applicable | 29 asked / 15 applied; 13 logged as kept, 1 rejected; failed count is unreliable |

## OPAL Sol: GPT small

Record [S]. Experimental Codex backend, gpt-6.1-sol high, a pre-release build. No compaction recorded. The loop timer and timestamp span differ slightly; both are retained. Plain ran on another machine with the same GPU type.

| Measurement | Static | OPAL |
|---|---|---|
| Run | op-gpt-499-plain | op-gpt-sol-b |
| Hardware | 1 × H100 80 GB | 1 × H100 80 GB |
| Training / loop clock | 186 s; script 190 s | 4,789 s (loop; includes waiting) |
| Loop less waiting | Not applicable | 226 s (derived; loop minus waiting) |
| Waiting for the agent | Not applicable | 4,563 s |
| Wall time | Not reported separately | 04:14:57–05:34:45 EDT; 4,788 s (derived) |
| GPU-hours | Not reported separately | 1.330 (derived from wall) |
| Agent cost | Not applicable | 20.20M input (19.64M cached); 110k output (57k reasoning) tokens |
| Decisions asked / applied / failed | Not applicable | 46 / 23 / 0 |

## OPAL Sol: MNIST diffusion

Record [S]. Experimental Codex backend, gpt-6.1-sol high, a pre-release build. No compaction recorded. Plain ran on another machine with the same GPU type.

| Measurement | Static | OPAL |
|---|---|---|
| Run | op-diff-499-plain | op-diff-sol-b |
| Hardware | 1 × H100 80 GB | 1 × H100 80 GB |
| Training / loop clock | 71 s (includes periodic EMA validation) | Not reported separately |
| Waiting for the agent | Not applicable | 2,079 s |
| Wall time | Not reported separately | 05:47:37–06:24:37 EDT; 2,220 s (derived) |
| GPU-hours | Not reported separately | 0.617 (derived from wall) |
| Agent cost | Not applicable | 7.27M input (7.06M cached); 50k output (23k reasoning) tokens |
| Decisions asked / applied / failed | Not applicable | 46 / 24 / 0 |

## Evidence records

The research records are not bundled in this package; run identifiers refer to our saved logs, metrics and OPAL event folders. Record letters: [I] image runs, [G] Goldfish runs, [T] modded-nanogpt Track 3 runs, [S] small examples. Derived values only subtract recorded waits or convert recorded timestamp spans to GPU-hours.

See [what we checked](checks.md) for test-data handling, evaluation integrity, keep mode, and rank consistency.
