# What we checked

These are checks behind the package examples, recorded for the October 21, 2026 release.
They cover data access, evaluation behaviour, unchanged training, and distributed
updates. [Run times and agent costs](times.md) are recorded separately.

## Test data stays out of the agent's messages

By design, the agent receives training observations and, when enabled, development
metrics. Test data and test scores are not sent to it. This is an interface rule,
not process isolation: an agent program runs inside the training process.

We repeated the overfit example with `--test-at-end`. The test arrays were loaded
only after the last update and `opt.close()`, and final and development-selected
models were scored afterwards. The result was test loss **0.6507**, accuracy
**80.42%** (`op-ofit-499-te2`), compared with **0.6576 / 80.00%** when the test set
was in the process (`op-ofit-499-b`). The saved messages of the latter run showed
development metrics only; its program scan found no test-data access.

This is one agent run in each condition. It checks that the improvement also
occurred with the test arrays absent during training; it does not measure the
effect of their presence independently of agent-to-agent variation.

## Saved agent programs

We scanned **200 run identifiers and 4,368 distinct complete programs**, deduplicated
by SHA-256. The scan recovered saved and overwritten programs from responses,
tool transcripts, workspaces, and source archives, then used Python syntax and
text checks with review of flagged code and its decision rationale.

The harmful programs were in `ss-o1`, `ss-o2`, and `ss-o3`: they temporarily
swapped weights or scaled the classifier around development evaluations, then
restored the training state. These accounted for **57 distinct complete programs**.
Two other runs, `dopt-t3-agent-opus-d10-c` and `dopt-t3-agent-opus-d30-c`, had
programs that inspected host optimizer source or function globals. None of these
runs supplied a published number in the blog/README corpus checked by the scan.

The scan found no agent program reading official development, validation, or test
samples or labels. This is a static finding, not a proof of isolation. Coverage
ended at the scan's local enumeration on October 20; later decisions are outside
it. One referenced run, a Goldfish Amharic run used only for a timing number, had no
recoverable local program history and was not checked. Saved or returned code is not proof that it was
activated during training.

## Evaluation integrity

With development reporting enabled, OPAL samples parameter values after updates
and checks the window around each reported evaluation. A temporary move followed
by a return raises an `eval_integrity` flag. A detected jump that remains on the
training trajectory is recorded as an informational `eval_jump`, not a failure.
The programs and event records still need to be read when interpreting either.

Both `op-ofit-499-b` and the test-at-end repeat `op-ofit-499-te2` recorded
**120 checks, 0 flagged**. The coordinator also recorded **64 `eval_jump` events**
for `op-ofit-499-b`; zero flags does not mean the parameters never jumped.

For the earlier overfit run `ss-o5`, an independent parameter-fingerprint check
recorded **42,120 update rows and 121 evaluations**: `n_evals_not_on_trajectory=0`
and `final_eval_on_trajectory=true`. The program review found permanent changes
to averaged weights, rather than weights swapped in only for an evaluation and
then restored. That external result applies to `ss-o5`, not automatically to
the later runs.

OPAL's check samples parameters over a finite window. It does not detect every
possible evaluation manipulation, changes to model outputs without parameter
moves, or data access. It flags a run; it does not stop it.

## Keep mode versus plain training

We ran MNIST on CPU both plain and with `--opal --backend keep`. Both printed
parameter checksum **`20f642dc05b6`**; comparison of all saved parameter tensors
also found bitwise equality after the release-fix integration.

Blank-slate Goldfish and LightningDiT integrations were checked on CPU at
a 0.3 pre-release, using both the original integration and the package's native interface
(`OpalTrainer` for Goldfish; the Accelerate optimizer passed directly to `wrap`
for LightningDiT). All paired losses and final parameter hashes matched bitwise
over **100 updates/steps**. A further native LightningDiT check at a 0.3 pre-release also
matched. These are CPU keep-mode checks; they do not establish identical GPU,
mixed-precision, distributed, or live-agent behaviour.

## Distributed rank consistency

The Track 3 harness periodically compared parameters across DDP ranks, every
**100 updates** in the overnight runs. The older runtime repaired a mismatch by
broadcasting rank zero's parameters. Such a repair changes the trajectory and
does not undo training already performed with different parameters; repaired
runs must be marked as such.

In this release, a candidate updater that makes ranks differ during its trial
window is rejected and rolled back; reports also flag runs that required repairs.
Outside that trial window, a periodic check can still repair a mismatch and flag
the run.
This checks sampled points in a run and candidates during
their trial windows; it is not a proof that ranks can never diverge later.

## Unchanged-run variation

The non-blocking small runs for CIFAR, GPT, and diffusion ended before their first
agent answer arrived. No program change was applied. Comparing them with their
plain runs on the same node gave these observed differences:

| Example | Plain | Unchanged non-blocking run | Absolute difference |
|---|---:|---:|---:|
| CIFAR test accuracy | 0.9449 | 0.9463 | 0.0014 |
| GPT validation loss | 4.4763 | 4.4798 | 0.0035 |
| Diffusion final EMA validation loss | 0.1696 | 0.1696 | 0 |

These are paired observations with non-deterministic training, not estimates of a
noise distribution. The CIFAR agent run's accuracy change is smaller than this
observed unchanged-run difference.

## Limits of the results

There is one run per setting. Agent-to-agent variation can be much larger than
unchanged-run variation: the overfit agent runs span about **72.4%–80.4%** test
accuracy, and the earlier Thai runs starting from record **#36** span about
**0.04** validation loss. These runs also include different package versions;
they do not isolate agent randomness. The decision-schedule checks are single
runs and do not establish a ranking between schedules.

OPAL has **no sandbox** for the optimizer and analysis programs: it executes them
in the training process. The program-boundary check flags suspicious constructs;
it does not block them or prevent access to objects reachable in that process.
The interface rules, static scan, evaluation checks, and test-at-end repeat address
different failure modes; none is a general security guarantee.

Comparisons here use the recipe the agent started from: record #36 settings are
compared with #36, and Goldfish with its starting recipe.

## Evidence records

The research records are named here for traceability; they are not bundled in this
package; they are our experiment logs, review notes and keep-mode comparisons.
