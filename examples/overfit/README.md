# examples/overfit

CIFAR-10 runs that overfit under a static AdamW, for OPAL 0.1.1-validation.

The question: a static AdamW run overfits (development and test loss rise, accuracy peaks then falls); does the agent,
shown the development set's loss and accuracy (wrap(validation=N)), steer the optimizer so that the run does not?
Baselines: static AdamW with its learning rate swept on the development set, and early stopping on the development
set (the test accuracy at the best development evaluation; this script logs everything needed and saves best.pt).

Settings (examples/overfit/splits.py; the same splits and labels in every run):
  label10  45,000 training images, 10% of them (4,500) with a wrong label (uniform over the other classes, fixed seed)
  small20  10,000 training images (20% of CIFAR-10's training set, 1,000 per class), clean labels
  Both: development set = 5,000 images (10% of each class of the official training set, clean labels), held out from
  training; test set = the official 10,000 test images, logged here only and never shown to the agent.
Model: ResNet-18 with a CIFAR stem (--width 64: 11.2M parameters; --width 128 for a wider, more overfit-prone one),
batch norm, no dropout. The model normalises its logits (a centred LogitNorm, τ = 0.04, as in the blog's run), so the
agent cannot set a temperature on the development set. No data augmentation by default (--augment adds flips and
4-pixel crops). Optimizer: AdamW (lr --lr, betas 0.9/0.999, weight decay --wd on every tensor) with a linear warmup
over --warmup-epochs and a cosine decay to 0 (--schedule constant: warmup, then flat). Cross-entropy, no label
smoothing. A new data order each epoch, from --data-seed; the model initialisation from --seed.

    python examples/overfit/train.py --prepare-data                                   # download CIFAR-10 once
    python examples/overfit/train.py --setting label10 --lr 1e-3                      # static AdamW
    python examples/overfit/train.py --setting label10 --opal --validation            # agent, sees the dev set
    python examples/overfit/train.py --setting label10 --opal                         # agent, no dev set shown
    python examples/overfit/train.py --setting small20 --data synthetic --device cpu --width 4 --epochs 2 \
        --batch-size 64 --opal --validation --backend keep                            # CPU smoke test

The blog's run (label10, 120 epochs, its goal sentence passed with --goal):

    GOAL="ResNet-18 (11.2M parameters) trained from scratch on 45,000 CIFAR-10 training images for 120 epochs; judged by the cross-entropy loss on the 10,000 CIFAR-10 test images after the last update (lower is better). Do not change the parameters for an evaluation and change them back afterwards."
    python examples/overfit/train.py --setting label10 --lr 3e-3 --epochs 120 --opal --validation \
        --max-cost-usd 40 --eval-seed 0 --goal "$GOAL"

That run was judged by the test cross-entropy loss, not by the accuracy that train.py's default goal names, and its
goal adds the rule against changing the parameters for an evaluation.

Agent settings: OPAL's own Claude backend defaults, as in the blog's runs: no effort is passed, and --model defaults
to the backend's default. Agent runs use the neutral prompt, 30 uniformly spaced decisions, blocking, and the
training script (this file) and the model source (resnet.py) as the agent's read-only reference (--no-share-code to
withhold them); splits.py is not shared.

Why the split into files: OPAL gives the agent read-only copies of the training script (train.py) and of the
model's source (resnet.py). Everything about the experiment itself (its question, the wrong labels of label10, the
test-set scores, early stopping, the result) lives in this README, splits.py and records.py, which the agent does
not see, and the default output folder has a neutral name (the agent's working directory lies inside it).

Outputs in --out: log.jsonl (a config line, one line per evaluation: train loss and accuracy since the previous
evaluation, development and test loss and accuracy, for label10 how many wrongly labelled training images the model
predicts with their wrong and with their true label; a result line), result.json, best.pt (the model at the best
development accuracy) and, for agent runs, opal/ (OPAL's run folder).

Evaluation times (0.2.2): the development set is evaluated after one update drawn uniformly at random in each epoch
(--evals-per-epoch K: one in each K-th part of the epoch), and after the last update; records.eval_updates draws them
from --eval-seed, and the config line lists them (eval_updates). The test set is evaluated at the same updates, for us
only. The draw lives in records.py so that train.py, which the agent reads, shows no cadence; the agent is told "about
every N updates, at times that are not announced" (N = updates per epoch / K). Until 0.2.1 the script evaluated every
--dev-every updates (default one epoch) and the prompt said "Every N updates": two agent runs on small20 timed
changes of the parameters to the evaluations (a classifier scale probe; a weight average swapped in for each
evaluation), which wrap(validation=N) now checks for (eval_integrity).

With --test-at-end, prepare the CIFAR-10 cache separately using --prepare-data. During training, only x_train and
y_train are read from the NPZ cache; no test arrays are loaded or scored. After the last update and opt.close(), the
final model and the models selected by development accuracy/loss are scored on the test set. Per-evaluation logs
omit test scores; result.json keeps its keys, with peak_test_acc_oracle and overfit_gap_test_acc set to null. The
two development-selected states are retained in CPU memory even with --no-save-best. OPAL's timing observations and
its copy of the training source change; this flag does not sandbox agent programs or restrict filesystem access.
