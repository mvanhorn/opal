# Goldfish 125M GPT on 1 GB of Swahili

A Goldfish language model (GPT-2 architecture, 12 layers, width 768, 124.8M parameters) trained from scratch on the
1,000 MB Swahili tier of the Goldfish corpus for 10 epochs (79,360 updates of 64 sequences of 512 tokens) on one GPU,
judged by FLORES-200 bits: the summed log₂ surprisal of the second half of each sentence given its first half (split
at the middle character, as upstream's `eval_code/flores_eval_goldfish.py` does with `ONLY_SECOND_HALF = True`),
averaged over the 2,009 sentences of the dev and devtest splits.

| FLORES-200 bits ↓ (one run each) | Official recipe | Reactive optimizer |
|---|---|---|
| after the last update | 79.37 | **77.60** |

In this run the agent ended 1.77 bits below the official recipe.

**Reproduced on OPAL 0.3.x:** 77.25 (one run, neutral prompt, 51 decisions of which 11 changed the program, $17.41,
5 h 39 min on one H100). The run above was not made with this package. It comes from the harness OPAL
grew from (October 2026): Claude Code (Opus 5.5, effort `high`) with that harness's neutral prompt, 100 decisions every
793 updates, training paused at each decision. Swahili has not been run on any released OPAL version.

## Official code and data

- Training: [tylerachang/goldfish](https://github.com/tylerachang/goldfish) at `d4a518a` with
  [tylerachang/word-acquisition-language-models](https://github.com/tylerachang/word-acquisition-language-models) at
  `1182df1` (Hugging Face Transformers 4.18 Trainer). The goldfish repository has no license file (the released models
  are Apache 2.0), so this folder ships none of its code.
- Data: [goldfish-models/fish-food](https://huggingface.co/datasets/goldfish-models/fish-food) at revision
  `78ebeecb8a650267094071d87fbc8241c36ec96f`, CC BY-NC-SA 4.0; the 1,000 MB tier is rebuilt with the official pipeline
  (our rebuild has 507,910 sequences against the official 507,877) and tokenized with the released 100 MB
  SentencePiece tokenizer (50,000 pieces, vocabulary padded to 51,200).
- Evaluation: the original FLORES-200 release, dev and devtest concatenated, scored as upstream's
  `flores_eval_goldfish.py` scores it: bits per sentence for its second half given its first half, the mean over the
  sentences. Our scorer computes the same quantity (it is not in this repository); its scoring of the released Swahili
  model gave 79.472 bits against the published 79.454.
- The recipe: AdamW (Hugging Face's, betas 0.9 and 0.999, ε 1e-6), learning rate 1e-4, weight decay 0.01 on all but
  biases and LayerNorm weights, linear warmup over the first 10% of updates from 0 and a linear decay to 0, gradient
  clipping at norm 1, 10 epochs, seed 43 in our runs.

## Wiring it to OPAL (untested on 0.3.x)

The official environment cannot install OPAL: word-acquisition-language-models was last tested on Python 3.7,
PyTorch 1.12 and Transformers 4.18 (its `requirements.txt` pins torch 1.7.1), and OPAL needs Python ≥ 3.9 and
PyTorch ≥ 2.3. This example therefore needs a PyTorch 2 port of the Trainer loop. Our runs used one (the same
optimizer, schedule, clipping and data order); it is not in this repository yet. In any plain loop with this recipe,
the wrap is:

```python
import os
import opal

opt = opal.wrap(
    opt, model, scheduler=sched, total_steps=79_360,
    goal="GPT-2-style language model (125M parameters) trained from scratch on 1 GB of Swahili text for 10 epochs; "
         "judged by the bits per sentence on the FLORES-200 Swahili dev and devtest sentences after the last update "
         "(lower is better)",
    clip_grad_norm=1.0,                  # delete clip_grad_norm_ from the loop; the starting program clips
    epochs=10, train_size="507,910 sequences of 512 tokens", batch_size="64 sequences of 512 tokens",
    same_order_each_epoch=True,          # the Trainer reads the pre-shuffled examples in the same order every epoch
    max_cost_usd=float(os.environ["OPAL_MAX_COST_USD"]) if "OPAL_MAX_COST_USD" in os.environ else None,
)
# per update, after the backward passes of its 8 accumulated micro-batches:
opt.step(loss=mean_loss_of_the_update)
```

Then run it once with `backend="keep"` for a few hundred updates: its losses must match the unwrapped run.

## Our runs

| | Official recipe | Reactive optimizer |
|---|---|---|
| Run | `dopt-gf-swa-official-v1` | `dopt-gf-swa-agent-v1` |
| Hardware | 1× H100 80GB | 1× H100 80GB |
| Training time (evaluations left out) | 4.8 h | 5.75 h, of which 0.77 h waiting for the agent |
| Agent cost | – | $17.79 |
| Decisions answered / that changed code | – | 100 / 4 (decisions 1, 2, 9 and 11) |
| FLORES-200 bits | 79.374 | 77.600 |

- **Reruns.** The agent's final program rerun from update 0 with no agent: 77.624. Its decisions replayed: 77.516.
- **Where the numbers come from:** each run's `mirror/metrics.jsonl` (the last `flores_bits`) and
  `agent/broker-state.json` (cost, decisions) in our experiment records.
