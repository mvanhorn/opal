# Goldfish 125M GPT on 1 GB of Swahili

We trained a Goldfish language model from scratch on the 1,000 MB Swahili tier of the Goldfish corpus.
It has GPT-2 architecture: 12 layers, width 768, 124.8M parameters.
Training ran for 10 epochs on one GPU, with 79,360 updates of 64 sequences of 512 tokens.

The score is FLORES-200 bits. For each sentence, it sums log₂ surprisal over the second half given the first half,
split at the middle character. This follows upstream's `eval_code/flores_eval_goldfish.py` with
`ONLY_SECOND_HALF = True`. The score averages over the 2,009 sentences in dev and devtest.

| FLORES-200 bits ↓ (one run each) | Official recipe | Reactive optimizer |
|---|---|---|
| after the last update | 79.37 | **77.25** |

In this run the agent ended 2.12 bits below the official recipe.

The agent run used OPAL 0.3 with the neutral prompt: 51 decisions of which 11 changed the program,
$17.41, 5 h 39 min on one H100. This wall time includes evaluations and waiting for the agent.

## Official code and data

- Training uses [tylerachang/goldfish](https://github.com/tylerachang/goldfish) at `d4a518a` with
  [tylerachang/word-acquisition-language-models](https://github.com/tylerachang/word-acquisition-language-models)
  at `1182df1`, using Hugging Face Transformers 4.18 Trainer. The Goldfish repository has no license file.
  Its released models are Apache 2.0. This folder includes none of its code.
- Data comes from [goldfish-models/fish-food](https://huggingface.co/datasets/goldfish-models/fish-food) at revision
  `78ebeecb8a650267094071d87fbc8241c36ec96f`, CC BY-NC-SA 4.0. We rebuilt the 1,000 MB tier with the official pipeline.
  It has 507,910 sequences against the official 507,877. Tokenization uses the released 100 MB SentencePiece tokenizer
  with 50,000 pieces and vocabulary padded to 51,200.
- Evaluation concatenates dev and devtest from the original FLORES-200 release.
  Like upstream's `flores_eval_goldfish.py`, it measures bits per sentence for the second half given the first,
  then averages over sentences. Our scorer computes the same quantity and is not in this repository.
  It scored the released Swahili model at 79.472 bits against the published 79.454.
- The recipe uses Hugging Face's AdamW with betas (0.9, 0.999), ε 1e-6, learning rate 1e-4 and weight decay 0.01
  except on biases and LayerNorm weights. Linear warmup runs from 0 over the first 10% of updates, then linear decay
  goes to 0. Gradient clipping is at norm 1. We used 10 epochs and seed 43.

## Wiring it to OPAL

This folder describes wiring only. It has no runnable training commands.
The official environment cannot install OPAL. word-acquisition-language-models was last tested on Python 3.7,
PyTorch 1.12 and Transformers 4.18, and its `requirements.txt` pins torch 1.7.1.
OPAL needs Python ≥ 3.10 and PyTorch ≥ 2.5.
This example needs a PyTorch 2 port of the Trainer loop. Our runs used one with the same optimizer, schedule,
clipping and data order. That port is not in this repository yet. In a plain loop with this recipe, wrap it as follows:

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
| Hardware | 1× H100 80GB | 1× H100 80GB |
| Reported run time, including agent waits | 4.8 h | 5.65 h, of which 0.67 h waiting for the agent |
| Agent cost | – | $17.41 |
| Decisions answered / that changed code | – | 51 / 11 |
| FLORES-200 bits | 79.374 | 77.250 |

The official 4.8 h sums logged update windows, including evaluations. The agent time includes its 0.67 h wait.
