# Changelog

## 0.3.4 (2026-10-21)

- Stop agent calls when the remaining cost budget is below the estimated call cost (#6).
- Apply the default waiting cap to the remaining schedule after early decisions, excluding agent waits from training time (#4).

## 0.3.3 (2026-10-21)

First public release.

- `opal.wrap` turns a PyTorch optimizer into a reactive optimizer: a coding agent (Claude Code by default) reads the
  training run and rewrites the update rule and its own analysis while the model trains.
- Examples on CPU and one GPU, and the reproductions in the README (SiT-B/2, RAEv2-B, Goldfish, modded-nanogpt).
- `opal report`, `opal show`, `opal estimate`, `opal doctor`; the Hugging Face Trainer through `opal.hf.OpalTrainer`.
- An experimental Codex backend (`backend="codex"`).
- [What we checked](docs/checks.md) and [run times and agent costs](docs/times.md).
