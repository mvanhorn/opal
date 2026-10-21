# Changelog

## 0.3.5 (2026-10-21)

- Redact API keys and bearer tokens from backend diagnostics, decision records, prompts, logs and HTML reports.
- Reject raw optimizer parameter-set changes after wrapping, before step or zero_grad (#5).
- Reject GradScaler steps on wrappers combining multiple optimizers.
- Include OPAL programs, versions and update counts in Accelerate save_state/load_state checkpoints.
- Share one version source across package metadata, runtime, Codex client and provenance.
- Update wrap docstrings for cost-budget stopping and the Codex backend.
- Advance truncated loss-window indices and record omitted observation counts.
- Settle cancelled decisions with a bounded wait and include late charges in reports.
- Discard pending wake reasons when no further agent call can use them.
- Reject unsupported or cyclic agent updater state before installation and checkpointing.
- Compare ordered parameter bytes with SHA-256 in DDP consistency checks.

- Resolve repository documentation from test files for non-editable installs.
- Set tested direct dependency lower bounds and require Python 3.10 or newer.
- Make the mailbox cancellation test use an instance-level wait.
- Publish complete mailbox claims atomically so concurrent brokers cannot run the same request twice.
- Use file rendezvous in the DDP crash-resume test to avoid ephemeral-port races.
- Include docs, skills, examples and the changelog in the source distribution.
- Add MNIST --max-cost-usd and --no-summarize flags.
- Run CPU CI across Linux, macOS and supported Python versions, including direct dependency minima.

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
