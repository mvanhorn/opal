"""opal.hf.OpalTrainer: a tiny GPT-2 on CPU. With backend="keep" the run is bitwise the plain Trainer's (which still
clips), and OPAL sees one loss per update, the mean of its micro-batches, also at an epoch's end; with the clipping
handed to OPAL it is the same up to rounding. Skipped when transformers is not installed (the "hf" extra)."""

import json

import pytest
import torch

transformers = pytest.importorskip("transformers")
from transformers import GPT2Config, GPT2LMHeadModel, Trainer, TrainingArguments  # noqa: E402

from opal.hf import OpalTrainer, agent_optimizer  # noqa: E402


class Tokens(torch.utils.data.Dataset):
    def __init__(self, n=20, length=16):
        g = torch.Generator().manual_seed(1)
        self.x = torch.randint(64, (n, length), generator=g)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return dict(input_ids=self.x[i], labels=self.x[i])


def train(tmp_path, name, opal=None, clip=1.0):
    torch.manual_seed(0)
    model = GPT2LMHeadModel(GPT2Config(n_layer=2, n_embd=32, n_head=2, n_positions=32, vocab_size=64))
    args = TrainingArguments(
        output_dir=str(tmp_path / name), per_device_train_batch_size=4, gradient_accumulation_steps=2,
        num_train_epochs=2, learning_rate=1e-3, warmup_steps=2, max_grad_norm=clip, logging_steps=1000,
        save_strategy="no", report_to=[], seed=0, use_cpu=True, dataloader_num_workers=0, disable_tqdm=True,
    )  # fmt: skip
    if opal is None:
        trainer = Trainer(model=model, args=args, train_dataset=Tokens())
    else:
        trainer = OpalTrainer(model=model, args=args, train_dataset=Tokens(), opal=opal)
    trainer.train()
    return torch.cat([p.detach().flatten() for p in model.parameters()]), trainer


def test_keep_is_bitwise_the_plain_trainer_and_sees_each_updates_loss(tmp_path):
    plain, _ = train(tmp_path, "plain")
    run = tmp_path / "r"
    wrapped, trainer = train(tmp_path, "opal", opal=dict(goal="g", backend="keep", every=1, run_dir=run, verbose=False))
    assert torch.equal(plain, wrapped)
    opt = agent_optimizer(trainer)
    assert opt.total == trainer.state.max_steps and opt._clip_grad_norm is None and trainer.args.max_grad_norm == 1.0
    losses = [x for r in map(json.loads, (run / "losses.jsonl").read_text().splitlines()) for x in r["losses"]]
    assert len(losses) == trainer.state.max_steps and all(0 < x < 10 for x in losses)
    assert min(losses) > 0.7 * max(losses)  # an epoch's last update (one micro-batch of two) is a mean, not halved


def test_the_clipping_handed_to_opal_starts_the_same_up_to_rounding(tmp_path):
    plain, _ = train(tmp_path, "plain")
    kw = dict(goal="g", backend="keep", clip_grad_norm=1.0, run_dir=tmp_path / "r", verbose=False)
    handed, trainer = train(tmp_path, "opal", opal=kw, clip=0.0)
    opt = agent_optimizer(trainer)
    assert opt._clip_grad_norm == 1.0 and torch.allclose(plain, handed, atol=2e-3)
    unclipped, _ = train(tmp_path, "unclipped", clip=0.0)
    assert (plain - handed).abs().max() < (plain - unclipped).abs().max()  # OPAL did clip


def test_clipping_twice_is_refused(tmp_path):
    with pytest.raises(ValueError, match="clipped twice"):
        train(tmp_path, "x", opal=dict(goal="g", backend="keep", clip_grad_norm=1.0, run_dir=tmp_path / "r"))


@pytest.mark.parametrize("path,stale", [
    ("docs/GUIDE.md", "Untested: Accelerate, HF Trainer"),
])
def test_integration_docs_reflect_cpu_coverage(path, stale):
    from pathlib import Path
    import opal

    text = (Path(opal.__file__).resolve().parents[1] / path).read_text()
    assert stale not in text


def test_a_skipped_accelerate_step_does_not_pollute_the_next_updates_loss(tmp_path, monkeypatch):
    from accelerate.optimizer import AcceleratedOptimizer

    training_step = OpalTrainer.training_step
    optimizer_step = AcceleratedOptimizer.step
    losses, calls = {}, []

    def record_loss(self, *args, **kwargs):
        loss = training_step(self, *args, **kwargs)
        losses.setdefault(self.state.global_step, []).append(float(loss))
        return loss

    def skip_once(self, *args, **kwargs):
        calls.append(1)
        self._is_overflow = len(calls) == 1  # GradScaler's skipped-step signal, emulated on CPU
        if not self._is_overflow:
            return optimizer_step(self, *args, **kwargs)

    monkeypatch.setattr(OpalTrainer, "training_step", record_loss)
    monkeypatch.setattr(AcceleratedOptimizer, "step", skip_once)
    _, trainer = train(tmp_path, "skipped", opal=dict(goal="g", backend="keep", every=1,
                                                   run_dir=tmp_path / "run", verbose=False))
    opt = agent_optimizer(trainer)
    opt.close(interrupted=True)
    actual = [x for r in map(json.loads, (opt.run.dir / "losses.jsonl").read_text().splitlines()) for x in r["losses"]]
    expected = [sum(losses[k]) for k in sorted(losses) if k > 0]
    assert len(actual) == len(expected) == len(calls) - 1
    assert torch.allclose(torch.tensor(actual), torch.tensor(expected))
