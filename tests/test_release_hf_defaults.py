"""The Trainer's implicit arguments must receive the same clipping validation as explicit ones."""

import pytest

pytest.importorskip("transformers")
pytest.importorskip("accelerate")
from transformers import GPT2Config, GPT2LMHeadModel  # noqa: E402

from opal.hf import OpalTrainer  # noqa: E402


def test_default_training_arguments_cannot_clip_twice(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    model = GPT2LMHeadModel(GPT2Config(n_layer=1, n_embd=8, n_head=1, vocab_size=8, bos_token_id=0, eos_token_id=0))
    with pytest.raises(ValueError, match="clipped twice"):
        OpalTrainer(model=model, opal=dict(goal="g", backend="keep", clip_grad_norm=1.0))
