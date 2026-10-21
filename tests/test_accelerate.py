"""Accelerate (CPU): wrap after accelerator.prepare. OPAL wraps the torch optimizer inside Accelerate's, follows the
torch scheduler inside its AcceleratedScheduler, and keep mode is bitwise the plain loop; a GradScaler or Accelerate's
gradient accumulation is refused. The model's description strips every wrapper. Skipped without accelerate."""

import pytest
import torch

accelerate = pytest.importorskip("accelerate")
from accelerate import Accelerator  # noqa: E402
from accelerate.state import AcceleratorState, GradientState  # noqa: E402

import opal  # noqa: E402
from opal import context  # noqa: E402
from test_core import adamw, make  # noqa: E402


@pytest.fixture(autouse=True)
def fresh_state():
    AcceleratorState._reset_state(True)
    GradientState._reset_state()
    yield
    AcceleratorState._reset_state(True)
    GradientState._reset_state()


def run(tmp_path, wrapped, steps=60, **acc_kw):
    m, data = make()
    o, s = adamw(m, steps)
    acc = Accelerator(cpu=True, **acc_kw)
    m, o, s = acc.prepare(m, o, s)
    if wrapped:
        o = opal.wrap(o, m, scheduler=s, total_steps=steps, goal="g", backend="keep", every=10, run_dir=tmp_path / "r",
                      verbose=False)  # fmt: skip
    for i in range(steps):
        x, y = data[i % len(data)]
        loss = torch.nn.functional.mse_loss(m(x), y)
        acc.backward(loss)
        o.step(loss=loss) if wrapped else o.step()
        s.step()
        o.zero_grad()
    return torch.cat([p.detach().flatten() for p in m.parameters()]), o


def test_keep_after_prepare_is_bitwise_the_plain_loop(tmp_path):
    plain, _ = run(tmp_path, False)
    AcceleratorState._reset_state(True)
    GradientState._reset_state()
    wrapped, o = run(tmp_path, True)
    assert torch.equal(plain, wrapped)
    assert type(o.user_optimizers[0]).__name__ == "AdamW" and type(o.user_schedulers[0]).__module__.startswith("torch")


def test_accumulation_through_accelerate_is_refused(tmp_path):
    with pytest.raises(ValueError, match="accumulates gradients over 2 step"):
        run(tmp_path, True, gradient_accumulation_steps=2)


def test_every_wrapper_of_the_model_is_stripped():
    m, _ = make()

    class Wrapper(torch.nn.Module):  # what DDP and Accelerate look like from outside: the model in .module
        def __init__(self, inner):
            super().__init__()
            self.module = inner

    assert context.bare_model(Wrapper(Wrapper(m))) is m and context.bare_model(m) is m
