"""A failing program cannot leave stale gradient arguments in the fallback update."""

import pytest
import torch

import opal
from test_recovery import program


@pytest.mark.parametrize("change", ["build", "step"])
def test_exact_undo_redoes_the_update_with_the_restored_gradients(tmp_path, change):
    model = torch.nn.Linear(2, 1)
    original = [p.detach().clone() for p in model.parameters()]
    previous = tmp_path / "previous"
    previous.mkdir()
    (previous / "updater.py").write_text(program())
    opt = opal.replay(
        previous, torch.optim.SGD(model.parameters(), lr=0.01), model,
        total_steps=10, every=5, run_dir=tmp_path / "run", verbose=False,
    )
    for p in model.parameters():
        p.grad = torch.ones_like(p)
    build = "pass"
    fail = "grads.clear()\n        raise RuntimeError('failed after using grads as scratch')"
    if change == "build":
        build = "import torch\n    for _, p in named_params:\n        p.grad = torch.zeros_like(p)"
        fail = "raise RuntimeError('failed after replacing gradients in build')"
    bad = program(build=build, fail=fail)
    try:
        opt._apply(dict(k=1, requested_at=0, files=dict(opt.files, **{"updater.py": bad})), 1)
        opt.step(loss=1.0)
        for p, before in zip(model.parameters(), original):
            assert torch.equal(p, before - 0.01)
        assert "undone exactly and redone" in opt.errors[-1]["where"]
    finally:
        opt.close(interrupted=True)
