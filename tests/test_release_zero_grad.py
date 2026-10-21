"""Torch optimizers allow gradient views, including DDP gradient bucket views."""

import torch

import opal


def test_zero_grad_can_clear_views_without_replacing_them(tmp_path):
    model = torch.nn.Linear(2, 1)
    opt = opal.wrap(
        torch.optim.SGD(model.parameters(), lr=0.1), model,
        total_steps=10, goal="g", backend=None, run_dir=tmp_path, verbose=False,
    )
    gradients = []
    for p in model.parameters():
        p.grad = torch.ones(p.numel() + 1)[1:].view_as(p)
        gradients.append(p.grad)
    try:
        opt.zero_grad(set_to_none=False)
        for p, grad in zip(model.parameters(), gradients):
            assert p.grad is grad and torch.count_nonzero(grad) == 0
        opt.zero_grad()
        assert all(p.grad is None for p in model.parameters())
    finally:
        opt.close(interrupted=True)
