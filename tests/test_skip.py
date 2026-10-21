"""Skipping updates: can_skip with a step that returns True, ctx.loss, the cap on skips in a row, DDP agreement."""

import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import opal
from test_core import events, flat, make

SKIP_EVERY_7TH = """
import torch

LOSSES = []

class Updater:
    can_skip = True

    def __init__(self, named_params, previous=None, spec=None):
        self.named = named_params
        self.buf = {n: torch.zeros_like(p) for n, p in named_params}

    def step(self, grads, update, ctx):
        LOSSES.append(None if ctx.loss is None else float(ctx.loss))
        for n, p in self.named:
            if grads[n] is not None:
                self.buf[n].mul_(0.9).add_(grads[n])
                p.add_(self.buf[n], alpha=-0.02)
        return torch.tensor(update % 7 == 0)

    def state_dict(self):
        return dict(buf=self.buf)

    def load_state_dict(self, s):
        for n in self.buf:
            self.buf[n].copy_(s["buf"][n])


def build(named_params, previous=None, spec=None):
    return Updater(named_params, previous, spec)
"""

ANALYSIS = "def analyze(obs, memory):\n    return {}\n"


def program(tmp_path, updater):
    d = tmp_path / "prog"
    d.mkdir()
    (d / "updater.py").write_text(updater)
    (d / "analysis.py").write_text(ANALYSIS)
    return d


def reference(steps, skip):
    """Momentum SGD by hand, leaving out the updates `skip` picks (their gradients are computed and dropped)."""
    m, data = make()
    buf = [torch.zeros_like(p) for p in m.parameters()]
    for i in range(steps):
        x, y = data[i % len(data)]
        m.zero_grad()
        torch.nn.functional.mse_loss(m(x), y).backward()
        if skip(i + 1):
            continue
        with torch.no_grad():
            for b, p in zip(buf, m.parameters()):
                b.mul_(0.9).add_(p.grad)
                p.add_(b, alpha=-0.02)
    return flat(m)


def run(tmp_path, updater, steps):
    m, data = make()
    opt = torch.optim.SGD(m.parameters(), lr=0.0)
    w = opal.replay(program(tmp_path, updater), opt, m, total_steps=steps, run_dir=tmp_path / "r", verbose=False)
    losses = []
    for i in range(steps):
        x, y = data[i % len(data)]
        loss = torch.nn.functional.mse_loss(m(x), y)
        losses.append(float(loss))
        w.zero_grad()
        loss.backward()
        w.step(loss=loss)
    return m, w, losses


def test_skipped_updates_leave_parameters_and_state_as_if_never_applied(tmp_path):
    steps = 100
    m, w, losses = run(tmp_path, SKIP_EVERY_7TH, steps)
    assert torch.equal(flat(m), reference(steps, lambda u: u % 7 == 0))
    assert w._skipped[:3] == [7, 14, 21]
    mod = next(
        v for k, v in __import__("sys").modules.items() if k.startswith("opal_program_") and hasattr(v, "LOSSES")
    )
    assert mod.LOSSES == losses  # ctx.loss is this update's loss


def test_skips_in_a_row_are_capped(tmp_path):
    steps = 40
    always = SKIP_EVERY_7TH.replace("return torch.tensor(update % 7 == 0)", "return True")
    m, w, _ = run(tmp_path, always, steps)
    # updates 1-10 skipped, 11 applied, 12-21 skipped, 22 applied, ...
    assert torch.equal(flat(m), reference(steps, lambda u: u % 11 != 0))
    assert any("skip more than" in e["error"] for e in w.errors)


def _ddp_loss_worker(rank, world, port, prog, out):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(8, 32), torch.nn.ReLU(), torch.nn.Linear(32, 1))
    ddp = torch.nn.parallel.DistributedDataParallel(model)
    opt = torch.optim.SGD(ddp.parameters(), lr=0.0)
    w = opal.replay(prog, opt, ddp, total_steps=30, run_dir=out + "/r", verbose=False)
    g = torch.Generator().manual_seed(rank)
    losses = []
    for _ in range(30):
        loss = ddp(torch.randn(16, 8, generator=g)).pow(2).mean()
        losses.append(float(loss))
        w.zero_grad()
        loss.backward()
        w.step(loss=loss)
    import sys

    mod = next(v for k, v in sys.modules.items() if k.startswith("opal_program_") and hasattr(v, "LOSSES"))
    torch.save(
        dict(seen=mod.LOSSES, own=losses, params=torch.cat([p.detach().flatten() for p in model.parameters()])),
        f"{out}/rank{rank}.pt",
    )
    dist.destroy_process_group()


def test_ddp_ranks_see_the_mean_loss_and_skip_together(tmp_path):
    prog = str(program(tmp_path, SKIP_EVERY_7TH))
    port = 29600 + os.getpid() % 1000
    mp.spawn(_ddp_loss_worker, args=(2, port, prog, str(tmp_path)), nprocs=2, join=True)
    a, b = (torch.load(tmp_path / f"rank{r}.pt") for r in (0, 1))
    assert a["seen"] == b["seen"]
    mean = [(x + y) / 2 for x, y in zip(a["own"], b["own"])]
    assert max(abs(s - m) for s, m in zip(a["seen"], mean)) < 1e-6
    assert torch.equal(a["params"], b["params"])
