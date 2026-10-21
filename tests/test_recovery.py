"""Fallbacks and agent code that misbehaves: a program replaced inside its trial window is not a fallback; a
state_dict() that raises at a swap or a checkpoint; the modules of replaced versions are freed; ranks agree on the copy
that undoes a first update; agent code that raises SystemExit fails like any error."""

import gc
import io
import os
import sys
import weakref

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import opal
from test_core import SGD_UPDATER, events, make
from test_review1 import scripted

PLAIN_SGD = """
class Updater:
    def __init__(self, named_params, previous=None, spec=None):
        self.named, self.calls = named_params, 0
    def step(self, grads, update):
        self.calls += 1
        {fail}
        for n, p in self.named:
            if grads[n] is not None:
                p.add_(grads[n], alpha=-0.01)
    def state_dict(self):
        return {{}}
    def load_state_dict(self, s):
        pass
def build(named_params, previous=None, spec=None):
    {build}
    return Updater(named_params, previous, spec)
"""


def program(fail="pass", build="pass"):
    return PLAIN_SGD.format(fail=fail, build=build)


def steps(m, w, data, n, sched=None, lrs=None):
    for i in range(n):
        x, y = data[i % len(data)]
        w.zero_grad()
        loss = torch.nn.functional.mse_loss(m(x), y)
        loss.backward()
        w.step(loss=loss)
        if lrs is not None:
            lrs.append(w.user_optimizers[0].param_groups[0]["lr"])
        if sched is not None:
            sched.step()


def test_a_program_replaced_in_its_trial_window_is_not_a_fallback(tmp_path):
    """A (applied at 5, fails at its 12th update) is still in its trial window when B (applied at 10) fails at its first
    update: the run goes back to the starting program, which passed one, with B's update undone exactly, and the
    user's schedule continues as in the plain run. Before 0.3.2, B fell back to A, and A's failure ended the run."""
    a = program(fail="if self.calls >= 12:\n            raise RuntimeError('A fails')")
    b = program(fail="raise RuntimeError('B fails')")
    plain = []
    m, data = make()
    o = torch.optim.AdamW(m.parameters(), lr=1e-2)
    s = torch.optim.lr_scheduler.CosineAnnealingLR(o, T_max=60)
    for i in range(60):
        plain.append(o.param_groups[0]["lr"])
        s.step()
    m, data = make()
    o = torch.optim.AdamW(m.parameters(), lr=1e-2)
    s = torch.optim.lr_scheduler.CosineAnnealingLR(o, T_max=60)
    w = opal.wrap(o, m, scheduler=s, total_steps=400, goal="t", backend=scripted({"updater.py": a}, {"updater.py": b}),
                  every=5, decisions=40, trial_steps=20, run_dir=tmp_path / "r", verbose=False)  # fmt: skip
    lrs = []
    steps(m, w, data, 60, sched=s, lrs=lrs)
    ev = events(tmp_path / "r")
    assert [e["applied_at"] for e in ev if e["kind"] == "applied"] == [5, 10]
    assert [(e["update"], e["version"], e["restored"]) for e in ev if e["kind"] == "reverted"] == [(10, 2, True)]
    assert w.digests[0] == w._starting_digest and w.version == 3
    assert lrs[9:] == plain[9:]  # from update 10 on, the user's cosine as in the plain run
    w.close()


def test_two_swaps_before_an_update_are_undone_with_the_copy_from_before_both(tmp_path):
    """A and B are both applied before update 1; A's build doubles the parameters, B fails at its first update. The
    copy taken before A undoes both builds, and the starting SGD redoes the update: the plain run's parameters."""
    a = program(build="[p.data.mul_(2) for _, p in named_params]")
    b = program(fail="raise RuntimeError('B fails')")
    m = torch.nn.Linear(3, 1)
    ref = torch.nn.Linear(3, 1)
    ref.load_state_dict(m.state_dict())
    w = opal.wrap(torch.optim.SGD(m.parameters(), lr=0.1), m, total_steps=10, goal="t", backend=None,
                  run_dir=tmp_path / "r", every=5, verbose=False)  # fmt: skip
    for p, q in zip(m.parameters(), ref.parameters()):
        p.grad, q.grad = torch.ones_like(p), torch.ones_like(q)
    w._apply(dict(k=1, requested_at=0, files=dict(w.files, **{"updater.py": a})), 1)
    w._apply(dict(k=2, requested_at=0, files=dict(w.files, **{"updater.py": b})), 1)
    w.step(loss=1.0)
    torch.optim.SGD(ref.parameters(), lr=0.1).step()
    assert w.digests[0] == w._starting_digest
    assert all(torch.equal(p, q) for p, q in zip(m.parameters(), ref.parameters()))
    assert "the update was undone exactly and redone" in w.errors[-1]["where"]
    w.close()


# a state_dict() that is fine while empty (the check at build) and raises once buffers of different shapes exist
STATE_RAISES = """
import torch
class Updater:
    def __init__(self, named_params, previous=None, spec=None):
        self.named, self.bufs = named_params, {}
    def step(self, grads, update):
        for n, p in self.named:
            if grads[n] is not None:
                b = self.bufs.setdefault(n, torch.zeros_like(p))
                b.mul_(0.9).add_(grads[n])
                p.add_(b, alpha=-0.01)
    def state_dict(self):
        return {"bufs": torch.stack(list(self.bufs.values()))} if self.bufs else {}
    def load_state_dict(self, s):
        pass
def build(named_params, previous=None, spec=None):
    return Updater(named_params, previous, spec)
"""


def state_raises_run(tmp_path, *answers, n=150):
    m, data = make()
    o = torch.optim.SGD(m.parameters(), lr=0.01)
    w = opal.wrap(o, m, total_steps=400, goal="t", backend=scripted(*answers), every=10, decisions=4,
                  schedule="uniform", run_dir=tmp_path / "r", verbose=False)  # fmt: skip
    steps(m, w, data, n)
    return m, data, w


def test_a_state_dict_that_raises_at_a_swap_means_no_copy(tmp_path):
    edited = STATE_RAISES + "\n# a small edit\n"
    m, data, w = state_raises_run(tmp_path, {"updater.py": STATE_RAISES}, {"updater.py": edited}, n=250)
    assert [e["applied_at"] for e in events(tmp_path / "r") if e["kind"] == "applied"] == [100, 200]
    assert w.version == 2 and w._snapshot is None
    assert any("before a new updater was built" in e["where"] for e in w.errors)
    w.close()


def test_a_state_dict_that_raises_at_a_checkpoint_is_left_out(tmp_path):
    m, data, w = state_raises_run(tmp_path, {"updater.py": STATE_RAISES})
    assert w.version == 1
    state = w.state_dict()
    assert state["updater"] is None and state["files"]["updater.py"] == STATE_RAISES
    assert any(
        e["where"] == "checkpointing your updater state" and "state_dict() raised" in e["error"] for e in w.errors
    )
    torch.save(state, io.BytesIO())
    w.close()
    m2, _ = make()
    w2 = opal.wrap(torch.optim.SGD(m2.parameters(), lr=0.01), m2, total_steps=400, goal="t", backend=None,
                   run_dir=tmp_path / "r2", every=10, verbose=False)  # fmt: skip
    w2.load_state_dict(state)  # the program restarts its state
    assert w2.files["updater.py"] == STATE_RAISES
    w2.close()


CACHED = """
import torch
CACHE = {}  # module-level scratch
class Updater:
    def __init__(self, named_params, previous=None, spec=None):
        self.opts = spec["user_optimizers"]
        CACHE["big"] = torch.zeros(1000)
    def step(self, grads, update):
        for o in self.opts:
            o.step()
    def state_dict(self):
        return dict(opts=[o.state_dict() for o in self.opts])
    def load_state_dict(self, state):
        for o, s in zip(self.opts, state["opts"]):
            o.load_state_dict(s)
def build(named_params, previous=None, spec=None):
    return Updater(named_params, previous, spec)
"""


def test_replaced_versions_are_dropped_from_sys_modules_and_freed(tmp_path):
    def agent(ws, msg, system):
        (ws / "updater.py").write_text(CACHED + f"# version {len(agent.calls)}\n")
        agent.calls.append(1)
        return "x"

    agent.calls = []
    m, data = make()
    w = opal.wrap(torch.optim.SGD(m.parameters(), lr=0.01), m, total_steps=300, goal="t", backend=agent, every=10,
                  decisions=8, schedule="uniform", trial_steps=5, run_dir=tmp_path / "r",
                  verbose=False)  # fmt: skip
    tensors, in_trial = [], 0
    for _ in range(300):
        steps(m, w, data, 1)
        big = getattr(sys.modules[type(w.updater).__module__], "CACHE", {}).get("big")  # none in the starting program
        if big is not None and not any(r() is big for r in tensors):
            tensors.append(weakref.ref(big))
        del big
        if w._prev_updater is not None:  # the fallback of a trial window keeps its modules
            in_trial += 1
            assert type(w._prev_updater).__module__ in sys.modules
    gc.collect()
    assert w.version >= 6 and len(tensors) == w.version and in_trial > 0
    assert [r() is not None for r in tensors] == [False] * (len(tensors) - 1) + [True]
    live = {n.split(".")[0] for n in sys.modules if n.startswith("opal_program_r0v")}
    assert {f"opal_program_r0v{v}" for v in range(w.version)} & live == set()  # this run's versions but the last
    assert f"opal_program_r0v{w.version}" in live


def _ddp_snapshot_worker(rank, world, port, run_dir, out):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    torch.manual_seed(0)
    model = torch.nn.Linear(4, 1)
    ddp = torch.nn.parallel.DistributedDataParallel(model)
    bad = program(fail="[p.add_(1.0) for _, p in self.named]\n        raise RuntimeError('fails')")
    agent = scripted({"updater.py": bad})
    # this rank's copy would fit, the other's would not (a sharded optimizer keeps different state on each rank)
    w = opal.wrap(torch.optim.SGD(ddp.parameters(), lr=0.01), ddp, total_steps=40, goal="t", decisions=1, every=10,
                  sync_every=5, rank_check_every=0, run_dir=run_dir, backend=agent if rank == 0 else None,
                  snapshot_limit_gb=2.0 if rank == 0 else 0, verbose=False)  # fmt: skip
    g = torch.Generator().manual_seed(rank)
    for i in range(40):
        loss = ddp(torch.randn(8, 4, generator=g)).pow(2).mean()
        w.zero_grad()
        loss.backward()
        w.step(loss=loss)
    torch.save(dict(params=torch.cat([p.detach().flatten() for p in model.parameters()])), f"{out}/rank{rank}.pt")
    dist.destroy_process_group()


def test_ddp_ranks_agree_on_the_copy_that_undoes_a_first_update(tmp_path):
    """Rank 1's copy is above its snapshot_limit_gb: no rank keeps one, so the failing first update is handled the same
    way on both (with no rank check, the parameters still agree)."""
    port = 29500 + (os.getpid() + 13) % 1000
    mp.spawn(_ddp_snapshot_worker, args=(2, port, str(tmp_path / "r"), str(tmp_path)), nprocs=2, join=True)
    a, b = (torch.load(tmp_path / f"rank{r}.pt") for r in (0, 1))
    assert torch.equal(a["params"], b["params"])
    rev = [e for e in events(tmp_path / "r") if e["kind"] == "reverted"]
    assert [e["restored"] for e in rev] == [False]


def test_agent_code_that_raises_systemexit_fails_like_any_error(tmp_path):
    build_exits = SGD_UPDATER.replace("    return Updater(named_params, previous, spec)", "    raise SystemExit(1)")
    step_exits = SGD_UPDATER.replace("RAN.append(update)", "RAN.append(update)\n        raise SystemExit(1)")
    analysis_exits = "def analyze(obs, memory):\n    raise SystemExit(1)\n"
    agent = scripted({"updater.py": build_exits}, {"updater.py": step_exits}, {"analysis.py": analysis_exits})
    m, data = make()
    w = opal.wrap(torch.optim.SGD(m.parameters(), lr=0.01), m, total_steps=200, goal="t", backend=agent, every=10,
                  decisions=4, schedule="uniform", run_dir=tmp_path / "r", verbose=False)  # fmt: skip
    steps(m, w, data, 200)
    kinds = [e["kind"] for e in events(tmp_path / "r")]
    assert w.update == 200 and "load_failed" in kinds and "reverted" in kinds
    assert any(e["where"] == "analysis.analyze" and "SystemExit" in e["error"] for e in w.errors)

    interrupts = SGD_UPDATER.replace("RAN.append(update)", "RAN.append(update)\n        raise KeyboardInterrupt")
    w = opal.wrap(torch.optim.SGD(m.parameters(), lr=0.01), m, total_steps=10, goal="t", backend=None,
                  run_dir=tmp_path / "r2", every=5, verbose=False)  # fmt: skip
    w._apply(dict(k=1, requested_at=0, files=dict(w.files, **{"updater.py": interrupts})), 1)
    with pytest.raises(KeyboardInterrupt):  # not agent failure: the person stopping the run
        steps(m, w, data, 1)
    w.close()


def test_a_failing_first_update_restores_parameter_shapes(tmp_path):
    m, data = make()
    original = [p.detach().clone() for p in m.parameters()]
    w = opal.wrap(torch.optim.SGD(m.parameters(), lr=0.01), m, total_steps=10, goal="t", backend=None,
                  run_dir=tmp_path / "r", every=5, verbose=False)
    bad = program(fail="self.named[0][1].data = self.named[0][1].flatten()\n        raise RuntimeError('bad reshape')")
    for p in m.parameters():
        p.grad = torch.ones_like(p)
    try:
        w._apply(dict(k=1, files=dict(w.files, **{"updater.py": bad})), 1)
        w.step(loss=1.0)
        assert all(torch.equal(p, before - 0.01) for p, before in zip(m.parameters(), original))
        assert [e["restored"] for e in events(tmp_path / "r") if e["kind"] == "reverted"] == [True]
    finally:
        w.close(interrupted=True)
