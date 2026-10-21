"""CPU tests with a fake agent: equivalence, async swaps, rejects, reverts, kept answers, resume, blocking, NaN, DDP."""

import json
import os
import time
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import opal


SGD_UPDATER = """
import torch
RAN = []

class Updater:
    def __init__(self, named_params, previous=None, spec=None):
        self.named = named_params
        self.lr = torch.tensor(0.05)

    def step(self, grads, update):
        RAN.append(update)
        for n, p in self.named:
            if grads[n] is not None:
                p.add_(grads[n], alpha=-float(self.lr))

    def state_dict(self):
        return dict(lr=self.lr)

    def load_state_dict(self, s):
        self.lr = s["lr"]


def build(named_params, previous=None, spec=None):
    return Updater(named_params, previous, spec)
"""

MOMENTUM_UPDATER = """
import torch

class Updater:
    def __init__(self, named_params, previous=None, spec=None):
        self.named = named_params
        self.buf = {n: torch.zeros_like(p) for n, p in named_params}

    def step(self, grads, update):
        lr = 0.02 * (1 - update / 1000)
        for n, p in self.named:
            if grads[n] is not None:
                self.buf[n].mul_(0.9).add_(grads[n])
                p.add_(self.buf[n], alpha=-lr)

    def state_dict(self):
        return dict(buf=self.buf)

    def load_state_dict(self, s):
        self.buf = s["buf"]


def build(named_params, previous=None, spec=None):
    return Updater(named_params, previous, spec)
"""


def make(seed=0):
    torch.manual_seed(seed)
    model = torch.nn.Sequential(torch.nn.Linear(8, 32), torch.nn.ReLU(), torch.nn.Linear(32, 1))
    data = [(torch.randn(16, 8), torch.randn(16, 1)) for _ in range(50)]
    return model, data


def train(model, opt, data, steps, sched=None, pass_loss=True, nan_at=None, start=0):
    for i in range(start, steps):
        x, y = data[i % len(data)]
        loss = torch.nn.functional.mse_loss(model(x), y)
        if nan_at is not None and i == nan_at:
            loss = loss * float("nan")
        opt.zero_grad()
        loss.backward()
        opt.step(loss=loss) if pass_loss else opt.step()
        if sched is not None:
            sched.step()
    return model


def flat(model):
    return torch.cat([p.detach().flatten() for p in model.parameters()])


def adamw(model, steps):
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.1)
    return opt, torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)


def events(run_dir):
    return [json.loads(line) for line in open(Path(run_dir) / "events.jsonl")]


def test_starting_program_is_bitwise_identical(tmp_path):
    steps = 120
    m1, data = make()
    o1, s1 = adamw(m1, steps)
    train(m1, o1, data, steps, sched=s1, pass_loss=False)
    m2, _ = make()
    o2, s2 = adamw(m2, steps)
    w = opal.wrap(
        o2, m2, scheduler=s2, total_steps=steps, goal="test", backend=None, run_dir=tmp_path / "r", verbose=False
    )
    train(m2, w, data, steps, sched=s2)
    assert torch.equal(flat(m1), flat(m2))
    assert s2.get_last_lr() == s1.get_last_lr()


def test_kept_answer_changes_nothing(tmp_path):
    steps = 200
    m1, data = make()
    o1, s1 = adamw(m1, steps)
    train(m1, o1, data, steps, sched=s1, pass_loss=False)
    m2, _ = make()
    o2, s2 = adamw(m2, steps)
    calls = []
    w = opal.wrap(
        o2,
        m2,
        scheduler=s2,
        total_steps=steps,
        goal="test",
        decisions=4,
        every=10,
        run_dir=tmp_path / "r",
        backend=lambda ws, msg, sys_: calls.append(msg) or "keep",
        verbose=False,
    )
    train(m2, w, data, steps, sched=s2)
    assert torch.equal(flat(m1), flat(m2))
    assert calls and w.version == 0
    assert not list((tmp_path / "r" / "programs").glob("v001"))
    assert any(e["kind"] == "kept" for e in events(tmp_path / "r"))


def test_async_edit_applies_later_and_runs(tmp_path):
    steps = 400

    def agent(ws, msg, sys_):
        time.sleep(0.3)
        (ws / "updater.py").write_text(SGD_UPDATER)
        return "switch to SGD"

    m, data = make()
    o, s = adamw(m, steps)
    w = opal.wrap(
        o,
        m,
        scheduler=s,
        total_steps=steps,
        goal="test",
        decisions=2,
        every=10,
        run_dir=tmp_path / "r",
        backend=agent,
        blocking=False,  # the opt-in mode under test
        verbose=False,
    )
    t0 = time.time()
    for i in range(steps):
        x, y = data[i % len(data)]
        loss = torch.nn.functional.mse_loss(m(x), y)
        w.zero_grad()
        loss.backward()
        w.step(loss=loss)
        s.step()
        time.sleep(0.002)
    ev = [e for e in events(tmp_path / "r") if e["kind"] == "applied"]
    assert ev, events(tmp_path / "r")
    assert ev[0]["applied_at"] > ev[0]["requested_at"] + 10  # training went on while the agent thought
    import sys

    mod = sys.modules["opal_program_r0v1.updater"]
    assert mod.RAN and mod.RAN[0] == ev[0]["applied_at"]
    assert w.version >= 1


def test_syntax_error_rejected_and_reported(tmp_path):
    steps = 300
    msgs = []

    def agent(ws, msg, sys_):
        msgs.append(msg)
        if len(msgs) == 1:
            (ws / "updater.py").write_text("def build(:\n")
        return "x"

    m, data = make()
    o, s = adamw(m, steps)
    w = opal.wrap(
        o,
        m,
        scheduler=s,
        total_steps=steps,
        goal="t",
        decisions=3,
        every=10,
        run_dir=tmp_path / "r",
        backend=agent,
        verbose=False,
    )
    train(m, w, data, steps, sched=s)
    assert w.version == 0
    assert len(msgs) >= 2 and "SyntaxError" in msgs[1]
    assert "def build(:" not in (tmp_path / "r" / "workspace" / "updater.py").read_text()


def test_exception_in_step_reverts(tmp_path):
    steps = 300
    bad = SGD_UPDATER.replace(
        "RAN.append(update)", "RAN.append(update)\n        if update > 0: raise ValueError('boom')"
    )

    def agent(ws, msg, sys_):
        if "boom" not in (ws / "updater.py").read_text() and "reverted" not in msg:
            (ws / "updater.py").write_text(bad)
        return "x"

    m, data = make()
    o, s = adamw(m, steps)
    w = opal.wrap(
        o,
        m,
        scheduler=s,
        total_steps=steps,
        goal="t",
        decisions=2,
        every=10,
        run_dir=tmp_path / "r",
        backend=agent,
        blocking=True,
        verbose=False,
    )
    train(m, w, data, steps, sched=s)
    kinds = [e["kind"] for e in events(tmp_path / "r")]
    assert "reverted" in kinds
    assert "boom" not in "".join(w.files.values())
    assert torch.isfinite(flat(m)).all()


def test_blocking_applies_next_update(tmp_path):
    steps = 200

    def agent(ws, msg, sys_):
        time.sleep(0.2)
        (ws / "updater.py").write_text(MOMENTUM_UPDATER)
        return "x"

    m, data = make()
    o, s = adamw(m, steps)
    w = opal.wrap(
        o,
        m,
        scheduler=s,
        total_steps=steps,
        goal="t",
        decisions=1,
        every=10,
        run_dir=tmp_path / "r",
        backend=agent,
        blocking=True,
        verbose=False,
    )
    train(m, w, data, steps, sched=s)
    ev = [e for e in events(tmp_path / "r") if e["kind"] == "applied"]
    assert ev and ev[0]["applied_at"] == ev[0]["requested_at"]  # applied before the step of the same update


def test_resume_is_bitwise(tmp_path):
    prog = tmp_path / "prog"
    prog.mkdir()
    (prog / "updater.py").write_text(MOMENTUM_UPDATER)
    steps = 160
    m1, data = make()
    o1, _ = adamw(m1, steps)
    w1 = opal.replay(prog, o1, m1, total_steps=steps, run_dir=tmp_path / "a", verbose=False)
    train(m1, w1, data, steps)

    m2, _ = make()
    o2, _ = adamw(m2, steps)
    w2 = opal.replay(prog, o2, m2, total_steps=steps, run_dir=tmp_path / "b", verbose=False)
    train(m2, w2, data, 80)
    ck = dict(model=m2.state_dict(), opt=w2.state_dict())
    torch.save(ck, tmp_path / "ck.pt")
    ck = torch.load(tmp_path / "ck.pt", map_location="cpu", weights_only=False)
    m3, _ = make(seed=1)
    m3.load_state_dict(ck["model"])
    o3, _ = adamw(m3, steps)
    w3 = opal.replay(prog, o3, m3, total_steps=steps, run_dir=tmp_path / "c", verbose=False)
    w3.load_state_dict(ck["opt"])
    train(m3, w3, data, steps, start=80)
    assert torch.equal(flat(m1), flat(m3))


def test_nonfinite_loss_wakes_and_pauses(tmp_path):
    steps = 200
    msgs = []

    def agent(ws, msg, sys_):
        msgs.append(msg)
        return "x"

    m, data = make()
    o, s = adamw(m, steps)
    w = opal.wrap(
        o,
        m,
        scheduler=s,
        total_steps=steps,
        goal="t",
        decisions=1,
        every=10,
        run_dir=tmp_path / "r",
        backend=agent,
        verbose=False,
    )
    train(m, w, data, steps, sched=s, nan_at=55)
    assert any("non-finite" in x for x in msgs)


def test_system_prompt_and_reference(tmp_path):
    steps = 50
    seen = {}

    def agent(ws, msg, sys_):
        seen["system"] = sys_
        seen["files"] = sorted(p.name for p in ws.iterdir())
        return "x"

    m, data = make()
    o, s = adamw(m, steps)
    w = opal.wrap(
        o,
        m,
        scheduler=s,
        total_steps=steps,
        goal="GOAL-SENTENCE",
        decisions=1,
        every=5,
        run_dir=tmp_path / "r",
        backend=agent,
        blocking=True,
        verbose=False,
    )
    train(m, w, data, steps, sched=s)
    assert "GOAL-SENTENCE" in seen["system"] and "0.weight" in seen["system"]
    assert {"analysis.py", "updater.py", "inbox", "reference"} <= set(seen["files"])


# ----- DDP on CPU (gloo, two processes) ------------------------------------------------------------------------


def _ddp_worker(rank, world, port, run_dir, out):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(8, 32), torch.nn.ReLU(), torch.nn.Linear(32, 1))
    ddp = torch.nn.parallel.DistributedDataParallel(model)
    opt = torch.optim.AdamW(ddp.parameters(), lr=3e-3)

    def agent(ws, msg, sys_):
        time.sleep(0.2)
        (ws / "updater.py").write_text(MOMENTUM_UPDATER)
        return "x"

    w = opal.wrap(
        opt,
        ddp,
        total_steps=300,
        goal="t",
        decisions=2,
        every=10,
        sync_every=5,
        rank_check_every=50,
        run_dir=run_dir,
        backend=agent if rank == 0 else None,
        verbose=False,
    )
    g = torch.Generator().manual_seed(rank)
    for i in range(300):
        x = torch.randn(16, 8, generator=g)
        loss = ddp(x).pow(2).mean()
        w.zero_grad()
        loss.backward()
        w.step(loss=loss)
        time.sleep(0.003)
    torch.save(
        dict(params=torch.cat([p.detach().flatten() for p in model.parameters()]), version=w.version),
        f"{out}/rank{rank}.pt",
    )
    dist.destroy_process_group()


def test_ddp_two_ranks_swap_together(tmp_path):
    port = 29500 + os.getpid() % 1000
    mp.spawn(_ddp_worker, args=(2, port, str(tmp_path / "r"), str(tmp_path)), nprocs=2, join=True)
    a, b = (torch.load(tmp_path / f"rank{r}.pt") for r in (0, 1))
    assert a["version"] == b["version"] >= 1
    assert torch.equal(a["params"], b["params"])
    ev = [e for e in events(tmp_path / "r") if e["kind"] == "applied"]
    assert ev and ev[0]["applied_at"] % 5 == 0


DRIFT_UPDATER = MOMENTUM_UPDATER.replace(
    "                p.add_(self.buf[n], alpha=-lr)",
    "                p.add_(self.buf[n], alpha=-lr)\n"
    "        import torch.distributed as dist\n"
    "        if dist.is_initialized() and dist.get_rank() == 1 and update > 100 and update % 7 == 0:\n"
    "            self.named[-1][1].add_(1e-3)  # per-rank drift in one tensor only (the last layer's bias)",
)


def _ddp_drift_worker(rank, world, port, run_dir, out):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(8, 32), torch.nn.ReLU(), torch.nn.Linear(32, 1))
    ddp = torch.nn.parallel.DistributedDataParallel(model)
    opt = torch.optim.AdamW(ddp.parameters(), lr=3e-3)

    def agent(ws, msg, sys_):
        (ws / "updater.py").write_text(DRIFT_UPDATER)
        return "x"

    w = opal.wrap(
        opt, ddp, total_steps=150, goal="t", decisions=1, every=10, sync_every=5, rank_check_every=60,
        run_dir=run_dir, backend=agent if rank == 0 else None, verbose=False,
    )
    g = torch.Generator().manual_seed(rank)
    for i in range(150):
        loss = ddp(torch.randn(16, 8, generator=g)).pow(2).mean()
        w.zero_grad()
        loss.backward()
        w.step(loss=loss)
        time.sleep(0.002)
    torch.save(dict(params=torch.cat([p.detach().flatten() for p in model.parameters()])), f"{out}/rank{rank}.pt")
    dist.destroy_process_group()


def test_ddp_rank_check_sees_every_tensor_and_the_last_update(tmp_path):
    """Drift in one tensor is caught (every tensor is compared) and resynchronised, also at the last update (150 is
    not a multiple of rank_check_every=60), so both ranks end with the same weights."""
    port = 29500 + (os.getpid() + 7) % 1000
    mp.spawn(_ddp_drift_worker, args=(2, port, str(tmp_path / "r"), str(tmp_path)), nprocs=2, join=True)
    a, b = (torch.load(tmp_path / f"rank{r}.pt") for r in (0, 1))
    assert torch.equal(a["params"], b["params"])
    ev = [e for e in events(tmp_path / "r") if e["kind"] == "ranks_resynced"]
    assert ev and ev[-1]["update"] == 150
    assert all(e["tensors"] == 1 and e["names"] == ["2.bias"] for e in ev), ev


def test_mailbox_broker_and_report(tmp_path):
    import threading

    from opal.backends.base import CallableBackend
    from opal.backends.mailbox import run_broker
    from opal.report import write_report

    box = tmp_path / "box"

    def agent(ws, msg, sys_):
        assert (ws / "reference").exists()
        if "Decision 1." in msg:
            (ws / "updater.py").write_text(MOMENTUM_UPDATER)
        return "Observation: loss flat. Change: momentum SGD with linear decay. Fixed schedule."

    th = threading.Thread(
        target=run_broker,
        args=(str(box), CallableBackend(agent), tmp_path / "broker"),
        kwargs=dict(poll=0.05, log=lambda *_: None),
        daemon=True,
    )
    steps = 300
    m, data = make()
    o, s = adamw(m, steps)
    w = opal.wrap(
        o,
        m,
        scheduler=s,
        total_steps=steps,
        goal="t",
        decisions=2,
        every=10,
        run_dir=tmp_path / "r",
        backend=f"mailbox:{box}",
        backend_options=dict(poll=0.05),
        verbose=False,
    )
    th.start()
    for i in range(steps):
        x, y = data[i % len(data)]
        loss = torch.nn.functional.mse_loss(m(x), y)
        w.zero_grad()
        loss.backward()
        w.step(loss=loss)
        s.step()
        time.sleep(0.003)
    th.join(timeout=10)
    assert not th.is_alive()  # the broker stops when the run writes `done`
    assert w.version == 1
    out = write_report(tmp_path / "r")
    page = out.read_text()
    assert "momentum SGD" in page and 'class="a"' in page


def test_compiled_model_never_recompiles_on_swap(tmp_path):
    import torch._dynamo

    torch._dynamo.reset()
    steps = 120

    def agent(ws, msg, sys_):
        (ws / "updater.py").write_text(MOMENTUM_UPDATER if "Decision 1." in msg else SGD_UPDATER)
        return "x"

    m, data = make()
    cm = torch.compile(m, backend="eager")
    o, s = adamw(m, steps)
    w = opal.wrap(
        o,
        cm,
        scheduler=s,
        total_steps=steps,
        goal="t",
        decisions=3,
        every=10,
        run_dir=tmp_path / "r",
        backend=agent,
        blocking=True,
        verbose=False,
    )
    for i in range(steps):
        x, y = data[i % len(data)]
        loss = torch.nn.functional.mse_loss(cm(x), y)
        w.zero_grad()
        loss.backward()
        w.step(loss=loss)
        s.step()
        if i == 5:
            frames = torch._dynamo.utils.counters["stats"]["unique_graphs"]
    assert w.version >= 2  # two swaps happened
    assert torch._dynamo.utils.counters["stats"]["unique_graphs"] == frames  # and the model graph was never recompiled
    assert "0.weight" in w.worker.system  # names are cleaned of _orig_mod.


def test_gradscaler_forwards_loss_and_matches_plain(tmp_path):
    steps = 60
    out = []
    for wrapped in (False, True):
        m, data = make()
        o, s = adamw(m, steps)
        scaler = torch.amp.GradScaler("cpu", init_scale=1024.0)
        opt = (
            opal.wrap(
                o,
                m,
                scheduler=s,
                total_steps=steps,
                goal="t",
                backend=None,
                every=10,
                run_dir=tmp_path / "r",
                verbose=False,
            )
            if wrapped
            else o
        )
        for i in range(steps):
            x, y = data[i % len(data)]
            loss = torch.nn.functional.mse_loss(m(x), y)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt, loss=loss) if wrapped else scaler.step(opt)
            scaler.update()
            s.step()
        out.append(flat(m))
        if wrapped:
            assert len([json.loads(x) for x in open(tmp_path / "r" / "losses.jsonl")]) == steps // 10
    assert torch.equal(out[0], out[1])


def test_checkpoint_with_scheduler_pickles_to_disk_and_resumes_in_same_folder(tmp_path):
    steps = 200

    def agent(ws, msg, sys_):
        if "Decision 1." in msg:
            (ws / "updater.py").write_text(
                MOMENTUM_UPDATER.replace("class Updater:", "class Updater:\n    follow_user_scheduler = False")
            )
        return "x"

    os.chdir(tmp_path)  # default run folder goes under ./opal_runs
    m, data = make()
    o, s = adamw(m, steps)
    w = opal.wrap(
        o,
        m,
        scheduler=s,
        total_steps=steps,
        goal="t",
        decisions=2,
        every=10,
        backend=agent,
        blocking=True,
        verbose=False,
    )
    train(m, w, data, 100, sched=s)
    assert w.version == 1
    torch.save(dict(model=m.state_dict(), opt=w.state_dict(), sched=s.state_dict()), tmp_path / "ck.pt")
    first_dir = w.run.dir

    ck = torch.load(tmp_path / "ck.pt", weights_only=False)
    m2, _ = make(seed=3)
    m2.load_state_dict(ck["model"])
    o2, s2 = adamw(m2, steps)
    time.sleep(1.1)  # a different auto-named folder
    w2 = opal.wrap(
        o2,
        m2,
        scheduler=s2,
        total_steps=steps,
        goal="t",
        decisions=2,
        every=10,
        backend=agent,
        blocking=True,
        verbose=False,
    )
    w2.load_state_dict(ck["opt"])
    s2.load_state_dict(ck["sched"])
    assert w2.run.dir == first_dir and w2.version == 1
    assert len(list((tmp_path / "opal_runs").iterdir())) == 1
    train(m2, w2, data, steps, sched=s2, start=100)
    kinds = [e["kind"] for e in events(first_dir)]
    assert "resumed" in kinds and kinds.count("start") == 1


SLOW_UPDATER = SGD_UPDATER.replace(
    "        RAN.append(update)\n",
    "        RAN.append(update)\n        for n, p in self.named:\n            for _ in range(20):\n                p.abs().max()\n",
)


def test_cost_check_profiles_each_new_updater(tmp_path):
    steps = 200
    answers, msgs = [SGD_UPDATER, SLOW_UPDATER], []

    def agent(ws, msg, sys_):
        msgs.append(msg)
        if answers:
            (ws / "updater.py").write_text(answers.pop(0))
        return "x"

    m, data = make()
    o, s = adamw(m, steps)
    w = opal.wrap(
        o,
        m,
        scheduler=s,
        total_steps=steps,
        goal="t",
        decisions=3,
        every=10,
        run_dir=tmp_path / "r",
        backend=agent,
        blocking=True,
        verbose=False,
    )
    train(m, w, data, steps, sched=s)
    audits = {e["version"]: e for e in events(tmp_path / "r") if e["kind"] == "audit"}
    assert {0, 1, 2} <= set(audits)
    assert audits[2]["operator_calls"] > audits[1]["operator_calls"] + 50  # 4 tensors x 20 x 2 extra calls
    assert "Profile of one update" in msgs[-1] and "operator calls per update" in msgs[-1]


def test_default_is_blocking(tmp_path):
    m, data = make()
    o, s = adamw(m, 50)
    w = opal.wrap(o, m, scheduler=s, total_steps=50, goal="t", run_dir=tmp_path / "r", backend="keep", verbose=False)
    assert w.blocking is True and w._auto is None
    assert "Training pauses until you finish, and your edits take effect from that update." in w.worker.system
    assert events(tmp_path / "r")[0]["blocking"] is True
