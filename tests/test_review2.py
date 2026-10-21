"""Regression tests, second set: the analysis interval, rejected builds and reverts, state types, the
user's schedulers, a NaN while a decision is in flight, resumes, the mailbox and DDP."""

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import opal
from test_core import MOMENTUM_UPDATER, SGD_UPDATER, adamw, events, flat, make, train
from test_review1 import scripted, wrapped

COUNT_ANALYSIS = """
EVERY = {every}
CALLS = []
def analyze(obs, memory):
    CALLS.append((obs.update, len(obs.losses)))
"""


def plain_run(steps):
    m, data = make()
    o, s = adamw(m, steps)
    train(m, o, data, steps, sched=s, pass_loss=False)
    return flat(m)


def test_n02_analysis_runs_every_EVERY_and_sees_all_losses(tmp_path):
    agent = scripted({"analysis.py": COUNT_ANALYSIS.format(every=70)})
    m, data, s, w = wrapped(tmp_path, agent, steps=400, decisions=1)
    train(m, w, data, 400, sched=s)
    calls = sys.modules[w.analysis.__name__].CALLS
    updates = [u for u, _ in calls]
    assert all(b - a == 70 for a, b in zip(updates, updates[1:-1])), updates
    assert all(n == 70 for _, n in calls[1:-1]), calls  # the first call follows the swap mid-interval


def test_n04_rejected_build_leaves_no_trace_on_the_users_optimizer(tmp_path):
    sneaky = MOMENTUM_UPDATER.replace(
        "self.buf = {",
        "spec['user_optimizers'][0].param_groups[0]['lr'] *= 10\n        self.bad = lambda: 0\n        self.buf = {",
    ).replace("return dict(buf=self.buf)", "return dict(buf=self.buf, bad=self.bad)")
    steps = 120
    m, data, s, w = wrapped(tmp_path, scripted({"updater.py": sneaky}), steps=steps, decisions=1)
    train(m, w, data, steps, sched=s)
    assert "load_failed" in [e["kind"] for e in events(tmp_path / "r")]
    assert torch.equal(flat(m), plain_run(steps))


def test_n04_first_step_failure_on_shared_optimizer_is_undone_exactly(tmp_path):
    shared = """
class Updater:
    def __init__(self, named_params, previous=None, spec=None):
        self.opt = spec["user_optimizers"][0]
    def step(self, grads, update):
        self.opt.step()
        raise ValueError("after stepping the shared AdamW")
    def state_dict(self):
        return {}
    def load_state_dict(self, s):
        pass
def build(named_params, previous=None, spec=None):
    return Updater(named_params, previous, spec)
"""
    steps = 120
    m, data, s, w = wrapped(tmp_path, scripted({"updater.py": shared}), steps=steps, decisions=1)
    train(m, w, data, steps, sched=s)
    rev = [e for e in events(tmp_path / "r") if e["kind"] == "reverted"]
    assert rev and rev[0]["restored"]
    assert torch.equal(flat(m), plain_run(steps))


def test_n08_revert_keeps_the_new_analysis(tmp_path):
    marker = "EVERY = 10\nMARKER = 1\ndef analyze(obs, memory):\n    return None\n"
    late = SGD_UPDATER.replace(
        "RAN.append(update)", "RAN.append(update)\n        if update > 75: raise ValueError('x')"
    )
    agent = scripted({"updater.py": late}, {"analysis.py": marker})
    m, data, s, w = wrapped(tmp_path, agent, steps=300, decisions=4, trial_steps=40, every=10)
    train(m, w, data, 300, sched=s)
    assert "MARKER" in w.files["analysis.py"] and getattr(w.analysis, "MARKER", 0) == 1
    assert "raise ValueError" not in w.files["updater.py"]


def test_n11_nonplain_state_types_are_rejected_and_checkpoints_never_raise(tmp_path):
    for kind, extra in [
        ("namedtuple", "collections.namedtuple('P', 'a b')(1, 2)"),
        ("defaultdict", "collections.defaultdict(lambda: 0)"),
        ("subclass", "type('Bag', (dict,), {})()"),
    ]:
        upd = MOMENTUM_UPDATER.replace("import torch", "import collections\nimport torch").replace(
            "return dict(buf=self.buf)", f"return dict(buf=self.buf, extra={extra})"
        )
        d = tmp_path / kind
        m, data, s, w = wrapped(d, scripted({"updater.py": upd}), steps=100, decisions=1)
        train(m, w, data, 100, sched=s)
        assert w.version == 0, kind
    # state that turns non-plain only later: the checkpoint leaves it out instead of raising
    later = (
        MOMENTUM_UPDATER.replace("import torch", "import collections\nimport torch")
        .replace("return dict(buf=self.buf)", "return dict(buf=self.buf, q=self.q if hasattr(self, 'q') else [])")
        .replace("lr = 0.02", "self.q = [lambda: update]\n        lr = 0.02")
    )
    m, data, s, w = wrapped(tmp_path / "later", scripted({"updater.py": later}), steps=100, decisions=1)
    train(m, w, data, 100, sched=s)
    assert w.version == 1
    sd = w.state_dict()
    assert sd["updater"] is None
    torch.save(sd, tmp_path / "ck.pt")


def test_n14_overlapping_groups_rejected(tmp_path):
    dup = MOMENTUM_UPDATER.replace(
        "def step(self, grads, update):",
        """@property
    def param_groups(self):
        ps = [p for _, p in self.named]
        return [dict(params=ps, lr=0.1), dict(params=ps[:1], lr=0.1)]

    def step(self, grads, update):""",
    )
    m, data, s, w = wrapped(tmp_path, scripted({"updater.py": dup}), steps=100, decisions=1)
    train(m, w, data, 100, sched=s)
    assert w.version == 0 and "load_failed" in [e["kind"] for e in events(tmp_path / "r")]


def test_n19_program_can_step_the_users_scheduler(tmp_path):
    own = (
        MOMENTUM_UPDATER.replace("class Updater:", "class Updater:\n    follow_user_scheduler = False")
        .replace("self.buf = {", "self.sched = spec['user_schedulers'][0]\n        self.buf = {")
        .replace("lr = 0.02 * (1 - update / 1000)", "self.sched.step()\n        lr = 0.02")
    )
    m, data, s, w = wrapped(tmp_path, scripted({"updater.py": own}), steps=100, decisions=1)
    train(m, w, data, 100, sched=s)
    assert w.version == 1 and s.last_epoch > 60  # the program's own calls run; the loop's calls did nothing


def test_n28_helper_named_like_stdlib_rejected(tmp_path):
    upd = MOMENTUM_UPDATER.replace("import torch", "import torch\nimport copy")
    m, data, s, w = wrapped(tmp_path, scripted({"updater.py": upd, "copy.py": "X = 1\n"}), steps=100, decisions=1)
    train(m, w, data, 100, sched=s)
    assert w.version == 0 and "rejected" in [e["kind"] for e in events(tmp_path / "r")]


def test_n07_nan_while_a_decision_is_in_flight_pauses_for_it(tmp_path):
    def agent(ws, msg, system):
        time.sleep(0.5 if "Decision 1." in msg else 0.0)
        return "x"

    m, data, s, w = wrapped(tmp_path, agent, steps=300, decisions=1, blocking=False)
    for i in range(300):
        x, y = data[i % len(data)]
        loss = torch.nn.functional.mse_loss(m(x), y)
        if i == 25:
            loss = loss * float("nan")
        w.zero_grad()
        loss.backward()
        w.step(loss=loss)
        s.step()
    req = [e for e in events(tmp_path / "r") if e["kind"] == "requested"]
    nan = [e for e in req if any("non-finite" in r for r in e["reasons"])]
    assert nan
    out = [e for e in events(tmp_path / "r") if e["kind"] in ("kept", "applied") and e.get("k") == nan[0]["k"]]
    assert out and out[0]["applied_at"] == nan[0]["update"]  # answered before training went on


def test_n18_resume_archives_the_lost_attempt(tmp_path):
    agent = scripted(
        {"updater.py": MOMENTUM_UPDATER}, None, {"analysis.py": "EVERY = 10\ndef analyze(o, m):\n    pass\n"}
    )
    m, data, s, w = wrapped(tmp_path, agent, steps=300, decisions=6)
    train(m, w, data, 60, sched=s)
    ck = w.state_dict()
    train(m, w, data, 200, sched=s, start=60)  # more decisions after the checkpoint, then a "crash"
    k_after = w.worker.k
    m2, _ = make()
    o2, s2 = adamw(m2, 300)
    w2 = opal.wrap(
        o2,
        m2,
        scheduler=s2,
        total_steps=300,
        goal="t",
        decisions=6,
        every=10,
        blocking=True,
        run_dir=tmp_path / "r",
        verbose=False,
        backend=scripted(),
    )
    w2.load_state_dict(ck)
    assert (tmp_path / "r" / "attempts").exists() and k_after > ck["decisions"]
    train(m2, w2, data, 300, sched=s2, start=60)
    msgs = tmp_path / "r" / "decisions"
    first_after = sorted(d for d in msgs.iterdir() if int(d.name) > ck["decisions"])[0]
    assert "resumed from a checkpoint" in (first_after / "message.txt").read_text()


def test_n10_crash_leaves_the_mailbox_waiting(tmp_path):
    box = tmp_path / "box"
    m, data = make()
    o, s = adamw(m, 200)
    w = opal.wrap(
        o,
        m,
        scheduler=s,
        total_steps=200,
        goal="t",
        decisions=2,
        every=10,
        run_dir=tmp_path / "r",
        backend=f"mailbox:{box}",
        backend_options=dict(poll=0.05, broker_wait=1, unavailable_timeout=0),
        verbose=False,
    )
    train(m, w, data, 30, sched=s)
    w.close(interrupted=True)  # what atexit does after a crash
    run = next(d for d in box.iterdir() if d.is_dir())
    assert (run / "interrupted").exists() and not (run / "done").exists()


def test_n21_long_agent_turn_through_the_mailbox_is_applied(tmp_path):
    from opal.backends.base import CallableBackend
    from opal.backends.mailbox import run_broker

    box = tmp_path / "box"

    def slow(ws, msg, system):
        time.sleep(4)
        (ws / "updater.py").write_text(MOMENTUM_UPDATER)
        return "slow answer"

    m, data = make()
    o, s = adamw(m, 100)
    w = opal.wrap(
        o,
        m,
        scheduler=s,
        total_steps=100,
        goal="t",
        decisions=1,
        every=10,
        blocking=True,
        run_dir=tmp_path / "r",
        backend=f"mailbox:{box}",
        backend_options=dict(poll=0.05, broker_wait=2),
        verbose=False,
    )
    th = threading.Thread(
        target=run_broker,
        args=(str(box), CallableBackend(slow), tmp_path / "b"),
        kwargs=dict(poll=0.05, log=lambda *_: None),
        daemon=True,
    )
    th.start()
    train(m, w, data, 100, sched=s)
    assert w.version == 1


def test_n21_no_broker_gives_up_quickly(tmp_path):
    m, data = make()
    o, s = adamw(m, 100)
    w = opal.wrap(
        o,
        m,
        scheduler=s,
        total_steps=100,
        goal="t",
        decisions=1,
        every=10,
        blocking=True,
        run_dir=tmp_path / "r",
        backend=f"mailbox:{tmp_path / 'box'}",
        backend_options=dict(poll=0.05, broker_wait=1, unavailable_timeout=0),
        verbose=False,
    )
    t0 = time.time()
    train(m, w, data, 100, sched=s)
    assert time.time() - t0 < 30
    resp = json.loads((tmp_path / "r" / "decisions" / "001" / "response.json").read_text())
    assert "no broker" in (resp["backend_error"] or "")


# ----- an answer that comes after the run stopped waiting for it ---------------


def _mailbox_run(tmp_path, steps=300):
    m, data = make()
    o, s = adamw(m, steps)
    w = opal.wrap(
        o,
        m,
        scheduler=s,
        total_steps=steps,
        goal="t",
        decisions=4,
        every=10,
        run_dir=tmp_path / "r",
        backend=f"mailbox:{tmp_path / 'box'}",
        backend_options=dict(poll=0.02, broker_wait=0.2, unavailable_timeout=0),
        blocking=False,  # late answers arrive while training continues
        verbose=False,
    )
    return m, data, s, w


def _until_given_up(m, w, data, s, k, start):
    """Train from update `start` until decision k is out and the run has stopped waiting for it (no broker)."""
    i = start
    while w.worker.k < k:
        train(m, w, data, i + 1, sched=s, start=i)
        i += 1
    w.worker.wait()
    assert w.worker.ready["gave_up"]
    return i


def test_f9_late_mailbox_answer_is_applied(tmp_path):
    """The broker came back after the run had stopped waiting for decision 13 and answered it; nobody read the answer
    and the next request reset the workspace, so the edit was lost without a word. An answer that comes before the next
    request is now applied like any other, and the next message says when."""
    from opal.backends.base import CallableBackend
    from opal.backends.mailbox import run_broker

    m, data, s, w = _mailbox_run(tmp_path)
    i = _until_given_up(m, w, data, s, 1, 0)
    be = w.worker.backend
    agent = scripted({"updater.py": SGD_UPDATER})
    threading.Thread(
        target=run_broker,
        args=(str(tmp_path / "box"), CallableBackend(agent), tmp_path / "b"),
        kwargs=dict(poll=0.02, run=be.run_id, log=lambda *_: None),
        daemon=True,
    ).start()
    t0 = time.time()
    while not (be.folder / "response-0001.json").exists() and time.time() - t0 < 30:
        time.sleep(0.05)
    train(m, w, data, 300, sched=s, start=i)
    applied = [e for e in events(tmp_path / "r") if e["kind"] == "applied"]
    assert w.version == 1 and [(e["k"], e.get("late")) for e in applied] == [(1, True)]
    msg = (tmp_path / "r" / "decisions" / "002" / "message.txt").read_text()
    u = applied[0]["applied_at"]
    assert f"Your edit from decision 1 (requested at update 20) took effect at update {u}, {u - 20} updates" in msg
    assert "it arrived after the run had stopped waiting for it, and was applied when it came" in msg


def test_f9_mailbox_answer_after_the_next_request_is_discarded_and_the_agent_is_told(tmp_path):
    m, data, s, w = _mailbox_run(tmp_path)
    i = _until_given_up(m, w, data, s, 1, 0)
    i = _until_given_up(m, w, data, s, 2, i)  # decision 2 went out before any answer to decision 1
    be = w.worker.backend
    files = dict(w.files, **{"updater.py": SGD_UPDATER})
    late = dict(run_id=be.run_id, attempt=be.attempt, k=1, files=files, text="late", returncode=0)
    (be.folder / "response-0001.json").write_text(json.dumps(late))  # what the broker that came back writes
    train(m, w, data, 300, sched=s, start=i)
    assert w.version == 0 and not [e for e in events(tmp_path / "r") if e["kind"] == "applied"]
    msg = (tmp_path / "r" / "decisions" / "002" / "message.txt").read_text()
    said = "Decision 1 (requested at update 20) was not answered in time: the run stopped waiting for it after 0 s"
    assert said + " (no broker is answering). If you answered it, your answer arrived after the run had stopped" in msg
    assert "stopped waiting and was discarded, and this folder was reset to the running programs." in msg


def test_f9_every_message_says_what_became_of_the_previous_answer(tmp_path):
    agent = scripted({"updater.py": "def build(:\n"}, None, {"updater.py": SGD_UPDATER}, None)
    m, data, s, w = wrapped(tmp_path, agent)  # blocking; decisions at updates 20, 90, 160, 230
    train(m, w, data, 300, sched=s)
    first, rejected, kept, applied = agent.calls
    assert "Decision 1: your edit was not applied and was discarded at update 20 (the errors below say why)" in rejected
    assert "Decision 2 (requested at update 90, answered by update 90): you kept the programs." in kept
    assert "Your edit from decision 3 (requested at update 160) took effect at update 160, 0 updates after" in applied
    assert "Why now: scheduled decision.\nThe next scheduled decision is at update 90.\nTraining loss" in first


def test_n13_agent_is_stopped_when_the_training_process_dies(tmp_path):
    code = f"""
import os, sys, time
sys.path.insert(0, {str(Path(__file__).parent)!r})
import torch, opal
from opal.backends.base import Backend
from test_core import make, adamw
class Slow(Backend):
    name = "slow"
    def decide(self, ws, msg, system, rec):
        p = self._run(["sh", "-c", "sleep 20; touch {tmp_path / "leak"}"], "", None, ws, 60)
        return dict(text="", returncode=p.returncode, error=None, seconds=0)
m, data = make()
o, s = adamw(m, 100)
w = opal.wrap(o, m, scheduler=s, total_steps=100, goal="t", decisions=1, every=10, run_dir={str(tmp_path / "r")!r},
                backend=Slow(), blocking=False, verbose=False)
for i in range(20):
    x, y = data[i]
    loss = torch.nn.functional.mse_loss(m(x), y); w.zero_grad(); loss.backward(); w.step(loss=loss)
time.sleep(1.0)
os._exit(1)   # dies without close(), like SIGKILL
"""
    subprocess.run([sys.executable, "-c", code], timeout=60)
    time.sleep(10)
    assert not (tmp_path / "leak").exists()


# ----- DDP: a program that raises after its trial window; the resumed run falls back -------------------------------


def _crash_worker(rank, world, port, run_dir, ck, phase, out):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(8, 32), torch.nn.ReLU(), torch.nn.Linear(32, 1))
    ddp = torch.nn.parallel.DistributedDataParallel(model)
    opt = torch.optim.AdamW(ddp.parameters(), lr=3e-3)
    opt.param_groups[0]["rank_tag"] = rank  # user settings that differ by rank must not split the ranks' decisions
    late = SGD_UPDATER.replace(
        "RAN.append(update)", "RAN.append(update)\n        if update > 90: raise ValueError('late')"
    )
    agent = scripted({"updater.py": late}) if phase == 1 else scripted()
    w = opal.wrap(
        opt,
        ddp,
        total_steps=200,
        goal="t",
        decisions=2,
        every=10,
        sync_every=5,
        blocking=True,
        trial_steps=5,
        run_dir=run_dir,
        backend=agent if rank == 0 else None,
        verbose=False,
    )
    start = 0
    if phase >= 2:
        state = torch.load(ck, weights_only=False)
        model.load_state_dict(state["model"])
        w.load_state_dict(state["opt"])
        start = state["update"]
    g = torch.Generator().manual_seed(rank)
    result = "finished"
    try:
        for u in range(start, 200):
            loss = ddp(torch.randn(16, 8, generator=g)).pow(2).mean()
            w.zero_grad()
            loss.backward()
            w.step(loss=loss)
            if phase == 1 and u + 1 == 80 and rank == 0:
                torch.save(dict(model=model.state_dict(), opt=w.state_dict(), update=80), ck)
    except RuntimeError as exc:
        result = "crashed: " + str(exc)[:80]
    torch.save(
        dict(
            result=result,
            params=torch.cat([p.detach().flatten() for p in model.parameters()]),
            updater=w.files["updater.py"],
        ),
        f"{out}/p{phase}-rank{rank}.pt",
    )
    dist.destroy_process_group()


def test_n06_ddp_crash_after_trial_then_resume_falls_back(tmp_path):
    port = 29700 + os.getpid() % 1000
    ck = str(tmp_path / "ck.pt")
    mp.spawn(_crash_worker, args=(2, port, str(tmp_path / "r"), ck, 1, str(tmp_path)), nprocs=2, join=True)
    a = torch.load(tmp_path / "p1-rank0.pt")
    assert a["result"].startswith("crashed")
    assert list((tmp_path / "r").glob("crashed-*-r*.json"))
    mp.spawn(_crash_worker, args=(2, port + 1, str(tmp_path / "r"), ck, 2, str(tmp_path)), nprocs=2, join=True)
    b0, b1 = (torch.load(tmp_path / f"p2-rank{r}.pt") for r in (0, 1))
    assert b0["result"] == b1["result"] == "finished"
    assert "late" not in b0["updater"] and torch.equal(b0["params"], b1["params"])
    # a second restart from the same checkpoint must fall back again (markers are never deleted)
    mp.spawn(_crash_worker, args=(2, port + 2, str(tmp_path / "r"), ck, 3, str(tmp_path)), nprocs=2, join=True)
    c0 = torch.load(tmp_path / "p3-rank0.pt")
    assert c0["result"] == "finished" and "late" not in c0["updater"]


# ----- DDP: the loss passed on rank 0 only must not break collectives ----------------------------------------------


def _rank0_loss_worker(rank, world, port, out):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(8, 32), torch.nn.ReLU(), torch.nn.Linear(32, 1))
    ddp = torch.nn.parallel.DistributedDataParallel(model)
    opt = torch.optim.AdamW(ddp.parameters(), lr=3e-3)
    w = opal.wrap(
        opt,
        ddp,
        total_steps=120,
        goal="t",
        decisions=2,
        every=10,
        backend="keep" if rank == 0 else None,
        run_dir=out + "/r",
        verbose=False,
    )
    g = torch.Generator().manual_seed(rank)
    for _ in range(120):
        loss = ddp(torch.randn(16, 8, generator=g)).pow(2).mean()
        w.zero_grad()
        loss.backward()
        if rank == 0:
            w.observe(loss)
        w.step()
    torch.save(torch.cat([p.detach().flatten() for p in model.parameters()]), f"{out}/rank{rank}.pt")
    dist.destroy_process_group()


def test_n17_ddp_loss_on_rank0_only(tmp_path):
    port = 29800 + os.getpid() % 1000
    mp.spawn(_rank0_loss_worker, args=(2, port, str(tmp_path)), nprocs=2, join=True)
    assert torch.equal(torch.load(tmp_path / "rank0.pt"), torch.load(tmp_path / "rank1.pt"))


def test_r01_user_schedulers_with_rich_state_can_be_wrapped(tmp_path):
    m, data = make()
    o = torch.optim.AdamW(m.parameters(), lr=3e-3)
    s = torch.optim.lr_scheduler.MultiStepLR(o, milestones=[30, 60], gamma=0.5)  # milestones is a Counter
    w = opal.wrap(
        o, m, scheduler=s, total_steps=100, goal="t", backend="keep", every=10, run_dir=tmp_path / "r", verbose=False
    )
    train(m, w, data, 100, sched=s)
    torch.save(dict(opt=w.state_dict(), sched=s.state_dict()), tmp_path / "ck.pt")


def test_claude_cost_is_per_decision_although_the_cli_reports_the_session_total(tmp_path):
    import types

    from opal.backends.claude import ClaudeBackend

    class Fake(ClaudeBackend):
        total = 0.0

        def _run(self, cmd, input, stdout, cwd, timeout, env=None):
            Fake.total += 0.5  # the CLI's total_cost_usd grows over a --resume session
            stdout.write(json.dumps(dict(type="system", subtype="init")) + "\n")
            stdout.write(json.dumps(dict(type="result", result="keep", total_cost_usd=Fake.total)) + "\n")
            stdout.flush()
            return types.SimpleNamespace(returncode=0, stderr="")

    be = Fake(sandbox=False)
    be.setup(tmp_path / "run", tmp_path / "ws")
    (tmp_path / "ws").mkdir(exist_ok=True)
    costs = []
    for k in range(3):
        rec = tmp_path / f"d{k}"
        rec.mkdir()
        costs.append(be.decide(tmp_path / "ws", "m", "s", rec)["cost_usd"])
    assert costs == [0.5, 0.5, 0.5]
    again = Fake(sandbox=False)
    again.restore(be.state())  # a resumed run or a restarted broker keeps counting from the session total
    again.setup(tmp_path / "run", tmp_path / "ws")
    rec = tmp_path / "d3"
    rec.mkdir()
    assert again.decide(tmp_path / "ws", "m", "s", rec)["cost_usd"] == 0.5
