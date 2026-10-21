"""Regression tests, first set: analysis-only edits, program files, reverts, NaN handling, close, the mailbox,
resumes and DDP."""

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

ANALYSIS_EVERY10 = """
EVERY = 10
def analyze(obs, memory):
    return {"report": {"u": obs.update}}
"""


def scripted(*answers):
    """A fake agent that writes the given files at its 1st, 2nd, ... decision (None = keep)."""
    calls = []

    def agent(ws, msg, system):
        i = len(calls)
        calls.append(msg)
        if i < len(answers) and answers[i]:
            for name, text in answers[i].items():
                (ws / name).write_text(text)
        return f"answer {i + 1}"

    agent.calls = calls
    return agent


def wrapped(tmp_path, agent, steps=300, **kw):
    m, data = make()
    o, s = adamw(m, steps)
    kw = dict(dict(decisions=4, every=10, blocking=True, verbose=False), **kw)
    w = opal.wrap(o, m, scheduler=s, total_steps=steps, goal="t", run_dir=tmp_path / "r", backend=agent, **kw)
    return m, data, s, w


def test_c1_analysis_only_edit_keeps_updater_state(tmp_path):
    agent = scripted({"updater.py": MOMENTUM_UPDATER}, {"analysis.py": ANALYSIS_EVERY10})
    m, data, s, w = wrapped(tmp_path, agent)
    train(m, w, data, 120, sched=s)
    assert w.version == 2
    applied = [e for e in events(tmp_path / "r") if e["kind"] == "applied"]
    assert applied[1]["changed"] == ["analysis.py"] and applied[1]["rebuilt_updater"] is False
    assert sum(float(b.norm()) for b in w.updater.buf.values()) > 0


def test_m5_scratch_files_are_not_program(tmp_path):
    agent = scripted({"check_grads.py": "print(1)\n", "try-it.py": "x = 1\n"})
    m, data, s, w = wrapped(tmp_path, agent)
    train(m, w, data, 100, sched=s)
    assert w.version == 0
    assert "check_grads.py" not in w.files


def test_m8_helpers_are_private_to_the_program(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "utils.py").write_text("WHO = 'user'\n")
    sys.path.insert(0, str(tmp_path / "pkg"))
    try:
        upd = MOMENTUM_UPDATER.replace("import torch", "import torch\nimport utils\nassert utils.WHO == 'agent'")
        agent = scripted({"updater.py": upd, "utils.py": "WHO = 'agent'\n"})
        m, data, s, w = wrapped(tmp_path, agent)
        train(m, w, data, 100, sched=s)
        assert w.version == 1
        import utils

        assert utils.WHO == "user"
        assert not any(str(tmp_path / "r") in p for p in sys.path)
    finally:
        sys.path.remove(str(tmp_path / "pkg"))
        sys.modules.pop("utils", None)


def test_h1_param_groups_always_have_lr_and_cover_everything(tmp_path):
    agent = scripted({"updater.py": MOMENTUM_UPDATER})
    m, data, s, w = wrapped(tmp_path, agent)
    train(m, w, data, 100, sched=s)
    assert w.version == 1
    assert all("lr" in g for g in w.param_groups)
    assert sum(len(g["params"]) for g in w.param_groups) == len(list(m.parameters()))


def test_h4_agent_class_in_state_is_rejected_and_memory_failure_does_not_crash(tmp_path):
    bad = MOMENTUM_UPDATER.replace("return dict(buf=self.buf)", "return dict(buf=self.buf, extra=Stats())").replace(
        "class Updater:", "class Stats:\n    pass\n\n\nclass Updater:"
    )
    lam = 'EVERY = 10\ndef analyze(obs, memory):\n    memory["f"] = lambda: 1\n    return None\n'
    agent = scripted({"updater.py": bad}, {"analysis.py": lam})
    m, data, s, w = wrapped(tmp_path, agent)
    train(m, w, data, 100, sched=s)
    kinds = [e["kind"] for e in events(tmp_path / "r")]
    assert "load_failed" in kinds and "Stats" in json.dumps(
        [e for e in events(tmp_path / "r") if e["kind"] == "load_failed"]
    )
    torch.save(w.state_dict(), tmp_path / "ck.pt")  # memory with a lambda is left out, not fatal
    code = f"import torch; sd = torch.load(r'{tmp_path / 'ck.pt'}', weights_only=False); print(sorted(sd)[:3])"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr


def test_h6_close_at_early_stop_and_gradscaler_skips(tmp_path):
    m, data, s, w = wrapped(tmp_path, scripted(), steps=300)
    train(m, w, data, 120, sched=s)  # stops early
    w.close()
    assert events(tmp_path / "r")[-1]["kind"] == "end"


def test_h6_agent_child_killed_at_close(tmp_path):
    from opal.backends.base import Backend

    class Slow(Backend):
        name = "slow"

        def decide(self, workspace, message, system, record_dir):
            p = self._run(["sh", "-c", f"sleep 30; touch {tmp_path / 'leak'}"], "", subprocess.DEVNULL, workspace, 60)
            return dict(text="", returncode=p.returncode, error=None, seconds=0.0)

    m, data, s, w = wrapped(tmp_path, Slow(), steps=120, blocking=False)
    train(m, w, data, 120, sched=s)
    time.sleep(1.0)
    assert not (tmp_path / "leak").exists()
    assert "cancelled" in [e["kind"] for e in events(tmp_path / "r")]


def test_h7_raising_every_keeps_decisions_and_nan_checks(tmp_path):
    lazy = "EVERY = 1000\ndef analyze(obs, memory):\n    return None\n"
    agent = scripted({"analysis.py": lazy})
    m, data, s, w = wrapped(tmp_path, agent, steps=600, decisions=6, blocking=True)
    train(m, w, data, 600, sched=s, nan_at=401)
    req = [e for e in events(tmp_path / "r") if e["kind"] == "requested"]
    assert len(req) >= 5
    nan_req = [e for e in req if any("non-finite" in r for r in e["reasons"])]
    assert nan_req and nan_req[0]["update"] <= 450


def test_m6_persistent_nan_wakes_once(tmp_path):
    m, data, s, w = wrapped(tmp_path, scripted(), steps=400, decisions=2)
    train(m, w, data, 400, sched=s, nan_at=100)
    req = [e for e in events(tmp_path / "r") if e["kind"] == "requested"]
    assert sum(any("non-finite" in r for r in e["reasons"]) for e in req) == 1


def test_m1_first_step_failure_is_undone_exactly(tmp_path):
    half = SGD_UPDATER.replace(
        "RAN.append(update)",
        "RAN.append(update)\n        list(self.named)[0][1].add_(1.0)\n        raise ValueError('half done')",
    )
    steps = 100
    m1, data = make()
    o1, s1 = adamw(m1, steps)
    train(m1, o1, data, steps, sched=s1, pass_loss=False)
    m2, data, s2, w = wrapped(tmp_path, scripted({"updater.py": half}), steps=steps, decisions=1)
    train(m2, w, data, steps, sched=s2)
    rev = [e for e in events(tmp_path / "r") if e["kind"] == "reverted"]
    assert rev and rev[0]["restored"]
    assert torch.equal(flat(m1), flat(m2))  # as if the failing program had never run


def test_m2_revert_never_crashes_on_carry_over(tmp_path):
    v1 = MOMENTUM_UPDATER.replace("self.buf = {", "self.opts = spec['user_optimizers']\n        self.buf = {")
    v2 = SGD_UPDATER.replace(
        "RAN.append(update)", "RAN.append(update)\n        if update > 160: raise ValueError('late')"
    )
    agent = scripted({"updater.py": v1}, {"updater.py": v2})
    m, data, s, w = wrapped(tmp_path, agent, steps=300, decisions=3, trial_steps=5)
    train(m, w, data, 300, sched=s)
    assert torch.isfinite(flat(m)).all()
    assert "reverted" in [e["kind"] for e in events(tmp_path / "r")]


def test_m3_answer_on_reverted_files_does_not_reinstall_them(tmp_path):
    boom = SGD_UPDATER.replace("RAN.append(update)", "RAN.append(update)\n        raise ValueError('boom')")
    state = {}

    def agent(ws, msg, system):
        n = state.setdefault("n", 0) + 1
        state["n"] = n
        if n == 1:
            (ws / "updater.py").write_text(boom)
        if n == 2:
            time.sleep(0.3)  # the boom program raises and is reverted while this decision runs
            (ws / "analysis.py").write_text(ANALYSIS_EVERY10)
        return "x"

    m, data, s, w = wrapped(tmp_path, agent, steps=400, decisions=4, blocking=False)
    for i in range(400):
        x, y = data[i % len(data)]
        loss = torch.nn.functional.mse_loss(m(x), y)
        w.zero_grad()
        loss.backward()
        w.step(loss=loss)
        s.step()
        time.sleep(0.002)
    assert "boom" not in w.files["updater.py"]
    assert sum(e["kind"] == "reverted" for e in events(tmp_path / "r")) == 1


def test_m4_edits_from_a_failed_call_are_not_applied(tmp_path):
    from opal.backends.base import Backend

    class Crashy(Backend):
        name = "crashy"

        def decide(self, workspace, message, system, record_dir):
            (workspace / "updater.py").write_text(MOMENTUM_UPDATER)
            return dict(text="", returncode=1, error="killed", seconds=0.0)

    m, data, s, w = wrapped(tmp_path, Crashy())
    train(m, w, data, 100, sched=s)
    assert w.version == 0
    assert "rejected" in [e["kind"] for e in events(tmp_path / "r")]


def test_m11_runs_in_the_same_second_get_different_folders(tmp_path):
    os.chdir(tmp_path)
    m, data = make()
    a = opal.wrap(adamw(m, 10)[0], m, total_steps=10, goal="t", backend="keep", verbose=False)
    b = opal.wrap(adamw(m, 10)[0], m, total_steps=10, goal="t", backend="keep", verbose=False)
    assert a.run.dir != b.run.dir


def test_m12_vector_log_is_ignored_not_fatal(tmp_path):
    m, data, s, w = wrapped(tmp_path, scripted())
    w.log(per_class=torch.ones(10), acc=torch.tensor(0.5), note="x")
    train(m, w, data, 60, sched=s)
    lines = [json.loads(x) for x in open(tmp_path / "r" / "metrics.jsonl")]
    assert lines[0]["acc"] == 0.5 and "per_class" not in lines[0]


def test_h2_mailbox_reuse_and_resume_never_take_stale_answers(tmp_path):
    from opal.backends.base import CallableBackend
    from opal.backends.mailbox import run_broker

    box = tmp_path / "box"

    def serve(agent, workdir):
        th = threading.Thread(
            target=run_broker,
            args=(str(box), CallableBackend(agent), workdir),
            kwargs=dict(poll=0.05, log=lambda *_: None),
            daemon=True,
        )
        th.start()
        return th

    # run 1: its agent switches to SGD
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
        blocking=True,
        run_dir=tmp_path / "r1",
        backend=f"mailbox:{box}",
        backend_options=dict(poll=0.05),
        verbose=False,
    )
    th = serve(scripted({"updater.py": SGD_UPDATER}), tmp_path / "b1")
    train(m, w, data, 200, sched=s)
    th.join(timeout=10)
    assert w.version == 1 and not th.is_alive()
    # run 2 in the same box: its agent keeps; it must not get run 1's answer
    m2, _ = make()
    o2, s2 = adamw(m2, 200)
    w2 = opal.wrap(
        o2,
        m2,
        scheduler=s2,
        total_steps=200,
        goal="t",
        decisions=2,
        every=10,
        blocking=True,
        run_dir=tmp_path / "r2",
        backend=f"mailbox:{box}",
        backend_options=dict(poll=0.05),
        verbose=False,
    )
    th2 = serve(scripted(), tmp_path / "b2")
    train(m2, w2, data, 200, sched=s2)
    th2.join(timeout=10)
    assert w2.version == 0
    texts = [
        json.loads((d / "response.json").read_text())["text"] for d in sorted((tmp_path / "r2" / "decisions").iterdir())
    ]
    assert texts and all(t.startswith("answer") for t in texts)
    assert (box.stat().st_mode & 0o077) == 0


def test_l1_resume_bookkeeping(tmp_path):
    agent = scripted({"updater.py": MOMENTUM_UPDATER})
    m, data, s, w = wrapped(tmp_path, agent, steps=400, decisions=8)
    train(m, w, data, 150, sched=s)
    ck = w.state_dict()
    m2, _ = make()
    o2, s2 = adamw(m2, 400)
    msgs = []
    w2 = opal.wrap(
        o2,
        m2,
        scheduler=s2,
        total_steps=400,
        goal="t",
        decisions=8,
        every=10,
        blocking=True,
        run_dir=tmp_path / "r",
        verbose=False,
        backend=lambda ws, msg, sy: msgs.append(msg) or "x",
    )
    w2.load_state_dict(ck)
    train(m2, w2, data, 400, sched=s2, start=150)
    assert msgs and "from update 151" in msgs[0]


# ----- DDP: an exception on one rank only, inside the trial window ------------------------------------------------


def _rank_local_worker(rank, world, port, run_dir, out):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(8, 32), torch.nn.ReLU(), torch.nn.Linear(32, 1))
    ddp = torch.nn.parallel.DistributedDataParallel(model)
    opt = torch.optim.AdamW(ddp.parameters(), lr=3e-3)
    oom = SGD_UPDATER.replace(
        "RAN.append(update)",
        "RAN.append(update)\n        if spec_rank == 1: raise torch.OutOfMemoryError('only rank 1')",
    ).replace(
        "def __init__(self, named_params, previous=None, spec=None):",
        "def __init__(self, named_params, previous=None, spec=None):\n        global spec_rank\n        spec_rank = spec['rank']",
    )
    agent = scripted({"updater.py": oom})
    w = opal.wrap(
        opt,
        ddp,
        total_steps=200,
        goal="t",
        decisions=2,
        every=10,
        sync_every=5,
        blocking=True,
        run_dir=run_dir,
        backend=agent if rank == 0 else None,
        verbose=False,
    )
    g = torch.Generator().manual_seed(rank)
    for _ in range(200):
        loss = ddp(torch.randn(16, 8, generator=g)).pow(2).mean()
        w.zero_grad()
        loss.backward()
        w.step(loss=loss)
    torch.save(
        dict(
            params=torch.cat([p.detach().flatten() for p in model.parameters()]),
            version=w.version,
            updater=w.files["updater.py"],
        ),
        f"{out}/rank{rank}.pt",
    )
    dist.destroy_process_group()


def test_h3_rank_local_exception_reverts_everywhere(tmp_path):
    port = 29600 + os.getpid() % 1000
    mp.spawn(_rank_local_worker, args=(2, port, str(tmp_path / "r"), str(tmp_path)), nprocs=2, join=True)
    a, b = (torch.load(tmp_path / f"rank{r}.pt") for r in (0, 1))
    assert torch.equal(a["params"], b["params"])
    assert "only rank 1" not in a["updater"] and "only rank 1" not in b["updater"]
    assert "reverted" in [e["kind"] for e in events(tmp_path / "r")]
