"""Regressions in the wrapper core: the user's LR schedule across reverts, fallbacks
and swaps (H1, H2), the exact undo of a failing build or first update (H3, F1), what a later failure in the trial
window leaves (P02), the loss a ctx program gets from observe() (L4), skips and resumes that keep the user's
optimizer state (M4), the analysis grid and the inbox after a resume (L5, B10), and two DDP cases (H4, M1)."""

import json
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import opal
from test_core import events, flat, make
from test_review1 import scripted


def train(model, opt, data, steps, sched=None, pass_loss=True, start=0, lrs=None, skip_at=None):
    for i in range(start, steps):
        x, y = data[i % len(data)]
        loss = torch.nn.functional.mse_loss(model(x), y)
        opt.zero_grad()
        loss.backward()
        if skip_at is None or i + 1 != skip_at:  # the plain run of a skipped update: no step at all
            opt.step(loss=loss) if pass_loss else opt.step()
        if lrs is not None:
            lrs.append(opt.param_groups[0]["lr"])
        if sched is not None:
            sched.step()
    return model


START_STEP = "        for o in self.opts:\n            o.step()"


def starting_updater(tmp_path):
    m, _ = make()
    o = torch.optim.AdamW(m.parameters(), lr=1e-2)
    s = torch.optim.lr_scheduler.CosineAnnealingLR(o, T_max=10)
    w = opal.wrap(o, m, scheduler=s, total_steps=10, goal="t", backend=None, run_dir=tmp_path / "s", verbose=False)
    w.close()
    return w.files["updater.py"]


def schedule_run(tmp_path, sched, agent, steps, **kw):
    m, data = make()
    o = torch.optim.AdamW(m.parameters(), lr=1e-2)
    s = sched(o)
    lrs = []
    if agent is None:
        train(m, o, data, steps, sched=s, pass_loss=False, lrs=lrs)
        return lrs, None
    kw = dict(dict(decisions=5, schedule="uniform", every=10, trial_steps=20, verbose=False), **kw)
    w = opal.wrap(o, m, scheduler=s, total_steps=steps, goal="t", run_dir=tmp_path / "r", backend=agent, **kw)
    train(m, w, data, steps, sched=s, lrs=lrs)
    return lrs, w


def test_h2_a_trial_revert_keeps_the_lr_on_the_users_schedule(tmp_path):
    """A program that follows the user's MultiStepLR raises at update 35, inside its trial window and after the
    milestone at 30: the LR stays on the user's schedule for the whole run."""
    raising = starting_updater(tmp_path).replace(
        START_STEP, "        if update == 35:\n            raise ValueError('boom')\n" + START_STEP
    )
    sched = lambda o: torch.optim.lr_scheduler.MultiStepLR(o, milestones=[30], gamma=0.1)  # noqa: E731
    plain, _ = schedule_run(tmp_path, sched, None, 100)
    lrs, w = schedule_run(tmp_path, sched, scripted({"updater.py": raising}), 100)
    rev = [e for e in events(tmp_path / "r") if e["kind"] == "reverted"]
    assert [(e["update"], e["restored"]) for e in rev] == [(35, False)]
    assert lrs == plain


OWNS = START_STEP.replace(
    "        for o", "        for g in self.param_groups:\n            g['lr'] = 0.05\n        for o"
)


def test_h1_a_late_fallback_continues_the_users_schedule(tmp_path):
    """A program that owns the schedule (sets the user's group LR) passes its trial window and raises at update 70:
    the run falls back to the starting program, and from there the LR is the plain run's (a chainable MultiStepLR)."""
    text = starting_updater(tmp_path).replace("follow_user_scheduler = True", "follow_user_scheduler = False")
    text = text.replace(START_STEP, "        if update == 70:\n            raise ValueError('late bug')\n" + OWNS)
    sched = lambda o: torch.optim.lr_scheduler.MultiStepLR(o, milestones=[150], gamma=0.1)  # noqa: E731
    plain, _ = schedule_run(tmp_path, sched, None, 200)
    lrs, w = schedule_run(tmp_path, sched, scripted({"updater.py": text}), 200)
    rev = [(e["update"], e["restored"]) for e in events(tmp_path / "r") if e["kind"] == "reverted"]
    assert rev == [(70, False)] and w.updater.follow_user_scheduler
    assert lrs[40:69] == [0.05] * 29  # the program's own rate while it ran
    assert lrs[69:] == plain[69:]


def test_h1_restoring_the_starting_updater_resumes_the_schedule_as_if_it_never_stopped(tmp_path):
    """Decision 1 installs a program that owns the LR, decision 2 restores the starting updater.py: the user's cosine
    continues as in the plain run (its missed scheduler.step() calls are replayed), not from where it stopped."""
    start = starting_updater(tmp_path)
    owns = start.replace("follow_user_scheduler = True", "follow_user_scheduler = False").replace(START_STEP, OWNS)
    sched = lambda o: torch.optim.lr_scheduler.CosineAnnealingLR(o, T_max=200)  # noqa: E731
    plain, _ = schedule_run(tmp_path, sched, None, 200)
    lrs, w = schedule_run(tmp_path, sched, scripted({"updater.py": owns}, {"updater.py": start}), 200)
    applied = [e["applied_at"] for e in events(tmp_path / "r") if e["kind"] == "applied"]
    assert applied == [40, 80] and w.files["updater.py"] == start
    assert lrs[40:79] == [0.05] * 39 and lrs[79:] == plain[79:]
    assert w.user_schedulers[0].last_epoch == 200  # stepped as often as in the plain run


def test_l2_get_last_lr_reports_the_programs_rate_for_each_user_group(tmp_path):
    two = """
import torch
class Updater:
    follow_user_scheduler = False
    def __init__(self, named_params, previous=None, spec=None):
        self.named = named_params
        self.opt = torch.optim.SGD([{"params": [p], "lr": 0.1 * (i + 1)} for i, (_, p) in enumerate(named_params)])
        self.param_groups = self.opt.param_groups
    def step(self, grads, update):
        self.opt.step()
    def state_dict(self):
        return self.opt.state_dict()
    def load_state_dict(self, s):
        self.opt.load_state_dict(s)
def build(named_params, previous=None, spec=None):
    return Updater(named_params, previous, spec)
"""
    m, data = make()
    o1 = torch.optim.SGD([m[0].weight], lr=0.01)
    o2 = torch.optim.SGD([m[0].bias, m[2].weight, m[2].bias], lr=0.01)
    s1, s2 = (torch.optim.lr_scheduler.StepLR(o, 1000) for o in (o1, o2))
    w = opal.wrap([o1, o2], m, scheduler=[s1, s2], total_steps=100, goal="t", decisions=2, every=10,
                     run_dir=tmp_path / "r", backend=scripted({"updater.py": two}), verbose=False)  # fmt: skip
    for i in range(100):
        x, y = data[i % len(data)]
        w.zero_grad()
        torch.nn.functional.mse_loss(m(x), y).backward()
        w.step()
        s1.step()
        s2.step()
    assert s1.get_last_lr() == [0.1] and s2.get_last_lr() == [0.2]  # one rate per group of each user optimizer


# ----- H3, F1, P02: what a failing build or update leaves ------------------------------------------------------------

MOVES_STATE = """
class Updater:
    def __init__(self, named_params, previous=None, spec=None):
        opt = spec["user_optimizers"][0]
        self.m = {n: opt.state[p]["exp_avg"].clone() for n, p in named_params if p in opt.state}
        opt.state.clear()          # frees the user's copy: the program owns the moments now
    def step(self, grads, update):
        raise ValueError("bug in the new rule")
    def state_dict(self):
        return dict(m=self.m)
    def load_state_dict(self, s):
        self.m = s["m"]
def build(named_params, previous=None, spec=None):
    return Updater(named_params, previous, spec)
"""
RESCALES = MOVES_STATE.replace(
    "opt.state.clear()          # frees the user's copy: the program owns the moments now",
    "[p.mul_(0.5) for _, p in named_params]; [p.grad.mul_(3) for _, p in named_params]",
)
FAILS_IN_BUILD = MOVES_STATE.replace(
    "opt.state.clear()          # frees the user's copy: the program owns the moments now",
    "opt.state.clear()\n        raise ValueError('bug later in build')",
)


def adamw_run(tmp_path, agent, steps=120):
    m, data = make()
    o = torch.optim.AdamW(m.parameters(), lr=3e-3)
    s = torch.optim.lr_scheduler.CosineAnnealingLR(o, T_max=steps)
    if agent is None:
        train(m, o, data, steps, sched=s, pass_loss=False)
        return flat(m), None
    w = opal.wrap(o, m, scheduler=s, total_steps=steps, goal="t", decisions=1, every=10, run_dir=tmp_path,
                     backend=agent, verbose=False)  # fmt: skip
    train(m, w, data, steps, sched=s)
    return flat(m), w


def test_h3_what_build_did_to_shared_objects_is_undone_exactly(tmp_path):
    ref, _ = adamw_run(tmp_path, None)
    for label, text in (("frees", MOVES_STATE), ("rescales", RESCALES)):
        got, w = adamw_run(tmp_path / label, scripted({"updater.py": text}))
        rev = [e for e in events(tmp_path / label) if e["kind"] == "reverted"]
        assert [e["restored"] for e in rev] == [True], label
        assert torch.equal(got, ref), label
    got, w = adamw_run(tmp_path / "fails", scripted({"updater.py": FAILS_IN_BUILD}))
    assert "load_failed" in [e["kind"] for e in events(tmp_path / "fails")]
    assert torch.equal(got, ref)


CAST_THEN_RAISE = """
class Updater:
    def __init__(self, named_params, previous=None, spec=None):
        self.named = named_params
    def step(self, grads, update):
        n0, p0 = self.named[0]
        p0.data = p0.data.bfloat16()
        p0.sub_(0.01 * (p0 @ grads[n0].t() @ p0))          # raises: bf16 @ fp32
    def state_dict(self):
        return {}
    def load_state_dict(self, s):
        pass
def build(named_params, previous=None, spec=None):
    return Updater(named_params, previous, spec)
"""


def test_f1_a_program_that_casts_and_then_raises_is_put_back_exactly(tmp_path):
    def run(agent):
        m, data = make()
        o = torch.optim.SGD(m.parameters(), lr=0.01)
        if agent is None:
            train(m, o, data, 200, pass_loss=False)
            return m, None
        w = opal.wrap(o, m, total_steps=200, goal="t", run_dir=tmp_path / "r", backend=agent, decisions=2, every=10,
                         verbose=False)  # fmt: skip
        train(m, w, data, 200)
        return m, w

    ref, _ = run(None)
    m, w = run(scripted({"updater.py": CAST_THEN_RAISE}))
    rev = [e for e in events(tmp_path / "r") if e["kind"] == "reverted"]
    assert [e["restored"] for e in rev] == [True] and w.update == 200
    assert m[0].weight.dtype == torch.float32 and torch.equal(flat(m), flat(ref))


ADDS_THEN_RAISES = """
import torch
class U:
    def __init__(self, named, previous=None, spec=None):
        self.named, self.calls = named, 0
    def step(self, grads, update):
        self.calls += 1
        self.named[0][1].add_(10)
        if self.calls == 2:
            raise RuntimeError("second-step failure")
    def state_dict(self):
        return dict(calls=self.calls)
    def load_state_dict(self, s):
        self.calls = s["calls"]
def build(named_params, previous=None, spec=None):
    return U(named_params, previous, spec)
"""


def test_p02_a_failure_after_the_first_update_of_the_window_is_reported_as_not_undone(tmp_path):
    """Only the first update of a new updater is undone exactly; a failure at its second update keeps what that
    update did, and the agent is told so in plain words."""
    m = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        m.weight.fill_(1)
    w = opal.wrap(torch.optim.SGD(m.parameters(), lr=0.1), m, total_steps=100, goal="t", backend=None,
                     run_dir=tmp_path / "r", every=10, verbose=False)  # fmt: skip
    files = dict(w.files, **{"updater.py": ADDS_THEN_RAISES})
    w._apply(dict(k=1, requested_at=0, files=files), 1)
    for _ in range(2):
        m.weight.grad = torch.ones_like(m.weight)
        w.step(loss=1.0)
    assert m.weight.item() == 21.0  # +10 twice, the second update not undone; the starting SGD did not redo it
    assert (
        w.errors[-1]["where"] == "updater.step of v001 (reverted: this update was not undone and may be partly applied)"
    )
    w.close()


def test_p02_a_first_update_without_a_copy_says_why(tmp_path):
    raising = ADDS_THEN_RAISES.replace("if self.calls == 2:", "if self.calls == 1:")
    m = torch.nn.Linear(1, 1, bias=False)
    w = opal.wrap(torch.optim.SGD(m.parameters(), lr=0.1), m, total_steps=100, goal="t", backend=None,
                     run_dir=tmp_path / "r", every=10, verbose=False, snapshot_limit_gb=0)  # fmt: skip
    w._apply(dict(k=1, requested_at=0, files=dict(w.files, **{"updater.py": raising})), 1)
    m.weight.grad = torch.ones_like(m.weight)
    w.step(loss=1.0)
    assert "not undone, because the copy that undoes it was above snapshot_limit_gb" in w.errors[-1]["where"]
    w.close()


# ----- L4, M4: ctx.loss from observe(); skips and resumes keep the user's optimizer state -----------------------------

CTX_SEEN = """
SEEN = []
class Updater:
    follow_user_scheduler = True
    def __init__(self, named_params, previous=None, spec=None):
        self.opt = spec["user_optimizers"][0]
    def step(self, grads, update, ctx):
        SEEN.append(None if ctx.loss is None else float(ctx.loss))
        self.opt.step()
    def state_dict(self):
        return {}
    def load_state_dict(self, s):
        pass
def build(named_params, previous=None, spec=None):
    return Updater(named_params, previous, spec)
"""


def test_l4_observe_gives_a_ctx_program_its_loss(tmp_path):
    m, data = make()
    w = opal.wrap(torch.optim.SGD(m.parameters(), lr=0.01), m, total_steps=60, goal="t", decisions=1, every=10,
                     run_dir=tmp_path / "r", backend=scripted({"updater.py": CTX_SEEN}), verbose=False)  # fmt: skip
    losses = []
    for i in range(60):
        x, y = data[i % len(data)]
        loss = torch.nn.functional.mse_loss(m(x), y)
        w.zero_grad()
        loss.backward()
        w.observe(loss)  # as a framework that calls step() itself
        losses.append(float(loss.detach()))
        w.step()
    seen = sys.modules[w.updater.__module__].SEEN
    assert seen and seen == losses[-len(seen) :]


SKIPS_AT_30 = """
class Updater:
    follow_user_scheduler = True
    can_skip = True
    def __init__(self, named_params, previous=None, spec=None):
        self.opt = spec["user_optimizers"][0]
    def step(self, grads, update):
        self.opt.step()
        return update == 30
    def state_dict(self):
        return {}                  # leaves the user's AdamW out
    def load_state_dict(self, s):
        pass
def build(named_params, previous=None, spec=None):
    return Updater(named_params, previous, spec)
"""


def test_m4_a_skip_also_undoes_the_users_optimizer_step(tmp_path):
    m, data = make()
    o = torch.optim.AdamW(m.parameters(), lr=3e-3)
    train(m, o, data, 80, pass_loss=False, skip_at=30)
    ref = flat(m)
    m, data = make()
    o = torch.optim.AdamW(m.parameters(), lr=3e-3)
    w = opal.wrap(o, m, total_steps=80, goal="t", decisions=1, every=10, run_dir=tmp_path / "r",
                     backend=scripted({"updater.py": SKIPS_AT_30}), verbose=False)  # fmt: skip
    train(m, w, data, 80)
    assert w._skipped == [30] and torch.equal(flat(m), ref)


DELEGATES = """
class Updater:
    follow_user_scheduler = True
    def __init__(self, named_params, previous=None, spec=None):
        self.opt = spec["user_optimizers"][0]
    def step(self, grads, update):
        self.opt.step()
    def state_dict(self):
        return {}
    def load_state_dict(self, s):
        pass
def build(named_params, previous=None, spec=None):
    return Updater(named_params, previous, spec)
"""

COUNT = """
EVERY = 10
CALLS = []
def analyze(obs, memory):
    CALLS.append(obs.update)
"""


def resumable(run_dir, steps, agent, **kw):
    m, data = make()
    o = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=0.1)
    s = torch.optim.lr_scheduler.CosineAnnealingLR(o, T_max=steps)
    kw = dict(dict(decisions=4, schedule="uniform", every=10, verbose=False), **kw)
    w = opal.wrap(o, m, scheduler=s, total_steps=steps, goal="t", run_dir=run_dir, backend=agent, **kw)
    return m, data, o, s, w


def resume(tmp_path, steps, at, first_agent, second_agent, lrs=None, **kw):
    m, data, o, s, w = resumable(tmp_path / "b", steps, first_agent, **kw)
    train(m, w, data, at, sched=s)
    ck = dict(model=m.state_dict(), opt=w.state_dict(), sched=s.state_dict())
    torch.save(ck, tmp_path / "ck.pt")
    w.close(interrupted=True)
    ck = torch.load(tmp_path / "ck.pt", weights_only=False)
    m2, data, o2, s2, w2 = resumable(tmp_path / "c", steps, second_agent, **kw)
    m2.load_state_dict(ck["model"])
    w2.load_state_dict(ck["opt"])
    s2.load_state_dict(ck["sched"])  # after the wrapper's, as some scripts do
    train(m2, w2, data, steps, sched=s2, start=at, lrs=lrs)
    return m2, w2


def test_m4_a_resume_keeps_the_users_optimizer_state_the_program_does_not_save(tmp_path):
    steps = 200
    m, data, o, s, w = resumable(tmp_path / "a", steps, scripted({"updater.py": DELEGATES}))
    train(m, w, data, steps, sched=s)
    m2, w2 = resume(tmp_path, steps, 120, scripted({"updater.py": DELEGATES}), lambda ws, msg, sy: "keep")
    assert w2.files["updater.py"] == DELEGATES and torch.equal(flat(m2), flat(m))


def test_h1_a_resume_while_a_program_owns_the_schedule_then_a_return_to_the_starting_program(tmp_path):
    """The ledger of the user's schedule is checkpointed: after a resume, restoring the starting updater still gives the
    plain run's learning rate."""
    steps = 200
    start = starting_updater(tmp_path)
    owns = start.replace("follow_user_scheduler = True", "follow_user_scheduler = False").replace(START_STEP, OWNS)
    m, data = make()
    o = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=0.1)
    s = torch.optim.lr_scheduler.CosineAnnealingLR(o, T_max=steps)
    plain = []
    train(m, o, data, steps, sched=s, pass_loss=False, lrs=plain)
    lrs = []
    m2, w2 = resume(tmp_path, steps, 105, scripted({"updater.py": owns}), scripted({"updater.py": start}), lrs=lrs)
    applied = [e["applied_at"] for e in events(tmp_path / "b") + events(tmp_path / "c") if e["kind"] == "applied"]
    assert applied == [50, 150]  # decision 1 took over the schedule; the resumed run's first decision restored it
    assert w2.files["updater.py"] == start and lrs[150 - 105 - 1 :] == plain[149:]


def test_l5_and_b10_the_analysis_grid_and_the_inbox_after_a_resume(tmp_path):
    steps = 200
    def agent(ws, msg, sy):
        if "Decision 1." in msg:
            (ws / "analysis.py").write_text(COUNT)
        return "x"

    m, data, o, s, w = resumable(tmp_path / "a", steps, agent, decisions=6, schedule="front")
    train(m, w, data, steps, sched=s)
    full = [u for u in sys.modules[w.analysis.__name__].CALLS if u > 85]
    m, data, o, s, w = resumable(tmp_path / "b", steps, agent, decisions=6, schedule="front")
    train(m, w, data, 85, sched=s)
    ck = w.state_dict()
    train(m, w, data, 150, sched=s, start=85)  # the lost attempt goes on, and asks more decisions
    lost = sorted(p.name for p in (tmp_path / "b" / "workspace" / "inbox").iterdir())
    w.close(interrupted=True)
    m2, data, o2, s2, w2 = resumable(tmp_path / "b", steps, agent, decisions=6, schedule="front")
    w2.load_state_dict(ck)  # into the same folder
    inbox = sorted(p.name for p in (tmp_path / "b" / "workspace" / "inbox").iterdir())
    assert len(lost) > ck["decisions"] and len(inbox) == ck["decisions"]
    assert any((tmp_path / "b" / "attempts").glob("*/inbox/decision-*.json"))
    train(m2, w2, data, steps, sched=s2, start=85)
    assert list(sys.modules[w2.analysis.__name__].CALLS) == full


# ----- DDP (gloo, 2 ranks): crash markers after a resume (H4); a loss passed on rank 0 only (M1) ------------------------

LATE_CRASH = """
class Updater:
    def __init__(self, named_params, previous=None, spec=None):
        self.named = named_params
    def step(self, grads, update):
        if update >= 120:
            raise ValueError("late crash")
        for n, p in self.named:
            if grads[n] is not None:
                p.add_(grads[n], alpha=-0.01)
    def state_dict(self):
        return {}
    def load_state_dict(self, s):
        pass
def build(named_params, previous=None, spec=None):
    return Updater(named_params, previous, spec)
"""

CTX_SGD = """
class Updater:
    def __init__(self, named_params, previous=None, spec=None):
        self.named = named_params
    def step(self, grads, update, ctx):
        for n, p in self.named:
            if grads[n] is not None:
                p.add_(grads[n] * ctx.loss, alpha=-0.01)
    def state_dict(self):
        return {}
    def load_state_dict(self, s):
        pass
def build(named_params, previous=None, spec=None):
    return Updater(named_params, previous, spec)
"""


def _free_port():
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _ddp_agent(text):
    def agent(ws, msg, sys_):
        if "Decision 1." in msg:
            (ws / "updater.py").write_text(text)
        return "x"

    return agent


def _h4_worker(rank, world, port, out, phase):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    rendezvous = (Path(out) / f"rendezvous-{phase}").resolve().as_uri()
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=world)
    os.chdir(out)  # the default run folder: ./opal_runs/<time>-<pid>-<random>
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.ReLU(), torch.nn.Linear(16, 1))
    ddp = torch.nn.parallel.DistributedDataParallel(model)
    w = opal.wrap(torch.optim.SGD(ddp.parameters(), lr=0.01), ddp, total_steps=200, goal="t", decisions=2, every=10,
                     sync_every=5, trial_steps=5, backend=_ddp_agent(LATE_CRASH), verbose=False)  # fmt: skip
    ck, start = Path(out) / "ck.pt", 0
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
            if phase == 1 and u + 1 == 80:
                if rank == 0:
                    torch.save(dict(model=model.state_dict(), opt=w.state_dict(), update=80), ck)
                w.close(interrupted=True)
                result = "stopped"
                break
    except RuntimeError as exc:
        result = "crashed: " + str(exc).splitlines()[0][:90]
    Path(out, f"p{phase}-rank{rank}.json").write_text(json.dumps(dict(result=result, run_path=str(w._run_path))))
    dist.destroy_process_group()


def test_h4_a_crash_after_a_resume_leaves_markers_and_the_next_restart_falls_back(tmp_path):
    results = []
    for phase in (1, 2, 3):
        mp.spawn(_h4_worker, args=(2, _free_port(), str(tmp_path), phase), nprocs=2, join=True)
        results.append([json.loads((tmp_path / f"p{phase}-rank{r}.json").read_text()) for r in (0, 1)])
    assert [r["result"] for r in results[0]] == ["stopped", "stopped"]
    assert all(r["result"].startswith("crashed") for r in results[1])
    assert all(Path(r["run_path"]).exists() for r in results[1])
    assert len(list((tmp_path / "opal_runs").rglob("crashed-*.json"))) == 2
    assert [r["result"] for r in results[2]] == ["finished", "finished"]


def _m1_worker(rank, world, port, out):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.ReLU(), torch.nn.Linear(16, 1))
    ddp = torch.nn.parallel.DistributedDataParallel(model)
    w = opal.wrap(torch.optim.SGD(ddp.parameters(), lr=0.01), ddp, total_steps=100, goal="t", decisions=2,
                     every=10, backend=_ddp_agent(CTX_SGD), run_dir=out + "/r", verbose=False)  # fmt: skip
    g = torch.Generator().manual_seed(rank)
    for _ in range(100):
        loss = ddp(torch.randn(16, 8, generator=g)).pow(2).mean()
        w.zero_grad()
        loss.backward()
        w.step(loss=loss) if rank == 0 else w.step()
    Path(out, f"m1-{rank}.json").write_text(json.dumps(dict(version=w.version, w=flat(model).tolist())))
    dist.destroy_process_group()


def test_m1_a_ctx_program_with_the_loss_on_rank_0_only(tmp_path):
    mp.spawn(_m1_worker, args=(2, _free_port(), str(tmp_path)), nprocs=2, join=True)
    r0, r1 = (json.loads((tmp_path / f"m1-{r}.json").read_text()) for r in (0, 1))
    assert r0["version"] == r1["version"] == 1 and r0["w"] == r1["w"]  # it ran, identically on both ranks
