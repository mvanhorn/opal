"""Regressions in the precision guard and the analysis: p.grad is checked, not the program's grads dict (F2);
devices are checked (F4); placement is checked after every update (F5); state created or recast later, the user's
optimizer state cast in build(), and same-shape parameters of two dtypes are caught (F8, P01), while
float32 state of bfloat16 parameters and the user optimizer's native state pass; an analysis that changes training is
caught and stopped (F3, DW2); an analysis.py without analyze() is rejected (F19); reports are stored as plain JSON
when they are made (F6)."""

import json
import os
import sys

import pytest
import torch

import opal
from test_core import MOMENTUM_UPDATER, adamw, events, flat, make, train
from test_review1 import scripted
from test_review2 import plain_run


def wrapped(tmp_path, model, opt, agent, steps, **kw):
    kw = dict(dict(decisions=2, every=10, blocking=True, verbose=False), **kw)
    return opal.wrap(opt, model, total_steps=steps, goal="t", run_dir=tmp_path / "r", backend=agent, **kw)


def reverted(tmp_path):
    return [e for e in events(tmp_path / "r") if e["kind"] == "reverted"]


UPCAST_INTO_DICT = """
import torch
class Updater:
    def __init__(self, named_params, previous=None, spec=None):
        self.named = named_params
        self.m = {n: torch.zeros_like(p, dtype=torch.float32) for n, p in named_params}   # fp32 state: allowed
    def step(self, grads, update):
        for n, p in self.named:
            if grads[n] is None:
                continue
            grads[n] = grads[n].float()                     # computes in float32, in the dict it was given
            self.m[n].mul_(0.9).add_(grads[n])
            p.add_(self.m[n].to(p.dtype), alpha=-0.01)
    def state_dict(self):
        return dict(m=self.m)
    def load_state_dict(self, s):
        self.m = s["m"]
def build(named_params, previous=None, spec=None):
    return Updater(named_params, previous, spec)
"""


class Float32Output(torch.nn.Module):
    def forward(self, x):
        return x.float()  # the loss in float32: torch < 2.4 has no bfloat16 mse_loss on the CPU


def bf16_model():
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.ReLU(), torch.nn.Linear(16, 1)).to(torch.bfloat16)
    model.append(Float32Output())
    data = [(torch.randn(16, 8, dtype=torch.bfloat16), torch.randn(16, 1)) for _ in range(50)]
    return model, data


def test_f2_writing_into_the_grads_dict_is_not_a_dtype_change(tmp_path):
    """Also a legitimate positive of P01: bfloat16 parameters with float32 state, checked at every trial update and
    check step."""
    model, data = bf16_model()
    w = wrapped(
        tmp_path, model, torch.optim.AdamW(model.parameters(), lr=1e-3), scripted({"updater.py": UPCAST_INTO_DICT}), 120
    )
    train(model, w, data, 120)
    assert not reverted(tmp_path) and w.version == 1


LATE_BF16 = """
import torch
class Updater:
    def __init__(self, named_params, previous=None, spec=None):
        self.named, self.buf, self.k = named_params, {}, 0
    def step(self, grads, update):
        self.k += 1
        for n, p in self.named:
            if grads[n] is None:
                continue
            if self.k >= START and n not in self.buf:      # momentum switched on after a warm-up
                self.buf[n] = torch.zeros_like(p, dtype=torch.bfloat16)
            if n in self.buf:
                self.buf[n].mul_(0.9).add_(grads[n].bfloat16())
                p.add_(self.buf[n].float(), alpha=-0.01)
            else:
                p.add_(grads[n], alpha=-0.01)
    def state_dict(self):
        return dict(buf=self.buf, k=self.k)
    def load_state_dict(self, s):
        self.buf, self.k = s["buf"], s["k"]
def build(named_params, previous=None, spec=None):
    return Updater(named_params, previous, spec)
"""

RECAST = """
import torch
class Updater:
    def __init__(self, named_params, previous=None, spec=None):
        self.named, self.k = named_params, 0
        self.buf = {n: torch.zeros_like(p) for n, p in named_params}
    def step(self, grads, update):
        self.k += 1
        if self.k == AT:
            self.buf = {n: t.bfloat16() for n, t in self.buf.items()}
        for n, p in self.named:
            if grads[n] is not None:
                self.buf[n].mul_(0.9).add_(grads[n].to(self.buf[n].dtype))
                p.add_(self.buf[n].to(p.dtype), alpha=-0.01)
    def state_dict(self):
        return dict(buf=self.buf, k=self.k)
    def load_state_dict(self, s):
        self.buf, self.k = s["buf"], s["k"]
def build(named_params, previous=None, spec=None):
    return Updater(named_params, previous, spec)
"""


@pytest.mark.parametrize(
    "src,where",
    [
        (LATE_BF16.replace("START", "3"), "trial"),  # created at the 3rd update, inside the trial window
        (LATE_BF16.replace("START", "26"), "check"),  # created after the window, found at the next check
        (RECAST.replace("AT", "2"), "trial"),  # float32 at the first update, recast at the second (P01)
        (RECAST.replace("AT", "33"), "check"),
    ],
)
def test_p01_state_created_or_recast_later_is_caught(tmp_path, src, where):
    m, data = make()
    w = wrapped(tmp_path, m, torch.optim.SGD(m.parameters(), lr=0.01), scripted({"updater.py": src}), 200)
    train(m, w, data, 200)
    rev = reverted(tmp_path)
    assert len(rev) == 1 and "precision contract" in rev[0]["error"] and "state_dict()['buf']" in rev[0]["error"]
    start = next(e["applied_at"] for e in events(tmp_path / "r") if e["kind"] == "applied")
    assert (rev[0]["update"] < start + 20) == (where == "trial")


USER_STATE_BF16 = """
class Updater:
    def __init__(self, named_params, previous=None, spec=None):
        self.opts = spec["user_optimizers"]
        for o in self.opts:
            for st in o.state.values():
                if "momentum_buffer" in st:
                    st["momentum_buffer"] = st["momentum_buffer"].bfloat16()   # "save memory"
    def step(self, grads, update):
        for o in self.opts:
            o.step()
    def state_dict(self):
        return dict(opts=[o.state_dict() for o in self.opts])
    def load_state_dict(self, s):
        for o, x in zip(self.opts, s["opts"]):
            o.load_state_dict(x)
def build(named_params, previous=None, spec=None):
    return Updater(named_params, previous, spec)
"""


def test_f8_casting_the_users_optimizer_state_in_build_is_caught_and_undone(tmp_path):
    m, data = make()
    opt = torch.optim.SGD(m.parameters(), lr=0.01, momentum=0.9, foreach=False)
    w = wrapped(tmp_path, m, opt, scripted({"updater.py": USER_STATE_BF16}), 120)
    train(m, w, data, 120)
    rev = reverted(tmp_path)
    assert [e["restored"] for e in rev] == [True] and "the training script's optimizer 0 state" in rev[0]["error"]
    assert {st["momentum_buffer"].dtype for st in opt.state.values()} == {torch.float32}
    m2, data = make()
    o2 = torch.optim.SGD(m2.parameters(), lr=0.01, momentum=0.9, foreach=False)
    train(m2, o2, data, 120, pass_loss=False)
    assert torch.equal(flat(m), flat(m2))  # undone exactly: the plain run


def test_p01_the_user_optimizers_native_state_passes(tmp_path):
    """A program that keeps driving the training script's AdamW of a bfloat16 parameter, whose state torch keeps in
    bfloat16, passes; so does state that optimizer creates later."""

    class Mixed(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(8, 1)
            self.scale = torch.nn.Parameter(torch.ones(8, dtype=torch.bfloat16))

        def forward(self, x):
            return self.lin(x * self.scale.float())

    delegates = """
class Updater:
    follow_user_scheduler = True
    def __init__(self, named_params, previous=None, spec=None):
        self.opts = spec["user_optimizers"]
    def step(self, grads, update):
        for o in self.opts:
            o.step()
    def state_dict(self):
        return dict(opts=[o.state_dict() for o in self.opts])
    def load_state_dict(self, s):
        for o, x in zip(self.opts, s["opts"]):
            o.load_state_dict(x)
def build(named_params, previous=None, spec=None):
    return Updater(named_params, previous, spec)
"""
    torch.manual_seed(0)
    m = Mixed()
    data = [(torch.randn(16, 8), torch.randn(16, 1)) for _ in range(50)]
    late = torch.optim.AdamW([m.scale], lr=1e-3)  # gets its state only after the program took over
    w = wrapped(
        tmp_path, m, [torch.optim.AdamW(m.lin.parameters(), lr=1e-3), late], scripted({"updater.py": delegates}), 120
    )
    for i in range(120):
        x, y = data[i % len(data)]
        w.zero_grad()
        loss = torch.nn.functional.mse_loss(m(x), y)
        loss.backward()
        if i < 30:
            m.scale.grad = None  # the second optimizer creates its state after the swap
        w.step(loss=loss)
    assert not reverted(tmp_path) and w.version == 1
    assert late.state[m.scale]["exp_avg"].dtype == torch.bfloat16


class Twin(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Parameter(torch.ones(8))  # float32
        self.b = torch.nn.Parameter(torch.ones(8, dtype=torch.bfloat16))  # bfloat16, the same shape

    def forward(self, x):
        return (x * self.a * self.b.float()).sum(-1, keepdim=True)


def test_f8_state_keyed_by_a_parameter_name_is_compared_with_that_parameter(tmp_path):
    torch.manual_seed(0)
    m = Twin()
    data = [(torch.randn(16, 8), torch.randn(16, 1)) for _ in range(50)]
    w = wrapped(
        tmp_path,
        m,
        torch.optim.SGD(m.parameters(), lr=0.01),
        scripted({"updater.py": LATE_BF16.replace("START", "1")}),
        120,
    )
    train(m, w, data, 120)
    rev = reverted(tmp_path)
    assert len(rev) == 1 and "the state of parameter a (float32)" in rev[0]["error"]


LATE_CAST = """
class Updater:
    def __init__(self, named_params, previous=None, spec=None):
        self.named, self.k = named_params, 0
    def step(self, grads, update):
        self.k += 1
        for n, p in self.named:
            if grads[n] is not None:
                p.add_(grads[n], alpha=-0.01)
        if self.k == 25:                       # after the 20-update trial window, not on a check step
            p0 = self.named[0][1]
            p0.data = p0.data.bfloat16()
    def state_dict(self):
        return dict(k=self.k)
    def load_state_dict(self, s):
        self.k = s["k"]
def build(named_params, previous=None, spec=None):
    return Updater(named_params, previous, spec)
"""


def test_f5_a_cast_after_the_trial_window_is_caught_before_the_next_forward(tmp_path):
    m, data = make()
    w = wrapped(tmp_path, m, torch.optim.SGD(m.parameters(), lr=0.01), scripted({"updater.py": LATE_CAST}), 200)
    train(m, w, data, 200)  # an MLP: the next forward would fail on mixed dtypes
    rev = reverted(tmp_path)
    start = next(e["applied_at"] for e in events(tmp_path / "r") if e["kind"] == "applied")
    assert [e["update"] for e in rev] == [start + 24]  # the update that cast it, not the next check step
    assert m[0].weight.dtype == torch.float32 and w.update == 200


@pytest.mark.skipif(os.environ.get("OPAL_CPU_ONLY") == "1" or not torch.backends.mps.is_available(),
                    reason="needs a second device (MPS); disabled by OPAL_CPU_ONLY=1")
def test_f4_a_device_change_is_caught_and_put_back(tmp_path):
    to_cpu = LATE_CAST.replace("self.k == 25", "self.k == 1").replace("p0.data.bfloat16()", "p0.data.cpu()")
    m, data = make()
    m = m.to("mps")
    data = [(x.to("mps"), y.to("mps")) for x, y in data]
    w = wrapped(tmp_path, m, torch.optim.SGD(m.parameters(), lr=0.01), scripted({"updater.py": to_cpu}), 120)
    train(m, w, data, 120)
    rev = reverted(tmp_path)
    assert len(rev) == 1 and "on mps:0 -> cpu" in rev[0]["error"] and m[0].weight.device.type == "mps"


COSINE_ANALYSIS = """
EVERY = 10
def analyze(obs, memory):
    for n, g in obs.grads.items():
        if g is not None:
            v = g.flatten()            # a view of the gradient
            v /= v.norm() + 1e-12      # changes the gradient
    return {"report": {"x": 1}}
"""
CAST_ANALYSIS = """
EVERY = 10
def analyze(obs, memory):
    p = obs.params["0.weight"]
    p.data = p.data.bfloat16()
    return {"report": {"x": 1}}
"""


@pytest.mark.parametrize("analysis", [COSINE_ANALYSIS, CAST_ANALYSIS], ids=["in-place", "cast"])
@pytest.mark.parametrize("with_updater", [False, True], ids=["starting-updater", "agent-updater"])
def test_f3_an_analysis_that_changes_training_is_stopped_and_named(tmp_path, analysis, with_updater):
    steps = 200
    answers = (
        ({"updater.py": MOMENTUM_UPDATER}, {"analysis.py": analysis}) if with_updater else ({"analysis.py": analysis},)
    )
    agent = scripted(*answers)
    m, data = make()
    o, s = adamw(m, steps)
    w = opal.wrap(o, m, scheduler=s, total_steps=steps, goal="t", run_dir=tmp_path / "r", backend=agent,
                     decisions=4, every=10, verbose=False)  # fmt: skip
    train(m, w, data, steps, sched=s)
    assert not reverted(tmp_path)  # the updater is not blamed
    told = [msg for msg in agent.calls if "analysis.analyze changed parameters or gradients" in msg]
    assert told and "not run again until analysis.py changes" in told[0]
    assert m[0].weight.dtype == torch.float32 and w._analysis_off and w.update == steps


def test_f19_an_analysis_without_analyze_is_rejected(tmp_path):
    agent = scripted({"analysis.py": "EVERY = 10\ndef analyse(obs, memory):\n    return None\n"})
    m, data = make()
    o, s = adamw(m, 120)
    w = opal.wrap(o, m, scheduler=s, total_steps=120, goal="t", run_dir=tmp_path / "r", backend=agent,
                     decisions=2, every=10, verbose=False)  # fmt: skip
    train(m, w, data, 120, sched=s)
    assert "load_failed" in [e["kind"] for e in events(tmp_path / "r")]
    assert any("analysis.py must define analyze(obs, memory)" in msg for msg in agent.calls)


EMA_UPDATER = """
import torch
class Updater:
    def __init__(self, named_params, previous=None, spec=None):
        self.named = named_params
        self.step_count = torch.zeros(())
    def step(self, grads, update):
        self.step_count.add_(1)
        for n, p in self.named:
            if grads[n] is not None:
                p.add_(grads[n], alpha=-0.01)
    def state_dict(self):
        return dict(step_count=self.step_count)
    def load_state_dict(self, s):
        self.step_count = s["step_count"]
def build(named_params, previous=None, spec=None):
    return Updater(named_params, previous, spec)
"""
REPORT_ANALYSIS = """
EVERY = 10
class Thing:
    pass
def analyze(obs, memory):
    return {"report": {"steps_seen": obs.updater.step_count, "thing": Thing()}}
"""


def test_f6_reports_are_plain_json_when_they_are_made(tmp_path):
    agent = scripted({"updater.py": EMA_UPDATER, "analysis.py": REPORT_ANALYSIS})
    m, data = make()
    o, s = adamw(m, 300)
    w = opal.wrap(o, m, scheduler=s, total_steps=300, goal="t", run_dir=tmp_path / "r", backend=agent,
                     decisions=3, every=10, verbose=False)  # fmt: skip
    train(m, w, data, 300, sched=s)
    inbox = json.loads((tmp_path / "r" / "workspace" / "inbox" / "decision-002.json").read_text())
    seen = [r["report"]["steps_seen"] for r in inbox["reports"]]
    assert seen == sorted(seen) and len(set(seen)) == len(seen) and all(isinstance(x, float) for x in seen)
    assert any("values that are not JSON were stored as text: Thing" in msg for msg in agent.calls)
