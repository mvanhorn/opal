"""The precision contract: a program that changes a parameter's or gradient's dtype, or keeps parameter-shaped state
in another dtype than the parameter's or float32, is reverted and the agent is told; programs that keep it pass."""

import torch

import opal
from test_core import SGD_UPDATER, adamw, events, flat, make, train
from test_review1 import scripted, wrapped
from test_review2 import plain_run

CAST_IN_STEP = SGD_UPDATER.replace(
    "RAN.append(update)", "RAN.append(update)\n        self.named[0][1].data = self.named[0][1].data.bfloat16()"
)
CAST_IN_BUILD = SGD_UPDATER.replace(
    "self.lr = torch.tensor(0.05)",
    "self.lr = torch.tensor(0.05)\n        named_params[0][1].data = named_params[0][1].data.bfloat16()",
)

MOMENTS = """
import torch

class Updater:
    def __init__(self, named_params, previous=None, spec=None):
        self.named = named_params
        self.buf = {}
        self.stat = torch.zeros((), dtype=torch.bfloat16)                  # a small statistic: any dtype

    def step(self, grads, update):
        for n, p in self.named:
            if grads[n] is None:
                continue
            if n not in self.buf:
                self.buf[n] = torch.zeros_like(p, dtype=DTYPE(p))
            self.buf[n].mul_(0.9).add_(grads[n].to(self.buf[n].dtype))
            p.add_(self.buf[n].to(p.dtype), alpha=-0.02)
            self.stat = grads[n].float().norm().to(self.stat.dtype)

    def state_dict(self):
        rows = {n: p.float().norm(dim=-1) for n, p in self.named if p.ndim == 2}  # per-row statistics
        return dict(buf=self.buf, stat=self.stat, rows=rows)

    def load_state_dict(self, s):
        self.buf, self.stat = s["buf"], s["stat"]


def build(named_params, previous=None, spec=None):
    return Updater(named_params, previous, spec)
"""
BF16_MOMENTS = MOMENTS.replace("import torch\n", "import torch\nDTYPE = lambda p: torch.bfloat16\n", 1)
FP32_MOMENTS = MOMENTS.replace("import torch\n", "import torch\nDTYPE = lambda p: torch.float32\n", 1)


class Mixed(torch.nn.Module):
    """A float32 layer and a bfloat16 parameter."""

    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(8, 1)
        self.scale = torch.nn.Parameter(torch.ones(8, dtype=torch.bfloat16))

    def forward(self, x):
        return self.lin(x * self.scale.float())


def precision_errors(msgs):
    return [m for m in msgs if "precision contract" in m]


def test_updater_that_casts_a_parameter_to_bf16_is_reverted_and_the_agent_is_told(tmp_path):
    steps = 120
    agent = scripted({"updater.py": CAST_IN_STEP})
    m, data, s, w = wrapped(tmp_path, agent, steps=steps, decisions=2)
    train(m, w, data, steps, sched=s)
    rev = [e for e in events(tmp_path / "r") if e["kind"] == "reverted"]
    assert rev and rev[0]["restored"] and "precision contract" in rev[0]["error"]
    assert all(p.dtype == torch.float32 for p in m.parameters())
    assert torch.equal(flat(m), plain_run(steps))  # the failing update was undone exactly and redone
    told = precision_errors(agent.calls[1:])
    assert told and "parameter 0.weight float32 -> bfloat16" in told[0] and "reverted" in told[0]


def test_build_that_casts_a_parameter_is_rejected_and_leaves_it_exactly(tmp_path):
    steps = 120
    agent = scripted({"updater.py": CAST_IN_BUILD})
    m, data, s, w = wrapped(tmp_path, agent, steps=steps, decisions=2)
    train(m, w, data, steps, sched=s)
    assert "load_failed" in [e["kind"] for e in events(tmp_path / "r")] and w.version == 0
    assert torch.equal(flat(m), plain_run(steps))
    assert precision_errors(agent.calls[1:])


def test_bf16_moments_of_fp32_parameters_are_reverted(tmp_path):
    steps = 120
    agent = scripted({"updater.py": BF16_MOMENTS})
    m, data, s, w = wrapped(tmp_path, agent, steps=steps, decisions=2)
    train(m, w, data, steps, sched=s)
    rev = [e for e in events(tmp_path / "r") if e["kind"] == "reverted"]
    assert rev and rev[0]["restored"] and "parameter-shaped state" in rev[0]["error"]
    assert "bfloat16" not in w.files["updater.py"]
    assert torch.equal(flat(m), plain_run(steps))
    told = precision_errors(agent.calls[1:])
    assert told and "state_dict()['buf']['0.weight'] is bfloat16" in told[0]


def test_starting_program_and_programs_that_keep_the_contract_pass(tmp_path):
    steps = 120
    torch.manual_seed(0)
    model = Mixed()
    data = [(torch.randn(16, 8), torch.randn(16, 1)) for _ in range(50)]
    o, s = adamw(model, steps)  # the starting program: torch AdamW, bfloat16 moments for the bfloat16 parameter
    agent = scripted(None, {"updater.py": FP32_MOMENTS})  # then float32 moments for every parameter, a bf16 scalar
    w = opal.wrap(
        o,
        model,
        scheduler=s,
        total_steps=steps,
        goal="t",
        run_dir=tmp_path / "r",
        backend=agent,
        decisions=3,
        every=10,
        blocking=True,
        verbose=False,
    )
    train(model, w, data, steps, sched=s)
    kinds = [e["kind"] for e in events(tmp_path / "r")]
    assert "applied" in kinds and "reverted" not in kinds and w.version == 1
    assert len(agent.calls) >= 3 and not precision_errors(agent.calls)
    assert model.scale.dtype == torch.bfloat16 and model.lin.weight.dtype == torch.float32
    assert all(b.dtype == torch.float32 for b in w.updater.buf.values())
