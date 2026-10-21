"""An incomplete agent updater is rejected before it can break the training loop."""

import pytest
import torch

import opal
from test_core import events


@pytest.mark.parametrize("missing", ["step", "load_state_dict"])
def test_missing_updater_methods_reject_the_edit_and_keep_training(tmp_path, missing):
    methods = {
        "step": "    def step(self, grads, update): pass\n",
        "state_dict": "    def state_dict(self): return {}\n",
        "load_state_dict": "    def load_state_dict(self, state): pass\n",
    }
    source = "class Updater:\n" + "".join(src for name, src in methods.items() if name != missing)
    source += "def build(named_params, previous=None, spec=None): return Updater()\n"

    def agent(ws, message, system):
        (ws / "updater.py").write_text(source)
        return "propose incomplete updater"

    model = torch.nn.Linear(1, 1)
    original = [p.detach().clone() for p in model.parameters()]
    opt = opal.wrap(
        torch.optim.SGD(model.parameters(), lr=0.1), model,
        total_steps=2, goal="g", backend=agent, schedule=[1], every=1, run_dir=tmp_path, verbose=False,
    )
    try:
        for _ in range(2):
            for p in model.parameters():
                p.grad = torch.ones_like(p)
            opt.step(loss=1.0)
        outcomes = [e for e in events(tmp_path) if e["kind"] == "load_failed"]
        assert len(outcomes) == 1 and missing in outcomes[0]["error"]
        for p, before in zip(model.parameters(), original):
            expected = before.clone().add_(-0.1).add_(-0.1)
            assert torch.equal(p, expected)
    finally:
        opt.close(interrupted=True)


def test_replay_keeps_existing_starting_programs_without_checkpoint_methods(tmp_path):
    source = tmp_path / "program"
    source.mkdir()
    (source / "updater.py").write_text(
        "class Updater:\n"
        "    def __init__(self, named, previous=None, spec=None): self.opts = spec['user_optimizers']\n"
        "    def step(self, grads, update):\n"
        "        for opt in self.opts: opt.step()\n"
        "def build(named, previous=None, spec=None): return Updater(named, previous, spec)\n"
    )
    model = torch.nn.Linear(1, 1)
    before = [p.detach().clone() for p in model.parameters()]
    opt = opal.replay(source, torch.optim.SGD(model.parameters(), lr=0.1), model,
                      total_steps=1, run_dir=tmp_path / "run", verbose=False)
    try:
        for p in model.parameters():
            p.grad = torch.ones_like(p)
        opt.step(loss=1.0)
        assert all(torch.equal(p, original - 0.1) for p, original in zip(model.parameters(), before))
    finally:
        opt.close(interrupted=True)
