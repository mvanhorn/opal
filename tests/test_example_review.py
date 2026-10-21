"""The example updater and the legacy prompt's inline
example: the example never frees the training script's optimizer state, so a fallback to the starting
program continues with it, on schedule; it says when it cannot reproduce the starting run (another optimizer class, a
scheduler stepped per epoch); it keeps the starting program's clipping; the legacy example carries AdamW moments over
from the starting program."""

import math
from pathlib import Path

import pytest
import torch

import opal
from test_core import events, flat, make
from test_review1 import scripted

EXAMPLE = (Path(opal.__file__).parent / "prompts" / "example_updater.py").read_text()


def fails_at(n):
    return (
        EXAMPLE
        + f"""

_Base = Updater


class Updater(_Base):
    def step(self, grads, update, ctx):
        self.n = getattr(self, "n", 0) + 1
        if self.n == {n}:
            raise RuntimeError("a bug in my rule")
        return super().step(grads, update, ctx)
"""
    )


def train(model, opt, data, steps, sched=None, sched_every=1, clip=None, hook=None):
    for i in range(steps):
        x, y = data[i % len(data)]
        loss = torch.nn.functional.mse_loss(model(x), y)
        opt.zero_grad()
        loss.backward()
        if clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step(loss=loss) if isinstance(opt, opal.AgentOptimizer) else opt.step()
        if sched is not None and (i + 1) % sched_every == 0:
            sched.step()
        if hook:
            hook(i + 1)


@pytest.mark.parametrize("n", [3, 400], ids=["in-the-trial-window", "after-it"])
def test_prompts1_a_fallback_from_the_example_continues_the_users_adamw(tmp_path, n):
    def run(agent, steps=600):
        m, data = make()
        o = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=0.1)
        s = torch.optim.lr_scheduler.CosineAnnealingLR(o, T_max=steps)
        if agent is None:
            lrs = []
            train(m, o, data, steps, sched=s, hook=lambda u: lrs.append(o.param_groups[0]["lr"]))
            return m, o, lrs, None
        w = opal.wrap(o, m, scheduler=s, total_steps=steps, goal="g", run_dir=tmp_path / "r", backend=agent,
                         decisions=4, every=10, verbose=False)  # fmt: skip
        lrs = []
        train(m, w, data, steps, sched=s, hook=lambda u: lrs.append(o.param_groups[0]["lr"]))
        return m, o, lrs, w

    _, o_plain, plain, _ = run(None)
    m, o, lrs, w = run(scripted({"updater.py": fails_at(n)}))
    rev = [e for e in events(tmp_path / "r") if e["kind"] == "reverted"]
    start = next(e["applied_at"] for e in events(tmp_path / "r") if e["kind"] == "applied")
    assert [e["update"] for e in rev] == [start + n - 1]
    p0 = next(iter(m.parameters()))
    # the user's AdamW kept its state while the example ran, and the starting program went on with it
    assert float(o.state[p0]["step"]) == 600 - n
    assert lrs[rev[0]["update"] :] == plain[rev[0]["update"] :]  # and with the user's cosine where it should be


def example_run(tmp_path, make_opt, steps=200, sched_every=1, clip=None):
    m1, data = make()
    o1, s1 = make_opt(m1)
    train(m1, o1, data, steps, sched=s1, sched_every=sched_every, clip=clip)
    m2, _ = make()
    o2, s2 = make_opt(m2)
    agent = scripted({"updater.py": EXAMPLE})
    w = opal.wrap(o2, m2, scheduler=s2, total_steps=steps, goal="g", run_dir=tmp_path / "r", backend=agent,
                     decisions=4, every=10, verbose=False, clip_grad_norm=clip)  # fmt: skip
    train(m2, w, data, steps, sched=s2, sched_every=sched_every)
    return m1, m2, w, events(tmp_path / "r"), agent


def test_prompts3_the_example_says_when_it_cannot_reproduce_the_starting_run(tmp_path):
    def sgd(m):
        o = torch.optim.SGD(m.parameters(), lr=0.1, momentum=0.9)
        return o, torch.optim.lr_scheduler.StepLR(o, step_size=1000)

    _, _, w, ev, agent = example_run(tmp_path / "sgd", sgd)
    assert "load_failed" in [e["kind"] for e in ev] and w.version == 0
    assert any("this example rebuilds torch.optim.AdamW" in msg and "uses SGD" in msg for msg in agent.calls)

    def per_epoch(m):  # 20 updates per epoch; the cosine is stepped once per epoch
        o = torch.optim.AdamW(m.parameters(), lr=3e-3)
        return o, torch.optim.lr_scheduler.CosineAnnealingLR(o, T_max=10)

    _, _, w, ev, agent = example_run(tmp_path / "epoch", per_epoch, sched_every=20)
    assert "load_failed" in [e["kind"] for e in ev]
    assert any("not once per update: write its schedule in starting_multipliers()" in msg for msg in agent.calls)


def test_prompts4_the_example_keeps_the_starting_programs_clipping(tmp_path):
    def adamw(m):
        o = torch.optim.AdamW(m.parameters(), lr=3e-3)
        return o, torch.optim.lr_scheduler.CosineAnnealingLR(o, T_max=200)

    m1, m2, w, ev, _ = example_run(tmp_path, adamw, clip=0.05)
    assert w.version == 1 and w.updater.clip == 0.05
    assert torch.allclose(flat(m1), flat(m2), rtol=1e-5, atol=1e-6), (flat(m1) - flat(m2)).abs().max()


def test_goldfish_documents_a_supported_minimum_torch():
    import re
    from importlib.metadata import metadata
    from pathlib import Path
    from packaging.requirements import Requirement

    text = (Path(opal.__file__).resolve().parents[1] / "examples/goldfish_swahili/README.md").read_text()
    version = re.search(r"PyTorch ≥ (\d+\.\d+)", text).group(1)
    requirement = next(Requirement(x) for x in metadata("opal-optimizer").get_all("Requires-Dist")
                       if Requirement(x).name == "torch")
    assert requirement.specifier.contains(version)
