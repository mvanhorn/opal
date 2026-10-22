"""OPAL 0.1.2-aggressive (prompt="aggressive"): the neutral prompt with the edits listed in opal/misc/aggressive.py; the decision message's last sentences; the example updater's learning-rate controller, off by default."""

import ast
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import opal
from opal.misc import aggressive
from test_core import adamw, flat, make, train
from test_example_updater import ALLOWED_NUMBERS
from test_review1 import scripted

PROMPTS = Path(opal.__file__).parent / "prompts"


def run(tmp_path, agent, steps=200, **kw):
    m, data = make()
    o, s = adamw(m, steps)
    kw = dict(dict(decisions=4, every=10, verbose=False), **kw)
    w = opal.wrap(o, m, scheduler=s, total_steps=steps, goal="t", run_dir=tmp_path / "r", backend=agent, **kw)
    train(m, w, data, steps, sched=s)
    return w, m


@pytest.mark.parametrize("validation", [None, 25])
def test_the_aggressive_prompt_is_the_neutral_one_with_only_the_approved_edits(tmp_path, validation):
    kw = dict(validation=validation) if validation else {}
    neutral = scripted()
    w_n, _ = run(tmp_path / "n", neutral, prompt="neutral", **kw)
    bold = scripted()
    w_a, _ = run(tmp_path / "a", bold, prompt="aggressive", **kw)
    expected = w_n.worker.system
    for old, new in aggressive.PROMPT_EDITS:
        assert expected.count(old) == 1
        expected = expected.replace(old, new)
    assert w_a.worker.system == expected
    # the option off: the neutral prompt and its messages are as before
    assert "Then review, for\neach parameter group" not in w_n.worker.system
    assert all("Edit the programs in place, or keep them (a valid decision)." in m for m in neutral.calls)
    assert all("the four parts of DECISIONS: observed, changed or kept, prediction, rules." in m for m in neutral.calls)
    for m in bold.calls:
        assert "keep them" not in m and m.endswith(
            "Edit the programs in place. End with your rationale in the five parts of DECISIONS: observed, review, "
            "changed, prediction, rules."
        )
    # everything else of the run is the neutral one's: starting analysis, next-decision line, reference files
    assert w_a.files == w_n.files and all("The next scheduled decision" in m for m in bold.calls[:2])
    keep = ["TASK", "Explore enough to learn what", "Complexity or novelty by itself is not the objective."]
    assert all(s in w_a.worker.system for s in keep)


def example_text(controller=None, decay=None, rule=None):
    text = aggressive.example((PROMPTS / "example_updater.py").read_text())
    if controller is not None:
        text = text.replace("LR_CONTROLLER = False", f"LR_CONTROLLER = {controller}")
        text = text.replace("SIGNAL_DECAY = None", f"SIGNAL_DECAY = {decay}")
    if rule is not None:
        text = text.replace("    return multiplier\n", rule, 1)
    return text


def test_the_aggressive_example_has_no_numbers_and_is_the_one_in_reference(tmp_path):
    text = example_text()
    found = {
        n.value
        for n in ast.walk(ast.parse(text))
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool)
    }
    assert found <= ALLOWED_NUMBERS, found
    w, _ = run(tmp_path, scripted(), prompt="aggressive", steps=60)
    assert (tmp_path / "r" / "workspace" / "reference" / "example_updater.py").read_text() == text
    w2, _ = run(tmp_path / "n", scripted(), prompt="neutral", steps=60)
    neutral_ref = (tmp_path / "n" / "r" / "workspace" / "reference" / "example_updater.py").read_text()
    assert neutral_ref == (PROMPTS / "example_updater.py").read_text()  # the option off: the example as it was


@pytest.mark.parametrize(
    "variant",
    [
        dict(),  # the controller off: the starting run
        dict(controller=True, decay=0.5),  # on, with the rule as written (it changes nothing)
    ],
    ids=["off", "on-identity"],
)
def test_the_aggressive_example_with_its_defaults_reproduces_the_starting_run(tmp_path, variant):
    steps = 200
    m1, data = make()
    o1, s1 = adamw(m1, steps)
    train(m1, o1, data, steps, sched=s1, pass_loss=False)
    w, m2 = run(tmp_path, scripted({"updater.py": example_text(**variant)}), steps=steps, prompt="aggressive")
    assert w.version == 1 and type(w.updater).__name__ == "Updater"
    assert (w.updater.controller is not None) == bool(variant)
    assert torch.allclose(flat(m1), flat(m2), rtol=1e-5, atol=1e-6), (flat(m1) - flat(m2)).abs().max()


def test_the_controller_acts_through_the_rule_and_its_state_is_checkpointed(tmp_path):
    frozen = example_text(controller=True, decay=0.5, rule="    return torch.zeros_like(multiplier)\n")
    seen = {}

    def agent(ws, msg, sy):
        if "Decision 1." in msg:
            (ws / "updater.py").write_text(frozen)
        return "x"

    m, data = make()
    o, s = adamw(m, 200)
    w = opal.wrap(o, m, scheduler=s, total_steps=200, goal="t", run_dir=tmp_path / "r", backend=agent, decisions=4,
                     every=10, prompt="aggressive", verbose=False)  # fmt: skip
    for i in range(200):
        x, y = data[i % len(data)]
        w.zero_grad()
        loss = torch.nn.functional.mse_loss(m(x), y)
        loss.backward()
        before = flat(m).clone()
        w.step(loss=loss)
        s.step()
        if w.version == 1:
            assert torch.equal(flat(m), before)  # including the update that installs the controller
        if w.version == 1 and "at" not in seen:
            seen["at"], seen["w"] = i, flat(m).clone()
    assert seen and torch.equal(flat(m), seen["w"])  # a zero multiplier: no update after the first one
    state = w.state_dict()["updater"]["lr_controller"]
    assert set(state) == {"signal", "mult"} and all(float(v) > 0 for v in state["signal"].values())
    assert all(float(v) == 0 for v in state["mult"].values())


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("multiplier", [0, 1])
def test_controller_preserves_exact_endpoints_despite_cancellation(dtype, multiplier):
    namespace = {"torch": torch, "SIGNAL_DECAY": 0.5}
    exec(aggressive.CONTROLLER, namespace)
    namespace["lr_rule"] = lambda signal, mult, update, total: torch.full_like(mult, multiplier)
    # The small value is lost when subtracting it from one; subtracting the delta
    # back from the updated parameter cannot recover the original bits.
    small = torch.finfo(dtype).eps / 8
    original = torch.tensor([small, 1, -small], dtype=dtype)
    proposed = torch.tensor([1, small, -1], dtype=dtype)
    parameter = original.clone()
    rule = SimpleNamespace(groups=[dict(key="g", params=[parameter])])
    controller = namespace["LRController"](rule, {})
    before = controller.before()
    parameter.copy_(proposed)
    controller.after(before, update=1, total=2)
    assert torch.equal(parameter, original if multiplier == 0 else proposed)
