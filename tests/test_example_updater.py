"""reference/example_updater.py (the neutral prompt's example): structure only. With its defaults it reproduces the
starting run; moving a tensor between rules keeps its state; it holds no numbers an agent could copy."""

import ast
import types
from pathlib import Path

import torch

import opal
from test_core import flat, make, train

EXAMPLE = Path(opal.__file__).parent / "prompts" / "example_updater.py"
ALLOWED_NUMBERS = {0, 1, 2}  # 0.0 and 1.0 compare equal to 0 and 1; -1 is the unary minus of 1


def setup(model, steps, schedule):
    decay = [p for n, p in model.named_parameters() if n.endswith("weight")]
    rest = [p for n, p in model.named_parameters() if not n.endswith("weight")]
    opt = torch.optim.AdamW([dict(params=decay, weight_decay=0.1), dict(params=rest, weight_decay=0.0)], lr=3e-3)
    if schedule == "lambda":  # warmup, then a linear decay
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 20) * max(0.0, (steps - s) / steps))
    else:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    return opt, sched


def test_example_defaults_reproduce_the_starting_run(tmp_path):
    steps = 300
    for schedule in ("lambda", "cosine"):
        m1, data = make()
        o1, s1 = setup(m1, steps, schedule)
        train(m1, o1, data, steps, sched=s1, pass_loss=False)
        m2, _ = make()
        o2, s2 = setup(m2, steps, schedule)
        calls = []

        def agent(ws, msg, sy):  # installs the example, unchanged, at the first decision
            calls.append(msg)
            if len(calls) == 1:
                (ws / "updater.py").write_text(EXAMPLE.read_text())
            return "x"

        w = opal.wrap(
            o2,
            m2,
            scheduler=s2,
            total_steps=steps,
            goal="g",
            run_dir=tmp_path / schedule,
            backend=agent,
            decisions=2,
            every=10,
            prompt="neutral",
            verbose=False,
        )
        train(m2, w, data, steps, sched=s2)
        assert w.version == 1 and type(w.updater).__name__ == "Updater"
        assert torch.allclose(flat(m1), flat(m2), rtol=1e-5, atol=1e-6), (schedule, (flat(m1) - flat(m2)).abs().max())
        # the training script's optimizer keeps its state, which a fallback to the starting program needs
        assert len(o2.state) == len(list(m2.parameters())) and w.updater.rules["adamw"].opt is not o2


def test_moving_a_tensor_between_rules_keeps_its_state():
    ns = {}
    exec(compile(EXAMPLE.read_text(), "example", "exec"), ns)
    m, _ = make()
    named = list(m.named_parameters())
    spec = dict(total_updates=100, user_optimizers=[torch.optim.AdamW(m.parameters(), lr=3e-3)], user_schedulers=[])
    u1 = ns["build"](named, None, spec)
    m(torch.randn(2, 8)).sum().backward()
    u1.step(None, 1, types.SimpleNamespace(loss=None, update=1))
    before = u1.state_dict()["adamw"]["0.weight"]
    ns["RULE_OF"]["0.weight"] = "yours"
    u2 = ns["build"](named, u1, spec)
    moved = u2.state_dict()["yours"]["0.weight"]
    assert "0.weight" not in u2.state_dict()["adamw"]
    assert (
        torch.equal(moved["exp_avg"], before["exp_avg"]) and moved["exp_avg"].data_ptr() != before["exp_avg"].data_ptr()
    )
    del ns["RULE_OF"]["0.weight"]
    u3 = ns["build"](named, u2, spec)
    assert torch.equal(u3.state_dict()["adamw"]["0.weight"]["exp_avg_sq"], before["exp_avg_sq"])


def test_example_has_no_numbers_to_copy():
    found = sorted(
        {
            node.value
            for node in ast.walk(ast.parse(EXAMPLE.read_text()))
            if isinstance(node, ast.Constant)
            and isinstance(node.value, (int, float))
            and not isinstance(node.value, bool)
        }
    )
    assert set(found) <= ALLOWED_NUMBERS, found


def test_neutral_with_the_example_in_reference_and_keep_is_bitwise(tmp_path):
    from test_review2 import plain_run

    m, data = make()
    o = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=0.1)
    s = torch.optim.lr_scheduler.CosineAnnealingLR(o, T_max=120)
    w = opal.wrap(
        o,
        m,
        scheduler=s,
        total_steps=120,
        goal="g",
        run_dir=tmp_path / "r",
        backend="keep",
        prompt="neutral",
        verbose=False,
    )
    train(m, w, data, 120, sched=s)
    assert (tmp_path / "r" / "workspace" / "reference" / "example_updater.py").exists()
    assert torch.equal(flat(m), plain_run(120))
