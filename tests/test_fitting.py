"""Fitting OPAL into a training script: clipping moved into the program, close() and the report, the starting
program's description, no profiler warning in the training log."""

import shutil
import warnings

import torch

import opal
from test_core import adamw, flat, make


def test_clip_grad_norm_in_the_program_equals_clipping_in_the_loop(tmp_path):
    steps = 80
    m1, data = make()
    o1, s1 = adamw(m1, steps)
    for i in range(steps):
        x, y = data[i % len(data)]
        o1.zero_grad()
        torch.nn.functional.mse_loss(m1(x), y).backward()
        torch.nn.utils.clip_grad_norm_(m1.parameters(), 0.3)
        o1.step()
        s1.step()
    m2, _ = make()
    o2, s2 = adamw(m2, steps)
    w = opal.wrap(
        o2, m2, scheduler=s2, total_steps=steps, goal="t", backend=None, clip_grad_norm=0.3, run_dir=tmp_path / "r",
        verbose=False,
    )  # fmt: skip
    for i in range(steps):
        x, y = data[i % len(data)]
        loss = torch.nn.functional.mse_loss(m2(x), y)
        w.zero_grad()
        loss.backward()
        w.step(loss=loss)
        s2.step()
    assert torch.equal(flat(m1), flat(m2))
    assert "CLIP_GRAD_NORM = 0.3" in w.files["updater.py"]


def test_report_is_written_when_close_returns_and_no_profiler_warning(tmp_path):
    m, data = make()
    o, s = adamw(m, 30)
    sched = torch.optim.lr_scheduler.LambdaLR(o, lambda step: min(1.0, (step + 1) / 10))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        w = opal.wrap(
            o, m, scheduler=sched, total_steps=30, goal="t", backend="keep", every=10, run_dir=tmp_path / "r",
            verbose=False,
        )  # fmt: skip
        for i in range(30):
            x, y = data[i % len(data)]
            loss = torch.nn.functional.mse_loss(m(x), y)
            w.zero_grad()
            loss.backward()
            w.step(loss=loss)
            sched.step()
        w.close()
    assert (tmp_path / "r" / "report.html").exists()
    shutil.rmtree(tmp_path / "r")  # nothing writes into the folder after close()
    assert not any("Profiler clears events" in str(c.message) for c in caught)
    assert "lr_lambdas[0] (source)" in w._starting_files["updater.py"] and "min(1.0" in w._starting_files["updater.py"]
