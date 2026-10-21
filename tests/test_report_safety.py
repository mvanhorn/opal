"""The rank-zero report is optional: filesystem, renderer and thread failures cannot fail training."""

import threading

import pytest
import torch

import opal
from opal import report
from test_core import adamw, flat, make, train


def stop(live):
    thread = live.thread
    live.finish(timeout=1)
    if thread is not None:
        thread.join(timeout=2)
        assert not thread.is_alive()


@pytest.mark.parametrize("contents", [None, "", '{"kind":', 'null\n[]\n{"kind":"start"}\n{"kind":'])
def test_bad_or_partial_run_never_escapes_live_report(tmp_path, contents, monkeypatch):
    errors, uncaught = [], []
    monkeypatch.setattr(threading, "excepthook", uncaught.append)
    if contents is not None:
        (tmp_path / "events.jsonl").write_text(contents)
    live = report.LiveReport(tmp_path, log=errors.append, idle=0.01)
    live.refresh()
    stop(live)
    assert not uncaught
    assert all("training is not affected" in line and "\n" not in line for line in errors)


def test_renderer_exception_does_not_change_training(tmp_path, monkeypatch):
    def broken(*args):
        raise RuntimeError("bad report\nsecond line")

    monkeypatch.setattr(report, "write_report", broken)
    steps = 40
    plain, data = make()
    opt, sched = adamw(plain, steps)
    train(plain, opt, data, steps, sched=sched, pass_loss=False)
    wrapped, _ = make()
    opt, sched = adamw(wrapped, steps)
    w = opal.wrap(
        opt,
        wrapped,
        scheduler=sched,
        total_steps=steps,
        goal="g",
        backend="keep",
        decisions=2,
        every=10,
        run_dir=tmp_path,
        verbose=False,
    )
    lines = []
    w._live.log = lines.append
    train(wrapped, w, data, steps, sched=sched)
    w.close()
    assert torch.equal(flat(plain), flat(wrapped))
    assert len(lines) == 1 and "RuntimeError: bad report second line" in lines[0]
    assert "training is not affected" in lines[0]


def test_thread_infrastructure_summary_and_log_exceptions_are_contained(tmp_path, monkeypatch):
    (tmp_path / "events.jsonl").write_text('{"kind":"start"}\n')
    lines, uncaught = [], []
    monkeypatch.setattr(threading, "excepthook", uncaught.append)
    live = report.LiveReport(tmp_path, summarizer=object(), log=lines.append, idle=0.01)

    def broken(*args):
        raise RuntimeError("injected failure")

    monkeypatch.setattr(live, "_summary", broken)
    monkeypatch.setattr(live, "_write", broken)  # outside write_report's own guard
    live.summarize(1, "reason", {}, {})
    stop(live)
    assert not uncaught
    assert {"summary", "report"} == {"summary" if "the summary" in line else "report" for line in lines}
    live.log = broken
    live._warned.clear()
    live.refresh()
    stop(live)
    assert not uncaught


def test_thread_start_failure_never_reaches_training(tmp_path, monkeypatch):
    lines = []
    live = report.LiveReport(tmp_path, log=lines.append, idle=0.01)

    def broken(*args):
        raise RuntimeError("cannot start thread")

    with monkeypatch.context() as m:
        m.setattr(threading.Thread, "start", broken)
        live.refresh()
        live.finish(timeout=0.01)
    assert len(lines) == 1 and live.thread is None
    # A later refresh can recover once the thread failure is gone.
    live.refresh()
    stop(live)
