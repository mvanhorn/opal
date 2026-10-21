"""A resumed report must exclude decisions from the discarded checkpoint suffix."""

import copy

import pytest

import opal
from opal.report import _jsonl, decision_counts, decision_rows
from test_core import adamw, make, train
from test_review1 import scripted
from test_recovery import program


@pytest.mark.parametrize("legacy", [False, True])
def test_resume_archives_decision_events_with_the_lost_attempt(tmp_path, legacy):
    run = tmp_path / "run"
    model, data = make()
    raw, scheduler = adamw(model, 300)
    opt = opal.wrap(
        raw, model, scheduler=scheduler, total_steps=300, goal="g", every=10,
        decisions=6, schedule="uniform", backend=scripted({"updater.py": program()}),
        run_dir=run, verbose=False,
    )
    train(model, opt, data, 60, sched=scheduler)
    checkpoint = copy.deepcopy(opt.state_dict())
    if legacy:
        checkpoint.pop("event_offset", None)
    train(model, opt, data, 200, sched=scheduler, start=60)
    opt.close(interrupted=True)
    assert decision_counts(_jsonl(run / "events.jsonl"))["asked"] == 4
    resumed_model, _ = make()
    raw2, scheduler2 = adamw(resumed_model, 300)
    resumed = opal.wrap(
        raw2, resumed_model, scheduler=scheduler2, total_steps=300, goal="g", every=10,
        decisions=6, schedule="uniform", backend="keep", run_dir=run, verbose=False,
    )
    try:
        resumed.load_state_dict(checkpoint)
        assert decision_counts(_jsonl(run / "events.jsonl"))["asked"] == checkpoint["decisions"]
        assert len(decision_rows(run)) == 1
        archived = list((run / "attempts").glob("*/events.jsonl"))
        assert archived and any(e["kind"] == "requested" for e in _jsonl(archived[0]))
    finally:
        resumed.close(interrupted=True)
