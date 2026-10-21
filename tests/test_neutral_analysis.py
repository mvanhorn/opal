"""The neutral prompt's starting analysis: raw loss-trend, update-to-weight, norm and cosine fields; the
one-epoch-earlier comparison only when epochs are given and the data repeat in order; a bounded snapshot."""

import json

import torch

import opal
from test_core import adamw, flat, make, train
from test_review2 import plain_run

FIELDS = {
    "loss_window_means",
    "loss_slope_per_1000_updates",
    "loss_std",
    "grad_norm_total",
    "grad_norm_by_role",
    "weight_norm_by_role",
    "grad_momentum_cosine_by_role",
    "update_to_weight_by_role",
    "update_to_weight_per_update_by_role",
    "grad_norm_by_block",
}


def run(tmp_path, steps=200, **kw):
    m, data = make()
    o, s = adamw(m, steps)
    w = opal.wrap(
        o,
        m,
        scheduler=s,
        total_steps=steps,
        goal="g",
        run_dir=tmp_path / "r",
        backend="keep",
        prompt="neutral",
        decisions=2,
        every=10,
        verbose=False,
        **kw,
    )
    train(m, w, data, steps, sched=s)
    reports = [json.loads(x)["report"] for x in open(tmp_path / "r" / "reports.jsonl")]
    return m, w, reports


def test_new_fields_bounded_snapshot_and_bitwise(tmp_path):
    m, w, reports = run(tmp_path)
    assert FIELDS <= set(reports[-1]) and "loss_minus_one_epoch_earlier" not in reports[-1]
    assert len(reports[-1]["loss_window_means"]) == 4 and isinstance(reports[-1]["grad_norm_by_block"], list)
    roles = set(reports[-1]["grad_norm_by_role"])
    assert set(reports[-1]["update_to_weight_by_role"]) == roles
    snap = w.memory["snapshot"]
    assert len(snap) == len(roles) and all(torch.is_tensor(t) for t in snap.values())  # one tensor per role
    assert torch.equal(flat(m), plain_run(200))  # the analysis never changes training


def test_epoch_comparison_only_with_epochs_and_the_same_order(tmp_path):
    _, _, reports = run(tmp_path / "a", epochs=4, same_order_each_epoch=True)
    late = [r for r in reports if r["update"] > 60]
    assert late and all("loss_minus_one_epoch_earlier" in r for r in late)
    assert len(run(tmp_path / "b")[1].memory.get("epoch_losses", [])) == 0
    _, w, reports = run(tmp_path / "c", epochs=4, same_order_each_epoch=False)
    assert not any("loss_minus_one_epoch_earlier" in r for r in reports)
    _, w, _ = run(tmp_path / "d", epochs=4, same_order_each_epoch=True)
    assert len(w.memory["epoch_losses"]) <= 50 + 10  # one epoch (200 / 4 updates) plus one window


def test_notes_md_persists_and_is_not_a_program_edit(tmp_path):
    seen, msgs = [], []

    def agent(ws, msg, sys_):
        msgs.append(msg)
        seen.append((ws / "NOTES.md").read_text() if (ws / "NOTES.md").exists() else None)
        (ws / "NOTES.md").write_text(f"decision {len(msgs)}: kept\n")
        return "x"

    m, data = make()
    o, s = adamw(m, 200)
    w = opal.wrap(
        o,
        m,
        scheduler=s,
        total_steps=200,
        goal="g",
        run_dir=tmp_path / "r",
        backend=agent,
        prompt="neutral",
        decisions=3,
        every=10,
        blocking=True,
        verbose=False,
    )
    train(m, w, data, 200, sched=s)
    assert seen[0] is None and seen[1] == "decision 1: kept\n"  # persists across decisions
    assert w.version == 0 and "NOTES.md" not in w.files  # not part of the program, nothing reloaded
    kinds = [json.loads(x)["kind"] for x in open(tmp_path / "r" / "events.jsonl")]
    assert "applied" not in kinds and "rejected" not in kinds
    assert msgs[0].endswith(
        "End with your rationale in the four parts of DECISIONS: observed, changed or kept, prediction, rules."
    )
