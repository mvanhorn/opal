"""opal compare: two run folders, or two result.json files, side by side."""

import json
from pathlib import Path

import pytest

from opal import cli, compare


def write_jsonl(path, records):
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


def write_run(
    folder,
    *,
    goal="A small run",
    backend="claude",
    blocking=True,
    outcome="applied",
    k=1,
    update=5,
    cost=0.25,
    wait=12.0,
    loss=1.0,
    result=None,
    extra_events=(),
    extra_requests=(),
):
    folder.mkdir(parents=True, exist_ok=True)
    events = [
        dict(
            kind="start",
            utc="2026-10-21T00:00:00Z",
            goal=goal,
            total_updates=30,
            schedule=[update],
            blocking=blocking,
            backend=backend,
        ),
        dict(kind="requested", k=k, update=update, utc="2026-10-21T00:00:01Z"),
        dict(
            kind=outcome,
            k=k,
            requested_at=update,
            applied_at=update + 5,
            version=1 if outcome == "applied" else 0,
            **(dict(changed=["updater.py"]) if outcome == "applied" else {}),
        ),
        *extra_requests,
        *extra_events,
        dict(
            kind="end",
            utc="2026-10-21T00:02:00Z",
            update=30,
            cost_usd=cost,
            agent_wait_seconds=wait,
        ),
    ]
    write_jsonl(folder / "events.jsonl", events)
    write_jsonl(folder / "losses.jsonl", [dict(from_update=1, to_update=3, losses=[3.0, 2.0, loss])])
    rec = folder / "decisions" / f"{k:03d}"
    rec.mkdir(parents=True)
    (rec / "response.json").write_text(json.dumps(dict(text="changed the rate", cost_usd=cost, seconds=15)))
    if result is not None:
        (folder / "result.json").write_text(json.dumps(result))
    return folder


def result_record(**kw):
    rec = dict(
        metric=dict(name="FID-50K", value=3.19, direction="lower"),
        train_seconds=13500.0,
        agent_wait_seconds=1764.0,
        agent_cost_usd=13.30,
        decisions=dict(asked=50, applied=6, failed=0),
        hardware="8 x H100",
        opal_commit="abc123",
        command="python train.py --opal",
        opal_run=None,
    )
    rec.update(kw)
    return rec


def test_compare_marks_a_decision_whose_outcome_differs(tmp_path, capsys):
    a = write_run(tmp_path / "a", outcome="applied", loss=0.25)
    b = write_run(tmp_path / "b", outcome="kept", loss=0.30, cost=0.0, wait=0.0, backend="keep")
    assert cli.main(["compare", str(a), str(b)]) == 0
    out = capsys.readouterr().out
    assert "a: " in out and "b: " in out
    assert "* backend" in out and "claude" in out and "keep" in out
    assert "decisions that differ:" in out
    assert "changed" in out and "kept" in out
    assert "last loss" in out and "0.25" in out


def test_compare_says_when_every_decision_matches(tmp_path, capsys):
    a = write_run(tmp_path / "a")
    b = write_run(tmp_path / "b", goal="A small run")
    assert cli.main(["compare", str(a), str(b)]) == 0
    out = capsys.readouterr().out
    assert "all 1 asked decisions have the same outcome." in out
    assert "decisions that differ:" not in out


def test_compare_json_lists_differing_decisions(tmp_path, capsys):
    a = write_run(tmp_path / "a", outcome="applied")
    b = write_run(tmp_path / "b", outcome="kept")
    assert cli.main(["compare", "--json", str(a), str(b)]) == 0
    obj = json.loads(capsys.readouterr().out)
    assert obj["a"]["kind"] == "run" and obj["b"]["kind"] == "run"
    assert obj["differing_decisions"] == [dict(k=1, a="changed", b="kept", asked_at=5)]
    assert obj["a"]["last_loss"] == 1.0 and obj["a"]["cost_usd"] == 0.25


def test_compare_result_json_files_print_the_metric_and_skip_decisions(tmp_path, capsys):
    fa = tmp_path / "official.json"
    fb = tmp_path / "opal.json"
    official = result_record(metric=dict(name="FID-50K", value=3.39, direction="lower"))
    official.update(agent_cost_usd=None, decisions=None)
    fa.write_text(json.dumps(official))
    fb.write_text(json.dumps(result_record()))
    assert cli.main(["compare", str(fa), str(fb)]) == 0
    out = capsys.readouterr().out
    assert "FID-50K 3.39 (lower is better)" in out
    assert "FID-50K 3.19 (lower is better)" in out
    assert "* metric" in out
    assert "decision-by-decision comparison needs two run folders" in out
    assert "python train.py --opal" in out


def test_compare_uses_result_json_inside_a_run_folder(tmp_path, capsys):
    rec = result_record(metric=dict(name="test accuracy", value=0.9456, direction="higher"))
    other = result_record(metric=dict(name="test accuracy", value=0.9449, direction="higher"))
    a = write_run(tmp_path / "a", result=rec, loss=0.2512)
    b = write_run(tmp_path / "b", result=other, loss=0.2576)
    assert cli.main(["compare", str(a), str(b)]) == 0
    out = capsys.readouterr().out
    assert "test accuracy 0.9456 (higher is better)" in out
    assert "test accuracy 0.9449 (higher is better)" in out


def test_compare_result_json_points_at_its_run_folder(tmp_path, capsys):
    a = write_run(tmp_path / "a", outcome="applied")
    b = write_run(tmp_path / "b", outcome="kept")
    fa = tmp_path / "a.json"
    fa.write_text(json.dumps(result_record(opal_run=str(a), metric=dict(name="loss", value=0.25, direction="lower"))))
    assert cli.main(["compare", str(fa), str(b)]) == 0
    out = capsys.readouterr().out
    assert "decisions that differ:" in out
    assert "changed" in out and "kept" in out


def test_compare_a_lost_pending_decision_is_not_the_same_as_kept(tmp_path, capsys):
    a = write_run(tmp_path / "a", outcome="kept")
    b = tmp_path / "b"
    b.mkdir()
    write_jsonl(
        b / "events.jsonl",
        [
            dict(kind="start", utc="2026-10-21T00:00:00Z", goal="x", total_updates=30, schedule=[5], blocking=True),
            dict(kind="requested", k=1, update=5, utc="2026-10-21T00:00:01Z"),
            dict(kind="end", utc="2026-10-21T00:02:00Z", update=30, cost_usd=0),
        ],
    )
    assert cli.main(["compare", str(a), str(b)]) == 0
    out = capsys.readouterr().out
    assert "kept" in out and "lost" in out
    assert "decisions that differ:" in out


def test_compare_missing_path_exits(tmp_path):
    with pytest.raises(SystemExit) as exited:
        cli.main(["compare", str(tmp_path / "missing"), str(tmp_path / "also")])
    assert exited.value.code
    assert "not a run folder" in str(exited.value)


def test_compare_rejects_a_json_file_without_metric(tmp_path):
    bad = tmp_path / "notes.json"
    bad.write_text(json.dumps(dict(hello=1)))
    good = tmp_path / "ok.json"
    good.write_text(json.dumps(result_record()))
    with pytest.raises(SystemExit) as exited:
        cli.main(["compare", str(bad), str(good)])
    assert "not a result.json" in str(exited.value)


def test_compare_a_parent_folder_names_the_runs_inside(tmp_path):
    write_run(tmp_path / "parent" / "r1")
    write_run(tmp_path / "parent" / "r2")
    with pytest.raises(SystemExit) as exited:
        cli.main(["compare", str(tmp_path / "parent"), str(tmp_path / "parent" / "r1")])
    assert "runs inside: r1, r2" in str(exited.value)


def test_the_guide_and_readme_name_compare():
    root = Path(__file__).resolve().parents[1]
    guide = (root / "docs" / "GUIDE.md").read_text()
    readme = (root / "README.md").read_text()
    assert "`opal compare a b`" in guide
    assert "opal compare opal_runs/<a> opal_runs/<b>" in readme
    assert "opal compare" in (root / "skills" / "train-with-agent" / "SKILL.md").read_text()


def test_load_side_reads_a_partial_events_line(tmp_path):
    folder = write_run(tmp_path / "a")
    with open(folder / "events.jsonl", "a") as f:
        f.write('{"kind": "requested", "k": 2, "upd')
    snap = compare.load_side(folder)
    assert [d["k"] for d in snap["decisions"]] == [1]
