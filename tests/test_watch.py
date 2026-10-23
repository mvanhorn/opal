"""opal watch: print a run's events as they are written."""

import json
from pathlib import Path

import pytest

from opal import cli, watch


def write_jsonl(path, records):
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


def write_run(folder, *, finished=True, extra=(), partial=False):
    folder.mkdir(parents=True, exist_ok=True)
    events = [
        dict(
            kind="start",
            utc="2026-10-21T00:00:00Z",
            goal="A small run",
            total_updates=30,
            schedule=[5, 20],
            blocking=True,
            backend="claude",
        ),
        dict(kind="requested", k=1, update=5, utc="2026-10-21T00:00:01Z"),
        dict(
            kind="applied",
            k=1,
            requested_at=5,
            applied_at=10,
            version=1,
            changed=["updater.py"],
            utc="2026-10-21T00:00:20Z",
        ),
        dict(kind="requested", k=2, update=20, utc="2026-10-21T00:01:00Z"),
        dict(kind="kept", k=2, applied_at=20, utc="2026-10-21T00:01:10Z"),
        *extra,
    ]
    if finished:
        events.append(dict(kind="end", utc="2026-10-21T00:02:00Z", update=30, cost_usd=0.25, decisions=2))
    write_jsonl(folder / "events.jsonl", events)
    if partial:
        with open(folder / "events.jsonl", "a") as f:
            f.write('{"kind": "requested", "k": 3, "upd')
    rec = folder / "decisions" / "001"
    rec.mkdir(parents=True)
    (rec / "response.json").write_text(
        json.dumps(dict(text="I lowered the rate because the loss plateaued.", cost_usd=0.25, seconds=15))
    )
    return folder


def test_watch_prints_a_finished_run_and_returns(tmp_path, capsys):
    run = write_run(tmp_path / "r")
    assert cli.main(["watch", str(run), "--timeout", "0"]) == 0
    out = capsys.readouterr().out
    assert f"watching {run.resolve()}" in out
    assert "start: 30 updates; backend claude; blocking" in out
    assert "goal: A small run" in out
    assert "requested  k=1  at update 5" in out
    assert "changed    k=1  at update 10" in out
    assert "$0.25" in out and "lowered the rate" in out
    assert "kept       k=2  at update 20" in out
    assert "finished at update 30; 2 decisions; $0.25" in out
    assert "still running" not in out


def test_watch_timeout_zero_on_an_unfinished_run_says_still_running(tmp_path, capsys):
    run = write_run(tmp_path / "r", finished=False)
    assert cli.main(["watch", str(run), "--timeout", "0"]) == 0
    out = capsys.readouterr().out
    assert "requested  k=2  at update 20" in out
    assert "finished at update" not in out
    assert "still running at update 20" in out


def test_watch_prints_an_event_written_while_it_waits(tmp_path, capsys):
    run = write_run(tmp_path / "r", finished=False)

    def sleep(_):
        rec = dict(kind="end", utc="2026-10-21T00:02:00Z", update=30, cost_usd=0.1, decisions=2)
        with open(run / "events.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")

    assert watch.follow(run, interval=0.01, sleep=sleep) == 0
    out = capsys.readouterr().out
    assert "finished at update 30; 2 decisions; $0.10" in out
    assert "still running" not in out


def test_watch_skips_a_partial_last_line(tmp_path, capsys):
    run = write_run(tmp_path / "r", partial=True)
    assert cli.main(["watch", str(run), "--timeout", "0"]) == 0
    out = capsys.readouterr().out
    assert "k=3" not in out
    assert "finished at update 30" in out


def test_watch_estimate_and_revert_lines(tmp_path, capsys):
    extra = [
        dict(kind="estimate", update=10, decisions=1, capped_from=38, utc="2026-10-21T00:00:15Z"),
        dict(kind="reverted", update=12, version=1, utc="2026-10-21T00:00:16Z"),
    ]
    run = write_run(tmp_path / "r", extra=extra)
    assert cli.main(["watch", str(run), "--timeout", "0"]) == 0
    out = capsys.readouterr().out
    assert "estimate at update 10: 1 scheduled; capped from 38" in out
    assert "reverted at update 12" in out


def test_watch_missing_folder_exits(tmp_path):
    with pytest.raises(SystemExit) as exited:
        cli.main(["watch", str(tmp_path / "missing"), "--timeout", "0"])
    assert "not a run folder" in str(exited.value)


def test_watch_rejects_a_non_positive_interval(capsys):
    with pytest.raises(SystemExit) as exited:
        cli.main(["watch", "r", "--interval", "0", "--timeout", "0"])
    assert exited.value.code == 2
    assert "interval" in capsys.readouterr().err


def test_the_guide_and_readme_name_watch():
    root = Path(__file__).resolve().parents[1]
    assert "`opal watch <folder>`" in (root / "docs" / "GUIDE.md").read_text()
    assert "opal watch   opal_runs/<folder>" in (root / "README.md").read_text()
    assert "opal watch <run>" in (root / "skills" / "train-with-agent" / "SKILL.md").read_text()
