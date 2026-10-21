"""A run resumed from a checkpoint continues in the checkpoint's run folder only when that folder is the same run,
interrupted: in this run's opal_runs/ and not finished. A checkpoint copied from a finished run, or from another place,
starts the resumed run in its own folder, and the original folder is left as it was."""

import json
import shutil

import torch

import opal
from test_core import adamw, make, train


def keep_run(steps=120):
    m, data = make()
    o, s = adamw(m, steps)
    w = opal.wrap(o, m, scheduler=s, total_steps=steps, goal="g", backend="keep", every=10, verbose=False)
    return m, data, s, w


def snapshot(folder):
    return {p.relative_to(folder).as_posix(): p.read_bytes() for p in sorted(folder.rglob("*")) if p.is_file()}


def test_a_copy_of_a_finished_runs_checkpoint_writes_to_its_own_folder(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    m, data, s, w = keep_run()
    train(m, w, data, 60, sched=s)
    torch.save(dict(model=m.state_dict(), opt=w.state_dict(), sched=s.state_dict()), tmp_path / "ck.pt")
    train(m, w, data, 120, sched=s, start=60)  # the run finishes
    first = w.run.dir
    before = snapshot(first)
    shutil.copy(tmp_path / "ck.pt", tmp_path / "copy.pt")
    ck = torch.load(tmp_path / "copy.pt", weights_only=False)
    m2, data2, s2, w2 = keep_run()
    m2.load_state_dict(ck["model"])
    s2.load_state_dict(ck["sched"])
    w2.load_state_dict(ck["opt"])
    train(m2, w2, data2, 120, sched=s2, start=60)
    assert w2.run.dir != first and w2.run.dir.parent == first.parent
    assert snapshot(first) == before and not (first / "attempts").exists()  # the finished run is untouched
    assert [json.loads(x)["kind"] for x in (w2.run.dir / "events.jsonl").read_text().splitlines()][-1] == "end"
    assert torch.equal(torch.cat([p.flatten() for p in m.parameters()]), torch.cat([p.flatten() for p in m2.parameters()]))


def test_an_interrupted_run_still_continues_in_its_folder(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    m, data, s, w = keep_run()
    train(m, w, data, 60, sched=s)
    state = w.state_dict()
    train(m, w, data, 80, sched=s, start=60)
    w.close(interrupted=True)  # a job's time limit: the same run resumes
    m2, data2, s2, w2 = keep_run()
    w2.load_state_dict(state)
    assert w2.run.dir == w.run.dir


def test_a_checkpoint_from_another_place_writes_to_its_own_folder(tmp_path, monkeypatch):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    monkeypatch.chdir(tmp_path / "a")
    m, data, s, w = keep_run()
    train(m, w, data, 60, sched=s)
    state = w.state_dict()
    w.close(interrupted=True)
    first = w.run.dir
    before = snapshot(first)
    monkeypatch.chdir(tmp_path / "b")
    m2, data2, s2, w2 = keep_run()
    w2.load_state_dict(state)
    assert w2.run.dir.parent == (tmp_path / "b" / "opal_runs").resolve() and snapshot(first) == before
    w2.close()


def test_four_digit_programs_and_decisions_are_archived_and_reported(tmp_path, monkeypatch):
    from opal.report import _versions

    monkeypatch.chdir(tmp_path)
    m, data, s, w = keep_run()
    try:
        for version in (999, 1000):
            folder = w.run.programs / f"v{version:03d}"
            folder.mkdir()
            (folder / "updater.py").write_text(f"# version {version}\n")
        (w.run.decisions / "1000").mkdir()
        assert "v1000" in _versions(w.run.dir)
        lost = w._archive_lost_attempt(999, 999, 0)
        assert (lost / "programs" / "v1000").is_dir()
        assert (lost / "decisions" / "1000").is_dir()
        assert (w.run.programs / "v999").is_dir()
    finally:
        w.close(interrupted=True)


def test_final_program_without_a_marker_sorts_versions_numerically(tmp_path):
    for version in (999, 1000):
        folder = tmp_path / "programs" / f"v{version:03d}"
        folder.mkdir(parents=True)
        (folder / "updater.py").write_text(f"# version {version}\n")
    assert opal.final_program(tmp_path).name == "v1000"
