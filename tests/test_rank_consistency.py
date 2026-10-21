"""Two-process CPU DDP: trial rejection is an exact optimizer rollback; later repairs stay visible."""

import copy
import json
from contextlib import redirect_stderr
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import opal
from opal.cli import _show
from opal.report import rank_flag, write_report
from test_core import events

LOCAL_STATE = """
import torch
class Updater:
    def __init__(self, named_params, previous=None, spec=None):
        self.named, self.opts, self.calls = named_params, spec['user_optimizers'], 0
        self.buf = [self.opts[0].state.get(p, {}).get('local_momentum', torch.zeros_like(p)).clone()
                    for _, p in named_params]
    def step(self, grads, update):
        self.calls += 1
        for opt in self.opts:
            opt.step()
        if self.calls >= DELAY:
            for (_, p), b in zip(self.named, self.buf):
                p.add_(b, alpha=0.125)
    def state_dict(self):
        return dict(calls=self.calls, buf=self.buf)
    def load_state_dict(self, state):
        self.calls, self.buf = state['calls'], state['buf']
def build(named_params, previous=None, spec=None):
    return Updater(named_params, previous, spec)
"""
CONSISTENT = LOCAL_STATE.replace("self.calls >= DELAY", "False")


def _same(a, b):
    if torch.is_tensor(a):
        return torch.is_tensor(b) and torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_same(v, b[k]) for k, v in a.items())
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


def _state(w):
    return copy.deepcopy(
        dict(
            params=[p.detach().clone() for _, p in w.named],
            grads=[p.grad for _, p in w.named],
            opts=[o.state_dict() for o in w.user_optimizers],
            updater=w.updater.state_dict(),
        )
    )


def _worker(rank, folder, case):
    torch.set_num_threads(1)
    root = Path(folder)
    dist.init_process_group(
        "gloo",
        init_method=(root / "rendezvous").absolute().as_uri(),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    with (root / f"stderr-{rank}.txt").open("w") as stderr, redirect_stderr(stderr):
        torch.manual_seed(42)
        model = torch.nn.Linear(3, 2)
        reference = copy.deepcopy(model)
        ddp = torch.nn.parallel.DistributedDataParallel(model)
        opt = torch.optim.SGD(ddp.parameters(), lr=0.01, momentum=0.9)
        # A shard on each rank, intentionally different optimizer state with identical model parameters.
        for i, p in enumerate(model.parameters()):
            if i == rank:
                opt.state[p]["local_momentum"] = torch.ones_like(p)
        plain = torch.optim.SGD(reference.parameters(), lr=0.01, momentum=0.9)
        plain.load_state_dict(copy.deepcopy(opt.state_dict()))

        def agent(ws, msg, system):
            (root / "next-message.txt").write_text(msg)
            return "keep"

        total = 4 if case in ("last", "consistent_short") else 6 if case == "preexisting" else 12
        periodic = 4 if case in ("late", "periodic_trial") else 0
        limit = 0 if rank == 1 and case in ("no_snapshot", "consistent_no_snapshot") else 2
        w = opal.wrap(
            opt,
            ddp,
            total_steps=total,
            goal="rank consistency test",
            every=10,
            sync_every=20,
            trial_steps=0 if case == "zero" else 4,
            rank_check_every=periodic,
            decisions=0,
            max_decisions=1,
            backend=agent if rank == 0 else None,
            snapshot_limit_gb=limit,
            summarize=False,
            verbose=False,
            run_dir=root / "r",
        )
        start = w.digests[0]
        before, restored, rejected_at, failure = None, False, None, None
        rng = torch.Generator().manual_seed(rank + 100)
        checks = []
        original_check = w._rank_differences

        def counted_check():
            checks.append(w.update)
            return original_check()

        w._rank_differences = counted_check
        for u in range(1, total + 1):
            w.zero_grad()
            loss = ddp(torch.randn(5, 3, generator=rng)).square().mean()
            loss.backward()
            if u == 2:
                before = _state(w)
                if case == "preexisting" and rank == 1:
                    with torch.no_grad():
                        next(model.parameters()).add_(1)
                if case == "analysis":
                    w._apply(
                        dict(
                            k=99,
                            requested_at=1,
                            files=dict(w.files, **{"analysis.py": w.files["analysis.py"] + "\n# analysis-only edit\n"}),
                        ),
                        u,
                    )
                elif case != "keep":
                    delay = 7 if case == "late" else 3 if case in ("endpoint", "periodic_trial", "last") else 1
                    source = (
                        CONSISTENT
                        if case in ("consistent", "consistent_no_snapshot", "consistent_short", "overlap", "fallback")
                        else LOCAL_STATE.replace("DELAY", str(delay))
                    )
                    if case == "build":
                        source = CONSISTENT.replace(
                            "return Updater(named_params, previous, spec)",
                            "u = Updater(named_params, previous, spec)\n    named_params[0][1].data.add_(spec['rank'])\n    return u",
                        )
                    w._apply(dict(k=99, requested_at=1, files=dict(w.files, **{"updater.py": source})), u)
            if case == "overlap" and u == 4:
                source = LOCAL_STATE.replace("DELAY", "3")
                w._apply(dict(k=100, requested_at=3, files=dict(w.files, **{"updater.py": source})), u)
            if case == "fallback" and u == 8:
                before, start = _state(w), w.digests[0]
                source = LOCAL_STATE.replace("DELAY", "1")
                w._apply(dict(k=100, requested_at=7, files=dict(w.files, **{"updater.py": source})), u)
            if case in ("keep", "analysis", "consistent", "consistent_no_snapshot", "consistent_short"):
                for p, q in zip(model.parameters(), reference.parameters()):
                    q.grad = p.grad.detach().clone()
                plain.step()
            try:
                w.step(loss=loss)
            except RuntimeError as exc:
                if case != "no_snapshot":
                    raise
                failure = str(exc)
                break
            if case in ("keep", "analysis", "consistent", "consistent_no_snapshot", "consistent_short"):
                assert all(torch.equal(p, q) for p, q in zip(model.parameters(), reference.parameters()))
                assert _same(opt.state_dict(), plain.state_dict())
            if (
                u >= 2
                and w.digests[0] == start
                and case not in ("keep", "analysis", "consistent", "consistent_no_snapshot", "consistent_short")
            ):
                restored, rejected_at = _same(_state(w), before), u
                # Observe the actual next agent message, through the worker and its normal message formatter.
                if rank == 0:
                    w._submit(u)
                w._blocking_point(u)
                break
        if case == "late" and rank == 0:
            torch.save(w.state_dict(), root / "checkpoint.pt")
        w.close(interrupted=w.update < total)
        snapshot_released = w._rank_snapshot is None
        torch.save(
            dict(
                params=[p.detach() for p in model.parameters()],
                restored=restored,
                rejected_at=rejected_at,
                failure=failure,
                checks=checks,
                snapshot_released=snapshot_released,
                ranks_repaired_at=w._ranks_repaired_at,
            ),
            root / f"rank{rank}.pt",
        )
    dist.destroy_process_group()


def _run(tmp_path, case):
    mp.spawn(_worker, args=(str(tmp_path), case), nprocs=2, join=True)
    return [torch.load(tmp_path / f"rank{r}.pt", weights_only=False) for r in (0, 1)]


@pytest.mark.parametrize(
    ("case", "update"),
    [
        ("immediate", 2),
        ("endpoint", 6),
        ("periodic_trial", 4),
        ("last", 4),
        ("overlap", 8),
        ("build", 2),
        ("zero", 2),
        ("fallback", 8),
    ],
)
def test_divergent_trial_restores_exact_pre_swap_state_and_tells_agent(tmp_path, case, update):
    ranks = _run(tmp_path, case)
    assert all(r["restored"] and r["rejected_at"] == update and r["snapshot_released"] for r in ranks)
    assert _same(ranks[0]["params"], ranks[1]["params"])
    ev = events(tmp_path / "r")
    divergence = [e for e in ev if e["kind"] == "rank_divergence"]
    assert len(divergence) == 1 and divergence[0]["update"] == update and divergence[0]["snapshot_available"]
    assert divergence[0]["names"] == (["weight"] if case == "build" else ["weight", "bias"])
    assert [e["restored"] for e in ev if e["kind"] == "reverted"] == [True]
    assert not any(e["kind"] == "ranks_resynced" for e in ev)
    msg = (tmp_path / "next-message.txt").read_text()
    assert "made the ranks' parameters differ" in msg and "rank-local state" in msg and "weight" in msg
    assert "restored exactly; no update was redone" in msg


@pytest.mark.parametrize("case", ["keep", "analysis", "consistent", "consistent_no_snapshot", "consistent_short"])
def test_consistent_ddp_is_bitwise_identical_to_plain_and_releases_snapshot(tmp_path, case):
    ranks = _run(tmp_path, case)
    assert _same(ranks[0]["params"], ranks[1]["params"])
    assert all(r["snapshot_released"] for r in ranks)
    assert not any(e["kind"] in ("rank_divergence", "reverted", "ranks_resynced") for e in events(tmp_path / "r"))
    # No new collectives per update: keep has none; a new consistent updater checks only before, first and end.
    expected = [] if case in ("keep", "analysis") else [1, 2, 4 if case == "consistent_short" else 6]
    assert all(r["checks"] == expected for r in ranks)
    assert "ranks repaired" not in (tmp_path / "r" / "report.html").read_text()


def test_late_divergence_is_repaired_recorded_and_printed_once_per_repair(tmp_path, capsys):
    ranks = _run(tmp_path, "late")
    assert _same(ranks[0]["params"], ranks[1]["params"])
    assert all(r["ranks_repaired_at"] == [8, 12] for r in ranks)
    ev = events(tmp_path / "r")
    repairs = [e for e in ev if e["kind"] == "ranks_resynced"]
    assert [e["update"] for e in repairs] == [8, 12]
    assert all(e["names"] == ["weight", "bias"] for e in repairs)
    assert not any(e["kind"] == "rank_divergence" for e in ev)
    assert (tmp_path / "stderr-0.txt").read_text().count("[opal] WARNING: parameters differed") == 2
    assert "[opal] WARNING: parameters differed" not in (tmp_path / "stderr-1.txt").read_text()
    assert "ranks repaired at updates 8, 12" in (tmp_path / "r" / "report.html").read_text()
    _show(tmp_path / "r")
    assert "WARNING: ranks repaired at updates 8, 12" in capsys.readouterr().out
    # The source folder need not accompany a checkpoint: its repaired trajectory must still be labelled.
    state = torch.load(tmp_path / "checkpoint.pt", weights_only=False)
    state["run_dir"] = str(tmp_path / "source-folder-no-longer-present")
    model = torch.nn.Linear(3, 2)
    resumed = opal.wrap(
        torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9),
        model,
        total_steps=20,
        goal="resume repair warning",
        backend=None,
        verbose=False,
        summarize=False,
        run_dir=tmp_path / "resumed",
    )
    resumed.load_state_dict(state)
    resumed.close(interrupted=True)
    assert resumed.state_dict()["ranks_repaired_at"] == [8, 12]
    assert "ranks repaired at updates 8, 12" in (tmp_path / "resumed" / "report.html").read_text()
    _show(tmp_path / "resumed")
    assert "WARNING: ranks repaired at updates 8, 12" in capsys.readouterr().out


def test_no_snapshot_stops_every_rank_and_leaves_checkpoint_recovery_markers(tmp_path):
    ranks = _run(tmp_path, "no_snapshot")
    assert all("no exact pre-swap snapshot is available" in r["failure"] for r in ranks)
    ev = events(tmp_path / "r")
    assert [e["snapshot_available"] for e in ev if e["kind"] == "rank_divergence"] == [False]
    assert not any(e["kind"] in ("reverted", "ranks_resynced") for e in ev)
    assert len(list((tmp_path / "r").glob("crashed-*.json"))) == 2


def test_preexisting_divergence_is_flagged_without_blame_or_false_exact_rollback(tmp_path):
    ranks = _run(tmp_path, "preexisting")
    assert _same(ranks[0]["params"], ranks[1]["params"])
    ev = events(tmp_path / "r")
    assert not any(e["kind"] == "rank_divergence" for e in ev)
    assert [e["update"] for e in ev if e["kind"] == "ranks_resynced"] == [1, 2, 6]


def test_repair_banner_persists_across_resume_and_reads_old_events(tmp_path, capsys):
    ev = [
        dict(kind="start", total_updates=200),
        dict(kind="ranks_resynced", update=100),
        dict(kind="resumed", update=150),
        dict(kind="boundary", update=150, version=2, file="updater.py", line=1, category="io"),
        dict(kind="ranks_resynced", update=200),
        dict(kind="end", update=200),
    ]
    (tmp_path / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in ev))
    page = write_report(tmp_path).read_text()
    assert "ranks repaired at updates 100, 200" in page.split('<h2>', 1)[0]
    assert "Boundary flags:" in page
    assert '<html lang="en">' in page and 'id="language"' not in page
    _show(tmp_path)
    assert "WARNING: ranks repaired at updates 100, 200" in capsys.readouterr().out
    assert rank_flag([]) == ""
