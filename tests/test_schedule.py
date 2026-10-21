"""wrap(schedule=...): "front" (the default) gives exactly the decision updates of earlier releases; "uniform", a list
of (end_fraction, count) segments and a list of update numbers; decisions= and a list must agree."""

import json

import pytest

import opal
from opal.schedule import count, decision_segments, decision_updates
from test_core import adamw, make, train


def old_decision_updates(total, n, early_fraction=0.05, early_share=0.2, cadence=1, spacing="front"):
    """opal/schedule.py as of 1f217e3 (decision_spacing), copied to prove that the defaults did not move."""
    if n <= 0 or total <= 1:
        return []
    if spacing == "uniform":
        return old_snap([max(1, total // n) * (i + 1) for i in range(n)], total, cadence)
    n_early = max(1, round(n * early_share))
    n_late = max(0, n - n_early)
    early_end = max(1, int(total * early_fraction))
    out = [round(early_end * (i + 1) / n_early) for i in range(n_early)]
    span = total - early_end
    out += [early_end + round(span * (i + 1) / (n_late + 1)) for i in range(n_late)]
    return old_snap(out, total, cadence)


def old_snap(updates, total, cadence):
    snapped = {-(-u // cadence) * cadence for u in updates}
    return sorted(u for u in snapped if 0 < u and total - u >= cadence)


TOTALS = [1, 2, 7, 19, 20, 99, 200, 1000, 2340, 3250, 9999, 64626, 68074, 400000]


@pytest.mark.parametrize("spacing", ["front", "uniform"])
def test_front_and_uniform_give_exactly_the_old_updates(spacing):
    for total in TOTALS:
        for n in (0, 1, 2, 3, 4, 5, 7, 10, 12, 42, 50, 100, 1000):
            for cadence in (1, 10, 50, 100):
                old = old_decision_updates(total, n, cadence=cadence, spacing=spacing)
                assert decision_updates(total, n, cadence, spacing) == old, (total, n, cadence)


def test_front_is_its_two_segments():
    for total in TOTALS[3:]:
        for n in (1, 5, 10, 50, 100):
            early = max(1, round(n * 0.2))
            segments = [(0.05, early), (1.0, max(0, n - early))]
            assert decision_updates(total, n, 50, segments) == decision_updates(total, n, 50, "front")


def test_segments_twenty_in_the_first_five_percent_then_thirty():
    first, rest = decision_segments(10_000, None, 1, [(0.05, 20), (1.0, 30)])
    assert first == [25 * (i + 1) for i in range(20)]  # evenly over updates 1-500, the last at the segment's end
    assert len(rest) == 30 and rest[0] > 500 and rest[-1] < 10_000  # the last one gap before the end of the run
    gaps = {b - a for a, b in zip([500] + rest, rest + [10_000])}
    assert max(gaps) - min(gaps) <= 1  # 9,500 updates in 31 equal gaps
    assert count([(0.05, 20), (1.0, 30)]) == 50
    # snapped to checks every 50 updates: the first 5% has only 10 checks, so 20 become 10
    at = decision_updates(10_000, None, 50, [(0.05, 20), (1.0, 30)])
    assert [u for u in at if u <= 500] == list(range(50, 501, 50)) and len(at) == 40
    # a segment that does not reach the end, and an empty one
    assert decision_updates(1000, None, 10, [(0.1, 2), (0.5, 0)]) == [50, 100]


def test_a_list_of_updates():
    assert decision_updates(6000, None, 50, [500, 1000, 5000]) == [500, 1000, 5000]
    assert decision_updates(6000, None, 50, [1000, 501, 520]) == [550, 1000]  # moved up to checks; two merge
    assert decision_updates(6000, None, 50, [5990]) == []  # in the last 50 updates: no answer could act
    assert decision_updates(6000, None, 50, []) == []
    with pytest.raises(ValueError, match="outside the run"):
        decision_updates(6000, None, 50, [0, 100])
    with pytest.raises(ValueError, match="outside the run"):
        decision_updates(6000, None, 50, [6001])


@pytest.mark.parametrize("update", [0, 2])
def test_explicit_updates_are_validated_even_for_a_one_step_run(update):
    with pytest.raises(ValueError, match="outside the run"):
        decision_updates(1, None, 1, [update])
    assert decision_updates(1, None, 1, [1]) == []


def test_decisions_and_a_list_must_agree():
    assert count("front") == 50 and count("uniform", 7) == 7 and count([10, 20], 2) == 2
    with pytest.raises(ValueError, match=r"decisions=10 but schedule gives 50 \(20 \+ 30\)"):
        count([(0.05, 20), (1.0, 30)], 10)
    with pytest.raises(ValueError, match=r"decisions=3 but schedule gives 2 \(2 updates\)"):
        count([10, 20], 3)
    for bad in ("even", 5, [(0.5, 2), 10], [(0.5, 2), (0.4, 1)], [(1.5, 2)], [(0.5, -1)], [(0.5, 2.0)], [True]):
        with pytest.raises(ValueError, match="schedule"):
            count(bad)
    with pytest.raises(ValueError, match="decisions must be an int"):
        count("front", -1)


def wrapped(tmp_path, steps=600, **kw):
    m, data = make()
    o, s = adamw(m, steps)
    calls = []
    w = opal.wrap(o, m, scheduler=s, total_steps=steps, goal="g", run_dir=tmp_path, verbose=False, every=10,
                  backend=lambda ws, msg, sy: calls.append(msg) or "keep", **kw)  # fmt: skip
    train(m, w, data, steps, sched=s)
    events = [json.loads(x) for x in (tmp_path / "events.jsonl").read_text().splitlines()]
    return w, [e["update"] for e in events if e["kind"] == "requested"]


def test_the_wrapper_asks_at_each_form(tmp_path):
    w, asked = wrapped(tmp_path / "front", decisions=12)
    assert w.schedule == old_decision_updates(600, 12, cadence=10) == asked
    w, asked = wrapped(tmp_path / "seg", schedule=[(0.05, 3), (1.0, 4)])
    assert w.schedule == asked == [10, 20, 30, 150, 260, 380, 490]
    assert (
        "You are called every 10 updates for the first 30 updates (3 decisions), then about every 113 updates from "
        "update 150 to update 490 (4 decisions); 7 scheduled decisions in all." in w.worker.system
    )
    w, asked = wrapped(tmp_path / "list", schedule=[100, 250, 400])
    assert w.schedule == asked == [100, 250, 400]
    assert "You are called at updates 100, 250, 400; 3 scheduled decisions in all." in w.worker.system
    with pytest.raises(ValueError, match="decisions=5 but schedule gives 3"):
        wrapped(tmp_path / "bad", schedule=[100, 250, 400], decisions=5)
