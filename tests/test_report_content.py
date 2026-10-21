"""Report contracts: deterministic narrative, publication filtering, incomplete records and offline detail."""

import html
import json
import re
from html.parser import HTMLParser

import pytest

from opal import cli, report, summaries
from opal.report_privacy import Privacy


def lines(path, records):
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


def sample(tmp_path):
    lines(
        tmp_path / "events.jsonl",
        [
            dict(
                kind="start",
                utc="2026-10-21T00:00:00Z",
                goal="A small run",
                total_updates=30,
                schedule=[5, 10, 20],
                blocking="auto",
                max_lag_updates=8,
            ),
            dict(kind="requested", k=1, update=5, wait="did not wait"),
            dict(kind="applied", k=1, requested_at=5, applied_at=10, lag_updates=5, version=1, changed=["updater.py"]),
            dict(kind="requested", k=2, update=20, wait="waited"),
            dict(kind="kept", k=2, applied_at=20, lag_updates=0, unavailable_seconds=60),
            dict(kind="end", utc="2026-10-21T00:02:00Z", update=30, cost_usd=0.25),
        ],
    )
    lines(tmp_path / "losses.jsonl", [dict(from_update=1, to_update=3, losses=[3.0, 2.0, 1.0])])
    for v, code in [("v000", "LR = 0.1\n"), ("v001", "LR = 0.01\n")]:
        p = tmp_path / "programs" / v
        p.mkdir(parents=True)
        (p / "updater.py").write_text(code)
    rec = tmp_path / "decisions" / "001"
    rec.mkdir(parents=True)
    (rec / "response.json").write_text(
        json.dumps(dict(text="I lowered the rate because the loss plateaued.", cost_usd=0.25, seconds=15))
    )
    (rec / "summary.json").write_text(
        json.dumps(dict(summary="Lowers the learning rate to 0.01.", tags=["common practice"], cost_usd=0.001))
    )
    (rec / "message.txt").write_text("REQUEST START\n" + "training context\n" * 500 + "REQUEST END")
    lines(
        rec / "backend.jsonl",
        [dict(type="system", subtype="init", model="recorded-model", cwd="/home/alice/old-run/workspace")],
    )
    return rec


class Structure(HTMLParser):
    def __init__(self, page):
        super().__init__()
        self.open_details = []
        self.scripts = 0
        self.external = []
        self.feed(page)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "details" and "open" in attrs:
            self.open_details.append(attrs.get("id"))
        if tag == "script":
            self.scripts += 1
        if tag in ("script", "link", "img", "iframe") and (attrs.get("src") or attrs.get("href")):
            self.external.append(attrs)


def test_narrative_uses_records_only_and_all_extra_detail_starts_closed(tmp_path, monkeypatch):
    sample(tmp_path)

    def no_model(*args, **kwargs):
        pytest.fail("Rendering must not call a model")

    monkeypatch.setattr(summaries, "ensure", no_model)
    page = report.write_report(tmp_path).read_text()
    assert report.write_report(tmp_path).read_text() == page
    top = page.split('<section id="overview"', 1)[1].split("</section>", 1)[0]
    assert "At update 10, decision 1" in top and "because the loss plateaued" in top
    assert "3 at update 1 to 1 at update 3" in top
    assert "$0.2500" in top and "$0.0010" in top and "2.0 min" in top
    assert "training waiting time: not recorded" in top  # never equate agent seconds with training wait
    assert "REQUEST END" in page and "recorded-model" in page
    assert "Waiting for agent availability" in page and "1.0 min" in page
    assert "OPAL commit" in page and "Agent effort" in page and "Not recorded" in page
    assert "Update requested → applied" in page and "5 → 10" in page
    structure = Structure(page)
    assert not structure.open_details and not structure.external and structure.scripts == 1
    assert 'http-equiv="refresh"' not in page
    assert "box-shadow" not in page and 'class="tag ' not in page


def test_full_answer_and_cached_summary_remain_in_closed_details(tmp_path):
    rec = sample(tmp_path)
    text = "I lowered the rate because the loss plateaued.\n" + "Full answer\n" * 800 + "ANSWER END"
    (rec / "response.json").write_text(json.dumps(dict(text=text)))
    page = report.write_report(tmp_path).read_text()
    assert "ANSWER END" in page
    assert '<details id="decision-details">' in page and '<details id="code-details">' in page
    assert "Lowers the learning rate to 0.01." in page
    assert "LR: 0.1 → 0.01" in page


def test_privacy_covers_prose_keys_messages_diffs_and_embedded_plot_data(tmp_path):
    rec = sample(tmp_path)
    secret = "alice on train-box.example.edu (hostname=train-box, username=alice), s3://private-bucket/key, alice@train-box, /etc/private/file, /home/alice/old-run/workspace/updater.py"
    (rec / "message.txt").write_text(secret)
    (rec / "response.json").write_text(json.dumps(dict(text=secret, files={"updater.py": secret})))
    (rec / "summary.json").write_text(json.dumps(dict(summary=secret, tags=["common practice", secret])))
    (tmp_path / "programs/v001/updater.py").write_text("LR = 0.01\n# " + secret)
    lines(tmp_path / "metrics.jsonl", [dict(update=1, **{secret: 2.0})])
    page = html.unescape(report.write_report(tmp_path).read_text())
    for value in ("alice", "train-box", "private-bucket", "/home/", "/etc/", str(tmp_path)):
        assert value not in page
    assert "workspace/updater.py" in page
    assert "[path omitted]" in page and "[remote location omitted]" in page


@pytest.mark.parametrize(
    "value",
    [
        r"C:\Users\alice\private\file.py",
        r"\\private-host\alice\data",
        "/Users/Jane Doe/private/file.py",
        "/home/alice/private/file.py",
        "s3://secret-bucket/a secret-bucket",
        "ssh://alice@private-host/a alice private-host",
        "user=alice host=train-box alice train-box",
        "alice@train.example.edu",
        "https://secret-bucket.s3.us-east-1.amazonaws.com/a secret-bucket",
        "arn:aws:s3:::secret-bucket/a secret-bucket",
        "ssh://alice@private-host:2222/a private-host alice",
        "~alice/private/file.py alice",
    ],
)
def test_private_locations_and_repeated_identities_are_removed(tmp_path, value):
    clean = Privacy(tmp_path, {"text": value}).clean(value)
    for secret in ("alice", "Jane Doe", "secret-bucket", "private-host", "train-box", "train.example.edu"):
        assert secret not in clean


def test_run_paths_are_relative_and_source_records_are_untouched(tmp_path):
    rec = sample(tmp_path)
    text = f"{tmp_path}/programs/v001/updater.py /home/alice/old-run/decisions/001/message.txt"
    (rec / "message.txt").write_text(text)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    page = report.write_report(tmp_path).read_text()
    assert "programs/v001/updater.py" in page and "decisions/001/message.txt" in page
    assert str(tmp_path) not in page and "/home/alice" not in page
    assert all(p.read_bytes() == data for p, data in before.items())
    assert not list(tmp_path.glob(".report.html.*.tmp"))


def test_html_and_script_injection_are_escaped(tmp_path):
    rec = sample(tmp_path)
    attack = '</script><img src="https://attacker.example.edu/a" onerror="alert(1)">'
    (rec / "response.json").write_text(json.dumps(dict(text=attack)))
    page = report.write_report(tmp_path).read_text()
    structure = Structure(page)
    assert structure.scripts == 1 and not structure.external
    assert "<img src=" not in page
    assert r"\u003c" in page


def test_partial_old_folder_malformed_lines_and_invalid_numbers_are_tolerated(tmp_path):
    lines(
        tmp_path / "events.jsonl",
        [
            None,
            [],
            7,
            {},
            dict(kind="requested"),
            dict(kind="start", schedule=None),
            dict(kind="applied", k=1, applied_at=5),
            dict(kind="eval_integrity", flagged=True, tensors=None),
            dict(kind="ranks_resynced"),
            dict(kind="ranks_resynced", update="bad"),
            dict(kind="resumed", ranks_repaired_at=None),
            dict(kind="resumed", ranks_repaired_at=[None, "bad", {}, -1, True]),
            dict(kind="end"),
        ],
    )
    with (tmp_path / "events.jsonl").open("ab") as f:
        f.write(b'bad JSON\n\xff\n{"kind":')
    lines(
        tmp_path / "losses.jsonl",
        [None, dict(losses=2), dict(from_update=1, to_update=6, losses=[1, float("nan"), float("inf"), True, None, 0])],
    )
    lines(tmp_path / "reports.jsonl", [dict(update=1, report=None)])
    lines(tmp_path / "metrics.jsonl", [dict(update="bad", acc=1), dict(update=1, acc=float("nan"))])
    rec = tmp_path / "decisions/001"
    rec.mkdir(parents=True)
    (rec / "response.json").write_text("[]")
    (rec / "summary.json").write_text("{broken")
    page = report.write_report(tmp_path).read_text()
    assert "diff unavailable" in page and "Training loss" in page
    payload = page.split("const L = ", 1)[1].split("const ns =", 1)[0]
    assert not re.search(r"\b(?:NaN|Infinity)\b", payload)
    assert report.summarize_missing(tmp_path, object(), log=lambda _: None) == 1
    assert cli.main(["report", str(tmp_path)]) == 0


def test_explicit_timing_and_provenance_and_reverted_changes(tmp_path):
    sample(tmp_path)
    (tmp_path / "result.json").write_text(json.dumps(dict(agent_wait_seconds=12, opal_commit="abc123+changes")))
    with (tmp_path / "events.jsonl").open("a") as f:
        f.write(json.dumps(dict(kind="reverted", version=1, update=22)) + "\n")
    page = report.write_report(tmp_path).read_text()
    assert "12.0 s" in page and "abc123+changes" in page
    assert "Reverted at update 22" in page


def test_logged_accuracy_in_overview_and_live_resumed_cost(tmp_path):
    sample(tmp_path)
    lines(
        tmp_path / "metrics.jsonl",
        [dict(update=5, test_loss=2, test_acc=0.6), dict(update=30, test_loss=1, test_acc=0.9)],
    )
    with (tmp_path / "events.jsonl").open("a") as f:
        f.write(json.dumps(dict(kind="resumed", update=30)) + "\n")
        f.write(json.dumps(dict(kind="requested", k=3, update=35)) + "\n")
        f.write(json.dumps(dict(kind="kept", k=3, applied_at=35, cost_usd=0.5)) + "\n")
    page = report.write_report(tmp_path).read_text()
    top = page.split('<section id="overview"', 1)[1].split("</section>", 1)[0]
    assert "test_acc: 0.6 → 0.9 (updates 5 → 30)" in top
    assert "$0.7500" in top  # the old end's .25 is not the resumed run's current total
    assert 'http-equiv="refresh"' in page


def test_reason_keeps_the_causal_intro_instead_of_an_unrelated_observation():
    text = (
        "My updater was taking 1.64 s per update, against 0.87 ms for the starting program. "
        "I replaced its orthogonalisation step. Observed: the loss fell to 0.16. "
        "That is different because other parameters move less."
    )
    assert report._excerpt(text, reason=True).startswith("My updater was taking 1.64 s")
    text = "The model fits the training data almost exactly, so I made the decay steeper. That is far more because AdamW shrinks its steps."
    assert report._excerpt(text, reason=True).startswith("The model fits the training data")


def test_copied_run_paths_with_spaces_are_relative(tmp_path):
    record = {"cwd": "/home/alice/old run/workspace", "text": "/home/alice/old run/workspace/updater.py"}
    clean = Privacy(tmp_path / "a copy", record).clean(record)
    assert clean["text"] == "workspace/updater.py"
    assert "alice" not in str(clean)
    assert Privacy(tmp_path, {"text": "/workspace/x /etc/private"}).text("/etc/private") == "[path omitted]"
