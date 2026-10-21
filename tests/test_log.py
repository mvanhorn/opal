"""opt.log is record-only by default: values go to metrics.jsonl and the report, never to the agent (its decision
messages, its inbox, obs.metrics), unless the training script passes show_agent=True."""

import json
import sys

import torch

from test_review1 import scripted, wrapped

SEEN = """
EVERY = 10
SEEN = []
def analyze(obs, memory):
    SEEN.extend(obs.metrics)
"""


def run_logging(tmp_path, steps=200):
    agent = scripted({"analysis.py": SEEN})
    m, data, s, w = wrapped(tmp_path, agent, steps=steps, decisions=4)
    for i in range(steps):
        x, y = data[i % len(data)]
        loss = torch.nn.functional.mse_loss(m(x), y)
        w.zero_grad()
        loss.backward()
        w.step(loss=loss)
        s.step()
        if i % 15 == 0:
            w.log(test_acc=0.123456, test_loss=torch.tensor(7.5))
            w.log(val_loss=0.5 + i / 1000, show_agent=True)
    run = tmp_path / "r"
    messages = [p.read_text() for p in sorted((run / "decisions").glob("*/message.txt"))]
    inbox = [p.read_text() for p in sorted((run / "workspace" / "inbox").glob("*.json"))]
    records = [json.loads(x) for x in open(run / "metrics.jsonl")]
    return messages, inbox, sys.modules[w.analysis.__name__].SEEN, records


def test_logged_values_are_record_only_by_default(tmp_path):
    messages, inbox, seen, records = run_logging(tmp_path)
    assert len(messages) >= 3 and seen
    assert sum("test_acc" in r for r in records) == 14 and any(r.get("test_loss") == 7.5 for r in records)
    for text in messages + inbox:
        assert "test_acc" not in text and "test_loss" not in text and "0.123456" not in text
    assert all("test_acc" not in r and "test_loss" not in r for r in seen)


def test_show_agent_values_reach_the_analysis_and_the_message(tmp_path):
    messages, inbox, seen, records = run_logging(tmp_path)
    assert sum("val_loss" in r for r in records) == 14
    assert any("val_loss" in r for r in seen)
    assert any("Values the training script chose to show you" in t and "val_loss" in t for t in messages)
    assert any("val_loss" in t for t in inbox)
