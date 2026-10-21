"""Records and tools: opt.log with a dict; `opal show` and `say`
on files being written; incremental reads of the run's records;
param_groups that follow the updater; no symbolic links followed for program files; a resume into a new
folder brings the inbox along."""

import json
import os

import torch

import opal
from opal import cli, programs, report
from test_core import adamw, make, train
from test_review1 import scripted


def keep_run(tmp_path, steps=100, **kw):
    m, data = make()
    o, s = adamw(m, steps)
    kw = dict(dict(decisions=2, every=10, verbose=False), **kw)
    w = opal.wrap(o, m, scheduler=s, total_steps=steps, goal="t", run_dir=tmp_path / "r",
                     backend=lambda ws, msg, sy: "keep", **kw)  # fmt: skip
    return m, data, s, w


def test_f13_log_takes_a_dict_and_says_when_it_kept_nothing(tmp_path, capsys):
    m, data, s, w = keep_run(tmp_path, verbose=True)
    train(m, w, data, 30, sched=s)
    w.log({"test_acc": 0.5})
    w.log(val=torch.tensor(1.5))
    w.log()
    try:
        w.log(True)
        raise AssertionError("a positional non-dict was accepted")
    except TypeError:
        pass
    train(m, w, data, 100, sched=s, start=30)
    recs = report._jsonl(tmp_path / "r" / "metrics.jsonl")
    assert [r for r in recs if "test_acc" in r] == [dict(update=30, test_acc=0.5)]
    assert [r["val"] for r in recs if "val" in r] == [1.5] and all(len(r) > 1 for r in recs)
    assert "opt.log was called with no number to keep" in capsys.readouterr().err


def test_f14_show_and_say_read_files_being_written(tmp_path, capsys):
    m, data, s, w = keep_run(tmp_path)
    train(m, w, data, 100, sched=s)
    r = tmp_path / "r"
    (r / "decisions" / "001" / "response.json").write_text("")  # between truncate and write
    with open(r / "events.jsonl", "a") as f:
        f.write('{"kind": "requested", "k": 3, "upd')  # a line being written
    assert cli.main(["show", str(r)]) == 0
    assert cli.main(["say", str(r), "a note"]) == 0
    out = capsys.readouterr().out
    assert "finished at update 100" in out and "noted, but this run has finished" in out


def test_f20_records_are_read_incrementally_and_reread_when_rewritten(tmp_path):
    f = tmp_path / "x.jsonl"
    f.write_text('{"a": 1}\n{"a": 2}\n{"a": 3')  # the last line is being written
    assert report._jsonl(f) == [dict(a=1), dict(a=2)]
    with open(f, "a") as h:
        h.write('}\n{"a": 4}\n')
    assert report._jsonl(f) == [dict(a=1), dict(a=2), dict(a=3), dict(a=4)]
    done = report._READ[str(f.resolve())][0]
    assert done == os.path.getsize(f)  # what was parsed is kept
    f.write_text('{"b": 1}\n{"b": 2}\n')  # rewritten, as a resume archives the lost attempt's lines
    assert report._jsonl(f) == [dict(b=1), dict(b=2)]
    f.write_text('{"c": 1}\n{"c": 2}\n{"c": 3}\n{"c": 4}\n')  # rewritten, and longer than before
    assert report._jsonl(f) == [dict(c=1), dict(c=2), dict(c=3), dict(c=4)]


FRESH_GROUPS = """
import torch
class Updater:
    follow_user_scheduler = False
    def __init__(self, named_params, previous=None, spec=None):
        self.named, self.total = named_params, spec["total_updates"]
        self.lr = 0.01
    @property
    def param_groups(self):  # new dicts at every call
        return [{"params": [p for _, p in self.named], "lr": self.lr}]
    def step(self, grads, update):
        self.lr = 0.01 * (1 - update / self.total)
        for n, p in self.named:
            if grads[n] is not None:
                p.add_(grads[n], alpha=-self.lr)
    def state_dict(self):
        return {}
    def load_state_dict(self, s):
        pass
def build(named_params, previous=None, spec=None):
    return Updater(named_params, previous, spec)
"""


def test_l1_param_groups_show_the_programs_current_rates(tmp_path):
    m, data = make()
    o, s = adamw(m, 200)
    w = opal.wrap(o, m, scheduler=s, total_steps=200, goal="t", run_dir=tmp_path / "r", decisions=2, every=10,
                     backend=scripted({"updater.py": FRESH_GROUPS}), verbose=False)  # fmt: skip
    seen = {}
    for i in range(200):
        x, y = data[i % len(data)]
        w.zero_grad()
        torch.nn.functional.mse_loss(m(x), y).backward()
        w.step()
        seen[i + 1] = w.param_groups[0]["lr"]
        s.step()
    assert abs(seen[150] - 0.01 * (1 - 150 / 200)) < 1e-12 and abs(seen[190] - 0.01 * (1 - 190 / 200)) < 1e-12
    lrs = [r["lr"][0] for r in report._jsonl(tmp_path / "r" / "losses.jsonl")]
    assert len(set(lrs[-5:])) == 5  # losses.jsonl and the report follow it too


def test_b6_program_files_are_not_read_or_written_through_links(tmp_path):
    secret = tmp_path / "metrics.jsonl"
    secret.write_text('{"test_acc": 0.91}\n')
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "updater.py").write_text("x = 1\n")
    (ws / "peek.py").symlink_to(secret)
    assert programs.read_dir(ws) == {"updater.py": "x = 1\n"}
    programs.write_dir(ws, {"updater.py": "x = 2\n", "peek.py": "y = 1\n"})
    assert secret.read_text() == '{"test_acc": 0.91}\n' and not (ws / "peek.py").is_symlink()
    (ws / "other.py").symlink_to(secret)
    programs.write_dir(ws, {"updater.py": "x = 3\n"})
    assert not (ws / "other.py").exists() and secret.exists()


def test_b8_a_resume_into_a_new_folder_brings_the_inbox(tmp_path):
    m, data, s, w = keep_run(tmp_path / "a", steps=200, decisions=4, schedule="uniform")
    train(m, w, data, 120, sched=s)
    ck = w.state_dict()
    w.close(interrupted=True)
    m2, data, s2, w2 = keep_run(tmp_path / "b", steps=200, decisions=4, schedule="uniform")
    w2.load_state_dict(ck)
    inbox = sorted(p.name for p in (tmp_path / "b" / "r" / "workspace" / "inbox").iterdir())
    assert inbox == [f"decision-{k:03d}.json" for k in range(1, ck["decisions"] + 1)] and ck["decisions"] == 2
    json.loads((tmp_path / "b" / "r" / "workspace" / "inbox" / "decision-001.json").read_text())
    w2.close()
