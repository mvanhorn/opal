"""OPAL's defaults are fixed (opal/defaults.py): defaults.json holds the hashes of the default prompt, the starting
programs and the code that writes them; a run warns when they were edited, and says when it uses a custom prompt or
program. (2026-10-20: a student's coding agent edited the default analysis.py and updater.py inside an editable
install, and his runs did not reproduce ours, with nothing to tell him.)"""

import json

import pytest
import torch

import opal
from opal import defaults
from test_core import adamw, events, make, train
from test_review2 import plain_run


def test_the_manifest_is_current():
    assert defaults.modified() == [], (
        "opal/defaults.json is stale: after a deliberate change to OPAL's defaults, run python -m opal.defaults --write"
    )


def run(tmp_path, steps=120, **kw):
    m, data = make()
    o, s = adamw(m, steps)
    agent = kw.pop("backend", lambda ws, msg, system: "keep")
    w = opal.wrap(o, m, scheduler=s, total_steps=steps, goal="g", run_dir=tmp_path / "r", backend=agent, every=10,
                  verbose=False, **kw)  # fmt: skip
    train(m, w, data, steps, sched=s)
    return m, w, events(tmp_path / "r")


def test_an_unmodified_install_says_nothing(tmp_path, capsys):
    m, w, ev = run(tmp_path)
    assert "modified" not in capsys.readouterr().err
    start = next(e for e in ev if e["kind"] == "start")
    assert "defaults_modified" not in start and "custom" not in json.dumps(start)


def test_an_edited_default_is_warned_about_and_recorded(tmp_path, capsys, monkeypatch):
    shipped = json.loads(defaults.MANIFEST.read_text())
    edited = tmp_path / "defaults.json"
    edited.write_text(json.dumps(dict(shipped, **{"misc/neutral_analysis.py": "0" * 64})))
    monkeypatch.setattr(defaults, "MANIFEST", edited)  # as if neutral_analysis.py had been edited after the release
    m, w, ev = run(tmp_path / "a", backend="keep", steps=120)
    err = capsys.readouterr().err
    assert "OPAL's own defaults were modified (misc/neutral_analysis.py)" in err and "reinstall OPAL" in err
    start = next(e for e in ev if e["kind"] == "start")
    assert start["defaults_modified"] is True and start["modified_files"] == ["misc/neutral_analysis.py"]
    assert "OPAL's own defaults were modified" in (tmp_path / "a" / "r" / "report.html").read_text()
    # keep mode is still the plain run, bit for bit
    assert torch.equal(torch.cat([p.detach().flatten() for p in m.parameters()]), plain_run(120))


def test_a_custom_prompt_or_program_is_said(tmp_path, capsys):
    template = tmp_path / "mine.md"
    template.write_text((defaults.HERE / "prompts" / "neutral.md").read_text())
    m, w, ev = run(tmp_path / "p", prompt=str(template))
    err = capsys.readouterr().err
    assert f"this run uses a custom prompt ({template})" in err and "the blog's results used OPAL's defaults" in err
    assert next(e for e in ev if e["kind"] == "start")["custom_prompt"] == str(template)
    m, w, ev = run(tmp_path / "q", program=str(tmp_path / "p" / "r"))
    assert "this run uses a custom starting program" in capsys.readouterr().err
    assert next(e for e in ev if e["kind"] == "start")["custom_program"] == str(tmp_path / "p" / "r")


@pytest.mark.parametrize("prompt", [None, "aggressive"])
def test_the_named_prompts_are_not_custom(tmp_path, capsys, prompt):
    run(tmp_path, prompt=prompt)
    assert "custom" not in capsys.readouterr().err
