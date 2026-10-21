"""Boundary diagnostics are static, best effort and have no authority over program execution."""

import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch

import opal
from opal import programs
from opal.agent import Run
from opal.misc import boundary, neutral_analysis
from opal.report import boundary_flag, write_report
from test_core import adamw, events, flat, make


def test_open_is_flagged_with_location_without_executing(tmp_path):
    flags = boundary.check({"updater.py": "# source only\nopen('/tmp/x')\n"})
    assert flags == [dict(file="updater.py", line=2, category="io", snippet="open('/tmp/x')")]
    target = tmp_path / "must-not-exist"
    boundary.check({"helper.py": f"open({str(target)!r}, 'w').write('executed')"})
    assert not target.exists()


def test_inspect_and_optimizer_globals_are_flagged():
    flags = boundary.check(
        {"analysis.py": "import inspect\nschedule = inspect.getmodule(o.step).__dict__\nx = o.step.__globals__\n"}
    )
    assert {f["line"] for f in flags if f["category"] == "introspection"} == {1, 2, 3}


@pytest.mark.parametrize(
    "source,category",
    [
        ("from pathlib import Path\nPath('x').read_text()", "io"),
        ("from pathlib import Path\np = Path('x')\np.write_bytes(b'x')", "io"),
        ("import numpy as np\nnp.load('x')", "io"),
        ("from torch import save as s\ns({}, 'x')", "io"),
        ("import torch\ntorch.jit.load('x')", "io"),
        ("import pickle as p\np.load(f)", "io"),
        ("import os as system\nsystem.listdir('.')", "io"),
        ("from os import path\npath.exists('x')", "io"),
        ("import glob\nglob.glob('*')", "io"),
        ("import shutil\nshutil.copy('x', 'y')", "io"),
        ("import subprocess\nsubprocess.run(['true'])", "io"),
        ("import socket\nsocket.socket()", "io"),
        ("from urllib.request import urlopen\nurlopen('https://example.org')", "io"),
        ("import requests as r\nr.get('https://example.org')", "io"),
        ("__import__('os')", "dynamic"),
        ("x = o.step.__builtins__", "dynamic"),
        ("getattr(o.step, '__builtins__')['open']('x')", "dynamic"),
        ("import importlib as i\ni.import_module('os')", "dynamic"),
        ("import builtins as b\nb.open('x')", "dynamic"),
        ("eval('1'); exec('x = 1'); compile('1', 'x', 'eval')", "dynamic"),
        ("import __main__ as m\nx = m.VALUE", "introspection"),
        ("import sys as s\nx = s.modules['__main__']", "introspection"),
        ("from sys import _getframe as frame\nx = frame().f_back.f_globals", "introspection"),
        ("from gc import get_objects, get_referrers\nget_referrers(get_objects())", "introspection"),
        ("import math as m\nx = m.__dict__", "introspection"),
        ("import xml.etree as e\nx = e.__dict__", "introspection"),
        ("x = getattr(o.step, '__globals__')", "introspection"),
        ("x = object.__subclasses__()", "introspection"),
        ("import math\nx = vars(math)", "introspection"),
        ("model.register_forward_hook(fn)", "model_mutation"),
        ("getattr(model, 'register_full_backward_hook')(fn)", "model_mutation"),
        ("model.forward = fn", "model_mutation"),
        ("loss_fn.__call__ = fn", "model_mutation"),
        ("setattr(criterion, 'forward', fn)", "model_mutation"),
        ("type(model).forward = fn", "model_mutation"),
        ("import torch\ntorch.nn.Module.forward = fn", "model_mutation"),
    ],
)
def test_common_boundary_patterns_and_import_aliases(source, category):
    assert any(f["category"] == category for f in boundary.check({"helper.py": source}))


def test_default_programs_and_local_object_access_are_clean(capsys):
    files = programs.starting_files([], [], {}, every=10)
    assert boundary.check(files) == []
    files["analysis.py"] = neutral_analysis.text(10, None)
    assert boundary.check(files) == []
    example = Path(opal.__file__).parent / "prompts" / "example_updater.py"
    assert boundary.check({"updater.py": example.read_text()}) == []
    assert boundary.check({"updater.py": "x = self.__dict__\ny = vars(obs.updater)\n# open('x')\ns = 'eval(x)'"}) == []
    assert capsys.readouterr().err == ""


def test_syntax_error_never_raises_and_logs_one_line(capsys):
    assert boundary.check({"updater.py": "def broken(:\n"}) == []
    err = capsys.readouterr().err
    assert "boundary check failed (SyntaxError)" in err and len(err.splitlines()) == 1


@pytest.mark.parametrize("error", [RuntimeError, MemoryError, SystemExit, KeyboardInterrupt])
def test_checker_failures_never_escape(error, monkeypatch, capsys):
    def fail(*args, **kw):
        raise error("injected")

    monkeypatch.setattr(boundary.ast, "parse", fail)
    assert boundary.check({"updater.py": "pass"}) == []
    assert len(capsys.readouterr().err.splitlines()) == 1


def test_recording_and_stderr_failures_never_escape(tmp_path, monkeypatch, capsys):
    run = Run(tmp_path)

    def fail(*args, **kw):
        raise RuntimeError("injected")

    monkeypatch.setattr(run, "event", fail)
    boundary.record({"updater.py": "open('x')"}, 2, run)
    assert len(capsys.readouterr().err.splitlines()) == 1
    monkeypatch.setattr(boundary, "print", fail, raising=False)
    boundary.record({"updater.py": "open('x')"}, 2, run)
    monkeypatch.setattr(boundary, "check", fail)
    boundary.record({"updater.py": "open('x')"}, 2, run)


def test_check_preserves_rng_states():
    py_state, np_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    boundary.check({"updater.py": "open('x')\nx = o.step.__globals__"})
    assert random.getstate() == py_state
    after = np.random.get_state()
    assert after[0] == np_state[0] and np.array_equal(after[1], np_state[1]) and after[2:] == np_state[2:]
    assert torch.equal(torch.get_rng_state(), torch_state)


def test_flagged_keep_run_is_bitwise_identical_with_check_disabled(tmp_path, monkeypatch, capsys):
    marker = tmp_path / "input.txt"
    marker.write_text("visible to the training process")
    folder = tmp_path / "program"
    files = programs.starting_files([], [], {}, every=5)
    files["updater.py"] = f"with open({str(marker)!r}) as f:\n    READ = f.read()\n" + files["updater.py"]
    programs.write_dir(folder, files)

    def train(name):
        model, data = make()
        opt, scheduler = adamw(model, 30)
        wrapped = opal.wrap(
            opt,
            model,
            scheduler=scheduler,
            total_steps=30,
            goal="boundary test",
            backend="keep",
            decisions=2,
            every=5,
            program=folder,
            run_dir=tmp_path / name,
            verbose=False,
        )
        losses = []
        for x, y in data[:30]:
            loss = torch.nn.functional.mse_loss(model(x), y)
            losses.append(loss.detach().clone())
            wrapped.zero_grad()
            loss.backward()
            wrapped.step(loss=loss)
            scheduler.step()
        wrapped._live.finish(timeout=30)
        assert wrapped.version == 0 and wrapped.update == 30
        # The flagged file read really ran, rather than being suppressed by the checker.
        assert wrapped.updater.step.__globals__["READ"] == marker.read_text()
        state = opt.state_dict()
        return flat(model).clone(), torch.stack(losses), scheduler.state_dict(), state

    checked = train("checked")
    flagged = [e for e in events(tmp_path / "checked") if e["kind"] == "boundary"]
    assert len(flagged) == 1 and flagged[0]["version"] == 0 and flagged[0]["line"] == 1
    assert "Boundary flags:" in (tmp_path / "checked" / "report.html").read_text()
    assert capsys.readouterr().err.count("[opal] boundary flags:") == 1
    monkeypatch.setattr(boundary, "record", lambda *args: None)
    unchecked = train("unchecked")
    assert torch.equal(checked[0], unchecked[0]) and torch.equal(checked[1], unchecked[1])
    assert checked[2] == unchecked[2]
    for key, state in checked[3]["state"].items():
        for field, value in state.items():
            assert torch.equal(value, unchecked[3]["state"][key][field])
    assert checked[3]["param_groups"] == unchecked[3]["param_groups"]
    assert not any(e["kind"] == "boundary" for e in events(tmp_path / "unchecked"))


def test_new_analysis_and_helper_are_checked_once_per_version(tmp_path, capsys):
    def agent(ws, message, system):
        (ws / "helper.py").write_text("def unused():\n    return open('/tmp/x')\n")
        (ws / "analysis.py").write_text("import helper\nEVERY = 5\ndef analyze(obs, memory):\n    return None\n")
        return "analysis-only edit"

    model, data = make()
    opt, scheduler = adamw(model, 30)
    wrapped = opal.wrap(
        opt,
        model,
        scheduler=scheduler,
        total_steps=30,
        goal="boundary test",
        backend=agent,
        decisions=2,
        every=5,
        run_dir=tmp_path / "r",
        verbose=False,
    )
    from test_core import train

    train(model, wrapped, data, 30, sched=scheduler)
    wrapped._live.finish(timeout=30)
    assert wrapped.version == 1  # next identical answer is kept, not checked again
    flagged = [e for e in events(tmp_path / "r") if e["kind"] == "boundary"]
    assert len(flagged) == 1 and flagged[0]["file"] == "helper.py" and flagged[0]["version"] == 1
    assert capsys.readouterr().err.count("[opal] boundary flags:") == 1


def test_report_note_is_optional_bounded_and_escaped(tmp_path):
    assert boundary_flag([]) == ""
    run = Run(tmp_path)
    run.event("start", total_updates=10, goal="test")
    for v in range(8):
        run.event("boundary", version=v, file="<updater>.py", line=1, category="io", snippet="<script>")
    page = write_report(tmp_path).read_text()
    assert "Boundary flags:</b> 8 static findings" in page and "&lt;updater&gt;.py" in page
    note = boundary_flag([json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()])
    assert "v005" in note and "v006" not in note and "<script>" not in note
