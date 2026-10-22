"""Regression witnesses adapted from opal-checks/review_035/repro (CPU/offline)."""
import copy
import html
import io
import json
import pickle
import threading
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import opal
from opal import programs
from opal.backends._privacy import redact_secrets
from opal.backends.base import Backend
from opal.backends.claude import ClaudeBackend
from opal.misc.ddp import DDPMixin
from opal.report import write_report
from opal.results import summary
from test_claude_backend import fake_cli
from test_release_035 import UPDATER, wrap


def reset_accelerate():
    from accelerate.state import AcceleratorState, GradientState
    AcceleratorState._reset_state(True)
    GradientState._reset_state()


def test_prepared_public_state_reset_is_bit_identical(tmp_path):
    from accelerate import Accelerator
    def run(wrapped):
        reset_accelerate()
        acc = Accelerator(cpu=True)
        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(1.)
        raw = torch.optim.SGD(model.parameters(), lr=.1, momentum=.9)
        model, prepared = acc.prepare(model, raw)
        opt = wrap(model, prepared, tmp_path / 'wrapped') if wrapped else prepared
        try:
            assert prepared.state is raw.state
            assert prepared.param_groups is raw.param_groups
            assert prepared.defaults is raw.defaults
            prepared.state = raw.state.copy()
            prepared.param_groups = list(raw.param_groups)
            prepared.defaults = dict(raw.defaults)
            assert prepared.state is raw.state and prepared.param_groups is raw.param_groups
            assert prepared.defaults is raw.defaults
            for i in range(4):
                if i == 2:
                    prepared.state.clear()
                opt.zero_grad()
                loss = model(torch.ones(1, 1)).sum()
                acc.backward(loss)
                opt.step(loss=loss) if wrapped else opt.step()
            # Writes must delegate too, including after a checkpoint replaced raw.state.
            state = copy.deepcopy(raw.state_dict())
            raw.load_state_dict(state)
            assert prepared.state is raw.state
            return model.weight.detach().clone()
        finally:
            if wrapped:
                opt.close(interrupted=True)
            reset_accelerate()
    assert torch.equal(run(False), run(True))


def _view_ranks(rank, folder, rendezvous, view, dtype):
    from pathlib import Path
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=30))
    try:
        results = []
        for wrapped in (False, True):
            model = torch.nn.Linear(2, 2, bias=False, dtype=dtype)
            value = torch.ones(2, 2, dtype=dtype) * ((1 + 1j) if dtype.is_complex else 1)
            model.weight = torch.nn.Parameter(value.conj() if view == 'conjugate' else torch._neg_view(value))
            try:
                ddp = torch.nn.parallel.DistributedDataParallel(model)
            except RuntimeError as exc:
                if dtype.is_complex and str(exc) == 'Invalid scalar type':
                    # Older Gloo rejects complex DDP before OPAL is constructed.
                    (Path(folder) / f'unsupported-{rank}').write_text(str(exc))
                    return
                raise
            raw = torch.optim.SGD(model.parameters(), lr=.1)
            opt = wrap(ddp, raw, Path(folder) / (view + str(wrapped)), rank_check_every=1) if wrapped else raw
            try:
                for _ in range(3):
                    opt.zero_grad()
                    loss = ddp(torch.ones(2, 2, dtype=dtype)).abs().square().mean()
                    loss.backward()
                    opt.step(loss=loss) if wrapped else opt.step()
                results.append(model.weight.detach().resolve_conj().resolve_neg().clone())
            finally:
                if wrapped:
                    opt.close(interrupted=True)
            dist.barrier()
        assert torch.equal(*results)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize('view,dtype', [('conjugate', torch.complex64), ('negative', torch.complex64),
                                       ('negative', torch.float32)])
def test_conjugate_and_negative_views_two_real_gloo_ranks(tmp_path, view, dtype):
    mp.spawn(_view_ranks, args=(str(tmp_path), (tmp_path / 'init').as_uri(), view, dtype), nprocs=2, join=True)
    if (tmp_path / 'unsupported-0').exists():
        assert (tmp_path / 'unsupported-1').exists()
        pytest.skip('installed PyTorch/Gloo cannot construct complex DDP (before OPAL)')


@pytest.mark.parametrize('value', [torch.Size([2, 3]), torch.float32, torch.device('cpu'),
                                   {1, 2}, torch.strided])
def test_weights_only_safe_state_types(value):
    stream = io.BytesIO()
    torch.save(value, stream)
    stream.seek(0)
    try:
        torch.load(stream, weights_only=True)
    except pickle.UnpicklingError:
        # The guard must reject library types the installed safe loader cannot read.
        assert programs.unpicklable(value)
    else:
        assert not programs.unpicklable(value)


def test_old_shape_checkpoint_and_analysis_only_starting_exemption(tmp_path):
    source = UPDATER.replace('dict(k=self.k)', 'dict(k=self.k, shape=self.named[0][1].shape)')
    def agent(ws, message, system):
        (ws / 'updater.py').write_text(source)
        return 'shape bookkeeping'
    model = torch.nn.Linear(2, 1)
    opt = wrap(model, torch.optim.SGD(model.parameters(), lr=.1), tmp_path / 'old', backend=agent, schedule=[1])
    try:
        opt.step(loss=1.)
        assert opt.version == 1
        state = opt.state_dict()
        stream = io.BytesIO()
        torch.save(state, stream)
        stream.seek(0)
        opt.load_state_dict(torch.load(stream, weights_only=True))
        assert opt.version == 1 and opt.updater.k == 1
    finally:
        opt.close(interrupted=True)


def test_analysis_only_edit_keeps_starting_updater_exemption(tmp_path):
    # A user's starting state can contain objects not on the agent allowlist.
    model = torch.nn.Linear(2, 1)
    raw = torch.optim.SGD([dict(params=list(model.parameters()), metadata={1: range(3)})], lr=.1)
    opt = wrap(model, raw, tmp_path / 'analysis')
    try:
        before = opt.state_dict()['updater']
        opt._apply(dict(k=99, requested_at=0, files=dict(opt.files, **{'analysis.py': opt.files['analysis.py'] + '\n# observer\n'})), 0)
        assert before is not None and opt.state_dict()['updater'] is not None
    finally:
        opt.close(interrupted=True)


def test_accelerate_raw_034_checkpoint_then_manual_opal(tmp_path, capsys):
    from accelerate import Accelerator
    reset_accelerate()
    acc = Accelerator(cpu=True)
    model = torch.nn.Linear(2, 1)
    raw = torch.optim.AdamW(model.parameters(), lr=.01)
    model, prepared = acc.prepare(model, raw)
    # 0.3.4 Accelerator wrote the original optimizer schema.
    model(torch.ones(1, 2)).sum().backward()
    prepared.step()
    acc.save_state(tmp_path / 'checkpoint')
    expected = copy.deepcopy(raw.state_dict())
    opt = wrap(model, prepared, tmp_path / 'run')
    try:
        opt.step(loss=1.)
        manual = copy.deepcopy(opt.state_dict())
        raw.state.clear()
        acc.load_state(tmp_path / 'checkpoint')
        acc.load_state(tmp_path / 'checkpoint')
        assert len(raw.state) == len(expected['state'])
        for p, saved in zip(raw.state.values(), expected['state'].values()):
            assert all(torch.equal(p[k], v) for k, v in saved.items())
        assert prepared.state is raw.state
        assert capsys.readouterr().err.count('no OPAL program or progress') == 1
        opt.load_state_dict(manual)
        assert opt.update == manual['update']
    finally:
        opt.close(interrupted=True)
        reset_accelerate()


@pytest.mark.parametrize('case', ['nested_unicode', 'nested_bearer', 'repr_bearer'])
def test_escaped_cli_credentials_all_exits(tmp_path, case):
    token = 'CanaryPrivateBearer1234567890'
    if case == 'nested_unicode':
        token = 'sk-CanaryPrivateKey1234567890'
        inner = json.dumps({'api_key': token}).replace('sk-', r'\u0073\u006b\u002d')
    else:
        inner = json.dumps({'Authorization': 'Bearer\n' + token})
    if case == 'repr_bearer':
        inner = repr(RuntimeError(inner))
    said = json.dumps(dict(type='result', is_error=True, result=inner))
    exe = fake_cli(tmp_path, mode='flaky', fails=100, half=0, say=said)
    backend = ClaudeBackend(binary=str(exe), sandbox=False, retries=0, unavailable_timeout=0)
    model = torch.nn.Linear(1, 1)
    opt = wrap(model, torch.optim.SGD(model.parameters(), lr=.1), tmp_path / 'run', backend=backend, schedule=[1, 2])
    try:
        for _ in range(3):
            opt.step(loss=1.)
    finally:
        opt.close(interrupted=True)
    for name in ['decisions/001/backend.jsonl', 'decisions/001/response.json',
                 'decisions/002/message.txt', 'report.html']:
        assert 'CanaryPrivate' not in html.unescape((opt.run.dir / name).read_text())
    # Only original secret spans are masked; surrounding nested JSON remains valid.
    cleaned = redact_secrets(json.dumps({'nested': inner}))
    json.loads(cleaned)
    assert 'CanaryPrivate' not in cleaned


@pytest.mark.parametrize('text', ['mask-generation', 'mask-rcnn-resnet50-fpn',
    'task-specific error: mask-generation failed', 'from huggingface_hub import hf_hub_download',
    'HF_DATASETS_CACHE=/cache/data', 'https://example.org/models/mask-generation',
    'sha256=' + 'abcdef0123456789' * 4, 'base64=YWJjZGVmZ2hpamtsbW5vcA==',
    'id=123456789012345678901234567890'])
def test_redaction_preserves_review_false_positives(text):
    assert redact_secrets(text) == text


def test_noncontiguous_hash_has_no_full_parameter_reshape(tmp_path, monkeypatch):
    tensor = torch.ones(1025, 1025).t()
    original = torch.Tensor.reshape
    def checked(t, *shape):
        assert t.is_contiguous() or t.numel() <= 1_048_576, 'unbounded noncontiguous copy'
        return original(t, *shape)
    monkeypatch.setattr(torch.Tensor, 'reshape', checked)
    monkeypatch.setattr(dist, 'all_reduce', lambda *a, **k: None)
    probe = DDPMixin()
    probe.named, probe.ctrl = [('weight', tensor)], None
    assert probe._rank_differences() == []


@pytest.mark.parametrize('charge', [None, .25])
def test_unknown_cancel_cost_persists_until_numeric_settlement(tmp_path, charge):
    class Cancel(Backend):
        name = 'offline-cost-probe'
        def __init__(self):
            self.started, self.release = threading.Event(), threading.Event()
        def decide(self, workspace, message, system, record_dir):
            self.started.set()
            assert self.release.wait(10)
            return dict(text='cancelled', error='cancelled', returncode=-1, cost_usd=charge, seconds=1)
        def cancel(self):
            pass
    backend = Cancel()
    model = torch.nn.Linear(1, 1)
    opt = wrap(model, torch.optim.SGD(model.parameters(), lr=.1), tmp_path / 'run',
               backend=backend, schedule=[1], blocking=False)
    try:
        opt.step(loss=1.)
        assert backend.started.wait(2)
        opt.close(interrupted=True)
        assert summary('loss', 1., 'lower', 1., opt)['agent_cost_usd'] is None
    finally:
        backend.release.set()
        if opt.worker.thread:
            opt.worker.thread.join(5)
        opt.close(interrupted=True)
    opt._live.finish(timeout=3)
    assert summary('loss', 1., 'lower', 1., opt)['agent_cost_usd'] == charge
    state = opt.state_dict()
    assert state['unknown_cost_decisions'] == ([1] if charge is None else [])
    opt.worker.unknown_cost_decisions.clear()
    opt.load_state_dict(state)
    assert summary('loss', 1., 'lower', 1., opt)['agent_cost_usd'] == charge
    page = write_report(opt.run.dir).read_text()
    assert ('Recorded agent cost: not recorded.' if charge is None else 'Recorded agent cost: $0.2500.') in page
    events = [json.loads(line) for line in (opt.run.dir / 'events.jsonl').read_text().splitlines()]
    assert [e for e in events if e['kind'] == 'cost_settled'][-1]['cost_usd'] == charge


@pytest.mark.parametrize('layers', [1, 2, 4, 7])
def test_redaction_nested_escape_depth_and_window_boundary(layers):
    token = 'sk-CanaryPrivateKey1234567890'
    encoded = token.replace('sk-', r'\u0073\u006b\u002d')
    for _ in range(layers - 1):
        encoded = json.dumps(encoded)[1:-1]
    text = 'normal code ' + 'x' * 61400 + ' ' + encoded + ' trailing'
    result = redact_secrets(text)
    assert 'CanaryPrivate' not in result
    assert result.startswith('normal code ' + 'x' * 61400 + ' ')
    assert result.endswith(' trailing')



@pytest.mark.parametrize('quote', ["'", '"'])
def test_adjacent_repr_fragments_cannot_reassemble_a_credential(tmp_path, quote):
    from opal.report_privacy import Privacy
    token = 'hf_' + 'Canary987' * 4
    diagnostic = repr(token[:10]) + ' ' + quote + token[10:] + quote
    assert token == __import__('ast').literal_eval(diagnostic)
    cleaned = redact_secrets(diagnostic)
    assert 'Canary' not in cleaned
    assert __import__('ast').literal_eval(cleaned).replace('<hidden>', '') == ''
    assert 'Canary' not in Privacy(tmp_path, {}).text(diagnostic)
    code = "value = 'YWJjZGVm' 'Z2hpamtsbW5vcA=='"
    assert redact_secrets(code) == code
    ordinary = "'hf_hub_download' 'failed to download'"
    assert redact_secrets(ordinary) == ordinary
