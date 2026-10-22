"""CPU/offline regression witnesses from review_035c/repro."""
import io
import json
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from opal import cli
from opal.backends import mailbox
from opal.backends.base import KeepBackend
from opal.misc import aggressive
from test_release_035 import UPDATER, wrap


@pytest.mark.parametrize('show_agent', [False, True])
@pytest.mark.parametrize('at', [4, 5])
def test_numpy_metrics_default_load_and_plain_resume(tmp_path, show_agent, at):
    model = torch.nn.Linear(3, 1)
    opt = wrap(model, torch.optim.SGD(model.parameters(), lr=.05), tmp_path / 'run', every=4, total_steps=12)
    try:
        for _ in range(at):
            opt.step(loss=1.)
        opt.log(acc=np.mean([.5, 1.]), count=np.int64(7), narrow=np.float32(.25), show_agent=show_agent)
        rec = opt._metrics_new[-1]
        assert type(rec['acc']) is float and type(rec['count']) is int and type(rec['narrow']) is float
        stream = io.BytesIO()
        torch.save(opt.state_dict(), stream)
        stream.seek(0)
        opt.load_state_dict(torch.load(stream))
        assert opt.update == at and opt._metrics_new[-1] == rec
    finally:
        opt.close(interrupted=True)


def test_numpy_metric_accelerate_load_state(tmp_path):
    from accelerate import Accelerator
    from accelerate.state import AcceleratorState, GradientState
    AcceleratorState._reset_state(True)
    GradientState._reset_state()
    acc = Accelerator(cpu=True)
    model = torch.nn.Linear(3, 1)
    model, prepared = acc.prepare(model, torch.optim.AdamW(model.parameters(), lr=.01))
    opt = wrap(model, prepared, tmp_path / 'run', every=4, total_steps=12)
    try:
        opt.step(loss=1.)
        opt.log(acc=np.mean([.5, 1.]))
        acc.save_state(tmp_path / 'checkpoint')
        acc.load_state(tmp_path / 'checkpoint')
        assert opt.update == 1 and opt._metrics_new[-1]['acc'] == .75
    finally:
        opt.close(interrupted=True)
        AcceleratorState._reset_state(True)
        GradientState._reset_state()


def test_observations_are_normalized_and_allowlisted(tmp_path):
    model = torch.nn.Linear(1, 1)
    opt = wrap(model, torch.optim.SGD(model.parameters(), lr=.1), tmp_path / 'run')
    class Unsupported:
        def __reduce__(self):
            raise AssertionError('must reject without serializing')
    try:
        opt._analysis_metrics.append(dict(acc=np.float64(.75), counts=np.array([1, 2])))
        opt.activations['unsupported'] = Unsupported()
        stream = io.BytesIO()
        torch.save(opt.state_dict(), stream)
        stream.seek(0)
        state = torch.load(stream)
        assert type(state['observations']['_analysis_metrics'][0]['acc']) is float
        assert torch.equal(state['observations']['_analysis_metrics'][0]['counts'], torch.tensor([1, 2]))
        assert state['observations']['activations'] == {}
    finally:
        opt.close(interrupted=True)


def test_ssh_lease_large_answer_uses_stdin(tmp_path, monkeypatch):
    box = mailbox._Folder(tmp_path, ssh='fake')
    seen = []
    def remote(cmd, data=None):
        seen.append((len(cmd), 0 if data is None else len(data)))
        return subprocess.run(['sh', '-c', cmd], input=data, capture_output=True, timeout=10)
    monkeypatch.setattr(box, '_sh', remote)
    assert mailbox._claim(box, 1, 'attempt', 'me', lambda _: None)
    held = box._held['claim-0001.json']
    body = dict(attempt='attempt', text='x' * 1_300_000)
    assert box.lease('claim-0001.json', 'answer', body=body, expected=held)
    assert json.loads(box.read('response-0001.json')) == body
    assert all(size < 131072 for size, _ in seen)
    assert seen[-2][1] > 1_300_000  # final call is read()


def test_failed_answer_retry_still_reads_new_requests(tmp_path, monkeypatch):
    calls, lines, retries = [], [], []
    state = dict(me='me', claim=None)
    def answer(folder, backend, work, ws, run, k, attempt, fresh, beat_state, log):
        calls.append(k)
        beat_state['claim'] = ('run/claim.json', dict(generation=str(k), attempt=attempt))
        return dict(k=k, returncode=0), json.dumps(dict(k=k)).encode()
    def lease(folder, name, action, body=None, expected=None):
        if name == 'claim-0001.json':
            retries.append(True)
            assert len(retries) < 5, 'failed writes blocked reading new requests'
            raise OSError('answer write is unavailable')
        return True
    monkeypatch.setattr(mailbox, '_answer_request', answer)
    monkeypatch.setattr(mailbox._Folder, 'lease', lease)
    monkeypatch.setattr(mailbox._Folder, 'read', lambda self, name: b'{}' if calls == [1, 2] else None)
    monkeypatch.setattr(mailbox, '_pending', lambda folder: None if calls == [1, 2] else (len(calls) + 1, 'a', 0))
    monkeypatch.setattr(mailbox, 'RETRY_SECONDS', .001)
    mailbox._serve(mailbox._Folder(tmp_path), KeepBackend(), tmp_path / 'work', None, .001,
                   'run', lines.append, state, False)
    assert calls == [1, 2]


def test_failure_after_claim_clears_heartbeat_and_releases(tmp_path, monkeypatch):
    state = dict(me='me', claim=None)
    held = dict(generation='g', attempt='a')
    calls, released = [], []
    class Finished(Exception):
        pass
    def answer(*args):
        calls.append(True)
        if len(calls) == 1:
            state['claim'] = ('run/claim-0001.json', held)
            raise OSError('ssh failed after claim')
        assert state['claim'] is None
        assert released == [('claim-0001.json', held)]
        raise Finished
    def lease(folder, name, action, body=None, expected=None):
        assert action == 'release'
        released.append((name, expected))
        raise OSError('best effort release also failed')
    monkeypatch.setattr(mailbox, '_answer_request', answer)
    monkeypatch.setattr(mailbox._Folder, 'lease', lease)
    monkeypatch.setattr(mailbox._Folder, 'read', lambda *a: None)
    monkeypatch.setattr(mailbox, '_pending', lambda *a: (1, 'a', 0))
    monkeypatch.setattr(mailbox, 'RETRY_SECONDS', .001)
    with pytest.raises(Finished):
        mailbox._serve(mailbox._Folder(tmp_path), KeepBackend(), tmp_path / 'work', None, .001,
                       'run', lambda _: None, state, False)


def test_same_broker_restart_reclaims_immediately_and_fences_old_generation(tmp_path):
    box = mailbox._Folder(tmp_path)
    assert mailbox._claim(box, 1, 'a', 'same-id', lambda _: None)
    old = box._held['claim-0001.json']
    restarted = mailbox._Folder(tmp_path)
    assert mailbox._claim(restarted, 1, 'a', 'same-id', lambda _: None)
    new = restarted._held['claim-0001.json']
    assert new['generation'] != old['generation']
    for action in ('renew', 'release', 'answer'):
        assert not box.lease('claim-0001.json', action, body={'late': True}, expected=old)
    assert restarted.lease('claim-0001.json', 'answer', body={'ok': True}, expected=new)


def test_backing_off_logged_once_per_request(tmp_path):
    owner, loser = mailbox._Folder(tmp_path), mailbox._Folder(tmp_path)
    assert mailbox._claim(owner, 1, 'a', 'owner', lambda _: None)
    lines = []
    for _ in range(5):
        assert not mailbox._claim(loser, 1, 'a', 'loser', lines.append)
    assert sum('backing off' in line for line in lines) == 1
    assert mailbox._claim(owner, 2, 'a', 'owner', lambda _: None)
    assert not mailbox._claim(loser, 2, 'a', 'loser', lines.append)
    assert sum('backing off' in line for line in lines) == 2


def test_remote_helper_python36_unlink(tmp_path, monkeypatch):
    original = Path.unlink
    def legacy_unlink(self):
        return original(self)
    monkeypatch.setattr(Path, 'unlink', legacy_unlink)
    name = tmp_path / 'claim-0001.json'
    held = dict(broker='me', generation='g', attempt='a', t=time.time())
    assert mailbox._lease_file(name, 'claim', held, None, 120)
    assert mailbox._lease_file(name, 'answer', {'ok': True}, held, 120)
    assert not name.exists()
    assert mailbox._lease_file(name, 'claim', held, None, 120)
    assert mailbox._lease_file(name, 'release', None, held, 120)


def test_doctor_mailbox_requires_python3(tmp_path, monkeypatch, capsys):
    import shutil
    box = tmp_path / 'box'
    box.mkdir()
    (box / 'broker-alive').write_text('{}')
    monkeypatch.setattr(shutil, 'which', lambda name: None)
    assert cli.main(['doctor', '--backend', f'mailbox:{box}']) == 1
    assert 'python3' in capsys.readouterr().out


@pytest.mark.parametrize('multiplier', [0., .5, 1.])
def test_controller_has_no_host_scalar_read_and_exact_endpoints(multiplier):
    class Trace(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.ops = []
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            self.ops.append(str(func))
            return func(*args, **(kwargs or {}))
    ns = dict(torch=torch, SIGNAL_DECAY=.5)
    exec(aggressive.CONTROLLER, ns)
    ns['lr_rule'] = lambda *a: torch.tensor(multiplier)
    old = [torch.tensor([1e-9, 1., -1e-9]), torch.tensor([1., 1e-9])]
    proposed = [torch.tensor([1., 1e-9, -1.]), torch.tensor([1e-9, 1.])]
    params = [v.clone() for v in old]
    controller = ns['LRController'](SimpleNamespace(groups=[dict(key='g', params=params)]), {})
    before = controller.before()
    for p, v in zip(params, proposed):
        p.copy_(v)
    with Trace() as trace:
        controller.after(before, 1, 2)
    assert not any('_local_scalar_dense' in op for op in trace.ops)
    for p, x, y in zip(params, old, proposed):
        assert torch.equal(p, x if multiplier == 0 else y if multiplier == 1 else torch.lerp(x, y, multiplier))


@pytest.mark.parametrize('value', [np.float64(.75), np.int64(7), np.array([1., 2.]), np.array([1., 2.])[::-1]])
def test_numpy_updater_state_normalized_before_allowlist_and_resume(tmp_path, value):
    model = torch.nn.Linear(1, 1)
    opt = wrap(model, torch.optim.SGD(model.parameters(), lr=.1), tmp_path / 'run')
    try:
        source = UPDATER.replace('dict(k=self.k)', 'dict(k=self.k, hist=self.hist)').replace(
            'self.named, self.k = named, 0', 'self.named, self.k = named, 0; self.hist = None').replace(
            "self.k = state['k']", "self.k = state['k']; self.hist = state['hist']")
        opt._apply(dict(k=1, requested_at=0, files=dict(opt.files, **{'updater.py': source})), 0)
        assert opt.version == 1
        opt.updater.hist = value
        saved = opt.state_dict()
        assert saved['updater'] is not None
        stream = io.BytesIO()
        torch.save(saved, stream)
        stream.seek(0)
        loaded = torch.load(stream)
        opt.load_state_dict(loaded)
        assert opt.updater.k == saved['updater']['k']
        h = opt.updater.hist
        assert torch.equal(h, torch.tensor(value.copy())) if type(value) is np.ndarray else h == value
    finally:
        opt.close(interrupted=True)


@pytest.mark.parametrize('where', ['updater', 'memory', 'observations', 'raised'])
def test_state_dropped_event_and_report_with_verbose_false(tmp_path, where):
    from collections import deque
    model = torch.nn.Linear(1, 1)
    opt = wrap(model, torch.optim.SGD(model.parameters(), lr=.1), tmp_path / 'run')
    try:
        opt._apply(dict(k=1, requested_at=0, files=dict(opt.files, **{'updater.py': UPDATER})), 0)
        assert opt.version == 1
        if where == 'updater':
            opt.updater.state_dict = lambda: {'unsupported': deque([1])}
        elif where == 'raised':
            def fail():
                raise ValueError('state failure')
            opt.updater.state_dict = fail
        elif where == 'memory':
            opt.memory = {'unsupported': lambda: None}
        else:
            opt.activations = {'unsupported': deque([1])}
        opt.state_dict()
    finally:
        opt.close(interrupted=True)
    events = [json.loads(row) for row in (opt.run.dir / 'events.jsonl').read_text().splitlines()]
    assert any(e['kind'] == 'state_dropped' and e.get('reason') for e in events)
    assert 'state_dropped' in (opt.run.dir / 'report.html').read_text()


@pytest.mark.parametrize('location', ['key', 'attributes', 'set'])
def test_normalizing_rejected_state_does_not_call_custom_container_code(tmp_path, location):
    from collections import Counter
    calls = []
    class Key:
        def __hash__(self):
            calls.append('hash')
            return 1
    class Attributes(dict):
        def __iter__(self):
            calls.append('iter')
            return super().__iter__()
        def keys(self):
            calls.append('keys')
            return super().keys()
    if location == 'key':
        value = {Key(): 1}
    elif location == 'set':
        value = {Key()}
    else:
        value = Counter()
        value.__dict__ = Attributes(x=1)
    model = torch.nn.Linear(1, 1)
    opt = wrap(model, torch.optim.SGD(model.parameters(), lr=.1), tmp_path / 'run')
    try:
        opt._apply(dict(k=1, requested_at=0, files=dict(opt.files, **{'updater.py': UPDATER})), 0)
        assert opt.version == 1
        opt.updater.state_dict = lambda: {'value': value}
        calls.clear()
        assert opt.state_dict()['updater'] is None
        assert calls == []
        events = [json.loads(row) for row in (opt.run.dir / 'events.jsonl').read_text().splitlines()]
        assert any(e['kind'] == 'state_dropped' for e in events)
    finally:
        opt.close(interrupted=True)
