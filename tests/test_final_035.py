"""Offline regression cases for the final 0.3.5 review."""
import ast
import io
import json
import multiprocessing
import time
from collections import Counter, OrderedDict

import pytest
import torch

from opal import programs
from opal.backends import mailbox
from opal.backends._privacy import redact_secrets, redact_stream
from test_release_035 import UPDATER, wrap


@pytest.mark.parametrize('hook', ['__reduce__', '__reduce_ex__', '__getstate__'])
@pytest.mark.parametrize('location', ['value', 'tensor', 'parameter', 'ordered', 'counter'])
def test_state_guard_never_runs_custom_serialization(hook, location):
    calls = []
    def reducer(*args):
        calls.append(True)
        return (torch.Size, ([2, 3],)) if len(calls) == 1 else (eval, ('1+1',))
    custom = type('Custom', (), {hook: reducer})()
    if location == 'value':
        value = {'custom': custom}
    else:
        value = {'tensor': torch.tensor(1), 'parameter': torch.nn.Parameter(torch.ones(1)),
                 'ordered': OrderedDict(), 'counter': Counter()}[location]
        value.custom = custom
    assert programs.unpicklable(value)
    assert calls == []


@pytest.mark.parametrize('kind', [torch.Tensor, torch.nn.Parameter, OrderedDict, Counter])
def test_state_guard_rejects_cycles_through_python_attributes(kind):
    value = kind(torch.ones(1)) if kind in (torch.Tensor, torch.nn.Parameter) else kind()
    value.loop = value
    assert 'cyclic' in programs.unpicklable(value)[0]


def test_state_guard_checks_nested_and_subclass_values_without_hooks():
    class Subdict(dict):
        def items(self):
            raise AssertionError('user code ran')
    assert programs.unpicklable(Subdict())
    value = 1
    for _ in range(102):
        value = [value]
    assert 'nesting' in programs.unpicklable(value)[0]
    assert not programs.unpicklable({'values': [None, True, 1, 1., 1j, 's', b'b',
        torch.Size([1]), torch.float32, torch.device('cpu'), torch.strided]})


def test_checkpoint_checks_the_single_state_that_it_returns(tmp_path):
    model = torch.nn.Linear(1, 1)
    def agent(ws, message, system):
        (ws / 'updater.py').write_text(UPDATER)
        return 'offline updater'
    opt = wrap(model, torch.optim.SGD(model.parameters(), lr=.1), tmp_path / 'run',
               backend=agent, schedule=[1])
    class Custom:
        def __reduce__(self):
            raise AssertionError('checkpoint ran a rejected reducer')
    try:
        opt.step(loss=1.)
        assert opt.version == 1
        calls = []
        def state():
            calls.append(True)
            return {'custom': Custom()}
        opt.updater.state_dict = state
        saved = opt.state_dict()
        assert len(calls) == 1 and saved['updater'] is None
        stream = io.BytesIO()
        torch.save(saved, stream)
        stream.seek(0)
        assert torch.load(stream, weights_only=True)['updater'] is None
    finally:
        opt.close(interrupted=True)


KEY = 'sk-ant-FAKE0000000000000000'


@pytest.mark.parametrize('quote', ["'", '"', "'''", '"""'])
@pytest.mark.parametrize('gap', ['', ' ', '\n'])
@pytest.mark.parametrize('prefix', ['', 'b'])
def test_redaction_literal_forms(quote, gap, prefix):
    diagnostic = prefix + quote + KEY[:3] + quote + gap + prefix + quote + KEY[3:] + quote
    # Newlines between literals are valid inside a parenthesized diagnostic expression.
    expression = '(' + diagnostic + ')'
    assert ast.literal_eval(expression) in (KEY, KEY.encode())
    cleaned = redact_secrets(expression)
    assert 'FAKE' not in cleaned
    assert ast.literal_eval(cleaned) in ('<hidden><hidden>', b'<hidden><hidden>')
    stream = io.StringIO()
    redact_stream(io.StringIO(json.dumps({'error': diagnostic}) + '\n'), stream)
    assert 'FAKE' not in stream.getvalue()
    json.loads(stream.getvalue())


@pytest.mark.parametrize('diagnostic', [
    'Authorization=Bearer+FAKE0000000000000000&step=2',
    'Authorization=bearer+FAKE0000000000000000&step=2',
    'access_token=FAKE0000000000000000&step=2',
    'access_token=FAKE%2B0000000000000000&step=2',
])
def test_redaction_form_fields(diagnostic):
    cleaned = redact_secrets(diagnostic)
    assert 'FAKE' not in cleaned and cleaned.endswith('&step=2')
    assert cleaned.startswith(diagnostic.split('=', 1)[0] + '=')


def _claim_racer(folder, barrier, queue, me):
    box = mailbox._Folder(folder)
    barrier.wait(15)
    log = []
    won = mailbox._claim(box, 1, 'attempt', me, log.append)
    queue.put((won, log, box._held if won else {}))


@pytest.mark.parametrize('same_broker', [False, True])
def test_stale_claim_has_one_winner_in_two_processes(tmp_path, same_broker):
    box = mailbox._Folder(tmp_path)
    old = dict(broker='old', attempt='attempt', t=time.time() - 1000, generation='old-generation')
    box.write('claim-0001.json', json.dumps(old).encode())
    ctx = multiprocessing.get_context('spawn')
    barrier, queue = ctx.Barrier(2), ctx.Queue()
    processes = [ctx.Process(target=_claim_racer, args=(str(tmp_path), barrier, queue,
                  'same' if same_broker else str(i))) for i in range(2)]
    try:
        for process in processes:
            process.start()
        rows = [queue.get(timeout=30) for _ in processes]
        for process in processes:
            process.join(30)
            assert process.exitcode == 0
        if same_broker:
            # A restart using the same stable id may replace a live generation immediately.
            assert all(row[0] for row in rows)
        else:
            assert sum(row[0] for row in rows) == 1
            assert any('backing off' in line for row in rows if not row[0] for line in row[1])
        winner = json.loads(box.read('claim-0001.json'))
        for won, _, held in rows:
            if won and held['claim-0001.json']['generation'] != winner['generation']:
                assert not box.lease('claim-0001.json', 'answer', body={'stale': True},
                                     expected=held['claim-0001.json'])
        assert not box.lease('claim-0001.json', 'renew', expected=old)
        assert not box.lease('claim-0001.json', 'release', expected=old)
        assert not box.lease('claim-0001.json', 'answer', body={'stale': True}, expected=old)
        assert box.read('response-0001.json') is None
        assert box.lease('claim-0001.json', 'answer', body={'winner': True}, expected=winner)
        assert json.loads(box.read('response-0001.json')) == {'winner': True}
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(5)
        queue.close()


@pytest.mark.parametrize('remote', [False, True])
def test_lease_fences_old_attempt_and_survives_owner_crash(tmp_path, monkeypatch, remote):
    import subprocess
    box = mailbox._Folder(tmp_path, ssh='fake' if remote else None)
    if remote:
        monkeypatch.setattr(box, '_sh', lambda cmd, data=None:
                            subprocess.run(['sh', '-c', cmd], input=data, capture_output=True, timeout=10))
    assert mailbox._claim(box, 1, 'old-attempt', 'broker', lambda msg: None)
    old = dict(box._held['claim-0001.json'])
    assert mailbox._claim(box, 1, 'new-attempt', 'broker', lambda msg: None)
    new = dict(box._held['claim-0001.json'])
    assert new['generation'] != old['generation']
    assert not box.lease('claim-0001.json', 'renew', expected=old)
    assert box.lease('claim-0001.json', 'renew', expected=new)
    assert box.lease('claim-0001.json', 'release', expected=new)
    assert mailbox._claim(box, 1, 'new-attempt', 'broker', lambda msg: None)


def test_resume_preserves_next_decision_observations_and_message(tmp_path):
    analysis = '''EVERY = 4
def analyze(obs, memory):
    return dict(report=dict(losses=list(obs.losses), metrics=list(obs.metrics)))
'''
    def agent(ws, message, system):
        return 'keep'
    def make(name):
        model = torch.nn.Linear(1, 1)
        opt = wrap(model, torch.optim.SGD(model.parameters(), lr=.1), tmp_path / name,
                   backend=agent, every=2, total_steps=12, schedule=[2, 8])
        opt._apply(dict(k=0, requested_at=0, files=dict(opt.files, **{'analysis.py': analysis})), 0)
        return opt
    uninterrupted, resumed = make('continuous'), make('resumed')
    try:
        for update in range(1, 6):
            uninterrupted.step(loss=torch.tensor(float(update)) if update % 2 else float(update))
            uninterrupted.log(visible=update, show_agent=True)
            uninterrupted.log(hidden=update * 10)
        stream = io.BytesIO()
        torch.save(uninterrupted.state_dict(), stream)
        stream.seek(0)
        saved = torch.load(stream, weights_only=True)
        resumed.load_state_dict(saved)
        # A resumed checkpoint includes a truthful resume note; it is unrelated to observations.
        resumed._resume_note = ''
        for opt in (uninterrupted, resumed):
            for update in range(6, 9):
                opt.step(loss=torch.tensor(float(update)) if update % 2 else float(update))
                opt.log(visible=update, show_agent=True)
                opt.log(hidden=update * 10)
        def inbox(opt):
            value = json.loads((opt.run.workspace / 'inbox/decision-002.json').read_text())
            value.pop('timing')
            return value
        assert inbox(uninterrupted) == inbox(resumed)
        assert inbox(resumed)['losses'] == list(range(3, 9))
        assert 'hidden' not in json.dumps(inbox(resumed))
        def message(opt):
            text = (opt.run.decisions / '002/message.txt').read_text()
            return '\n'.join(line for line in text.splitlines()
                             if not line.startswith(('Timing:', 'Profile of one update:')))
        assert message(uninterrupted) == message(resumed)
        assert 'visible' in message(resumed)
    finally:
        uninterrupted.close(interrupted=True)
        resumed.close(interrupted=True)


def test_legacy_checkpoint_outside_new_allowlist_still_restores(tmp_path):
    model = torch.nn.Linear(1, 1)
    opt = wrap(model, torch.optim.SGD(model.parameters(), lr=.1), tmp_path / 'legacy')
    try:
        legacy = opt.state_dict()
        # 0.3.4 restored this bytearray bookkeeping through its weights-only checkpoint loader.
        legacy['files'] = dict(legacy['files'], **{'updater.py': UPDATER.replace(
            'dict(k=self.k)', "dict(k=self.k, metadata=bytearray(b'ab'))")})
        legacy['updater'] = dict(k=3, metadata=bytearray(b'ab'))
        legacy['version'] = 1
        stream = io.BytesIO()
        torch.save(legacy, stream)
        stream.seek(0)
        opt.load_state_dict(torch.load(stream, weights_only=True))
        assert opt.version == 1 and opt.updater.k == 3
        # New outgoing checkpoints continue to enforce the requested closed allowlist.
        assert opt.state_dict()['updater'] is None
    finally:
        opt.close(interrupted=True)


@pytest.mark.parametrize('value', [bytearray(b'ab'), torch.per_tensor_affine])
def test_safe_loader_types_outside_closed_allowlist_are_rejected_for_new_state(value):
    stream = io.BytesIO()
    torch.save(value, stream)
    stream.seek(0)
    torch.load(stream, weights_only=True)
    assert programs.unpicklable(value)


@pytest.mark.parametrize('kind', [Counter, OrderedDict])
def test_state_guard_does_not_call_overridden_container_methods(kind):
    calls = []
    class Custom:
        def __getstate__(self):
            calls.append(True)
            return []
    value = kind()
    value.items = Custom().__getstate__
    assert programs.unpicklable(value)
    assert calls == []


def test_closed_allowlist_uses_type_identity_without_custom_metaclass_code():
    calls = []
    class Meta(type):
        def __eq__(cls, other):
            calls.append('equality')
            return True
        def __getattribute__(cls, name):
            if name in ('__module__', '__qualname__'):
                calls.append('type metadata')
            return super().__getattribute__(name)
    class Custom(metaclass=Meta):
        def __reduce__(self):
            calls.append('reducer')
            return torch.Size, ([2, 3],)
    assert "Custom" in programs.unpicklable(Custom())[0]
    assert calls == []
