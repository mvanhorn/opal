"""Mailbox cancellation must be bounded and scoped to the request's attempt."""

import json

import pytest

from opal.backends import mailbox
from opal.backends.base import KeepBackend
from test_mailbox_backend import workspace


@pytest.mark.parametrize("cancelled_attempt", ["old-attempt", "current-attempt"])
def test_a_cancellation_only_blocks_its_own_attempt(tmp_path, cancelled_attempt):
    folder = mailbox._Folder(tmp_path / "box")
    folder.make()
    attempt = "current-attempt"
    folder.write("cancel-0001.json", json.dumps(dict(attempt=cancelled_attempt)).encode())
    req = dict(run_id="run", attempt=attempt, k=1, files={"updater.py": "# program"}, manifest={})
    folder.write("request-0001.json", json.dumps(req).encode())
    work, ws = tmp_path / "work", workspace(tmp_path)
    backend = KeepBackend()
    backend.setup(work, ws)
    state = dict(me="test-broker", claim=None)
    result = mailbox._answer_request(folder, backend, work, ws, "run", 1, attempt, False, state, lambda _: None)
    assert (result is None) == (cancelled_attempt == attempt)
