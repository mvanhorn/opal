"""Closing a training run also stops its background mailbox wait."""

from test_mailbox_backend import mailbox_for, workspace


def test_cancelling_a_mailbox_wait_stops_the_poll_loop(tmp_path, monkeypatch):
    ws = workspace(tmp_path)
    backend = mailbox_for(tmp_path, tmp_path / "box", poll=0.01, broker_wait=0, timeout=1e6)
    polls = []

    def cancel_on_poll(seconds):
        polls.append(seconds)
        backend.cancel()
        assert len(polls) < 3, "cancelled mailbox kept polling"

    monkeypatch.setattr(backend, "_wait", cancel_on_poll)
    out = backend.decide(ws, "message", "system", tmp_path)
    assert out["returncode"] != 0 and "cancel" in out["error"]
    assert len(polls) == 1
    assert backend.late is None
