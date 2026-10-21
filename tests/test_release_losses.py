"""Release regressions: all accepted loss types reach checks, including non-finite losses."""

import json
import math

import torch

import opal


def test_mixed_tensor_and_number_losses_keep_their_order_and_nonfinite_alarm(tmp_path):
    model = torch.nn.Linear(1, 1)
    messages = []
    opt = opal.wrap(
        torch.optim.SGD(model.parameters(), lr=0.1), model, total_steps=12, goal="g",
        every=4, schedule=[], max_decisions=1, backend=lambda ws, msg, system: messages.append(msg),
        run_dir=tmp_path, verbose=False,
    )
    try:
        for loss in (0.5, torch.tensor(1.0), float("nan"), torch.tensor(3.0)):
            opt.step(loss=loss)
        recorded = json.loads((tmp_path / "losses.jsonl").read_text().splitlines()[0])["losses"]
        assert len(recorded) == 4
        assert recorded[:2] == [0.5, 1.0] and math.isnan(recorded[2]) and recorded[3] == 3.0
        assert len(messages) == 1 and "non-finite training loss" in messages[0]
    finally:
        opt.close(interrupted=True)
