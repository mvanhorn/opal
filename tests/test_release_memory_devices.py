"""The analysis memory embedded in a checkpoint must follow accelerator remapping too."""

import io

import torch

import opal


def test_cpu_resume_remaps_embedded_accelerator_memory(tmp_path, monkeypatch):
    model = torch.nn.Linear(1, 1)
    opt = opal.wrap(
        torch.optim.SGD(model.parameters(), lr=0.1), model,
        total_steps=10, goal="g", backend=None, run_dir=tmp_path, verbose=False,
    )
    try:
        state = opt.state_dict()
        buf = io.BytesIO()
        # Simulate memory saved on another machine's accelerator without requiring that accelerator in CI.
        with monkeypatch.context() as patch:
            patch.setattr(torch.serialization, "location_tag", lambda storage: "cuda:9999")
            torch.save({"snapshot": torch.ones(2)}, buf)
        state["memory"] = buf.getvalue()
        opt.load_state_dict(state)
        assert opt.memory["snapshot"].device == torch.device("cpu")
        assert torch.equal(opt.memory["snapshot"], torch.ones(2))
    finally:
        opt.close(interrupted=True)
