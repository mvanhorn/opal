"""Join must reject uneven CPU DDP inputs on every rank within a bounded time."""

import copy
import json
import subprocess
import sys
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.algorithms.join import Join
from torch.nn.parallel import DistributedDataParallel as DDP

import opal


ERROR = (
    "opal: uneven inputs across DDP ranks (torch.distributed.algorithms.Join) are not supported; "
    "give every rank the same number of steps"
)


def _worker(rank, folder, case):
    torch.set_num_threads(1)
    root = Path(folder)
    dist.init_process_group(
        "gloo", init_method=(root / "rendezvous").as_uri(), rank=rank, world_size=2,
        timeout=timedelta(seconds=15),
    )
    opt = None
    try:
        torch.manual_seed(42)
        model = DDP(torch.nn.Linear(3, 2))
        initial = copy.deepcopy(model.state_dict())
        data = [(torch.randn(4, 3) + rank, torch.randn(4, 2)) for _ in range(4)]

        def train(optimizer, batches):
            for x, y in batches:
                optimizer.zero_grad()
                loss = (model(x) - y).square().mean()
                loss.backward()
                if isinstance(optimizer, opal.AgentOptimizer):
                    optimizer.step(loss=loss)
                else:
                    optimizer.step()

        raw = torch.optim.AdamW(model.parameters(), lr=0.01)
        train(raw, data)
        expected = copy.deepcopy(model.state_dict())
        model.load_state_dict(initial)
        raw = torch.optim.AdamW(model.parameters(), lr=0.01)
        opt = opal.wrap(
            raw, model, total_steps=8, goal="Join regression", backend="keep", every=2,
            decisions=2, sync_every=1, rank_check_every=1, share_code=False,
            summarize=False, verbose=False, run_dir=root / "run",
        )
        steps = 4
        if case.startswith("uneven"):
            _, short_rank, short_steps = case.split("-")
            if rank == int(short_rank):
                steps = int(short_steps)
        result = dict(error=None)
        try:
            with Join([model, opt]) if case != "normal" else nullcontext():
                train(opt, data[:steps])
        except RuntimeError as exc:
            result["error"] = str(exc)
        result["updates"] = opt.update
        result["equal"] = all(
            torch.equal(value.view(torch.uint8), expected[key].view(torch.uint8))
            for key, value in model.state_dict().items()
        )
        (root / f"rank-{rank}.json").write_text(json.dumps(result))
    finally:
        if opt is not None:
            opt.close()
        dist.destroy_process_group()


def _run(tmp_path, case):
    # A subprocess deadline also bounds failures inside the Join context's shadow collectives.
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), str(tmp_path), case],
        capture_output=True, text=True, timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return [json.loads((tmp_path / f"rank-{rank}.json").read_text()) for rank in range(2)]


@pytest.mark.parametrize("short_rank", [0, 1])
@pytest.mark.parametrize("short_steps", [0, 2])
def test_join_uneven_inputs_stop_every_rank(tmp_path, short_rank, short_steps):
    ranks = _run(tmp_path, f"uneven-{short_rank}-{short_steps}")
    assert [rank["error"] for rank in ranks] == [ERROR, ERROR]
    assert [rank["updates"] for rank in ranks] == [short_steps, short_steps]


@pytest.mark.parametrize("case", ["normal", "join"])
def test_even_ddp_keep_is_bit_identical(tmp_path, case):
    ranks = _run(tmp_path, case)
    assert all(rank["error"] is None and rank["equal"] and rank["updates"] == 4 for rank in ranks)


if __name__ == "__main__":
    mp.spawn(_worker, args=(sys.argv[1], sys.argv[2]), nprocs=2, join=True)
