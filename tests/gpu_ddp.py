"""GPU data-parallel check (NCCL), run with torchrun on 2+ GPUs:

    torchrun --nproc_per_node=2 tests/gpu_ddp.py --mode plain|keep|agent [--compile] [--steps 600] [--out result.json]

plain: no wrapper. keep: wrapper with the starting program, every decision keeps it. agent: a fake agent that sleeps
and rewrites updater.py twice (asynchronously); checks that every rank swaps at the same update and that parameters
stay identical across ranks. Prints median steps/s and writes a JSON summary from rank 0.
"""

import argparse
import copy
import hashlib
import json
import os
import statistics
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

FOREACH_ADAMW = """
import copy

import torch


class Updater:
    follow_user_scheduler = False

    def __init__(self, named_params, previous=None, spec=None):
        self.named = named_params
        self.total = spec["total_updates"]
        self.opt = torch.optim.AdamW([p for _, p in named_params], lr=LR, betas=(0.9, 0.95), weight_decay=0.1, foreach=True)
        if previous is not None and hasattr(previous, "opts"):
            try:
                self.opt.load_state_dict(copy.deepcopy(previous.opts[0].state_dict()))
            except (ValueError, KeyError):
                pass
        for g in self.opt.param_groups:
            g["lr"] = LR

    @property
    def param_groups(self):
        return self.opt.param_groups

    def step(self, grads, update):
        scale = max(0.0, 1.0 - update / self.total)
        for g in self.opt.param_groups:
            g["lr"] = LR * scale
        self.opt.step()

    def state_dict(self):
        return self.opt.state_dict()

    def load_state_dict(self, s):
        self.opt.load_state_dict(s)


def build(named_params, previous=None, spec=None):
    return Updater(named_params, previous, spec)
"""


class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, h, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        y = self.ln1(x)
        mask = torch.triu(torch.ones(x.shape[1], x.shape[1], dtype=torch.bool, device=x.device), 1)
        x = x + self.attn(y, y, y, attn_mask=mask, need_weights=False)[0]
        return x + self.mlp(self.ln2(x))


class TinyLM(nn.Module):
    def __init__(self, vocab=4096, d=384, layers=6, heads=6, ctx=256):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.pos = nn.Parameter(torch.zeros(1, ctx, d))
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(layers))
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)

    def forward(self, idx):
        x = self.emb(idx) + self.pos[:, : idx.shape[1]]
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["plain", "keep", "agent"], required=True)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--run_dir", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--agent_delay", type=float, default=5.0, help="seconds the fake agent takes per decision")
    ap.add_argument(
        "--deterministic", action="store_true", help="deterministic kernels, for bitwise plain-vs-keep checks"
    )
    a = ap.parse_args()

    if a.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local)
    dev = torch.device("cuda", local)
    torch.manual_seed(0)
    model = TinyLM().to(dev)
    ddp = nn.parallel.DistributedDataParallel(model, device_ids=[local])
    fwd = torch.compile(ddp) if a.compile else ddp
    lr = 1e-3
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1, fused=True)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 50))

    if a.mode != "plain":
        import opal

        def agent(ws, msg, system):
            time.sleep(a.agent_delay)
            if "Decision 1." in msg or "Decision 3." in msg:
                (ws / "updater.py").write_text(
                    FOREACH_ADAMW.replace("LR", repr(lr * (1.0 if "Decision 1." in msg else 0.5)))
                )
            return "fake agent"

        backend = "keep" if a.mode == "keep" else agent
        opt = opal.wrap(
            opt,
            ddp,
            scheduler=sched,
            total_steps=a.steps,
            goal="GPU DDP test",
            backend=backend,
            decisions=6,
            every=25,
            sync_every=20,
            rank_check_every=200,
            run_dir=a.run_dir,
        )

    g = torch.Generator(device=dev).manual_seed(1000 + rank)
    times, losses = [], []
    torch.cuda.synchronize()
    t_start = time.time()
    for step in range(a.steps):
        t0 = time.perf_counter()
        idx = torch.randint(0, 4096, (a.batch, 257), device=dev, generator=g)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = fwd(idx[:, :-1])
        loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), idx[:, 1:].reshape(-1))
        loss.backward()
        opt.step(loss=loss) if a.mode != "plain" else opt.step()
        opt.zero_grad(set_to_none=True)
        sched.step()
        losses.append(loss.detach())
        if step % 50 == 49:
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    torch.cuda.synchronize()
    wall = time.time() - t_start
    flat = torch.cat([p.detach().float().flatten() for p in model.parameters()])
    s = flat.sum().reshape(1)
    lo, hi = s.clone(), s.clone()
    dist.all_reduce(lo, op=dist.ReduceOp.MIN)
    dist.all_reduce(hi, op=dist.ReduceOp.MAX)
    steady = sorted(times[100:]) if len(times) > 200 else sorted(times)
    res = dict(
        mode=a.mode,
        compile=a.compile,
        world=world,
        steps=a.steps,
        wall_s=round(wall, 2),
        median_steps_per_s=round(1 / statistics.median(steady), 2),
        ranks_identical=bool(torch.equal(lo, hi)),
        params_sha=hashlib.sha256(flat.cpu().numpy().tobytes()).hexdigest()[:16],
        loss_sha=hashlib.sha256(torch.stack(losses).float().cpu().numpy().tobytes()).hexdigest()[:16],
        final_loss=float(torch.stack(losses[-20:]).mean()),
        peak_mem_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2),
        version=getattr(opt, "version", None),
        gpu=torch.cuda.get_device_name(dev),
    )
    if rank == 0:
        print(json.dumps(res))
        if a.out:
            with open(a.out, "w") as f:
                json.dump(res, f, indent=1)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
