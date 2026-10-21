#!/usr/bin/env python3
"""Small GPT on FineWeb (GPT-2 tokens), one GPU, AdamW with linear warmup and cosine decay.

A plain single-file PyTorch script. Model: 33.5M parameters (8 layers, d=384, 6 heads), pre-norm, RoPE, causal
flash attention through scaled_dot_product_attention, GELU MLP, tied input/output embeddings, bf16 autocast.
Data: the GPT-2-tokenized FineWeb shards used by modded-nanogpt (Hugging Face dataset kjj0/fineweb10B-gpt2,
200 MB per 100M-token shard), downloaded on first use and checked by sha256. The training tokens are kept on the
GPU and read in a fixed order, so a run has no data-loader work and the same seed gives the same batches.

    python gpt_small.py                  # plain run
    python gpt_small.py --compile        # torch.compile(model)
    python gpt_small.py --opal           # optimizer wrapped by opal; nothing else changes
    python gpt_small.py --data synthetic --device cpu --steps 20 --n_layer 2 --d_model 64 --n_head 2 \
        --seq_len 64 --batch_size 4 --val_tokens 4096          # seconds on a laptop CPU

At the end it prints and (with --out) writes: final train loss (mean of the last 50 steps), held-out val loss,
median steps/s over steady-state steps (CUDA events, no per-step host sync), wall times, peak memory, and sha256
checksums of the final parameters and of the per-step loss sequence, for bitwise comparisons between runs.
"""

import argparse
import functools
import hashlib
import json
import math
import os
import socket
import sys
import time
import urllib.request

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("--steps", type=int, default=6000)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--seq_len", type=int, default=1024)
parser.add_argument("--n_layer", type=int, default=8)
parser.add_argument("--n_head", type=int, default=6)
parser.add_argument("--d_model", type=int, default=384)
parser.add_argument("--lr", type=float, default=1.5e-3)
parser.add_argument("--min_lr_frac", type=float, default=0.1, help="final LR as a fraction of --lr")
parser.add_argument("--warmup_steps", type=int, default=200)
parser.add_argument("--weight_decay", type=float, default=0.1)
parser.add_argument("--grad_clip", type=float, default=1.0, help="0 disables clipping")
parser.add_argument("--val_tokens", type=int, default=2**21)
parser.add_argument("--data", choices=["fineweb", "synthetic"], default="fineweb")
parser.add_argument("--data_dir", default="./data/fineweb10B-gpt2")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--device", default="cuda")
parser.add_argument("--compile", action="store_true", help="torch.compile(model)")
parser.add_argument(
    "--deterministic", action="store_true", help="deterministic kernels (for bitwise run-to-run comparisons; slower)"
)
parser.add_argument("--opal", action="store_true", help="wrap the optimizer with opal.wrap(...)")
parser.add_argument("--backend", default="claude", help="claude, keep, or mailbox:<dir>")
parser.add_argument("--model", default=None, help="agent model (default: claude-opus-5-5, at effort high)")
parser.add_argument(
    "--schedule", choices=["front", "uniform"], default="front", help="when the decisions are asked (wrap's schedule)"
)
parser.add_argument("--effort", default=None, help="agent effort (default: high)")
parser.add_argument("--save", default=None, help="write the final weights (model.pt) into this folder")
parser.add_argument("--every", type=int, default=50, help="updates between checks (losses, analysis, decisions)")
parser.add_argument("--decisions", type=int, default=50, help="how many times the agent is asked over the run")
parser.add_argument(
    "--blocking",
    nargs="?",
    const="true",
    default="true",
    choices=["true", "false", "auto"],
    help="true (default): training waits for each answer; false: it keeps going; auto: it waits "
    "only when an answer would land late",
)
parser.add_argument("--max_cost_usd", type=float, default=15.0, help="stop asking the agent after this much spend")
parser.add_argument(
    "--opal_kwargs", default="{}", help="JSON dict of extra keyword arguments for opal.wrap"
)
parser.add_argument("--log_every", type=int, default=100)
parser.add_argument("--timing_skip", type=int, default=50, help="first steps left out of the steps/s median")
parser.add_argument("--out", default=None, help="write the result JSON here")
parser.add_argument("--prepare_data", action="store_true", help="download the data, then exit")
args = parser.parse_args()

if args.deterministic:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device = torch.device(args.device)
script_t0 = time.perf_counter()

# ----------------------------------------------------------------------------------------------------------- data
HF_BASE = (
    os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/") + "/datasets/kjj0/fineweb10B-gpt2/resolve/main/"
)
SHARD_SHA256 = {  # LFS sha256 published by Hugging Face (x-linked-etag), checked 2026-10-16
    "fineweb_val_000000.bin": "5b95c8e0966f0861685b307b23dc5ae42b228ef74b28cb499784ae021f201640",
    "fineweb_train_000001.bin": "771fa4a99b9fe0946ffb6e848b4ba5c6a9b0fe87860ebf03bc2c1c7e45f8178e",
}
# llm.c / modded-nanogpt shard format: 256 int32 header (magic, version, n_tokens), then uint16
SHARD_MAGIC = 20240520
SHARD_VERSION = 1
HEADER_INTS = 256


def download(name, path):
    url, part = HF_BASE + name, path + ".part"
    print(f"downloading {url}", flush=True)
    h, t0 = hashlib.sha256(), time.time()
    with urllib.request.urlopen(url, timeout=60) as r, open(part, "wb") as f:
        while chunk := r.read(1 << 22):
            f.write(chunk)
            h.update(chunk)
    want = SHARD_SHA256.get(name)
    if want is not None and h.hexdigest() != want:
        os.remove(part)
        sys.exit(f"sha256 mismatch for {name}: got {h.hexdigest()}, expected {want}")
    os.replace(part, path)
    print(f"  {os.path.getsize(path) / 1e6:.0f} MB in {time.time() - t0:.0f} s", flush=True)


def read_shard(path):
    header = np.fromfile(path, dtype=np.int32, count=HEADER_INTS)
    assert header[0] == SHARD_MAGIC and header[1] == SHARD_VERSION, f"bad shard header in {path}"
    return np.memmap(path, dtype=np.uint16, mode="r", offset=HEADER_INTS * 4, shape=(int(header[2]),))


def load_tokens(split, n):
    """First n tokens of the split (train shards start at 000001, val is 000000), as int32 on the device."""
    if args.data == "synthetic":
        g = torch.Generator().manual_seed(args.seed + (0 if split == "train" else 1))
        return torch.randint(0, 50257, (n,), generator=g, dtype=torch.int32).to(device)
    os.makedirs(args.data_dir, exist_ok=True)
    parts, have, idx = [], 0, (1 if split == "train" else 0)
    while have < n:
        name = f"fineweb_{split}_{idx:06d}.bin"
        path = os.path.join(args.data_dir, name)
        if not os.path.exists(path):
            download(name, path)
        shard = read_shard(path)
        take = min(n - have, len(shard))
        parts.append(np.asarray(shard[:take]))
        have += take
        idx += 1
        if split == "val" and have < n:
            sys.exit(f"val split has only {have} tokens, asked for {n}")
    return torch.from_numpy(np.concatenate(parts).astype(np.int32)).to(device)


B, T = args.batch_size, args.seq_len
train_tokens = load_tokens("train", args.steps * B * T + 1)
val_tokens = load_tokens("val", (args.val_tokens // (B * T)) * B * T + 1)
if args.prepare_data:
    print(f"data ready: {len(train_tokens)} train and {len(val_tokens)} val tokens")
    sys.exit(0)


def get_batch(tokens, i):
    """Batch i of a fixed sequential order: B contiguous rows of T tokens, targets shifted by one."""
    chunk = tokens[i * B * T : (i + 1) * B * T + 1].long()
    return chunk[:-1].view(B, T), chunk[1:].view(B, T)


# ---------------------------------------------------------------------------------------------------------- model
def rope_tables(seq_len, head_dim, base=10000.0):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    # freqs: T, head_dim/2; the returned tables are 1, T, 1, head_dim/2
    freqs = torch.outer(torch.arange(seq_len, dtype=torch.float32), inv_freq)
    return freqs.cos()[None, :, None, :], freqs.sin()[None, :, None, :]


def apply_rope(x, cos, sin):  # x: B, T, H, head_dim
    x1, x2 = x.chunk(2, dim=-1)
    cos, sin = cos[:, : x.shape[1]].to(x.dtype), sin[:, : x.shape[1]].to(x.dtype)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class Block(nn.Module):
    def __init__(self, d, n_head):
        super().__init__()
        self.n_head = n_head
        self.ln1 = nn.LayerNorm(d, bias=False)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.attn_proj = nn.Linear(d, d, bias=False)
        self.ln2 = nn.LayerNorm(d, bias=False)
        self.fc = nn.Linear(d, 4 * d, bias=False)
        self.mlp_proj = nn.Linear(4 * d, d, bias=False)

    def forward(self, x, cos, sin):
        b, t, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(b, t, 3, self.n_head, d // self.n_head).unbind(2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
        x = x + self.attn_proj(y.transpose(1, 2).reshape(b, t, d))
        return x + self.mlp_proj(F.gelu(self.fc(self.ln2(x))))


class GPT(nn.Module):
    def __init__(self, vocab_size, n_layer, n_head, d_model, seq_len):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model, bias=False)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight  # tied
        cos, sin = rope_tables(seq_len, d_model // n_head)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        # GPT-2 init; residual projections scaled by 1/sqrt(2 n_layer)
        for name, p in self.named_parameters():
            if p.dim() == 2:
                std = 0.02 / math.sqrt(2 * n_layer) if name.endswith(("attn_proj.weight", "mlp_proj.weight")) else 0.02
                nn.init.normal_(p, mean=0.0, std=std)

    def forward(self, idx, targets):
        x = self.wte(idx)
        for block in self.blocks:
            x = block(x, self.cos, self.sin)
        logits = self.lm_head(self.ln_f(x)).float()
        return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))


torch.manual_seed(args.seed)
# vocab 50257 padded to 50304
raw_model = GPT(50304, args.n_layer, args.n_head, args.d_model, T).to(device)
n_params = sum(p.numel() for p in raw_model.parameters())

# ------------------------------------------------------------------------------------------- optimizer, schedule
decay = [p for p in raw_model.parameters() if p.dim() >= 2]
no_decay = [p for p in raw_model.parameters() if p.dim() < 2]
opt = torch.optim.AdamW(
    [{"params": decay, "weight_decay": args.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
    lr=args.lr,
    betas=(0.9, 0.95),
    eps=1e-8,
    fused=(device.type == "cuda"),
)


def warmup_cosine(step, warmup, total, min_frac):
    """LR multiplier: linear warmup over `warmup` steps, then cosine decay to `min_frac` at `total`."""
    if step < warmup:
        return (step + 1) / warmup
    t = min(1.0, (step - warmup) / max(1, total - warmup))
    return min_frac + (1.0 - min_frac) * 0.5 * (1.0 + math.cos(math.pi * t))


sched = torch.optim.lr_scheduler.LambdaLR(
    opt, functools.partial(warmup_cosine, warmup=args.warmup_steps, total=args.steps, min_frac=args.min_lr_frac)
)

GOAL = (
    f"Small GPT ({args.n_layer} layers, d={args.d_model}, {n_params / 1e6:.1f}M parameters) trained on FineWeb "
    f"GPT-2 tokens for {args.steps} steps of {B}x{T} tokens on one GPU; judged by validation loss at the last step."
)
if args.opal:
    import opal

    opt = opal.wrap(
        opt,
        raw_model,
        scheduler=sched,
        total_steps=args.steps,
        goal=GOAL,
        **{
            "backend": args.backend,
            "max_cost_usd": args.max_cost_usd,
            "every": args.every,
            "decisions": args.decisions,
            "schedule": args.schedule,
            "blocking": {"true": True, "false": False}.get(args.blocking, args.blocking),
            "backend_options": {k: v for k, v in dict(model=args.model, effort=args.effort).items() if v} or None,
            **json.loads(args.opal_kwargs),
        },
    )

model = torch.compile(raw_model) if args.compile else raw_model


# ------------------------------------------------------------------------------------------------------- training
class StepTimer:
    """Per-step durations without host syncs: CUDA events on a GPU, perf_counter on a CPU."""

    def __init__(self, n):
        self.cuda = device.type == "cuda"
        self.marks = [torch.cuda.Event(enable_timing=True) for _ in range(n + 1)] if self.cuda else [0.0] * (n + 1)

    def mark(self, i):
        if self.cuda:
            self.marks[i].record()
        else:
            self.marks[i] = time.perf_counter()

    def seconds(self):
        if self.cuda:
            torch.cuda.synchronize()
            return [a.elapsed_time(b) / 1e3 for a, b in zip(self.marks[:-1], self.marks[1:])]
        return [b - a for a, b in zip(self.marks[:-1], self.marks[1:])]


def sync():
    if device.type == "cuda":
        torch.cuda.synchronize()


def autocast():
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda"))


def current_lr():
    groups = getattr(opt, "param_groups", None)
    return groups[0]["lr"] if groups else float("nan")


@torch.no_grad()
def evaluate():  # eager model for evaluation, so eager and compiled runs are scored by the same code
    raw_model.eval()
    n = (len(val_tokens) - 1) // (B * T)
    total = torch.zeros((), device=device)
    for i in range(n):
        x, y = get_batch(val_tokens, i)
        with autocast():
            total += raw_model(x, y)
    raw_model.train()
    return (total / n).item()


print(
    f"gpt_small: {n_params / 1e6:.2f}M params, {args.steps} steps x {B}x{T} tokens, device "
    f"{torch.cuda.get_device_name(device) if device.type == 'cuda' else device}, torch {torch.__version__}, "
    f"compile={args.compile} opal={args.opal} deterministic={args.deterministic}",
    flush=True,
)
if device.type == "cuda":
    torch.cuda.reset_peak_memory_stats(device)
timer = StepTimer(args.steps)
loss_hist = torch.zeros(args.steps, device=device)
log = []
sync()
train_t0 = time.perf_counter()
for step in range(args.steps):
    timer.mark(step)
    x, y = get_batch(train_tokens, step)
    with autocast():
        loss = model(x, y)
    loss.backward()
    if args.grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(raw_model.parameters(), args.grad_clip)
    lr_used = current_lr()  # the LR this update uses (read before the scheduler moves it)
    if args.opal:
        opt.step(loss=loss)
    else:
        opt.step()
    opt.zero_grad(set_to_none=True)
    sched.step()
    loss_hist[step] = loss.detach()
    if step % args.log_every == 0 or step == args.steps - 1:
        entry = {"step": step, "loss": loss.item(), "lr": lr_used, "seconds": time.perf_counter() - train_t0}
        log.append(entry)
        print(
            f"step {step:5d}/{args.steps}  loss {entry['loss']:.4f}  lr {entry['lr']:.2e}  {entry['seconds']:.0f}s",
            flush=True,
        )
timer.mark(args.steps)
sync()
train_seconds = time.perf_counter() - train_t0

step_s = timer.seconds()
steady = step_s[args.timing_skip :] if len(step_s) > args.timing_skip else step_s
median_step = float(np.median(steady))
peak_alloc = torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else None
peak_reserved = torch.cuda.max_memory_reserved(device) / 2**30 if device.type == "cuda" else None
losses = loss_hist.cpu()
val_loss = evaluate()
if args.opal:  # into the run's records and report; the agent does not see it
    opt.log(val_loss=val_loss)


def sha256_tensors(tensors):
    h = hashlib.sha256()
    for t in tensors:
        h.update(t.detach().reshape(-1).cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


from opal.results import save_weights, summary  # noqa: E402

result = {
    **summary("val_loss", val_loss, "lower", train_seconds, opt if args.opal else None, device),
    "script": "gpt_small.py",
    "args": vars(args),
    "argv": sys.argv,
    "host": socket.gethostname(),
    "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "n_params": n_params,
    "tokens_per_step": B * T,
    "goal": GOAL,
    "final_train_loss_last50": float(losses[-50:].mean()),
    "val_loss": val_loss,
    "steps_per_s_median": 1.0 / median_step,
    "tokens_per_s_median": B * T / median_step,
    "timing_steps_used": len(steady),
    "train_loop_seconds": train_seconds,
    "first_step_seconds": step_s[0],
    "total_seconds": time.perf_counter() - script_t0,
    "peak_mem_allocated_gib": peak_alloc,
    "peak_mem_reserved_gib": peak_reserved,
    "param_sha256": sha256_tensors(p for _, p in raw_model.named_parameters()),
    "loss_sha256": sha256_tensors([losses]),
    "log": log,
}
print(
    f"RESULT train_loss_last50={result['final_train_loss_last50']:.4f} val_loss={val_loss:.4f} "
    f"steps/s={result['steps_per_s_median']:.2f} tokens/s={result['tokens_per_s_median']:.0f} "
    f"train_loop={train_seconds:.0f}s total={result['total_seconds']:.0f}s peak_alloc_gib={peak_alloc} "
    f"param_sha256={result['param_sha256'][:16]} loss_sha256={result['loss_sha256'][:16]}",
    flush=True,
)
if args.save:
    save_weights(args.save, model=raw_model.state_dict())
if args.out:
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)
