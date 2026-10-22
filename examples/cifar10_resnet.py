#!/usr/bin/env python3
"""Fast CIFAR-10 ResNet-18 on one GPU, SGD with Nesterov momentum and a one-cycle LR schedule.

A plain single-file PyTorch script in the airbench style: the whole dataset lives on the GPU, augmentation (random
crop with 4-pixel reflect padding, horizontal flip) runs on the GPU once per epoch from a seeded generator, and there
is no DataLoader. Model: CIFAR ResNet-18 (3x3 stem, no max-pool, 11.2M parameters), bf16 autocast, channels_last.
Data: CIFAR-10 from Hugging Face (uoft-cs/cifar10 parquet, sha256-checked), with the Toronto binary archive
(md5-checked) as a fallback; decoded once into a uint8 .npz cache.

    python cifar10_resnet.py                  # plain run
    python cifar10_resnet.py --compile        # torch.compile(model)
    python cifar10_resnet.py --opal           # optimizer wrapped by opal; nothing else changes
    python cifar10_resnet.py --data synthetic --device cpu --epochs 1 --width 8 --batch_size 64   # laptop smoke test

At the end it prints and (with --out) writes: test accuracy and loss, final train loss, median steps/s over
steady-state steps (CUDA events, no per-step host sync), wall times, peak memory, and sha256 checksums of the final
parameters and of the per-step loss sequence, for bitwise comparisons between runs.
"""

import argparse
import hashlib
import io
import json
import os
import socket
import sys
import tarfile
import time
import urllib.request

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("--epochs", type=int, default=60)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--lr", type=float, default=0.4, help="peak LR of the one-cycle schedule")
parser.add_argument("--warmup_frac", type=float, default=0.2, help="fraction of steps spent warming up")
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--weight_decay", type=float, default=5e-4, help="on conv/linear weights; 0 on BN and biases")
parser.add_argument("--label_smoothing", type=float, default=0.1)
parser.add_argument("--width", type=int, default=64, help="channels of the first stage (64 = ResNet-18)")
parser.add_argument("--data", choices=["cifar10", "synthetic"], default="cifar10")
parser.add_argument("--data_dir", default="./data/cifar10")
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
    choices=["true", "false"],
    help="true (default): training waits for each answer; false: it keeps going",
)
parser.add_argument("--max_cost_usd", type=float, default=15.0, help="stop asking the agent after this much spend")
parser.add_argument(
    "--opal_kwargs", default="{}", help="JSON dict of extra keyword arguments for opal.wrap"
)
parser.add_argument("--timing_skip", type=int, default=50, help="first steps left out of the steps/s median")
parser.add_argument("--out", default=None, help="write the result JSON here")
parser.add_argument("--prepare_data", action="store_true", help="download the data, then exit")
args = parser.parse_args()

if args.deterministic:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = not args.deterministic
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device = torch.device(args.device)
script_t0 = time.perf_counter()

# ----------------------------------------------------------------------------------------------------------- data
HF_BASE = (
    os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    + "/datasets/uoft-cs/cifar10/resolve/main/plain_text/"
)
HF_FILES = {  # LFS sha256 published by Hugging Face, checked 2026-10-16
    "train-00000-of-00001.parquet": "8428b53a88a11ac374111006708df51469e315a22ac6d66470afd9c78d2ae883",
    "test-00000-of-00001.parquet": "841389e6f2d64f28bf17310e430aebac20ec3ba611a3c5e231dc93c645ce84de",
}
TORONTO_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
TORONTO_MD5 = "c32a1d4ab5d03f1284b67883e8d87530"  # published on the CIFAR page


def download(url, path, algo, want):
    part = path + ".part"
    print(f"downloading {url}", flush=True)
    h, t0 = hashlib.new(algo), time.time()
    with urllib.request.urlopen(url, timeout=60) as r, open(part, "wb") as f:
        while chunk := r.read(1 << 22):
            f.write(chunk)
            h.update(chunk)
    if h.hexdigest() != want:
        os.remove(part)
        raise RuntimeError(f"{algo} mismatch for {url}: got {h.hexdigest()}, expected {want}")
    os.replace(part, path)
    print(f"  {os.path.getsize(path) / 1e6:.0f} MB in {time.time() - t0:.0f} s", flush=True)


def from_hf(data_dir):
    import pyarrow.parquet as pq
    from PIL import Image

    out = []
    for name, sha in HF_FILES.items():
        path = os.path.join(data_dir, name)
        if not os.path.exists(path):
            download(HF_BASE + name, path, "sha256", sha)
        table = pq.read_table(path).to_pydict()
        images = np.stack([np.asarray(Image.open(io.BytesIO(d["bytes"])).convert("RGB")) for d in table["img"]])
        out += [images, np.asarray(table["label"], dtype=np.int64)]
    return out  # x_train, y_train, x_test, y_test


def from_toronto(data_dir):
    path = os.path.join(data_dir, "cifar-10-binary.tar.gz")
    if not os.path.exists(path):
        download(TORONTO_URL, path, "md5", TORONTO_MD5)
    # read members in memory; nothing is extracted to disk
    with tarfile.open(path) as tar:

        def batch(name):
            raw = np.frombuffer(tar.extractfile(f"cifar-10-batches-bin/{name}").read(), dtype=np.uint8)
            raw = raw.reshape(-1, 3073)
            return raw[:, 1:].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1), raw[:, 0].astype(np.int64)

        train = [batch(f"data_batch_{i}.bin") for i in range(1, 6)]
        x_test, y_test = batch("test_batch.bin")
    return (np.concatenate([x for x, _ in train]), np.concatenate([y for _, y in train]), x_test, y_test)


def load_cifar10():
    if args.data == "synthetic":
        g = np.random.default_rng(args.seed)
        return (
            g.integers(0, 256, (2048, 32, 32, 3), dtype=np.uint8),
            g.integers(0, 10, 2048),
            g.integers(0, 256, (512, 32, 32, 3), dtype=np.uint8),
            g.integers(0, 10, 512),
        )
    os.makedirs(args.data_dir, exist_ok=True)
    cache = os.path.join(args.data_dir, "cifar10_uint8.npz")
    if not os.path.exists(cache):
        try:
            arrays = from_hf(args.data_dir)
        except Exception as e:  # noqa: BLE001 - any failure falls back to the original archive
            print(f"Hugging Face source failed ({e!r}); using the Toronto archive", flush=True)
            arrays = from_toronto(args.data_dir)
        x_tr, y_tr, x_te, y_te = arrays
        assert x_tr.shape == (50000, 32, 32, 3) and x_te.shape == (10000, 32, 32, 3), (x_tr.shape, x_te.shape)
        assert (np.bincount(y_tr, minlength=10) == 5000).all() and (np.bincount(y_te, minlength=10) == 1000).all()
        np.savez(cache + ".tmp.npz", x_train=x_tr, y_train=y_tr, x_test=x_te, y_test=y_te)
        os.replace(cache + ".tmp.npz", cache)
    with np.load(cache) as z:
        return z["x_train"], z["y_train"], z["x_test"], z["y_test"]


MEAN, STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)


def to_device(x_uint8):
    """N,H,W,C uint8 -> N,C,H,W float32, normalized, channels_last memory."""
    x = torch.from_numpy(x_uint8).to(device).permute(0, 3, 1, 2).float().div_(255)
    mean, std = torch.tensor(MEAN, device=device), torch.tensor(STD, device=device)
    return ((x - mean[:, None, None]) / std[:, None, None]).contiguous(memory_format=torch.channels_last)


x_train_np, y_train_np, x_test_np, y_test_np = load_cifar10()
if args.prepare_data:
    print(f"data ready: {len(x_train_np)} train and {len(x_test_np)} test images")
    sys.exit(0)
x_train, x_test = to_device(x_train_np), to_device(x_test_np)
y_train, y_test = (torch.from_numpy(y_train_np).to(device), torch.from_numpy(y_test_np).to(device))
padded_train = F.pad(x_train, (4, 4, 4, 4), mode="reflect")
N_TRAIN, B = len(x_train), args.batch_size
steps_per_epoch = N_TRAIN // B  # the last partial batch of each epoch is dropped
total_steps = args.epochs * steps_per_epoch
data_gen = torch.Generator(device=device).manual_seed(args.seed)


def augmented_epoch():
    """Random 32x32 crops of the padded images, flipped half the time, in a random order; one gather on the GPU."""
    n = len(padded_train)
    ox = torch.randint(0, 9, (n,), device=device, generator=data_gen)
    oy = torch.randint(0, 9, (n,), device=device, generator=data_gen)
    flip = torch.rand(n, device=device, generator=data_gen) < 0.5
    ar = torch.arange(32, device=device)
    rows, cols = oy[:, None] + ar, ox[:, None] + ar
    cols = torch.where(flip[:, None], cols.flip(1), cols)
    crops = padded_train[torch.arange(n, device=device)[:, None, None], :, rows[:, :, None], cols[:, None, :]]
    order = torch.randperm(n, device=device, generator=data_gen)
    return crops.permute(0, 3, 1, 2), order  # N,C,32,32 with channels_last strides


# ---------------------------------------------------------------------------------------------------------- model
class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.shortcut = nn.Sequential()
        if stride != 1 or cin != cout:
            self.shortcut = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        return F.relu(self.bn2(self.conv2(out)) + self.shortcut(x))


class ResNet18(nn.Module):
    def __init__(self, width=64, num_classes=10):
        super().__init__()
        w = [width, 2 * width, 4 * width, 8 * width]
        self.stem = nn.Sequential(nn.Conv2d(3, w[0], 3, 1, 1, bias=False), nn.BatchNorm2d(w[0]), nn.ReLU())
        layers, cin = [], w[0]
        for i, cout in enumerate(w):
            layers += [BasicBlock(cin, cout, 1 if i == 0 else 2), BasicBlock(cout, cout, 1)]
            cin = cout
        self.layers = nn.Sequential(*layers)
        self.fc = nn.Linear(w[-1], num_classes)

    def forward(self, x):
        x = self.layers(self.stem(x))
        return self.fc(torch.flatten(F.adaptive_avg_pool2d(x, 1), 1))


torch.manual_seed(args.seed)
raw_model = ResNet18(args.width).to(device).to(memory_format=torch.channels_last)
n_params = sum(p.numel() for p in raw_model.parameters())

# ------------------------------------------------------------------------------------------- optimizer, schedule
decay = [p for p in raw_model.parameters() if p.dim() > 1]
no_decay = [p for p in raw_model.parameters() if p.dim() <= 1]
opt = torch.optim.SGD(
    [{"params": decay, "weight_decay": args.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
    lr=args.lr,
    momentum=args.momentum,
    nesterov=True,
)
sched = torch.optim.lr_scheduler.OneCycleLR(
    opt,
    max_lr=args.lr,
    total_steps=total_steps,
    pct_start=args.warmup_frac,
    anneal_strategy="linear",
    cycle_momentum=False,
)

GOAL = (
    f"ResNet-18 ({n_params / 1e6:.1f}M parameters) on CIFAR-10 for {args.epochs} epochs ({total_steps} steps "
    f"of batch {B}) on one GPU; judged by test accuracy at the end."
)
if args.opal:
    import opal

    opt = opal.wrap(
        opt,
        raw_model,
        scheduler=sched,
        total_steps=total_steps,
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
    correct, loss_sum = torch.zeros((), device=device), torch.zeros((), device=device)
    for i in range(0, len(x_test), 1000):
        with autocast():
            logits = raw_model(x_test[i : i + 1000]).float()
        loss_sum += F.cross_entropy(logits, y_test[i : i + 1000], reduction="sum")
        correct += (logits.argmax(1) == y_test[i : i + 1000]).sum()
    raw_model.train()
    return (correct / len(x_test)).item(), (loss_sum / len(x_test)).item()


print(
    f"cifar10_resnet: {n_params / 1e6:.2f}M params, {args.epochs} epochs x {steps_per_epoch} steps of batch {B}, "
    f"device {torch.cuda.get_device_name(device) if device.type == 'cuda' else device}, torch {torch.__version__}, "
    f"compile={args.compile} opal={args.opal} deterministic={args.deterministic}",
    flush=True,
)
if device.type == "cuda":
    torch.cuda.reset_peak_memory_stats(device)
timer = StepTimer(total_steps)
loss_hist = torch.zeros(total_steps, device=device)
log = []
sync()
train_t0 = time.perf_counter()
step = 0
for epoch in range(args.epochs):
    images, order = augmented_epoch()
    for i in range(steps_per_epoch):
        timer.mark(step)
        idx = order[i * B : (i + 1) * B]
        x, y = images[idx].contiguous(memory_format=torch.channels_last), y_train[idx]
        with autocast():
            loss = F.cross_entropy(model(x), y, label_smoothing=args.label_smoothing)
        loss.backward()
        lr_used = current_lr()  # the LR this update uses (read before the scheduler moves it)
        if args.opal:
            opt.step(loss=loss)
        else:
            opt.step()
        opt.zero_grad(set_to_none=True)
        sched.step()
        loss_hist[step] = loss.detach()
        step += 1
    entry = {
        "epoch": epoch,
        "step": step,
        "train_loss": loss_hist[step - steps_per_epoch : step].mean().item(),
        "lr": lr_used,
        "seconds": time.perf_counter() - train_t0,
    }
    log.append(entry)
    print(
        f"epoch {epoch + 1:3d}/{args.epochs}  train loss {entry['train_loss']:.4f}  lr {entry['lr']:.3e}  "
        f"{entry['seconds']:.0f}s",
        flush=True,
    )
timer.mark(total_steps)
sync()
train_seconds = time.perf_counter() - train_t0

step_s = timer.seconds()
steady = step_s[args.timing_skip :] if len(step_s) > args.timing_skip else step_s
median_step = float(np.median(steady))
peak_alloc = torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else None
peak_reserved = torch.cuda.max_memory_reserved(device) / 2**30 if device.type == "cuda" else None
losses = loss_hist.cpu()
test_acc, test_loss = evaluate()
if args.opal:  # into the run's records and report; the agent does not see it
    opt.log(test_acc=test_acc, test_loss=test_loss)


def sha256_tensors(tensors):
    h = hashlib.sha256()
    for t in tensors:
        h.update(t.detach().reshape(-1).cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


from opal.results import save_weights, summary  # noqa: E402

result = {
    **summary("test_accuracy", test_acc, "higher", train_seconds, opt if args.opal else None, device),
    "script": "cifar10_resnet.py",
    "args": vars(args),
    "argv": sys.argv,
    "host": socket.gethostname(),
    "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "cudnn": torch.backends.cudnn.version(),
    "n_params": n_params,
    "steps": total_steps,
    "goal": GOAL,
    "test_acc": test_acc,
    "test_loss": test_loss,
    "final_train_loss_last_epoch": log[-1]["train_loss"] if log else None,
    "steps_per_s_median": 1.0 / median_step,
    "images_per_s_median": B / median_step,
    "timing_steps_used": len(steady),
    "train_loop_seconds": train_seconds,
    "first_step_seconds": step_s[0],
    "total_seconds": time.perf_counter() - script_t0,
    "peak_mem_allocated_gib": peak_alloc,
    "peak_mem_reserved_gib": peak_reserved,
    "param_sha256": sha256_tensors(p for _, p in raw_model.named_parameters()),
    "buffer_sha256": sha256_tensors(b for _, b in raw_model.named_buffers()),
    "loss_sha256": sha256_tensors([losses]),
    "log": log,
}
print(
    f"RESULT test_acc={test_acc:.4f} test_loss={test_loss:.4f} steps/s={result['steps_per_s_median']:.2f} "
    f"images/s={result['images_per_s_median']:.0f} train_loop={train_seconds:.0f}s "
    f"total={result['total_seconds']:.0f}s peak_alloc_gib={peak_alloc} "
    f"param_sha256={result['param_sha256'][:16]} loss_sha256={result['loss_sha256'][:16]}",
    flush=True,
)
if args.save:
    save_weights(args.save, model=raw_model.state_dict())
if args.out:
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)
