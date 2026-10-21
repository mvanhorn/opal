#!/usr/bin/env python3
"""A small diffusion model (flow matching) on MNIST, with an EMA of the weights, on one GPU or CPU.

    python examples/diffusion_mnist.py                               # plain PyTorch run
    python examples/diffusion_mnist.py --opal --backend keep         # wrapped, no agent: must match the plain run
    python examples/diffusion_mnist.py --opal                        # wrapped, Claude Code rewrites the optimizer
    opal report opal_runs/<time>

The model predicts the velocity x1 - x0 along x_t = t * x1 + (1 - t) * x0 between noise x0 and an image x1 (the
SiT/rectified-flow objective). The run is judged by the validation loss of the EMA weights on held-out images at a fixed
grid of t and fixed noise, so it is comparable between runs; a grid of samples is written at the end. About 10 minutes
on one GPU with the defaults. Everything specific to opal is marked with "# opal".
"""

import argparse
import copy
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=6000)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--lr", type=float, default=2e-4)
parser.add_argument("--width", type=int, default=64)
parser.add_argument("--ema", type=float, default=0.999)
parser.add_argument("--eval_every", type=int, default=1000)
parser.add_argument("--data_dir", default="./data")
parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--samples", default="diffusion_samples.png")
parser.add_argument("--opal", action="store_true", help="wrap the optimizer with opal")  # opal
parser.add_argument("--backend", default="claude", help="claude, keep, or mailbox:<dir>")  # opal
parser.add_argument("--model", default=None, help="agent model (default: claude-opus-5-5, at effort high)")  # opal
parser.add_argument(
    "--schedule", choices=["front", "uniform"], default="front", help="when the decisions are asked (wrap's schedule)"
)  # opal
parser.add_argument("--effort", default=None, help="agent effort (default: high)")  # opal
parser.add_argument("--out", default=None, help="write the result JSON here")
parser.add_argument("--save", default=None, help="write the final weights (model.pt, ema.pt) into this folder")
parser.add_argument(
    "--every", type=int, default=50, help="updates between checks (losses, analysis, decisions)"
)  # opal
parser.add_argument(
    "--decisions", type=int, default=50, help="how many times the agent is asked over the run"
)  # opal
parser.add_argument(
    "--blocking",
    nargs="?",
    const="true",
    default="true",
    choices=["true", "false", "auto"],
    help="true (default): training waits for each answer; false: it keeps going; auto: it waits "
    "only when an answer would land late",
)  # opal
args = parser.parse_args()

torch.manual_seed(args.seed)
device = torch.device(args.device)


def load(train):
    from torchvision import datasets, transforms

    ds = datasets.MNIST(args.data_dir, train=train, download=True, transform=transforms.ToTensor())
    x = torch.stack([ds[i][0] for i in range(len(ds))])
    return (x * 2 - 1).to(device)  # images in [-1, 1], shape (N, 1, 28, 28)


x_train, x_val = load(True), load(False)[:2000]


class TimeEmbed(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim))

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        ang = 1000 * t[:, None] * freqs[None]
        return self.mlp(torch.cat([ang.sin(), ang.cos()], 1))


class Block(nn.Module):
    def __init__(self, cin, cout, tdim):
        super().__init__()
        self.norm1, self.conv1 = (nn.GroupNorm(8, cin), nn.Conv2d(cin, cout, 3, padding=1))
        self.norm2, self.conv2 = (nn.GroupNorm(8, cout), nn.Conv2d(cout, cout, 3, padding=1))
        self.temb = nn.Linear(tdim, cout)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, temb):
        h = self.conv1(F.silu(self.norm1(x))) + self.temb(temb)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class UNet(nn.Module):
    """28x28 -> 14x14 -> 7x7 and back, two residual blocks per level."""

    def __init__(self, w):
        super().__init__()
        self.temb = TimeEmbed(w)
        self.inp = nn.Conv2d(1, w, 3, padding=1)
        self.d1, self.d2, self.d3 = (Block(w, w, w), Block(w, 2 * w, w), Block(2 * w, 2 * w, w))
        self.mid = Block(2 * w, 2 * w, w)
        self.u3, self.u2, self.u1 = (Block(4 * w, 2 * w, w), Block(4 * w, w, w), Block(2 * w, w, w))
        self.out = nn.Sequential(nn.GroupNorm(8, w), nn.SiLU(), nn.Conv2d(w, 1, 3, padding=1))

    def forward(self, x, t):
        e = self.temb(t)
        h1 = self.d1(self.inp(x), e)
        h2 = self.d2(F.avg_pool2d(h1, 2), e)
        h3 = self.d3(F.avg_pool2d(h2, 2), e)
        h = self.mid(h3, e)
        h = self.u3(torch.cat([h, h3], 1), e)
        h = self.u2(torch.cat([F.interpolate(h, scale_factor=2), h2], 1), e)
        h = self.u1(torch.cat([F.interpolate(h, scale_factor=2), h1], 1), e)
        return self.out(h)


def flow_loss(net, x1, t, x0):
    xt = t[:, None, None, None] * x1 + (1 - t[:, None, None, None]) * x0
    return F.mse_loss(net(xt, t), x1 - x0)


model = UNet(args.width).to(device)
ema = copy.deepcopy(model).requires_grad_(False)
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 500))  # warmup, then constant

if args.opal:  # opal
    import opal

    opt = opal.wrap(
        opt,
        model,
        scheduler=sched,
        total_steps=args.steps,
        goal="Small UNet flow-matching diffusion model on MNIST with an EMA (decay 0.999) of the weights; judged by "
        "the validation flow-matching loss of the EMA weights at the last update",
        backend=args.backend,
        backend_options={k: v for k, v in dict(model=args.model, effort=args.effort).items() if v} or None,
        every=args.every,
        decisions=args.decisions,
        schedule=args.schedule,
        blocking={"true": True, "false": False}.get(args.blocking, args.blocking),
        max_cost_usd=10,
    )

gv = torch.Generator(device="cpu").manual_seed(1234)
val_t = torch.linspace(0.02, 0.98, len(x_val)).to(device)
val_noise = torch.randn(x_val.shape, generator=gv).to(device)


@torch.no_grad()
def val_loss(net):
    return sum(
        flow_loss(net, x_val[i : i + 500], val_t[i : i + 500], val_noise[i : i + 500]).item()
        for i in range(0, len(x_val), 500)
    ) / (len(x_val) // 500)


g = torch.Generator(device="cpu").manual_seed(args.seed)
t0 = time.time()
for step in range(1, args.steps + 1):
    idx = torch.randint(0, len(x_train), (args.batch_size,), generator=g).to(device)
    x1 = x_train[idx]
    x0 = torch.randn(x1.shape, generator=g).to(device)
    t = torch.rand(len(x1), generator=g).to(device)
    loss = flow_loss(model, x1, t, x0)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step(loss=loss) if args.opal else opt.step()  # opal: pass the loss
    sched.step()
    with torch.no_grad():
        for pe, p in zip(ema.parameters(), model.parameters()):
            pe.lerp_(p, 1 - args.ema)
    if step % args.eval_every == 0 or step == args.steps:
        v = val_loss(ema)
        if args.opal:  # opal: record what the run is judged by (in the report; the agent never sees it)
            opt.log(val_loss_ema=v)
        print(f"step {step}: train loss {loss.item():.4f}  EMA val loss {v:.4f}  ({time.time() - t0:.0f} s)")

print(f"final EMA validation loss {v:.4f}")
train_seconds = time.time() - t0

with torch.no_grad():  # 64 samples with 50 Euler steps from noise (t = 0) to data (t = 1)
    x = torch.randn((64, 1, 28, 28), generator=gv).to(device)
    for i in range(50):
        t = torch.full((64,), i / 50, device=device)
        x = x + ema(x, t) / 50
    try:
        from torchvision.utils import save_image

        save_image((x.clamp(-1, 1) + 1) / 2, args.samples, nrow=8)
        print(f"samples written to {args.samples}")
    except ImportError:
        pass
if args.opal:  # opal
    opt.close()
if args.out or args.save:
    from opal.results import save_weights, summary

    if args.save:
        save_weights(args.save, model=model.state_dict(), ema=ema.state_dict())
    if args.out:
        import json

        result = summary("val_loss_ema", v, "lower", train_seconds, opt if args.opal else None, device)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=1)
