#!/usr/bin/env python3
"""MNIST with a small MLP: the shortest complete example of putting an agent inside a training run.

    python examples/mnist_mlp.py                               # plain PyTorch run
    python examples/mnist_mlp.py --opal --backend keep         # wrapped, no agent: must match the plain run
    python examples/mnist_mlp.py --opal                        # wrapped, Claude Code rewrites the optimizer
    opal report opal_runs/<time>                               # afterwards: what the agent did

Runs on a laptop CPU in about 15 s (after the first run's MNIST download), or on one GPU. At the end it prints a
checksum of the final parameters: the plain run and the backend="keep" run print the same one. Everything that is
specific to opal is marked with "# opal".
"""

import argparse
import hashlib
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--data_dir", default="./data")
parser.add_argument(
    "--data", choices=["mnist", "synthetic"], default="mnist", help="synthetic: random images, no download"
)
parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--opal", action="store_true", help="wrap the optimizer with opal")  # opal
parser.add_argument("--backend", default="claude", help="claude, keep, or mailbox:<dir>")  # opal
parser.add_argument("--model", default=None, help="agent model (default: claude-opus-5-5, at effort high)")  # opal
parser.add_argument(
    "--schedule", choices=["front", "uniform"], default="front", help="when the decisions are asked (wrap's schedule)"
)  # opal
parser.add_argument("--effort", default=None, help="agent effort (default: high)")  # opal
parser.add_argument("--out", default=None, help="write the result JSON here")
parser.add_argument("--save", default=None, help="write the final weights (model.pt) into this folder")
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
    if args.data == "synthetic":  # random images and labels, for smoke tests
        g = torch.Generator().manual_seed(args.seed + int(train))
        n = 1024 if train else 256
        return torch.randn(n, 784, generator=g).to(device), torch.randint(10, (n,), generator=g).to(device)
    from torchvision import datasets, transforms

    ds = datasets.MNIST(args.data_dir, train=train, download=True, transform=transforms.ToTensor())
    x = torch.stack([ds[i][0] for i in range(len(ds))]).reshape(len(ds), -1)
    y = torch.tensor([ds[i][1] for i in range(len(ds))])
    return ((x - 0.1307) / 0.3081).to(device), y.to(device)


x_train, y_train = load(True)
x_test, y_test = load(False)

model = nn.Sequential(nn.Linear(784, 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 10)).to(device)
steps_per_epoch = len(x_train) // args.batch_size
total_steps = args.epochs * steps_per_epoch
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)

if args.opal:  # opal
    import opal

    opt = opal.wrap(
        opt,
        model,
        scheduler=sched,
        total_steps=total_steps,
        goal="3-layer MLP on MNIST; judged by test accuracy after the last update",
        backend=args.backend,
        backend_options={k: v for k, v in dict(model=args.model, effort=args.effort).items() if v} or None,
        every=args.every,
        decisions=args.decisions,
        schedule=args.schedule,
        blocking={"true": True, "false": False}.get(args.blocking, args.blocking),
        max_cost_usd=5,
    )


@torch.no_grad()
def evaluate():
    model.eval()
    logits = torch.cat([model(x_test[i : i + 2000]) for i in range(0, len(x_test), 2000)])
    model.train()
    return F.cross_entropy(logits, y_test).item(), (logits.argmax(1) == y_test).float().mean().item()


g = torch.Generator(device="cpu").manual_seed(args.seed)
t0 = time.time()
step = 0
for epoch in range(args.epochs):
    perm = torch.randperm(len(x_train), generator=g).to(device)
    for i in range(steps_per_epoch):
        idx = perm[i * args.batch_size : (i + 1) * args.batch_size]
        loss = F.cross_entropy(model(x_train[idx]), y_train[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step(loss=loss) if args.opal else opt.step()  # opal: pass the loss
        sched.step()
        step += 1
    test_loss, test_acc = evaluate()
    if args.opal:  # opal: record what the run is judged by (in the report; the agent never sees it)
        opt.log(test_loss=test_loss, test_acc=test_acc)
    print(
        f"epoch {epoch + 1}: train loss {loss.item():.4f}  test loss {test_loss:.4f}  test acc {test_acc:.4f}  "
        f"({time.time() - t0:.0f} s)"
    )

print(f"final test accuracy {test_acc:.4f}")
train_seconds = time.time() - t0
if args.opal:  # opal
    opt.close()
h = hashlib.sha256()  # sha256 of the final parameters' bytes: equal checksums mean byte-identical parameters
for p in model.parameters():
    h.update(p.detach().reshape(-1).cpu().contiguous().view(torch.uint8).numpy().tobytes())
print(f"parameter checksum {h.hexdigest()[:12]}")
if args.out or args.save:
    from opal.results import save_weights, summary

    if args.save:
        save_weights(args.save, model=model.state_dict())
    if args.out:
        import json

        result = summary("test_accuracy", test_acc, "higher", train_seconds, opt if args.opal else None, device)
        result.update(test_loss=test_loss, parameter_checksum=h.hexdigest()[:12])
        with open(args.out, "w") as f:
            json.dump(result, f, indent=1)
