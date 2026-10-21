"""examples/mnist_mlp.py, the CPU install check: on synthetic data (no download), the plain run and the wrapped run with
backend="keep" (no agent) print the same losses, accuracies and parameter checksum, and the wrapped run writes its
records."""

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cwd, *extra):
    cmd = [sys.executable, str(ROOT / "examples" / "mnist_mlp.py"), "--device", "cpu", "--data", "synthetic"]
    out = subprocess.run(
        cmd + ["--epochs", "2", *extra], cwd=cwd, capture_output=True, text=True, timeout=300, check=False
    )
    assert out.returncode == 0, out.stderr[-3000:]
    lines = [line for line in out.stdout.splitlines() if line.startswith(("epoch", "final", "parameter checksum"))]
    return [re.sub(r"\(\d+ s\)", "", line) for line in lines]  # without the wall time


def test_mnist_mlp_dry_run_matches_the_plain_run(tmp_path):
    plain = run(tmp_path)
    kept = run(tmp_path, "--opal", "--backend", "keep", "--every", "2", "--decisions", "3")
    assert len(plain) == 4 and plain[-2].startswith("final test accuracy")
    assert re.fullmatch(r"parameter checksum [0-9a-f]{12}", plain[-1])
    assert kept == plain  # the same losses, accuracies and parameter checksum
    (run_dir,) = (tmp_path / "opal_runs").iterdir()
    events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines()]
    assert [e["kind"] for e in events].count("kept") == 3 and events[-1]["kind"] == "end"
    assert (run_dir / "report.html").exists() and (run_dir / "metrics.jsonl").read_text().count("test_acc") == 2


def test_readmes_full_suite_install_includes_optional_test_dependencies():
    from importlib.metadata import metadata
    from packaging.requirements import Requirement

    paragraph = (ROOT / "README.md").read_text().split("The whole test suite", 1)[1].split("\n\n", 1)[0]
    extras = re.search(r'pip install -e "\.\[([^]]+)\]"', paragraph).group(1).split(",")
    installed = set()
    for line in metadata("opal-optimizer").get_all("Requires-Dist"):
        requirement = Requirement(line)
        if requirement.marker is None or any(requirement.marker.evaluate({"extra": e}) for e in extras):
            installed.add(requirement.name)
    assert {"pytest", "torchvision", "transformers", "accelerate"} <= installed
