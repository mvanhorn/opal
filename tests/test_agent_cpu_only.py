"""The agent's shell sees no GPU: CUDA_VISIBLE_DEVICES and HIP_VISIBLE_DEVICES are empty in the Claude backend's process
env and in its Claude Code settings env, with and without the sandbox. The broker runs the same backend, so this holds
wherever the agent's shell runs. (A real sandboxed Claude shell is not started here: the test checks the settings and
runs the training interpreter with the agent's env.)"""

import json
import os
import subprocess
import sys

import pytest

from opal.backends.base import agent_env
from opal.backends.claude import SANDBOX_NOTE, ClaudeBackend, _login_prefixes


@pytest.mark.parametrize("sandbox", [True, False])
def test_agent_shell_env_hides_the_gpus(tmp_path, sandbox):
    before = dict(os.environ)
    b = ClaudeBackend(sandbox=sandbox)
    b.setup(tmp_path / "run", tmp_path / "ws")
    settings = json.loads((b.control / "settings.json").read_text())
    for env in (settings["env"], b._env()):
        assert env["CUDA_VISIBLE_DEVICES"] == "" and env["HIP_VISIBLE_DEVICES"] == ""
    code = "import os, torch; print(repr(os.environ['CUDA_VISIBLE_DEVICES']), torch.cuda.is_available())"
    out = subprocess.run(
        [sys.executable, "-c", code], env=agent_env(_login_prefixes(), b._env()), capture_output=True, text=True
    )
    assert out.stdout.split() == ["''", "False"], out.stderr[-500:]
    assert dict(os.environ) == before  # the training process keeps its own GPUs
    assert "on CPU only: no GPU is visible" in SANDBOX_NOTE
