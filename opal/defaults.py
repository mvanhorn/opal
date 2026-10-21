"""OPAL's defaults are fixed: the default prompt, the starting programs, and the code that writes them.

The blog's and the README's results were made with these files as released. A coding agent that edits them inside an
(editable) install changes every later run without a word, so their sha256 hashes ship in defaults.json, and a run
(rank 0) compares the files it is about to use with them when it starts: an edited default prints a warning, and the
run's records (the start event of events.jsonl, report.html) say which files differ. Customise through wrap()'s
arguments instead: prompt=<a template file>, program=<a program folder>, task_facts=, and the rest.

tests/test_defaults.py fails when defaults.json is stale; after a deliberate change to one of these files, write it
again with `python -m opal.defaults --write`.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
MANIFEST = HERE / "defaults.json"
PROMPT_FILES = ("prompts/neutral.md", "misc/aggressive.py")  # the named prompts (not used with prompt=<a path>)
PROGRAM_FILES = ("prompts/example_updater.py", "misc/neutral_analysis.py")  # starting analysis, example (not with program=)
# what fills the prompt and writes the messages and the starting updater, whatever prompt or program is used
SHARED_FILES = ("misc/setup.py", "misc/taskfacts.py", "programs.py", "agent.py", "context.py")
FILES = PROMPT_FILES + PROGRAM_FILES + SHARED_FILES


def hashes(names=FILES) -> dict:
    return {n: hashlib.sha256((HERE / n).read_bytes()).hexdigest() for n in names}


def modified(names=FILES) -> list:
    """The files among `names` whose content differs from the released one (defaults.json), or that are missing."""
    try:
        shipped = json.loads(MANIFEST.read_text())
    except (OSError, ValueError):
        return ["defaults.json"]
    out = []
    for n in names:
        try:
            now = hashlib.sha256((HERE / n).read_bytes()).hexdigest()
        except OSError:
            now = None
        if now != shipped.get(n):
            out.append(n)
    return out


if __name__ == "__main__":
    if sys.argv[1:] != ["--write"]:
        sys.exit("usage: python -m opal.defaults --write   (after a deliberate change to OPAL's defaults)")
    MANIFEST.write_text(json.dumps(hashes(), indent=1, sort_keys=True) + "\n")
    print(f"wrote {MANIFEST}")
