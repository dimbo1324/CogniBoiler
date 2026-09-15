"""Check that the developer scripts themselves still work.

Two things, both cheap: the stdlib unit tests under ``scripts/`` (which cover
the orchestrator, the shared toolkit and the AGENTS.md renderer), and a load of the orchestrator catalog, which fails loudly on any bad hand-edit
of the JSON.

Worth running after touching anything in ``scripts/`` — a broken catalog is invisible
until the moment it matters.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from scripts._toolkit.config import load_config, repo_root
from scripts._toolkit.steps import run_steps

SCRIPT_DIR = Path(__file__).resolve().parent


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="selftest",
        description="Run the developer scripts' own tests and catalog validation.",
    )
    parser.parse_args(argv)

    config = load_config(SCRIPT_DIR, "steps.json")
    steps = []
    for entry in config["steps"]:
        argv_entry = list(entry["argv"])
        # `python` on PATH may be a different interpreter than the one running this, or
        # on Windows a Store stub that does nothing. Use our own.
        if argv_entry and argv_entry[0] == "python":
            argv_entry[0] = sys.executable
        steps.append({**entry, "argv": argv_entry})

    return run_steps(steps, repo_root(), title="selftest")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
