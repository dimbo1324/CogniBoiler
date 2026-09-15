"""Run the project's quality gate: the one verification path humans, agents and CI share.

The step list lives in config/steps.json. Every step runs even when an earlier one
failed, so a single run names every problem instead of hiding the rest behind the
first. Frontend steps need apps/web/node_modules; without it they are skipped with a
notice locally and fail under CI.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from scripts._toolkit.config import load_config, repo_root
from scripts._toolkit.steps import run_steps, select_steps

SCRIPT_DIR = Path(__file__).resolve().parent


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="quality-gate",
        description="Run the full quality gate, or the quick subset.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="skip the test suites and the frontend build — the minimum before a push",
    )
    args = parser.parse_args(argv)

    root = repo_root()
    config = load_config(SCRIPT_DIR, "steps.json")
    steps, skipped = select_steps(
        config["quick_steps" if args.quick else "steps"], root
    )
    title = "quality-gate (quick)" if args.quick else "quality-gate (full)"
    return run_steps(steps, root, title=title, keep_going=True, skipped=skipped)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
