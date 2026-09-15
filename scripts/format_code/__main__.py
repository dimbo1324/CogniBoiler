"""Format every source file in the repository, or check that it is already formatted.

ruff owns Python — the version pinned in uv.lock, the same one the pre-commit hook and
the gate run — and Prettier owns the frontend. The step lists live in
config/steps.json.
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
        prog="format-code",
        description="Format Python with ruff and the frontend with Prettier.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="report what would change without rewriting anything (what the gate does)",
    )
    args = parser.parse_args(argv)

    root = repo_root()
    config = load_config(SCRIPT_DIR, "steps.json")
    steps, skipped = select_steps(
        config["check_steps" if args.check else "steps"], root
    )
    title = "format-code (check only)" if args.check else "format-code"
    return run_steps(steps, root, title=title, keep_going=True, skipped=skipped)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
