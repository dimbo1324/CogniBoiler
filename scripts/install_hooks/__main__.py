"""Install the pre-commit hook and pre-build its environments.

The hook formats staged Python with the ruff version pinned in uv.lock and checks file
hygiene. Run once per clone.
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
        prog="install-hooks",
        description="Install the repository's pre-commit hook.",
    )
    parser.parse_args(argv)

    root = repo_root()
    config = load_config(SCRIPT_DIR, "steps.json")
    steps, skipped = select_steps(config["steps"], root)
    return run_steps(steps, root, title="install-hooks", skipped=skipped)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
