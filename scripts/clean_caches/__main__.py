"""List regenerable caches and build output, and delete them only with --apply.

What counts as a cache and what is protected lives in config/clean.json.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from scripts._toolkit.config import load_config, repo_root
from scripts._toolkit.console import fail, heading, info, ok
from scripts.clean_caches.plan import Rules, plan

SCRIPT_DIR = Path(__file__).resolve().parent


def remove(targets: list[Path]) -> list[tuple[Path, str]]:
    failures: list[tuple[Path, str]] = []
    for target in targets:
        try:
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
        except OSError as error:
            failures.append((target, str(error)))
    return failures


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="clean-caches",
        description="List tool caches and build output; delete them with --apply.",
    )
    parser.add_argument(
        "--apply", action="store_true", help="actually delete what the plan lists"
    )
    args = parser.parse_args(argv)

    root = repo_root()
    rules = Rules.from_config(load_config(SCRIPT_DIR, "clean.json"))
    targets = plan(root, rules)

    heading("clean-caches" if args.apply else "clean-caches (dry run)")
    if not targets:
        ok("nothing to clean")
        return 0
    for target in targets:
        info(str(target.relative_to(root)))

    if not args.apply:
        info(f"{len(targets)} item(s) would be deleted; rerun with --apply")
        return 0

    failures = remove(targets)
    for target, reason in failures:
        fail(f"{target.relative_to(root)}: {reason}")
    ok(f"deleted {len(targets) - len(failures)} of {len(targets)} item(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
