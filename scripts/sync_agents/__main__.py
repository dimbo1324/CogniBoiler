"""Regenerate AGENTS.md from the .ai/ rule modules, or verify that it is current.

AGENTS.md is never edited by hand: edit a module, then run this. The quality gate runs
``--check``, so a module change without its regenerated entry point fails the build.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from scripts._toolkit.config import load_config, repo_root
from scripts._toolkit.console import fail, ok
from scripts.sync_agents.render import (
    SyncError,
    collect_modules,
    normalize,
    render,
    size_kib,
)

SCRIPT_DIR = Path(__file__).resolve().parent


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="sync-agents",
        description="Regenerate AGENTS.md from .ai/, or verify it is in sync.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail when AGENTS.md is out of date instead of rewriting it",
    )
    args = parser.parse_args(argv)

    root = repo_root()
    config = load_config(SCRIPT_DIR, "sync.json")
    limit = float(config["size_limit_kib"])

    try:
        modules = collect_modules(root, config["groups"])
    except SyncError as error:
        fail(str(error))
        return 1

    content = render(modules, "\n".join(config["banner"]))
    size = size_kib(content)
    if size > limit:
        fail(
            f"assembled AGENTS.md is {size:.1f} KiB; the budget is {limit:.0f} KiB. "
            "Tighten a module, or mark a situational one with "
            "`<!-- tier: extended -->` and give it a `> **Essence.**` line."
        )
        return 1

    target = root / config["target"]
    current = normalize(target.read_text(encoding="utf-8")) if target.exists() else ""

    if args.check:
        if current == content:
            ok(f"AGENTS.md is in sync with .ai/ modules ({size:.1f} KiB).")
            return 0
        fail(
            "AGENTS.md is out of sync with .ai/ modules. "
            "Run: python dev_tools_scripts_runner.py sync-agents"
        )
        return 1

    if current == content:
        ok(f"AGENTS.md already up to date ({size:.1f} KiB).")
        return 0
    target.write_text(content, encoding="utf-8", newline="\n")
    ok(
        f"AGENTS.md regenerated from {len(modules)} modules "
        f"({size:.1f} KiB of {limit:.0f} KiB budget)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
