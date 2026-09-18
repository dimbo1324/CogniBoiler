"""Run the console's Playwright checks against a running stack.

The Vite dev server is started by Playwright itself when no ``--url`` is given; with
``--url`` the checks go to that console, for example the stack's nginx entry. The demo
passwords come from .env, read by the checks themselves. Chromium is installed on first
use; ``--no-install`` skips that step, for machines that manage browsers themselves.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from scripts._toolkit.config import load_config, repo_root
from scripts._toolkit.console import fail, heading
from scripts._toolkit.processes import NOT_FOUND, run

SCRIPT_DIR = Path(__file__).resolve().parent


def build_commands(
    config: dict[str, object], install: bool, extra: list[str]
) -> list[list[str]]:
    web = str(config["web_dir"])
    commands = []
    if install:
        commands.append(
            ["pnpm", "--dir", web, "exec", "playwright", "install", "chromium"]
        )
    commands.append(["pnpm", "--dir", web, "exec", "playwright", "test", *extra])
    return commands


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="console-e2e",
        description="Run the console's Playwright checks against a running stack.",
    )
    parser.add_argument("--url", help="console URL (default: the Vite dev server)")
    parser.add_argument(
        "--no-install", action="store_true", help="do not install Chromium first"
    )
    parser.add_argument(
        "playwright_args", nargs=argparse.REMAINDER, help="passed to playwright test"
    )
    args = parser.parse_args(argv)

    root = repo_root()
    config = load_config(SCRIPT_DIR, "console_e2e.json")
    if not (root / str(config["web_dir"]) / "node_modules").is_dir():
        fail("apps/web/node_modules is missing — run: pnpm --dir apps/web install")
        return 1

    extra = [value for value in args.playwright_args if value != "--"]
    env = {"CONSOLE_URL": args.url} if args.url else None
    heading(f"console-e2e — {args.url or 'Vite dev server'}")
    for command in build_commands(config, not args.no_install, extra):
        result = run(command, root, env=env)
        if result.returncode == NOT_FOUND:
            fail("pnpm is not on PATH — install Node and enable corepack")
            return 1
        if not result.ok:
            return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
