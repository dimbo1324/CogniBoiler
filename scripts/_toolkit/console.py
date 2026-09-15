"""Consistent console output, so every script reads the same way."""

from __future__ import annotations

import sys

_WIDTH = 78


def heading(text: str) -> None:
    print(f"\n{'=' * _WIDTH}")
    print(text)
    print("=" * _WIDTH, flush=True)


def step(text: str) -> None:
    print(f"\n--- {text} ---", flush=True)


def info(text: str) -> None:
    print(f"  {text}", flush=True)


def ok(text: str) -> None:
    print(f"  OK    {text}", flush=True)


def warn(text: str) -> None:
    print(f"  WARN  {text}", flush=True)


def fail(text: str) -> None:
    """Failures go to stderr so a caller can separate them from the transcript."""
    print(f"  FAIL  {text}", file=sys.stderr, flush=True)


def summary(title: str, rows: list[tuple[str, str]]) -> None:
    heading(title)
    width = max((len(name) for name, _ in rows), default=0)
    for name, value in rows:
        print(f"  {name.ljust(width)}   {value}")
    print("=" * _WIDTH, flush=True)


def confirm(question: str, *, assume_yes: bool) -> bool:
    """Ask before doing something irreversible.

    ``assume_yes`` short-circuits for non-interactive callers. When there is no
    terminal and consent was not given explicitly, the answer is **no**: a script that
    deletes files must not treat an empty pipe as approval.
    """
    if assume_yes:
        return True
    try:
        if not sys.stdin.isatty():
            fail("no terminal to confirm on — pass the explicit flag to proceed")
            return False
    except AttributeError, ValueError:
        return False
    try:
        answer = input(f"{question} [y/N]: ").strip().lower()
    except EOFError:
        # The stream ended instead of answering. For a prompt guarding deletion that is
        # not consent, so it reads the same as "no".
        print()
        return False
    return answer in ("y", "yes")
