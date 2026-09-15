"""Running a declared list of commands, and reporting honestly which ones failed.

Several scripts are a named sequence of commands with a summary at the end. That shape
belongs here once; the sequences themselves stay in each script's JSON.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .console import fail, ok, step, summary, warn
from .processes import NOT_FOUND, run


def select_steps(
    steps: list[dict[str, Any]], root: Path
) -> tuple[list[dict[str, Any]], list[tuple[str, str]]]:
    """Resolve a JSON step list against this checkout.

    ``"python"`` as a program means the interpreter running this script: ``python`` on
    PATH may be a different version, or on Windows a Store stub that does nothing.

    A step declaring ``requires_path`` needs an optional part of the checkout, such as
    the frontend's ``node_modules``. Without it the step is skipped with a reason —
    unless the ``CI`` variable is set, where a silent skip would let unchecked code
    through, so the step runs and fails on its own terms.
    """
    in_ci = bool(os.environ.get("CI"))
    selected: list[dict[str, Any]] = []
    skipped: list[tuple[str, str]] = []
    for entry in steps:
        argv = list(entry["argv"])
        if argv and argv[0] == "python":
            argv[0] = sys.executable
        required_path = entry.get("requires_path")
        if required_path and not (root / required_path).exists() and not in_ci:
            skipped.append((entry["label"], f"no {required_path}"))
            continue
        selected.append({**entry, "argv": argv})
    return selected, skipped


def run_steps(
    steps: list[dict[str, Any]],
    root: Path,
    *,
    title: str,
    keep_going: bool = False,
    skipped: Sequence[tuple[str, str]] = (),
) -> int:
    """Run every step, then summarise.

    By default a failed required step stops the run, because the steps after it would
    be meaningless. With ``keep_going`` every step runs regardless, which is what a
    gate wants: one run that names every problem rather than only the first.

    A missing optional tool never stops anything. ``skipped`` rows, produced by
    [`select_steps`], are reported in the summary so a shorter run never reads as a
    complete one.

    Returns a process exit code: 0 if every required step passed.
    """
    rows: list[tuple[str, str]] = [
        (label, f"skipped ({reason})") for label, reason in skipped
    ]
    for label, reason in skipped:
        warn(f"{label} — skipped: {reason}")
    failed = False

    for index, entry in enumerate(steps):
        label = entry["label"]
        argv = entry["argv"]
        required = bool(entry.get("required", True))

        step(label)
        result = run(list(argv), root)

        if result.ok:
            ok(label)
            rows.append((label, "ok"))
            continue

        if result.returncode == NOT_FOUND:
            note = f"tool not found: {argv[0]}"
            if not required:
                warn(f"{label} — {note}, skipped")
                rows.append((label, "missing (optional)"))
                continue
            fail(f"{label} — {note}")
            rows.append((label, "MISSING (required)"))
        elif required:
            fail(f"{label} — exit {result.returncode}")
            rows.append((label, f"FAILED ({result.returncode})"))
        else:
            warn(f"{label} — exit {result.returncode}, not required")
            rows.append((label, f"failed ({result.returncode}), optional"))
            continue

        failed = True
        if not keep_going:
            rows.extend(_not_reached(steps[index + 1 :]))
            break

    summary(title, rows)
    return 1 if failed else 0


def _not_reached(remaining: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """Rows for the steps a required failure prevented from running.

    Omitting them made the summary read as though the run had been shorter than it was
    declared to be — a reader comparing a failing summary against a passing one saw
    lines disappear and had to guess whether they were skipped or removed. Naming them
    as not run is the honest report the summary is for.
    """
    return [(entry["label"], "not run (earlier step failed)") for entry in remaining]
