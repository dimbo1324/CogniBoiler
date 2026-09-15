"""Loading a script's own JSON configuration.

Each script keeps its settings in ``scripts/<name>/config/*.json`` so paths, command
lines, and thresholds can be changed without touching Python. This module is the one
place that reads them, so a malformed file always raises the same exception, naming the
file and the key.

Unlike the orchestrator's own catalog — whose errors are caught and printed as one line,
because the catalog decides what gets *launched* — a per-script config error reaches the
terminal as a traceback. That is deliberate for now: these run as ``python -m`` in their
own process, the traceback names the file, and adding a handler to all eight entry points
would buy a tidier line and one more thing to keep in sync.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ScriptConfigError(RuntimeError):
    """A script's own config file is missing, malformed, or lacks a required key."""


def repo_root() -> Path:
    """The repository root, derived from this file's location.

    scripts/_toolkit/config.py -> scripts/_toolkit -> scripts -> root. Deriving it
    rather than trusting the current directory means a script behaves the same however
    it was launched.
    """
    return Path(__file__).resolve().parents[2]


def load_config(script_dir: Path, filename: str) -> dict[str, Any]:
    """Read ``<script_dir>/config/<filename>`` as a JSON object."""
    path = Path(script_dir) / "config" / filename
    if not path.exists():
        raise ScriptConfigError(f"missing config file: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ScriptConfigError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ScriptConfigError(f"{path}: expected a JSON object at the top level")
    return data


def require(config: dict[str, Any], key: str, where: str) -> Any:
    if key not in config:
        raise ScriptConfigError(f"{where}: missing required key {key!r}")
    return config[key]
