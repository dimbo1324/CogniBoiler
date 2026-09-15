"""Re-running a script inside the uv-managed environment when it needs a dependency.

The orchestrator itself is stdlib-only and may be launched by any Python on PATH. A few
scripts need a package only the project environment has — ``cryptography`` for
dev-secrets, ``grpc_tools`` for generate-proto. Rather than fail with an import error,
such a script hands itself to ``uv run`` once.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

from .processes import NOT_FOUND, run

MARKER = "COGNIBOILER_REEXEC"


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def reexec_under_uv(module: str, argv: list[str], root: Path) -> int | None:
    """Run ``python -m <module> <argv>`` through ``uv run`` and return its exit code.

    Returns ``None`` when this process already is the re-executed one, or when uv is
    not on PATH, so the caller reports the missing dependency instead of recursing.
    """
    if os.environ.get(MARKER):
        return None
    result = run(
        ["uv", "run", "--no-sync", "python", "-m", module, *argv],
        root,
        echo=False,
        env={MARKER: "1"},
    )
    if result.returncode == NOT_FOUND:
        return None
    return result.returncode
