"""Dev-tools orchestrator: all logic behind ``dev_tools_scripts_runner.py``, the thin
shim at the repository root that never changes beyond importing ``main`` from here.

Module map:
  models.py         Text/Category/ScriptInfo/Session — pure data
  exceptions.py     RunnerError/ConfigValidationError
  config_loader.py  ConfigLoader — reads and validates config/*.json
  registry.py       ScriptRegistry — queryable, read-only catalog
  execution.py      ScriptRunner — `python -m <module>` launches
  rendering.py      MenuRenderer — every print
  interactive.py    InteractiveShell — every input
  main.py           CliApp + main(argv) — argv dispatch
  config/*.json     the hand-edited catalog this package operates on

Adding a script means: create ``scripts/<name>/`` with a ``__main__.py``, then add one
entry to ``config/scripts.json``. No Python in this package changes.
"""

from __future__ import annotations

from .main import CliApp, main

__all__ = ["CliApp", "main"]
