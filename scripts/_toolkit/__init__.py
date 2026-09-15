"""Shared infrastructure for the developer scripts — never anybody's business logic.

Scripts depend on this; scripts never depend on each other. The dependency points one
way, which is what keeps every script individually readable, editable, and deletable.

  processes.py  find/require tools, run and capture commands, portably
  config.py     load a script's own config/*.json, and the repository root
  console.py    headings, steps, results, and the confirm prompt
  terminal.py   output that cannot crash on a legacy console code page

It has no ``__main__.py`` on purpose: the orchestrator's loader rejects any catalog
entry without one, so this package can never be registered as a runnable script.
"""

from __future__ import annotations

from . import config, console, processes, terminal

__all__ = ["config", "console", "processes", "terminal"]
