"""Cross-platform developer scripts, driven by the orchestrator in ``scripts/runner``.

Layout rules, which the orchestrator's config loader partly enforces:

* One directory per script, each with a ``__main__.py`` so ``python -m
  scripts.<name>`` runs it. Small scripts stay a single module; large ones decompose
  freely inside their own directory.
* **Scripts never import each other.** A script is meant to be readable, editable, and
  deletable on its own. The only shared code is ``scripts/_toolkit`` — infrastructure
  (process launching, config loading, output), never anybody's business logic. That
  dependency points one way, the same rule the service workspace follows.
* Behaviour lives in each script's own ``config/*.json``, not in its Python. Paths,
  command lines, and thresholds are data.

Importing this package hardens the process's console output (see ``_toolkit.terminal``).
That is a side effect of an import, which is normally worth avoiding — but it has to
happen before the first ``print`` anywhere, including one from a module-level failure,
and every entry point in this tree goes through this package. The alternative was the
same call repeated in nine entry points, where the tenth would eventually forget it and
regain a crash on cp866 consoles.
"""

from scripts._toolkit.terminal import enable_safe_output

enable_safe_output()
