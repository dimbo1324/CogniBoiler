"""Every script in the catalog is importable and exposes a runnable entry point.

The catalog loader only checks that a declared module has a *directory* — so a syntax
error, a bad import, or a missing `main` in any `__main__.py` passed `selftest` cleanly
and was discovered by the next person who ran that script. `selftest` claimed to import
every script module; this is what makes the claim true.

Importing a `__main__.py` runs its module-level code, so nothing in these files may act
at import time. That is a constraint worth enforcing anyway: a script that does work
merely by being imported cannot be introspected safely.

Run with:  python -m unittest discover -s scripts -t .
"""

from __future__ import annotations

import ast
import importlib
import unittest
from pathlib import Path

from scripts.runner.config_loader import ConfigLoader

ROOT = Path(__file__).resolve().parents[2].parent


def _imported_modules(source: Path) -> set[str]:
    """Every module name ``source`` imports, however it spells the import.

    Covers `import a.b`, `from a.b import c` (the module is `a.b.c` if `c` is itself a
    module, so both are reported), and a literal `importlib.import_module("a.b")`.
    """
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module)
            found.update(f"{node.module}.{alias.name}" for alias in node.names)
        elif isinstance(node, ast.Call):
            target = getattr(node.func, "attr", None)
            if target == "import_module" and node.args:
                first = node.args[0]
                if isinstance(first, ast.Constant) and isinstance(first.value, str):
                    found.add(first.value)
    return found


class CatalogImportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = ConfigLoader(ROOT).load()

    def test_the_shipped_catalog_loads(self) -> None:
        self.assertTrue(self.registry.scripts, "the catalog declares no scripts")

    def test_every_declared_module_imports(self) -> None:
        for script in self.registry.scripts:
            with self.subTest(script=script.title):
                importlib.import_module(f"{script.module}.__main__")

    def test_every_script_exposes_a_main_taking_argv(self) -> None:
        for script in self.registry.scripts:
            with self.subTest(script=script.title):
                module = importlib.import_module(f"{script.module}.__main__")
                entry = getattr(module, "main", None)
                self.assertTrue(
                    callable(entry), f"{script.module} has no callable main"
                )

    def test_every_script_config_directory_is_present(self) -> None:
        """Settings live in JSON by design, so a script whose config is gone is broken
        even though it imports."""
        for script in self.registry.scripts:
            with self.subTest(script=script.title):
                directory = ROOT / Path(*script.module.split("."))
                config = directory / "config"
                self.assertTrue(config.is_dir(), f"{script.module} has no config/")
                self.assertTrue(
                    any(config.glob("*.json")), f"{script.module}/config holds no JSON"
                )

    def test_scripts_do_not_import_one_another(self) -> None:
        """The owner's rule: script implementations must not be interdependent. Only
        `scripts._toolkit` is shared, and `scripts.runner` is the orchestrator, not a
        script's dependency.

        Parsed, not grepped. A substring check for `import scripts.doctor` misses
        `from scripts.doctor import main` — the form anyone would actually write — so the
        test asserted a rule it could not enforce.
        """
        forbidden = {s.module for s in self.registry.scripts} | {"scripts.runner"}
        for script in self.registry.scripts:
            directory = ROOT / Path(*script.module.split("."))
            allowed = {script.module}
            for source in directory.rglob("*.py"):
                # The rule constrains implementations. A test legitimately imports the
                # loader to ask what the catalog declares — this file does exactly that.
                if "tests" in source.relative_to(directory).parts:
                    continue
                for imported in _imported_modules(source):
                    offending = next(
                        (
                            name
                            for name in forbidden - allowed
                            if imported == name or imported.startswith(f"{name}.")
                        ),
                        None,
                    )
                    with self.subTest(script=script.title, source=source.name):
                        self.assertIsNone(
                            offending,
                            f"{source.relative_to(ROOT)} imports {imported!r}; scripts may "
                            "share only scripts._toolkit",
                        )


if __name__ == "__main__":
    unittest.main()
