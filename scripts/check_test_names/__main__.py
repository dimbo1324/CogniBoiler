"""Fail when two modules of the pytest suites would be imported under one name.

The suites share one rootdir and their ``tests/`` directories are not packages, so pytest
imports each test module, and each helper beside it, by its bare file name. Two
``tests/test_boundaries.py`` in two services stopped collection of the whole run; two
helpers of one name would let one shadow the other. This check reads the test paths from
``[tool.pytest.ini_options] testpaths`` and computes each module's import name the way
pytest's default import mode does, so the clash is caught in milliseconds, by name,
before a test run.

Standard library only: it runs on the orchestrator's interpreter.
"""

from __future__ import annotations

import argparse
import os
import sys
import tomllib
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

from scripts._toolkit.config import ScriptConfigError, load_config, repo_root
from scripts._toolkit.console import fail, ok

SCRIPT_DIR = Path(__file__).resolve().parent


def import_name(module: Path) -> str:
    """The name pytest imports ``module`` under: its stem, prefixed by every enclosing
    package (a directory holding ``__init__.py``)."""
    parts = [module.stem]
    directory = module.parent
    while (directory / "__init__.py").is_file():
        parts.append(directory.name)
        directory = directory.parent
    return ".".join(reversed(parts))


def collect_modules(
    root: Path,
    testpaths: Iterable[str],
    skip_dirs: frozenset[str],
    exempt_names: frozenset[str],
    test_dir_name: str = "tests",
) -> list[Path]:
    """Every Python module inside a test directory below the given test paths."""
    found: list[Path] = []
    for testpath in testpaths:
        for current, dirnames, filenames in os.walk(root / testpath):
            dirnames[:] = sorted(name for name in dirnames if name not in skip_dirs)
            here = Path(current)
            in_tests = test_dir_name in here.relative_to(root).parts
            for name in sorted(filenames):
                if not name.endswith(".py") or name in exempt_names:
                    continue
                if in_tests or name.startswith("test_") or name.endswith("_test.py"):
                    found.append(here / name)
    return found


def clashes(modules: Iterable[Path]) -> dict[str, list[Path]]:
    """Import names claimed by more than one file."""
    by_name: dict[str, list[Path]] = defaultdict(list)
    for module in modules:
        by_name[import_name(module)].append(module)
    return {name: paths for name, paths in sorted(by_name.items()) if len(paths) > 1}


def _testpaths(pyproject: Path) -> list[str]:
    try:
        settings = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ScriptConfigError(f"cannot read {pyproject}: {error}") from error
    testpaths = settings.get("tool", {}).get("pytest", {}).get("ini_options", {})
    paths = testpaths.get("testpaths")
    if not isinstance(paths, list) or not all(isinstance(p, str) for p in paths):
        raise ScriptConfigError(f"{pyproject}: no testpaths list for pytest")
    return paths


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="test-names",
        description="Fail when two pytest modules would be imported under one name.",
    )
    parser.parse_args(argv)

    root = repo_root()
    config = load_config(SCRIPT_DIR, "test_names.json")
    try:
        testpaths = _testpaths(root / str(config["pyproject"]))
    except ScriptConfigError as error:
        fail(str(error))
        return 1
    modules = collect_modules(
        root,
        testpaths,
        frozenset(config["skip_dirs"]),
        frozenset(config["exempt_names"]),
        str(config["test_dir_name"]),
    )
    found = clashes(modules)
    for name, paths in found.items():
        listed = ", ".join(path.relative_to(root).as_posix() for path in paths)
        fail(f"{name!r} is imported from {len(paths)} files: {listed}")
    if found:
        fail("rename all but one: pytest imports a module of a tests/ folder by name")
        return 1
    ok(f"{len(modules)} test modules under {', '.join(testpaths)}, every name unique")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
