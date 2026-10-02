"""The check that two pytest modules never share an import name.

Run with:  python -m unittest discover -s scripts -t .
"""

from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.check_test_names import __main__ as check
from scripts.check_test_names.__main__ import clashes, collect_modules, import_name

SKIP = frozenset({"node_modules", "__pycache__"})
EXEMPT = frozenset({"__init__.py", "conftest.py"})


def _touch(root: Path, *relatives: str) -> None:
    for relative in relatives:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")


class ImportNameTest(unittest.TestCase):
    def test_a_module_outside_a_package_is_imported_by_its_basename(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _touch(root, "apps/gw/tests/test_boundaries.py")
            module = root / "apps/gw/tests/test_boundaries.py"
            self.assertEqual(import_name(module), "test_boundaries")

    def test_a_module_inside_packages_carries_their_names(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _touch(root, "svc/tests/__init__.py", "svc/tests/unit/__init__.py")
            _touch(root, "svc/tests/unit/test_x.py")
            module = root / "svc/tests/unit/test_x.py"
            self.assertEqual(import_name(module), "tests.unit.test_x")


class ClashTest(unittest.TestCase):
    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory()
        self.root = Path(self._temp.name)

    def tearDown(self) -> None:
        self._temp.cleanup()

    def _clashes(self, *testpaths: str) -> dict[str, list[str]]:
        modules = collect_modules(self.root, list(testpaths), SKIP, EXEMPT)
        return {
            name: [path.relative_to(self.root).as_posix() for path in paths]
            for name, paths in clashes(modules).items()
        }

    def test_two_test_modules_of_one_name_clash(self) -> None:
        # The case that stopped a whole wave: two tests/test_boundaries.py.
        _touch(
            self.root,
            "apps/gw/tests/test_boundaries.py",
            "apps/opc/tests/test_boundaries.py",
            "apps/opc/tests/test_server.py",
        )
        self.assertEqual(
            self._clashes("apps"),
            {
                "test_boundaries": [
                    "apps/gw/tests/test_boundaries.py",
                    "apps/opc/tests/test_boundaries.py",
                ]
            },
        )

    def test_shared_helper_modules_clash_too(self) -> None:
        # A helper imported by basename is shadowed silently by its namesake.
        _touch(self.root, "apps/a/tests/fakes.py", "shared/b/tests/fakes.py")
        self.assertIn("fakes", self._clashes("apps", "shared/b/tests"))

    def test_conftest_and_package_markers_may_repeat(self) -> None:
        # pytest drops a rootless conftest from sys.modules before importing the next.
        _touch(self.root, "apps/a/tests/conftest.py", "apps/b/tests/conftest.py")
        self.assertEqual(self._clashes("apps"), {})

    def test_modules_in_differently_named_packages_do_not_clash(self) -> None:
        _touch(
            self.root,
            "apps/a/tests/alpha/__init__.py",
            "apps/a/tests/alpha/test_x.py",
            "apps/b/tests/beta/__init__.py",
            "apps/b/tests/beta/test_x.py",
        )
        self.assertEqual(self._clashes("apps"), {})

    def test_skipped_directories_are_not_searched(self) -> None:
        _touch(
            self.root,
            "apps/web/node_modules/pkg/tests/test_x.py",
            "apps/a/tests/test_x.py",
        )
        self.assertEqual(self._clashes("apps"), {})

    def test_source_modules_outside_test_directories_are_ignored(self) -> None:
        _touch(self.root, "apps/a/src/a/server.py", "apps/b/src/b/server.py")
        self.assertEqual(self._clashes("apps"), {})


class MainTest(unittest.TestCase):
    def test_the_repository_has_no_clash(self) -> None:
        with contextlib.redirect_stdout(io.StringIO()) as printed:
            self.assertEqual(check.main([]), 0)
        self.assertIn("test modules", printed.getvalue())

    def test_a_clash_fails_and_names_both_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "pyproject.toml").write_text(
                '[tool.pytest.ini_options]\ntestpaths = ["apps"]\n', encoding="utf-8"
            )
            _touch(root, "apps/a/tests/test_x.py", "apps/b/tests/test_x.py")
            with (
                mock.patch.object(check, "repo_root", return_value=root),
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()) as errors,
            ):
                self.assertEqual(check.main([]), 1)
        self.assertIn("apps/a/tests/test_x.py", errors.getvalue())
        self.assertIn("apps/b/tests/test_x.py", errors.getvalue())


if __name__ == "__main__":
    unittest.main()
