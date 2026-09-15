"""What clean-caches may select, and that --apply deletes nothing else.

Run with:  python -m unittest discover -s scripts -t .
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.clean_caches.__main__ import remove
from scripts.clean_caches.plan import Rules, plan

RULES = Rules(
    directory_names=frozenset({"__pycache__", ".pytest_cache"}),
    file_names=frozenset({".coverage"}),
    extra_paths=("apps/web/dist", "node_modules/pkg/dist"),
    protected_names=frozenset({".venv", "node_modules", ".git"}),
)


def _touch(root: Path, relative: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x", encoding="utf-8")


class PlanTest(unittest.TestCase):
    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory()
        self.root = Path(self._temp.name).resolve()
        for relative in (
            "apps/svc/src/__pycache__/mod.pyc",
            "apps/svc/src/keep.py",
            ".venv/lib/__pycache__/site.pyc",
            "node_modules/pkg/__pycache__/x.pyc",
            "node_modules/pkg/dist/index.js",
            ".git/objects/__pycache__/y",
            ".pytest_cache/v/cache",
            ".coverage",
            "apps/web/dist/index.html",
        ):
            _touch(self.root, relative)

    def tearDown(self) -> None:
        self._temp.cleanup()

    def _relative(self) -> set[str]:
        return {
            target.relative_to(self.root).as_posix()
            for target in plan(self.root, RULES)
        }

    def test_caches_and_build_output_are_selected(self) -> None:
        self.assertEqual(
            self._relative(),
            {"apps/svc/src/__pycache__", ".pytest_cache", ".coverage", "apps/web/dist"},
        )

    def test_nothing_under_a_protected_directory_is_selected(self) -> None:
        selected = self._relative()
        for protected in (".venv", "node_modules", ".git"):
            with self.subTest(protected=protected):
                self.assertFalse(any(path.startswith(protected) for path in selected))

    def test_apply_removes_exactly_the_plan(self) -> None:
        failures = remove(plan(self.root, RULES))
        self.assertEqual(failures, [])
        self.assertTrue((self.root / "apps/svc/src/keep.py").exists())
        self.assertTrue((self.root / ".venv/lib/__pycache__/site.pyc").exists())
        self.assertFalse((self.root / "apps/svc/src/__pycache__").exists())
        self.assertFalse((self.root / ".coverage").exists())


if __name__ == "__main__":
    unittest.main()
