"""A script's config may name paths, and a path it deletes must stay in the repository.

Run with:  python -m unittest discover -s scripts -t .
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts._toolkit.config import ScriptConfigError, is_inside, resolve_inside


class ResolveInsideTest(unittest.TestCase):
    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory()
        self.root = Path(self._temp.name).resolve() / "repo"
        self.root.mkdir()

    def tearDown(self) -> None:
        self._temp.cleanup()

    def test_a_relative_path_below_the_root_is_resolved(self) -> None:
        self.assertEqual(
            resolve_inside(self.root, "apps/web/readme-results/raw"),
            self.root / "apps" / "web" / "readme-results" / "raw",
        )

    def test_a_path_climbing_out_is_refused(self) -> None:
        for relative in ("..", "apps/../../elsewhere", str(self.root.parent)):
            with self.subTest(relative=relative):
                with self.assertRaisesRegex(ScriptConfigError, "outside"):
                    resolve_inside(self.root, relative)

    def test_the_root_itself_is_refused(self) -> None:
        # A config that resolves to the repository root would hand the whole checkout
        # to a deletion.
        with self.assertRaises(ScriptConfigError):
            resolve_inside(self.root, ".")

    def test_is_inside_answers_without_raising(self) -> None:
        self.assertTrue(is_inside(self.root, self.root / "a"))
        self.assertFalse(is_inside(self.root, self.root.parent / "b"))


if __name__ == "__main__":
    unittest.main()
