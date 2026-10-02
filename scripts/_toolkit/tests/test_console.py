"""Console helpers every script shares.

Run with:  python -m unittest discover -s scripts -t .
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts._toolkit.console import display_path


class DisplayPathTest(unittest.TestCase):
    def test_a_path_inside_the_root_is_shown_relative(self) -> None:
        root = Path(tempfile.gettempdir()) / "repo"
        self.assertEqual(display_path(root / "backups" / "x", root), "backups/x")

    def test_a_path_outside_the_root_is_shown_whole_instead_of_raising(self) -> None:
        with (
            tempfile.TemporaryDirectory() as root,
            tempfile.TemporaryDirectory() as away,
        ):
            shown = display_path(Path(away) / "backups", Path(root))
        self.assertEqual(shown, (Path(away) / "backups").as_posix())


if __name__ == "__main__":
    unittest.main()
