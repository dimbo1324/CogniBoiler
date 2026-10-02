"""Console helpers every script shares.

Run with:  python -m unittest discover -s scripts -t .
"""

from __future__ import annotations

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts._toolkit.console import confirm, display_path


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


class _Terminal(io.StringIO):
    """A stdin that claims to be (or not to be) a terminal."""

    def __init__(self, interactive: bool) -> None:
        super().__init__()
        self._interactive = interactive

    def isatty(self) -> bool:
        super().isatty()  # raises ValueError once closed, like a real stream
        return self._interactive


class ConfirmTest(unittest.TestCase):
    """The prompt that guards deleting files and overwriting databases."""

    def _ask(self, *, interactive: bool, answer: str | BaseException) -> bool:
        reply = mock.Mock(side_effect=[answer])
        with (
            mock.patch.object(sys, "stdin", _Terminal(interactive)),
            mock.patch("builtins.input", reply),
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            return confirm("Delete it?", assume_yes=False)

    def test_assume_yes_short_circuits_without_asking(self) -> None:
        with mock.patch("builtins.input") as asked:
            self.assertTrue(confirm("Delete it?", assume_yes=True))
        asked.assert_not_called()

    def test_no_terminal_means_no(self) -> None:
        self.assertFalse(self._ask(interactive=False, answer="y"))

    def test_the_stream_ending_means_no(self) -> None:
        self.assertFalse(self._ask(interactive=True, answer=EOFError()))

    def test_only_y_or_yes_is_consent(self) -> None:
        for answer, consent in (
            ("y", True),
            ("YES", True),
            ("  yes ", True),
            ("n", False),
            ("", False),
            ("maybe", False),
            ("yes please", False),
        ):
            with self.subTest(answer=answer):
                self.assertEqual(self._ask(interactive=True, answer=answer), consent)

    def test_a_closed_stdin_means_no(self) -> None:
        closed = _Terminal(True)
        closed.close()
        with (
            mock.patch.object(sys, "stdin", closed),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            self.assertFalse(confirm("Delete it?", assume_yes=False))


if __name__ == "__main__":
    unittest.main()
