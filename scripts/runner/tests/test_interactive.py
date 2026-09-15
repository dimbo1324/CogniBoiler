"""Tests for the interactive shell's handling of a stream that ends.

Ctrl+Z on Windows and Ctrl+D elsewhere are ordinary ways to leave a menu. They used to
end the session with an ``EOFError`` traceback printed across the terminal.
"""

from __future__ import annotations

import contextlib
import io
import unittest
from pathlib import Path

from scripts.runner.main import CliApp

ROOT = Path(__file__).resolve().parents[3]


class _TtyThatEndsImmediately(io.StringIO):
    """A stream that claims to be a terminal but has nothing to read.

    The orchestrator only enters the interactive shell when stdin is a tty, so a plain
    pipe cannot reach the code path this exercises.
    """

    def isatty(self) -> bool:
        return True

    def readline(self, *_args: object) -> str:
        return ""


@contextlib.contextmanager
def _stdin(stream: io.StringIO):
    import sys

    real = sys.stdin
    sys.stdin = stream
    try:
        yield
    finally:
        sys.stdin = real


def _run_menu(stream: io.StringIO) -> int:
    app = CliApp(ROOT)
    with _stdin(stream), contextlib.redirect_stdout(io.StringIO()):
        return app.main([])


class EndOfInputTest(unittest.TestCase):
    def test_eof_at_the_top_menu_leaves_cleanly(self) -> None:
        self.assertEqual(_run_menu(_TtyThatEndsImmediately()), 0)

    def test_eof_inside_a_category_leaves_cleanly(self) -> None:
        class PicksCategoryThenEnds(_TtyThatEndsImmediately):
            _answers = iter(["1\n"])

            def readline(self, *_args: object) -> str:
                return next(self._answers, "")

        self.assertEqual(_run_menu(PicksCategoryThenEnds()), 0)

    def test_eof_in_the_help_browser_leaves_cleanly(self) -> None:
        class OpensHelpThenEnds(_TtyThatEndsImmediately):
            _answers = iter(["h\n"])

            def readline(self, *_args: object) -> str:
                return next(self._answers, "")

        self.assertEqual(_run_menu(OpensHelpThenEnds()), 0)


if __name__ == "__main__":
    unittest.main()
