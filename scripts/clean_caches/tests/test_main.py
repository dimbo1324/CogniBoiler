"""clean-caches only lists without --apply, and deletes exactly its plan with it.

Run with:  python -m unittest discover -s scripts -t .
"""

from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.clean_caches import __main__ as clean_caches


class MainTest(unittest.TestCase):
    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory()
        self.root = Path(self._temp.name).resolve()
        self.cache = self.root / "apps" / "svc" / "__pycache__"
        self.cache.mkdir(parents=True)
        (self.cache / "mod.pyc").write_bytes(b"x")
        self.source = self.root / "apps" / "svc" / "mod.py"
        self.source.write_text("x = 1\n", encoding="utf-8")

    def tearDown(self) -> None:
        self._temp.cleanup()

    def _clean(self, *argv: str) -> tuple[int, str]:
        with (
            mock.patch.object(clean_caches, "repo_root", return_value=self.root),
            contextlib.redirect_stdout(io.StringIO()) as printed,
            contextlib.redirect_stderr(io.StringIO()),
        ):
            code = clean_caches.main(list(argv))
        return code, printed.getvalue()

    def test_without_apply_nothing_is_deleted(self) -> None:
        code, printed = self._clean()
        self.assertEqual(code, 0)
        self.assertTrue((self.cache / "mod.pyc").is_file())
        self.assertIn("dry run", printed)
        self.assertIn("apps/svc/__pycache__", printed.replace("\\", "/"))

    def test_apply_deletes_the_plan_and_nothing_else(self) -> None:
        code, _ = self._clean("--apply")
        self.assertEqual(code, 0)
        self.assertFalse(self.cache.exists())
        self.assertTrue(self.source.is_file())


if __name__ == "__main__":
    unittest.main()
