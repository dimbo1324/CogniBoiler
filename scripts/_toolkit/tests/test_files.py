"""Files that hold secrets: written whole, and readable by their owner only.

Run with:  python -m unittest discover -s scripts -t .
"""

from __future__ import annotations

import os
import stat
import tempfile
import unittest
from pathlib import Path

from scripts._toolkit.files import make_private_dir, write_private


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


class WritePrivateTest(unittest.TestCase):
    def test_the_text_replaces_the_file_whole_and_leaves_no_temporary(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / ".env"
            target.write_text("OLD=1\n", encoding="utf-8")
            write_private(target, "NEW=2\nPEM=a\nb\n")
            self.assertEqual(target.read_bytes(), b"NEW=2\nPEM=a\nb\n")
            self.assertEqual([path.name for path in Path(temp).iterdir()], [".env"])

    @unittest.skipUnless(os.name == "posix", "file modes are POSIX permissions")
    def test_a_new_file_is_readable_by_its_owner_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / ".env"
            write_private(target, "KEY=value\n")
            self.assertEqual(_mode(target), 0o600)

    @unittest.skipUnless(os.name == "posix", "file modes are POSIX permissions")
    def test_a_rewrite_does_not_loosen_a_tightened_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / ".env"
            target.write_text("KEY=value\n", encoding="utf-8")
            target.chmod(0o600)
            write_private(target, "KEY=other\n")
            self.assertEqual(_mode(target), 0o600)

    @unittest.skipUnless(os.name == "posix", "file modes are POSIX permissions")
    def test_a_leftover_world_readable_temporary_is_not_inherited(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / ".env"
            leftover = Path(temp) / ".env.tmp"
            leftover.write_text("stale", encoding="utf-8")
            leftover.chmod(0o644)
            write_private(target, "KEY=value\n")
            self.assertEqual(_mode(target), 0o600)


class MakePrivateDirTest(unittest.TestCase):
    def test_the_directory_is_created_with_its_parents(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "backups" / "20260928T120000Z"
            make_private_dir(target)
            self.assertTrue(target.is_dir())

    def test_an_existing_directory_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(FileExistsError):
                make_private_dir(Path(temp))

    @unittest.skipUnless(os.name == "posix", "file modes are POSIX permissions")
    def test_only_the_owner_may_enter_it(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "backup"
            make_private_dir(target)
            self.assertEqual(_mode(target), 0o700)


if __name__ == "__main__":
    unittest.main()
