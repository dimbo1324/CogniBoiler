"""How generate-proto --check decides that the committed stubs are out of date.

Run with:  python -m unittest discover -s scripts -t .
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.generate_proto.__main__ import drifted


class DriftTest(unittest.TestCase):
    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory()
        self.fresh = Path(self._temp.name) / "fresh"
        self.committed = Path(self._temp.name) / "committed"
        self.fresh.mkdir()
        self.committed.mkdir()

    def tearDown(self) -> None:
        self._temp.cleanup()

    def _write(self, directory: Path, name: str, data: bytes) -> None:
        (directory / name).write_bytes(data)

    def test_identical_stubs_have_not_drifted(self) -> None:
        for directory in (self.fresh, self.committed):
            self._write(directory, "cogniboiler_pb2.py", b"stub\n")
            self._write(directory, "cogniboiler_pb2_grpc.py", b"grpc\n")
        self.assertEqual(drifted(self.fresh, self.committed), [])

    def test_a_checkout_with_crlf_endings_has_not_drifted(self) -> None:
        self._write(self.fresh, "cogniboiler_pb2.py", b"a\nb\n")
        self._write(self.committed, "cogniboiler_pb2.py", b"a\r\nb\r\n")
        self.assertEqual(drifted(self.fresh, self.committed), [])

    def test_a_changed_stub_is_named(self) -> None:
        self._write(self.fresh, "cogniboiler_pb2.py", b"new field\n")
        self._write(self.committed, "cogniboiler_pb2.py", b"old\n")
        self.assertEqual(drifted(self.fresh, self.committed), ["cogniboiler_pb2.py"])

    def test_a_stub_on_one_side_only_is_named(self) -> None:
        self._write(self.fresh, "cogniboiler_pb2_grpc.py", b"grpc\n")
        self._write(self.committed, "stale_pb2.py", b"gone\n")
        self.assertEqual(
            drifted(self.fresh, self.committed),
            ["cogniboiler_pb2_grpc.py", "stale_pb2.py"],
        )

    def test_files_that_are_not_stubs_are_ignored(self) -> None:
        self._write(self.committed, "__init__.py", b"")
        self._write(self.committed, "README.md", b"notes")
        self.assertEqual(drifted(self.fresh, self.committed), [])


if __name__ == "__main__":
    unittest.main()
