"""Tests for tool resolution, timeouts, and raw output capture."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

from scripts._toolkit import processes

_HERE = Path(__file__).resolve().parent


class FindToolTest(unittest.TestCase):
    def test_the_running_interpreter_is_found_by_absolute_path(self) -> None:
        self.assertIsNotNone(processes.find_tool(sys.executable))

    def test_a_tool_that_does_not_exist_is_none(self) -> None:
        self.assertIsNone(processes.find_tool("cogniboiler-no-such-tool-xyz"))

    def test_the_shim_suffixes_hold_no_unreachable_entry(self) -> None:
        # An empty entry was skipped by the loop that consumes this tuple, so it looked
        # like coverage of the bare-name case that it did not provide.
        self.assertTrue(all(suffix for suffix in processes._WINDOWS_SHIM_SUFFIXES))


class CaptureTest(unittest.TestCase):
    def test_a_missing_tool_reports_not_found_rather_than_raising(self) -> None:
        code, out, err = processes.capture_bytes(
            ["cogniboiler-no-such-tool-xyz"], _HERE
        )
        self.assertEqual(code, processes.NOT_FOUND)
        self.assertEqual((out, err), (b"", b""))

    def test_a_hung_tool_times_out_instead_of_blocking_forever(self) -> None:
        # The Windows Store's `python` stub is the real-world case: it resolves, opens a
        # shop window, and never answers. Before the timeout this hung `doctor`.
        code, _out, _err = processes.capture_bytes(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            _HERE,
            timeout=0.5,
        )
        self.assertEqual(code, processes.TIMED_OUT)

    def test_raw_bytes_are_returned_undecoded(self) -> None:
        # The deletion planner needs the exact bytes: a lossy decode would hand the
        # protection rules a name that is not the real one.
        code, out, _err = processes.capture_bytes(
            [sys.executable, "-c", r"import sys; sys.stdout.buffer.write(b'\xff\xfe')"],
            _HERE,
        )
        self.assertEqual(code, 0)
        self.assertEqual(out, b"\xff\xfe")

    def test_text_capture_replaces_undecodable_bytes_instead_of_raising(self) -> None:
        code, out = processes.capture(
            [sys.executable, "-c", r"import sys; sys.stdout.buffer.write(b'ok\xff')"],
            _HERE,
        )
        self.assertEqual(code, 0)
        self.assertIn("ok", out)


if __name__ == "__main__":
    unittest.main()
