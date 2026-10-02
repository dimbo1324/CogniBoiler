"""The smoke report's bookkeeping and the fresh-history query.

The tolerant JSON lookup moved with the gateway client: _toolkit/tests/test_gateway.py.

Run with:  python -m unittest discover -s scripts -t .
"""

from __future__ import annotations

import contextlib
import io
import unittest

from scripts.smoke.__main__ import FRESH_TELEMETRY_S, Report, history_path


class ReportTest(unittest.TestCase):
    def test_one_failure_marks_the_whole_report_failed(self) -> None:
        report = Report()
        with (
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            report.record("first", True, "fine")
            report.record("second", False, "HTTP 500")
            report.record("third", True, "fine")
        self.assertTrue(report.failed)
        self.assertEqual(
            [name for name, _ in report.rows], ["first", "second", "third"]
        )
        self.assertIn("HTTP 500", report.rows[1][1])


class HistoryPathTest(unittest.TestCase):
    def test_only_points_from_the_fresh_window_count(self) -> None:
        path = history_path(1_789_700_000_000)
        self.assertIn("measurement=boiler_sensors", path)
        self.assertIn(f"start_ms={1_789_700_000_000 - FRESH_TELEMETRY_S * 1000}", path)
        self.assertNotIn("end_ms", path)


if __name__ == "__main__":
    unittest.main()
