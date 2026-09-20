"""What the demo script decides on its own: its clock, its audit lines and the log scan.

Run with:  python -m unittest discover -s scripts -t .
"""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from scripts.demo.__main__ import (
    Demo,
    Reply,
    audit_line,
    clock,
    error_lines,
    power_mw,
    severities,
)


class ClockTest(unittest.TestCase):
    def test_elapsed_time_reads_like_a_demo_script(self) -> None:
        self.assertEqual(clock(0), "0:00")
        self.assertEqual(clock(9.7), "0:09")
        self.assertEqual(clock(140), "2:20")
        self.assertEqual(clock(3601), "60:01")


class AuditLineTest(unittest.TestCase):
    def test_a_row_names_the_moment_to_the_second_the_user_and_the_outcome(
        self,
    ) -> None:
        line = audit_line(
            {
                "timestamp_ms": 1_789_840_051_080,
                "username": "operator",
                "method": "POST",
                "endpoint": "/api/v1/commands/load",
                "outcome": "accepted",
            }
        )
        self.assertIn("2026-09-19T17:47:31+00:00", line)
        self.assertIn("operator", line)
        self.assertIn("POST /api/v1/commands/load → accepted", line)

    def test_a_row_without_an_outcome_falls_back_to_the_status(self) -> None:
        line = audit_line({"timestamp_ms": 0, "response_status": 403})
        self.assertIn("1970-01-01T00:00:00+00:00", line)
        self.assertIn("403", line)


class ReplyTest(unittest.TestCase):
    def test_only_an_accepted_answer_counts_as_accepted(self) -> None:
        self.assertTrue(Reply(200, {"accepted": True}).accepted)
        self.assertFalse(Reply(200, {"accepted": False, "reason": "E-Stop"}).accepted)
        self.assertFalse(Reply(503, {"detail": "PLCService is unavailable."}).accepted)

    def test_a_refusal_carries_its_reason(self) -> None:
        self.assertEqual(
            Reply(200, {"accepted": False, "reason": "E-Stop active"}).refusal,
            "HTTP 200: E-Stop active",
        )
        self.assertEqual(
            Reply(503, {"detail": "PLCService is unavailable."}).refusal,
            "HTTP 503: PLCService is unavailable.",
        )
        self.assertEqual(Reply(500, None).refusal, "HTTP 500")


class PlantReadingTest(unittest.TestCase):
    def test_power_is_read_in_megawatts_and_survives_a_missing_field(self) -> None:
        self.assertAlmostEqual(
            power_mw({"turbine": {"electrical_power_w": 250_400_000.0}}), 250.4
        )
        self.assertNotEqual(power_mw({"turbine": {}}), power_mw({"turbine": {}}))

    def test_severities_are_collected_from_the_open_alarms(self) -> None:
        self.assertEqual(
            severities([{"severity": "warning"}, {"severity": "critical"}]),
            {"warning", "critical"},
        )
        self.assertEqual(severities([]), set())


class ErrorLinesTest(unittest.TestCase):
    def _write(
        self, directory: Path, name: str, records: list[dict[str, object]]
    ) -> None:
        text = "\n".join(json.dumps(record) for record in records)
        (directory / name).write_text(text + "\n", encoding="utf-8")

    def test_only_errors_written_during_the_demo_are_reported(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            logs = Path(temp)
            self._write(
                logs,
                "plc-controller.log",
                [
                    {
                        "timestamp": "2026-09-20T03:00:00Z",
                        "level": "error",
                        "event": "old",
                    },
                    {
                        "timestamp": "2026-09-20T04:10:00Z",
                        "level": "warning",
                        "event": "EMERGENCY STOP triggered",
                    },
                    {
                        "timestamp": "2026-09-20T04:11:00Z",
                        "level": "error",
                        "event": "queue full",
                    },
                ],
            )
            self._write(
                logs,
                "historian.log",
                [
                    {
                        "timestamp": "2026-09-20T04:12:00Z",
                        "level": "critical",
                        "event": "lost",
                    }
                ],
            )
            (logs / "notes.txt").write_text("not a log", encoding="utf-8")

            found = error_lines(logs, "2026-09-20T04:00:00")

        self.assertEqual(
            [(service, record["event"]) for service, record in found],
            [("historian", "lost"), ("plc-controller", "queue full")],
        )

    def test_a_broken_line_and_an_empty_directory_are_tolerated(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            logs = Path(temp)
            (logs / "api-gateway.log").write_text("{ not json\n[]\n", encoding="utf-8")
            self.assertEqual(error_lines(logs, "2026-09-20T00:00:00"), [])
            self.assertEqual(error_lines(logs / "missing", "2026-09-20T00:00:00"), [])


class DemoNarratorTest(unittest.TestCase):
    def test_a_failed_step_marks_the_run_and_keeps_its_reason(self) -> None:
        demo = Demo("http://localhost:8080")
        with (
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()) as errors,
        ):
            demo.step("the operator asks for 300 MW", True, "accepted by the PLC")
            demo.step("the unit comes back", False, "PLC estop")
        self.assertTrue(demo.failed)
        self.assertEqual([name for name, _ in demo.rows][1], "the unit comes back")
        self.assertIn("FAILED (PLC estop)", demo.rows[1][1])
        self.assertIn("the unit comes back", errors.getvalue())

    def test_waiting_ends_as_soon_as_the_plant_is_ready(self) -> None:
        demo = Demo("http://localhost:8080")
        readings = iter([200.0, 260.0, 301.0])
        with contextlib.redirect_stdout(io.StringIO()):
            passed = demo.wait_for(
                "the regulators bring the unit up",
                lambda: next(readings),
                lambda mw: mw >= 295.0,
                lambda mw: f"{mw:.1f} MW",
                timeout_s=30.0,
            )
        self.assertTrue(passed)
        self.assertFalse(demo.failed)

    def test_waiting_gives_up_and_says_what_it_last_saw(self) -> None:
        demo = Demo("http://localhost:8080")
        with (
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            passed = demo.wait_for(
                "the interlock trips the unit",
                lambda: 250.0,
                lambda mw: mw > 400.0,
                lambda mw: f"{mw:.1f} MW",
                timeout_s=0.0,
            )
        self.assertFalse(passed)
        self.assertTrue(demo.failed)
        self.assertIn("250.0 MW", demo.rows[0][1])


if __name__ == "__main__":
    unittest.main()
