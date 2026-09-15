"""Tests for the shared step runner's honesty about what it did and did not run."""

from __future__ import annotations

import contextlib
import io
import unittest
from pathlib import Path

from scripts._toolkit import steps
from scripts._toolkit.processes import CommandResult

_HERE = Path(__file__).resolve().parent


@contextlib.contextmanager
def _fake_runs(results: dict[str, int]):
    """Make every step return a canned exit code, keyed by its first argv token."""
    real = steps.run
    steps.run = lambda argv, _root: CommandResult(  # type: ignore[assignment]
        argv=argv, returncode=results.get(argv[0], 0)
    )
    try:
        yield
    finally:
        steps.run = real  # type: ignore[assignment]


def _run(step_list: list[dict], results: dict[str, int]) -> tuple[int, str]:
    """Run the steps with both streams captured.

    stderr is captured too, not just stdout: `console.fail` writes there on purpose, and
    an expected failure printing across the test run reads like a broken suite.
    """
    buffer = io.StringIO()
    with (
        _fake_runs(results),
        contextlib.redirect_stdout(buffer),
        contextlib.redirect_stderr(buffer),
    ):
        code = steps.run_steps(step_list, _HERE, title="test")
    return code, buffer.getvalue()


class RunStepsTest(unittest.TestCase):
    def test_every_step_passing_is_a_zero_exit(self) -> None:
        code, output = _run(
            [{"label": "one", "argv": ["a"]}, {"label": "two", "argv": ["b"]}], {}
        )
        self.assertEqual(code, 0)
        self.assertIn("one", output)

    def test_a_required_failure_stops_the_run(self) -> None:
        code, _ = _run(
            [{"label": "one", "argv": ["a"]}, {"label": "two", "argv": ["b"]}], {"a": 3}
        )
        self.assertEqual(code, 1)

    def test_the_steps_that_never_ran_are_named_in_the_summary(self) -> None:
        # Omitting them made a failing summary shorter than the run was declared to be,
        # so a reader could not tell a skipped step from a deleted one.
        _code, output = _run(
            [
                {"label": "first", "argv": ["a"]},
                {"label": "second", "argv": ["b"]},
                {"label": "third", "argv": ["c"]},
            ],
            {"a": 3},
        )
        self.assertIn("second", output)
        self.assertIn("third", output)
        self.assertIn("not run", output)

    def test_an_optional_failure_does_not_stop_the_run(self) -> None:
        code, output = _run(
            [
                {"label": "optional", "argv": ["a"], "required": False},
                {"label": "after", "argv": ["b"]},
            ],
            {"a": 1},
        )
        self.assertEqual(code, 0)
        self.assertIn("after", output)

    def test_a_missing_required_tool_is_reported_as_missing_not_as_a_failure(
        self,
    ) -> None:
        _code, output = _run(
            [{"label": "needs a tool", "argv": ["a"]}], {"a": steps.NOT_FOUND}
        )
        self.assertIn("MISSING", output)


if __name__ == "__main__":
    unittest.main()
