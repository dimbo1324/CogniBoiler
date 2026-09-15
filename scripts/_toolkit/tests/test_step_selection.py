"""Step selection against the checkout, and the gate's run-everything mode."""

from __future__ import annotations

import contextlib
import io
import os
import sys
import tempfile
import unittest
from collections.abc import Iterator
from pathlib import Path
from unittest import mock

from scripts._toolkit import steps
from scripts._toolkit.processes import CommandResult


@contextlib.contextmanager
def _fake_runs(results: dict[str, int]) -> Iterator[list[str]]:
    launched: list[str] = []

    def fake(argv: list[str], _root: Path) -> CommandResult:
        launched.append(argv[0])
        return CommandResult(argv=argv, returncode=results.get(argv[0], 0))

    with mock.patch.object(steps, "run", fake):
        yield launched


def _quiet() -> contextlib.ExitStack:
    stack = contextlib.ExitStack()
    buffer = io.StringIO()
    stack.enter_context(contextlib.redirect_stdout(buffer))
    stack.enter_context(contextlib.redirect_stderr(buffer))
    return stack


class KeepGoingTest(unittest.TestCase):
    def test_every_step_runs_after_a_required_failure(self) -> None:
        declared = [
            {"label": "first", "argv": ["a"]},
            {"label": "second", "argv": ["b"]},
            {"label": "third", "argv": ["c"]},
        ]
        with _fake_runs({"a": 1, "b": 2}) as launched, _quiet():
            code = steps.run_steps(declared, Path("."), title="t", keep_going=True)
        self.assertEqual(code, 1)
        self.assertEqual(launched, ["a", "b", "c"])

    def test_a_clean_run_with_keep_going_is_still_zero(self) -> None:
        with _fake_runs({}), _quiet():
            code = steps.run_steps(
                [{"label": "only", "argv": ["a"]}],
                Path("."),
                title="t",
                keep_going=True,
            )
        self.assertEqual(code, 0)

    def test_skipped_rows_reach_the_summary(self) -> None:
        buffer = io.StringIO()
        with (
            _fake_runs({}),
            contextlib.redirect_stdout(buffer),
            contextlib.redirect_stderr(buffer),
        ):
            steps.run_steps(
                [], Path("."), title="t", skipped=[("frontend lint", "no node_modules")]
            )
        self.assertIn("frontend lint", buffer.getvalue())
        self.assertIn("skipped (no node_modules)", buffer.getvalue())


class SelectStepsTest(unittest.TestCase):
    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory()
        self.root = Path(self._temp.name)
        self.declared = [
            {"label": "python step", "argv": ["python", "-m", "x"]},
            {"label": "needs web", "argv": ["pnpm", "lint"], "requires_path": "web/nm"},
        ]

    def tearDown(self) -> None:
        self._temp.cleanup()

    def test_python_means_the_running_interpreter(self) -> None:
        with mock.patch.dict(os.environ, {"CI": ""}):
            selected, _ = steps.select_steps(self.declared, self.root)
        self.assertEqual(selected[0]["argv"][0], sys.executable)

    def test_a_step_whose_path_is_absent_is_skipped_locally(self) -> None:
        with mock.patch.dict(os.environ, {"CI": ""}):
            selected, skipped = steps.select_steps(self.declared, self.root)
        self.assertEqual([entry["label"] for entry in selected], ["python step"])
        self.assertEqual(skipped, [("needs web", "no web/nm")])

    def test_under_ci_the_step_runs_instead_of_being_skipped(self) -> None:
        with mock.patch.dict(os.environ, {"CI": "true"}):
            selected, skipped = steps.select_steps(self.declared, self.root)
        self.assertEqual(len(selected), 2)
        self.assertEqual(skipped, [])

    def test_a_present_path_keeps_the_step(self) -> None:
        (self.root / "web" / "nm").mkdir(parents=True)
        with mock.patch.dict(os.environ, {"CI": ""}):
            selected, skipped = steps.select_steps(self.declared, self.root)
        self.assertEqual(len(selected), 2)
        self.assertEqual(skipped, [])


if __name__ == "__main__":
    unittest.main()
