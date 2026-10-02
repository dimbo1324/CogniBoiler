"""Which steps the quality gate runs, full and --quick.

The gate is the one verification path CI runs, so its step lists are tested as data: a
section dropped from the JSON would otherwise disappear from CI without a sign.

Run with:  python -m unittest discover -s scripts -t .
"""

from __future__ import annotations

import unittest
from typing import Any
from unittest import mock

from scripts._toolkit.config import load_config
from scripts.quality_gate import __main__ as gate

CONFIG = load_config(gate.SCRIPT_DIR, "steps.json")


def _labels(steps: list[dict[str, Any]]) -> list[str]:
    return [str(step["label"]) for step in steps]


def _find(steps: list[dict[str, Any]], *argv_part: str) -> dict[str, Any] | None:
    for step in steps:
        argv = list(step["argv"])
        if all(part in argv for part in argv_part):
            return step
    return None


class StepListTest(unittest.TestCase):
    def test_every_step_has_a_label_an_argv_and_is_required(self) -> None:
        for name in ("steps", "quick_steps"):
            for step in CONFIG[name]:
                with self.subTest(list=name, step=step.get("label")):
                    self.assertIsInstance(step["label"], str)
                    self.assertTrue(step["argv"])
                    self.assertIs(step["required"], True)

    def test_labels_are_unique(self) -> None:
        for name in ("steps", "quick_steps"):
            labels = _labels(CONFIG[name])
            self.assertEqual(len(labels), len(set(labels)), name)

    def test_the_quick_gate_is_a_subset_of_the_full_gate(self) -> None:
        full = {step["label"]: step["argv"] for step in CONFIG["steps"]}
        for step in CONFIG["quick_steps"]:
            with self.subTest(step=step["label"]):
                self.assertEqual(full.get(step["label"]), step["argv"])

    def test_the_full_gate_runs_every_kind_of_check(self) -> None:
        steps = CONFIG["steps"]
        for argv_part in (
            ("sync", "--locked"),
            ("ruff", "format", "--check"),
            ("ruff", "check"),
            ("mypy",),
            ("pytest",),
            ("scripts.generate_proto", "--check"),
            ("scripts.generate_openapi", "--check"),
            ("scripts.sync_agents", "--check"),
            ("unittest", "discover"),
        ):
            with self.subTest(argv_part=argv_part):
                self.assertIsNotNone(_find(steps, *argv_part))

    def test_the_quick_gate_skips_the_test_suites(self) -> None:
        quick = CONFIG["quick_steps"]
        self.assertIsNone(_find(quick, "pytest"))
        self.assertIsNotNone(_find(quick, "mypy"))

    def test_the_scripts_tests_run_in_the_project_environment(self) -> None:
        step = _find(CONFIG["steps"], "unittest", "discover")
        assert step is not None
        self.assertEqual(step["argv"][:3], ["uv", "run", "--no-sync"])
        self.assertIn("error", step["argv"])


class MainTest(unittest.TestCase):
    def _run(self, *argv: str) -> mock.Mock:
        with mock.patch.object(gate, "run_steps", return_value=0) as run_steps:
            self.assertEqual(gate.main(list(argv)), 0)
        return run_steps

    def _selected(self, run_steps: mock.Mock) -> set[str]:
        call = run_steps.call_args
        return set(_labels(call.args[0])) | {
            label for label, _ in call.kwargs["skipped"]
        }

    def test_the_full_gate_keeps_going_after_a_failure(self) -> None:
        run_steps = self._run()
        self.assertTrue(run_steps.call_args.kwargs["keep_going"])
        self.assertEqual(run_steps.call_args.kwargs["title"], "quality-gate (full)")
        self.assertEqual(self._selected(run_steps), set(_labels(CONFIG["steps"])))

    def test_quick_selects_the_quick_list(self) -> None:
        run_steps = self._run("--quick")
        self.assertEqual(run_steps.call_args.kwargs["title"], "quality-gate (quick)")
        self.assertEqual(self._selected(run_steps), set(_labels(CONFIG["quick_steps"])))


if __name__ == "__main__":
    unittest.main()
