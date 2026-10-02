"""format-code rewrites by default and only checks with --check.

Run with:  python -m unittest discover -s scripts -t .
"""

from __future__ import annotations

import unittest
from unittest import mock

from scripts.format_code import __main__ as format_code


def _run(*argv: str) -> mock.Mock:
    with mock.patch.object(format_code, "run_steps", return_value=0) as run_steps:
        format_code.main(list(argv))
    return run_steps


def _argvs(run_steps: mock.Mock) -> list[list[str]]:
    selected = [list(step["argv"]) for step in run_steps.call_args.args[0]]
    return selected


class FormatCodeTest(unittest.TestCase):
    def test_a_default_run_rewrites_the_files(self) -> None:
        argvs = _argvs(_run())
        self.assertIn("--fix", argvs[0])
        self.assertTrue(
            any("format" in argv and "--check" not in argv for argv in argvs)
        )

    def test_check_rewrites_nothing(self) -> None:
        run_steps = _run("--check")
        self.assertEqual(
            run_steps.call_args.kwargs["title"], "format-code (check only)"
        )
        argvs = _argvs(run_steps)
        self.assertTrue(argvs)
        for argv in argvs:
            with self.subTest(argv=argv):
                self.assertNotIn("--fix", argv)
                self.assertFalse("format" in argv and "--check" not in argv)


if __name__ == "__main__":
    unittest.main()
