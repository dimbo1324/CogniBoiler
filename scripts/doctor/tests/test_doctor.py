"""What doctor reports about tools, without launching anything real.

Run with:  python -m unittest discover -s scripts -t .
"""

from __future__ import annotations

import contextlib
import io
import unittest
from pathlib import Path
from unittest import mock

from scripts._toolkit.config import ScriptConfigError
from scripts.doctor import __main__ as doctor

HERE = Path(__file__).resolve().parent


def _tool(name: str, required: bool, version_args: object = None) -> dict[str, object]:
    return {
        "name": name,
        "required": required,
        "needed_for": "a test",
        "version_args": ["--version"] if version_args is None else version_args,
    }


class CheckToolsTest(unittest.TestCase):
    def _check(self, tools: list[dict[str, object]]) -> tuple[list[str], list[str]]:
        found = {"git": "/usr/bin/git"}
        with (
            mock.patch.object(doctor, "find_tool", side_effect=found.get),
            mock.patch.object(doctor, "capture", return_value=(0, "git 2.50\n")),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            return doctor._check_tools({"tools": tools}, HERE)

    def test_missing_tools_are_sorted_by_whether_they_are_required(self) -> None:
        missing = self._check(
            [_tool("git", True), _tool("uv", True), _tool("helm", False)]
        )
        self.assertEqual(missing, (["uv"], ["helm"]))

    def test_version_arguments_must_be_a_list(self) -> None:
        with self.assertRaises(ScriptConfigError):
            self._check([_tool("git", True, version_args="--version")])

    def test_the_first_non_empty_line_is_the_version(self) -> None:
        self.assertEqual(doctor._first_line("\n  git 2.50 \nmore"), "git 2.50")
        self.assertEqual(doctor._first_line(""), "(no version output)")


if __name__ == "__main__":
    unittest.main()
