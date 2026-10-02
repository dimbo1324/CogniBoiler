"""Tests for how a typed argument string becomes argv.

`shlex` defaults to POSIX rules, where a backslash escapes the next character. On
Windows it is the path separator, so the default silently ate it: typing
``--out C:\\Users\\dev\\build`` produced ``['--out', 'C:Usersdevbuild']`` and the script
was handed a path that does not exist, with nothing reported.
"""

from __future__ import annotations

import os
import unittest

from scripts.runner.execution import split_arguments


class SplitArgumentsTest(unittest.TestCase):
    def test_empty_input_is_no_arguments(self) -> None:
        self.assertEqual(split_arguments(""), [])
        self.assertEqual(split_arguments("   "), [])

    def test_plain_flags_split_on_whitespace(self) -> None:
        self.assertEqual(
            split_arguments("--scope artifacts --yes"),
            ["--scope", "artifacts", "--yes"],
        )

    def test_a_quoted_argument_stays_one_token_without_its_quotes(self) -> None:
        self.assertEqual(
            split_arguments('--label "two words"'), ["--label", "two words"]
        )

    # The Windows rules are selected explicitly, so these run on the Linux CI too
    # instead of being skipped everywhere but a developer's Windows machine.
    def test_a_windows_path_keeps_its_backslashes(self) -> None:
        self.assertEqual(
            split_arguments(r"--out C:\Users\dev\build", windows=True),
            ["--out", r"C:\Users\dev\build"],
        )

    def test_a_quoted_windows_path_with_spaces_survives_whole(self) -> None:
        self.assertEqual(
            split_arguments(r'--out "C:\Program Files\CogniBoiler"', windows=True),
            ["--out", r"C:\Program Files\CogniBoiler"],
        )

    def test_posix_rules_treat_a_backslash_as_an_escape(self) -> None:
        self.assertEqual(
            split_arguments(r"--label two\ words", windows=False),
            ["--label", "two words"],
        )

    def test_the_platform_picks_the_rules_by_default(self) -> None:
        raw = r"--out C:\Users\dev"
        self.assertEqual(
            split_arguments(raw), split_arguments(raw, windows=os.name == "nt")
        )


if __name__ == "__main__":
    unittest.main()
