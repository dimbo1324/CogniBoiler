"""The command lines console-e2e hands to pnpm.

Run with:  python -m unittest discover -s scripts -t .
"""

from __future__ import annotations

import unittest

from scripts.console_e2e.__main__ import build_commands


class BuildCommandsTest(unittest.TestCase):
    def test_chromium_is_installed_before_the_checks_run(self) -> None:
        commands = build_commands({"web_dir": "apps/web"}, True, ["--grep", "trends"])
        self.assertEqual(
            commands,
            [
                [
                    "pnpm",
                    "--dir",
                    "apps/web",
                    "exec",
                    "playwright",
                    "install",
                    "chromium",
                ],
                [
                    "pnpm",
                    "--dir",
                    "apps/web",
                    "exec",
                    "playwright",
                    "test",
                    "--grep",
                    "trends",
                ],
            ],
        )

    def test_the_install_step_can_be_skipped(self) -> None:
        commands = build_commands({"web_dir": "apps/web"}, False, [])
        self.assertEqual(
            commands, [["pnpm", "--dir", "apps/web", "exec", "playwright", "test"]]
        )


if __name__ == "__main__":
    unittest.main()
