"""The docker compose command lines every Docker-facing script builds the same way.

Run with:  python -m unittest discover -s scripts -t .
"""

from __future__ import annotations

import unittest

from scripts._toolkit.compose import compose_argv, exec_sh

CONFIG = {"project_name": "cogniboiler", "compose_file": "docker-compose.yml"}
PREFIX = [
    "docker",
    "compose",
    "--project-name",
    "cogniboiler",
    "--file",
    "docker-compose.yml",
]


class ComposeArgvTest(unittest.TestCase):
    def test_the_project_and_the_file_are_always_named(self) -> None:
        self.assertEqual(compose_argv(CONFIG, "ps"), [*PREFIX, "ps"])

    def test_a_profile_comes_before_the_subcommand(self) -> None:
        self.assertEqual(
            compose_argv(CONFIG, "up", "--detach", profile="core"),
            [*PREFIX, "--profile", "core", "up", "--detach"],
        )

    def test_a_shell_script_runs_inside_the_service_without_a_terminal(self) -> None:
        self.assertEqual(
            exec_sh(CONFIG, "postgresql", 'psql --username "$POSTGRES_USER"'),
            [
                *PREFIX,
                "exec",
                "-T",
                "postgresql",
                "sh",
                "-c",
                'psql --username "$POSTGRES_USER"',
            ],
        )


if __name__ == "__main__":
    unittest.main()
