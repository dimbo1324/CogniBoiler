"""The docker compose command lines `stack` builds.

Run with:  python -m unittest discover -s scripts -t .
"""

from __future__ import annotations

import argparse
import unittest

from scripts.stack.__main__ import build_command

CONFIG = {
    "project_name": "cogniboiler",
    "compose_file": "docker-compose.yml",
    "wait_timeout_s": 300,
    "infra_services": ["mosquitto", "postgresql"],
    "profiles": ["observability"],
}


def args(**values: object) -> argparse.Namespace:
    defaults: dict[str, object] = {
        "action": "up",
        "infra_only": False,
        "no_build": False,
        "volumes": False,
    }
    defaults.update(values)
    return argparse.Namespace(**defaults)


class BuildCommandTest(unittest.TestCase):
    def test_up_builds_waits_and_enables_the_profiles(self) -> None:
        command = build_command(args(), CONFIG)
        self.assertEqual(
            command[:6],
            [
                "docker",
                "compose",
                "--project-name",
                "cogniboiler",
                "--file",
                "docker-compose.yml",
            ],
        )
        self.assertEqual(command[6:8], ["--profile", "observability"])
        self.assertIn("--wait", command)
        self.assertEqual(command[-1], "--build")

    def test_infra_only_names_the_backing_services(self) -> None:
        command = build_command(args(infra_only=True, no_build=True), CONFIG)
        self.assertEqual(command[-2:], ["mosquitto", "postgresql"])
        self.assertNotIn("--build", command)

    def test_down_stops_the_profiled_services_too(self) -> None:
        command = build_command(args(action="down", volumes=True), CONFIG)
        self.assertIn("--profile", command)
        self.assertEqual(command[-2:], ["down", "--volumes"])


if __name__ == "__main__":
    unittest.main()
