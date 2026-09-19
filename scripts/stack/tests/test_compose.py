"""The docker compose command lines `stack` builds, and the log directory it prepares.

Run with:  python -m unittest discover -s scripts -t .
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import tempfile
import unittest
from pathlib import Path

from scripts.stack.__main__ import build_command, prepare_log_dir

CONFIG = {
    "project_name": "cogniboiler",
    "compose_file": "docker-compose.yml",
    "wait_timeout_s": 300,
    "default_profile": "full",
}


def args(**values: object) -> argparse.Namespace:
    defaults: dict[str, object] = {
        "action": "up",
        "profile": None,
        "infra_only": False,
        "no_build": False,
        "volumes": False,
    }
    defaults.update(values)
    return argparse.Namespace(**defaults)


class BuildCommandTest(unittest.TestCase):
    def test_up_starts_the_full_profile_by_default(self) -> None:
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
        self.assertEqual(command[6:8], ["--profile", "full"])
        self.assertIn("--wait", command)
        self.assertEqual(command[-1], "--build")

    def test_a_chosen_profile_and_infra_only(self) -> None:
        core = build_command(args(profile="core", no_build=True), CONFIG)
        self.assertEqual(core[6:8], ["--profile", "core"])
        self.assertNotIn("--build", core)
        infra = build_command(args(infra_only=True), CONFIG)
        self.assertEqual(infra[6:8], ["--profile", "infra"])

    def test_down_stops_every_service(self) -> None:
        command = build_command(args(action="down", volumes=True), CONFIG)
        self.assertEqual(command[6:8], ["--profile", "full"])
        self.assertEqual(command[-2:], ["down", "--volumes"])


class PrepareLogDirTest(unittest.TestCase):
    def test_creates_a_directory_the_services_can_write(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp) / "logs"
            self.assertTrue(prepare_log_dir(directory))
            self.assertTrue(prepare_log_dir(directory))
            self.assertTrue(directory.is_dir())
            if os.name == "posix":
                self.assertEqual(directory.stat().st_mode & 0o777, 0o777)

    def test_a_path_taken_by_a_file_is_reported_instead_of_raised(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            taken = Path(temp) / "logs"
            taken.write_text("not a directory", encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()) as printed:
                self.assertFalse(prepare_log_dir(taken))
            self.assertIn("standard output only", printed.getvalue())


if __name__ == "__main__":
    unittest.main()
