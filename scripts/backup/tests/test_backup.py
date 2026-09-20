"""The command lines `backup` builds, its folder names and its manifest.

Run with:  python -m unittest discover -s scripts -t .
"""

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path

from scripts.backup.__main__ import (
    directory_size,
    folder_name,
    influx_backup_command,
    influx_cleanup_command,
    influx_copy_command,
    manifest,
    postgres_dump_command,
)

CONFIG = {
    "project_name": "cogniboiler",
    "compose_file": "docker-compose.yml",
    "backup_dir": "backups",
    "postgres_service": "postgresql",
    "influx_service": "influxdb",
}

MOMENT = datetime(2026, 9, 20, 7, 6, 38, tzinfo=UTC)


class FolderNameTest(unittest.TestCase):
    def test_a_backup_is_named_by_the_utc_instant_it_started(self) -> None:
        self.assertEqual(folder_name(MOMENT), "20260920T070638Z")

    def test_names_sort_in_the_order_the_backups_were_taken(self) -> None:
        later = datetime(2026, 9, 20, 7, 6, 39, tzinfo=UTC)
        self.assertLess(folder_name(MOMENT), folder_name(later))


class CommandTest(unittest.TestCase):
    def test_every_command_names_the_project_and_the_compose_file(self) -> None:
        for command in (
            postgres_dump_command(CONFIG),
            influx_backup_command(CONFIG),
            influx_copy_command(CONFIG, Path("backups/x/influxdb")),
            influx_cleanup_command(CONFIG),
        ):
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

    def test_postgres_is_dumped_inside_its_container_without_a_password(self) -> None:
        command = postgres_dump_command(CONFIG)
        self.assertEqual(command[6:10], ["exec", "-T", "postgresql", "sh"])
        script = command[-1]
        self.assertIn("pg_dump", script)
        self.assertIn('"$POSTGRES_USER"', script)
        self.assertIn("--clean --if-exists", script)
        self.assertNotIn("PGPASSWORD", " ".join(command))

    def test_influx_reads_its_token_from_the_container_environment(self) -> None:
        script = influx_backup_command(CONFIG)[-1]
        self.assertIn("influx backup /tmp/cogniboiler-backup", script)
        self.assertIn('"$DOCKER_INFLUXDB_INIT_ADMIN_TOKEN"', script)
        self.assertNotIn("--token 0", script)

    def test_the_backup_is_copied_out_and_the_container_is_left_clean(self) -> None:
        copy = influx_copy_command(CONFIG, Path("backups/20260920T070638Z/influxdb"))
        self.assertEqual(copy[6], "cp")
        self.assertEqual(copy[7], "influxdb:/tmp/cogniboiler-backup")
        self.assertIn("influxdb", copy[8])
        self.assertEqual(
            influx_cleanup_command(CONFIG)[6:],
            [
                "exec",
                "-T",
                "influxdb",
                "rm",
                "-rf",
                "/tmp/cogniboiler-backup",
            ],
        )


class ManifestTest(unittest.TestCase):
    def test_the_manifest_says_when_and_what_without_a_secret(self) -> None:
        written = json.loads(
            manifest(MOMENT, CONFIG, {"postgres.sql": 12, "influxdb": 34})
        )
        self.assertEqual(written["taken_at"], "2026-09-20T07:06:38+00:00")
        self.assertEqual(written["project"], "cogniboiler")
        self.assertEqual(written["files"], {"postgres.sql": 12, "influxdb": 34})
        self.assertNotIn("token", json.dumps(written).lower())


class DirectorySizeTest(unittest.TestCase):
    def test_every_file_below_the_folder_is_counted(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "a").write_bytes(b"12345")
            (root / "nested").mkdir()
            (root / "nested" / "b").write_bytes(b"678")
            self.assertEqual(directory_size(root), 8)


if __name__ == "__main__":
    unittest.main()
