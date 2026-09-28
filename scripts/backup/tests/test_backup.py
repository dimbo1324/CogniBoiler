"""The command lines `backup` builds, its folder names and its manifest.

Run with:  python -m unittest discover -s scripts -t .
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import stat
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from unittest import mock

from scripts._toolkit.processes import NOT_FOUND, CommandResult
from scripts.backup import __main__ as backup
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


class FakeDocker:
    """Stands in for docker compose: a dump on stdout, a folder for `cp`."""

    def __init__(self, *, missing: bool = False) -> None:
        self.missing = missing
        self.commands: list[list[str]] = []

    def run(self, argv: list[str], cwd: Path, **_: object) -> CommandResult:
        self.commands.append(argv)
        if self.missing:
            return CommandResult(argv, NOT_FOUND)
        if "cp" in argv:
            copied = Path(argv[-1])
            copied.mkdir()
            (copied / "shard").write_bytes(b"influx")
        return CommandResult(argv, 0)

    def run_piped(
        self, argv: list[str], cwd: Path, *, stdout: Path | None = None, **_: object
    ) -> CommandResult:
        if not self.missing and stdout is not None:
            stdout.write_bytes(b"-- dump\n")
        return self.run(argv, cwd)


class MainTest(unittest.TestCase):
    def _backup(self, root: Path, docker: FakeDocker | None = None) -> int:
        docker = docker or FakeDocker()
        with (
            mock.patch.object(backup, "repo_root", return_value=root),
            mock.patch.object(backup, "run", side_effect=docker.run),
            mock.patch.object(backup, "run_piped", side_effect=docker.run_piped),
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            return backup.main([])

    def test_without_docker_nothing_is_left_behind(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.assertEqual(self._backup(root, FakeDocker(missing=True)), NOT_FOUND)
            self.assertEqual(list((root / "backups").iterdir()), [])

    def test_a_complete_backup_has_both_halves_and_a_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.assertEqual(self._backup(root), 0)
            (folder,) = (root / "backups").iterdir()
            self.assertEqual((folder / "postgres.sql").read_bytes(), b"-- dump\n")
            self.assertTrue((folder / "influxdb" / "shard").is_file())
            written = json.loads((folder / "manifest.json").read_text("utf-8"))
            self.assertEqual(written["project"], "cogniboiler")

    @unittest.skipUnless(os.name == "posix", "file modes are POSIX permissions")
    def test_only_the_owner_may_read_the_backup(self) -> None:
        # The dump holds password hashes, sessions and the audit log; the InfluxDB
        # backup holds its API tokens.
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.assertEqual(self._backup(root), 0)
            (folder,) = (root / "backups").iterdir()
            self.assertEqual(stat.S_IMODE(folder.stat().st_mode), 0o700)


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
