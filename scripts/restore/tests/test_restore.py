"""Which backup `restore` picks, and the command lines it builds.

Run with:  python -m unittest discover -s scripts -t .
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.restore.__main__ import (
    backup_folders,
    chosen_folder,
    influx_copy_command,
    influx_restore_command,
    postgres_restore_command,
    start_command,
    stop_command,
)

CONFIG = {
    "project_name": "cogniboiler",
    "compose_file": "docker-compose.yml",
    "backup_dir": "backups",
    "postgres_service": "postgresql",
    "influx_service": "influxdb",
    "dependent_services": ["api-gateway", "alert-manager", "historian", "opcua-server"],
    "wait_timeout_s": 180,
}


def make_backups(base: Path, names: list[str], *, complete: bool = True) -> None:
    for name in names:
        folder = base / name
        (folder / "influxdb").mkdir(parents=True)
        if complete:
            (folder / "postgres.sql").write_text("-- dump", encoding="utf-8")


class ChoiceTest(unittest.TestCase):
    def test_the_newest_complete_backup_is_restored_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            make_backups(base, ["20260919T100000Z", "20260920T070638Z"])
            make_backups(base, ["20260920T080000Z"], complete=False)
            self.assertEqual(
                [folder.name for folder in backup_folders(base)],
                ["20260919T100000Z", "20260920T070638Z"],
            )
            picked = chosen_folder(base, None)
            self.assertIsNotNone(picked)
            assert picked is not None
            self.assertEqual(picked.name, "20260920T070638Z")

    def test_a_named_backup_is_taken_by_name_or_by_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            make_backups(base, ["20260919T100000Z"])
            by_name = chosen_folder(base, "20260919T100000Z")
            by_path = chosen_folder(base, str(base / "20260919T100000Z"))
            self.assertEqual(by_name, base / "20260919T100000Z")
            self.assertEqual(by_path, base / "20260919T100000Z")

    def test_nothing_to_restore_is_reported_as_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            self.assertEqual(backup_folders(base), [])
            self.assertIsNone(chosen_folder(base, None))
            self.assertIsNone(chosen_folder(base, "does-not-exist"))
            self.assertEqual(backup_folders(base / "missing"), [])


class CommandTest(unittest.TestCase):
    def test_the_services_holding_connections_are_stopped_and_waited_for(self) -> None:
        stop = stop_command(CONFIG)
        start = start_command(CONFIG)
        self.assertEqual(stop[6], "stop")
        self.assertEqual(stop[7:], CONFIG["dependent_services"])
        self.assertEqual(start[6:10], ["up", "--detach", "--wait", "--wait-timeout"])
        self.assertEqual(start[10], "180")
        self.assertEqual(start[11:], CONFIG["dependent_services"])

    def test_postgres_is_restored_inside_its_container_and_stops_on_an_error(
        self,
    ) -> None:
        command = postgres_restore_command(CONFIG)
        self.assertEqual(command[6:10], ["exec", "-T", "postgresql", "sh"])
        script = command[-1]
        self.assertIn("psql", script)
        self.assertIn('"$POSTGRES_USER"', script)
        self.assertIn("ON_ERROR_STOP=1", script)
        self.assertNotIn("PGPASSWORD", " ".join(command))

    def test_influx_is_copied_in_and_restored_with_its_own_token(self) -> None:
        copy = influx_copy_command(CONFIG, Path("backups/x/influxdb"))
        self.assertEqual(copy[6], "cp")
        self.assertEqual(copy[8], "influxdb:/tmp/cogniboiler-restore")
        script = influx_restore_command(CONFIG)[-1]
        self.assertIn("influx restore /tmp/cogniboiler-restore --full", script)
        self.assertIn('"$DOCKER_INFLUXDB_INIT_ADMIN_TOKEN"', script)
        self.assertIn("rm -rf /tmp/cogniboiler-restore", script)


if __name__ == "__main__":
    unittest.main()
