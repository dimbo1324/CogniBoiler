"""Which backup `restore` picks, the command lines it builds, and when it stops.

Run with:  python -m unittest discover -s scripts -t .
"""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts._toolkit.processes import NOT_FOUND, CommandResult
from scripts.restore import __main__ as restore
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


def make_backups(
    base: Path,
    names: list[str],
    *,
    complete: bool = True,
    project: str | None = "cogniboiler",
) -> None:
    for name in names:
        folder = base / name
        (folder / "influxdb").mkdir(parents=True)
        if complete:
            (folder / "postgres.sql").write_text("-- dump", encoding="utf-8")
        if project is not None:
            (folder / "manifest.json").write_text(
                json.dumps({"project": project}), encoding="utf-8"
            )


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
            self.assertEqual(chosen_folder(base, None), base / "20260920T070638Z")

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

    def test_a_failing_statement_rolls_the_whole_dump_back(self) -> None:
        # The dump starts by dropping every table (--clean): without one transaction a
        # failure halfway leaves the tables dropped and half the data back.
        self.assertIn("--single-transaction", postgres_restore_command(CONFIG)[-1])

    def test_influx_is_copied_in_and_restored_with_its_own_token(self) -> None:
        copy = influx_copy_command(CONFIG, Path("backups/x/influxdb"))
        self.assertEqual(copy[6], "cp")
        self.assertEqual(copy[8], "influxdb:/tmp/cogniboiler-restore")
        script = influx_restore_command(CONFIG)[-1]
        self.assertIn("influx restore /tmp/cogniboiler-restore --full", script)
        self.assertIn('"$DOCKER_INFLUXDB_INIT_ADMIN_TOKEN"', script)
        self.assertIn("rm -rf /tmp/cogniboiler-restore", script)

    def test_the_influx_token_never_reaches_an_argv_in_the_container(self) -> None:
        script = influx_restore_command(CONFIG)[-1]
        self.assertNotIn("--token", script)
        self.assertNotIn(" -t ", script)
        self.assertIn(
            'INFLUX_TOKEN="$DOCKER_INFLUXDB_INIT_ADMIN_TOKEN" influx restore', script
        )


class FakeDocker:
    """Stands in for docker compose; ``failing`` names the step that fails."""

    def __init__(self, failing: str | None = None) -> None:
        self.failing = failing
        self.steps: list[str] = []

    def _step(self, argv: list[str]) -> str:
        script = argv[-1]
        if "psql" in script:
            return "postgres"
        if "influx restore" in script:
            return "influx"
        if "cp" in argv:
            return "copy"
        return "stop" if "stop" in argv else "start"

    def run(self, argv: list[str], cwd: Path, **_: object) -> CommandResult:
        step = self._step(argv)
        self.steps.append(step)
        if self.failing == "docker":
            return CommandResult(argv, NOT_FOUND)
        return CommandResult(argv, 1 if step == self.failing else 0)

    def run_piped(self, argv: list[str], cwd: Path, **_: object) -> CommandResult:
        return self.run(argv, cwd)


class MainTest(unittest.TestCase):
    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory()
        self.root = Path(self._temp.name)
        self.backups = self.root / "backups"

    def tearDown(self) -> None:
        self._temp.cleanup()

    def _restore(
        self, docker: FakeDocker, *argv: str, consent: bool = True
    ) -> tuple[int, str]:
        with (
            mock.patch.object(restore, "repo_root", return_value=self.root),
            mock.patch.object(restore, "confirm", return_value=consent) as asked,
            mock.patch.object(restore, "run", side_effect=docker.run),
            mock.patch.object(restore, "run_piped", side_effect=docker.run_piped),
            contextlib.redirect_stdout(io.StringIO()) as out,
            contextlib.redirect_stderr(io.StringIO()) as err,
        ):
            code = restore.main(list(argv))
        self.asked = asked
        return code, out.getvalue() + err.getvalue()

    def test_a_refused_confirmation_runs_nothing(self) -> None:
        make_backups(self.backups, ["20260920T070638Z"])
        docker = FakeDocker()
        code, _ = self._restore(docker, consent=False)
        self.assertEqual(code, 1)
        self.assertEqual(docker.steps, [])
        self.assertFalse(self.asked.call_args.kwargs["assume_yes"])

    def test_backups_outside_the_repository_can_be_listed(self) -> None:
        with tempfile.TemporaryDirectory() as away:
            make_backups(Path(away), ["20260920T070638Z"])
            docker = FakeDocker()
            code, printed = self._restore(docker, "--list", "--into", away)
        self.assertEqual(code, 0)
        self.assertIn("20260920T070638Z", printed)
        self.assertEqual(docker.steps, [])

    def test_a_complete_restore_stops_restores_both_and_starts(self) -> None:
        make_backups(self.backups, ["20260920T070638Z"])
        docker = FakeDocker()
        code, _ = self._restore(docker, "--yes")
        self.assertEqual(code, 0)
        self.assertEqual(docker.steps, ["stop", "postgres", "copy", "influx", "start"])
        self.assertTrue(self.asked.call_args.kwargs["assume_yes"])

    def test_a_refused_dump_leaves_influxdb_alone_and_restarts_the_services(
        self,
    ) -> None:
        make_backups(self.backups, ["20260920T070638Z"])
        docker = FakeDocker(failing="postgres")
        code, printed = self._restore(docker)
        self.assertEqual(code, 1)
        self.assertEqual(docker.steps, ["stop", "postgres", "start"])
        self.assertIn("rolled back", printed)

    def test_services_that_did_not_stop_leave_both_databases_alone(self) -> None:
        make_backups(self.backups, ["20260920T070638Z"])
        docker = FakeDocker(failing="stop")
        code, _ = self._restore(docker)
        self.assertEqual(code, 1)
        self.assertEqual(docker.steps, ["stop", "start"])

    def test_a_failed_influx_restore_is_reported_and_the_services_start(self) -> None:
        make_backups(self.backups, ["20260920T070638Z"])
        docker = FakeDocker(failing="influx")
        code, printed = self._restore(docker)
        self.assertEqual(code, 1)
        self.assertEqual(docker.steps, ["stop", "postgres", "copy", "influx", "start"])
        self.assertIn("PostgreSQL is already restored", printed)

    def test_without_docker_the_run_stops_at_once(self) -> None:
        make_backups(self.backups, ["20260920T070638Z"])
        docker = FakeDocker(failing="docker")
        code, _ = self._restore(docker)
        self.assertEqual(code, NOT_FOUND)
        self.assertEqual(docker.steps, ["stop"])

    def test_a_folder_without_a_manifest_is_refused_before_asking(self) -> None:
        make_backups(self.backups, ["20260920T070638Z"], project=None)
        docker = FakeDocker()
        code, printed = self._restore(docker)
        self.assertEqual(code, 1)
        self.assertEqual(docker.steps, [])
        self.asked.assert_not_called()
        self.assertIn("--force", printed)

    def test_a_backup_of_another_project_is_refused(self) -> None:
        make_backups(self.backups, ["20260920T070638Z"], project="elsewhere")
        docker = FakeDocker()
        code, printed = self._restore(docker)
        self.assertEqual(code, 1)
        self.assertEqual(docker.steps, [])
        self.assertIn("elsewhere", printed)

    def test_force_restores_a_folder_the_manifest_check_refused(self) -> None:
        make_backups(self.backups, ["20260920T070638Z"], project=None)
        docker = FakeDocker()
        code, _ = self._restore(docker, "--force")
        self.assertEqual(code, 0)
        self.assertEqual(docker.steps, ["stop", "postgres", "copy", "influx", "start"])


if __name__ == "__main__":
    unittest.main()
