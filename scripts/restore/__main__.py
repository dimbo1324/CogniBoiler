"""Put a folder written by `backup` back into the running stack's databases.

This overwrites live data, so it asks first, and only for a folder whose manifest says it
is a backup of this stack (``--force`` overrides that check). The services that hold
connections are stopped while the databases are replaced and started again afterwards:
PostgreSQL refuses to drop tables another session is using, and InfluxDB should not be
written to mid-restore.

Each step runs only when the one before it succeeded. The dump is replayed in a single
transaction, so a failing statement rolls PostgreSQL back to what it held before; the
services are started again whatever happened.

Both commands run inside their container, which already holds the credentials, so no
password or token is ever passed on a command line.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from scripts._toolkit.compose import compose_argv, exec_sh
from scripts._toolkit.config import load_config, repo_root
from scripts._toolkit.console import (
    confirm,
    display_path,
    fail,
    heading,
    info,
    ok,
    summary,
    warn,
)
from scripts._toolkit.processes import NOT_FOUND, run, run_piped

SCRIPT_DIR = Path(__file__).resolve().parent

CONTAINER_TEMP = "/tmp/cogniboiler-restore"  # noqa: S108 — inside the container, not here


def backup_folders(base: Path) -> list[Path]:
    """Every folder that holds both halves of a backup, newest last."""
    if not base.is_dir():
        return []
    return sorted(
        item
        for item in base.iterdir()
        if item.is_dir() and (item / "postgres.sql").is_file()
    )


def chosen_folder(base: Path, name: str | None) -> Path | None:
    if name:
        candidate = Path(name) if Path(name).is_absolute() else base / name
        return candidate if (candidate / "postgres.sql").is_file() else None
    folders = backup_folders(base)
    return folders[-1] if folders else None


def manifest_problem(folder: Path, project: str) -> str | None:
    """Why ``folder`` does not look like a backup of ``project``, or None when it does."""
    path = folder / "manifest.json"
    if not path.is_file():
        return "it has no manifest.json"
    try:
        written = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        return f"its manifest.json cannot be read ({error})"
    named = written.get("project") if isinstance(written, dict) else None
    if named != project:
        return f"its manifest names project {named!r}, not {project!r}"
    return None


def stop_command(config: dict[str, Any]) -> list[str]:
    return compose_argv(config, "stop", *config["dependent_services"])


def start_command(config: dict[str, Any]) -> list[str]:
    return compose_argv(
        config,
        "up",
        "--detach",
        "--wait",
        "--wait-timeout",
        str(config["wait_timeout_s"]),
        *config["dependent_services"],
    )


def postgres_restore_command(config: dict[str, Any]) -> list[str]:
    """psql inside the container, reading the dump from this process's stdin.

    The dump drops every table first (``pg_dump --clean``). One transaction makes a
    failure anywhere roll all of it back instead of leaving the tables half restored.
    """
    return exec_sh(
        config,
        str(config["postgres_service"]),
        'psql --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" '
        "--quiet --set ON_ERROR_STOP=1 --single-transaction",
    )


def influx_copy_command(config: dict[str, Any], source: Path) -> list[str]:
    return compose_argv(
        config,
        "cp",
        str(source),
        f"{config['influx_service']}:{CONTAINER_TEMP}",
    )


def influx_restore_command(config: dict[str, Any]) -> list[str]:
    """influx restore inside the container; the token travels as ``INFLUX_TOKEN``."""
    return exec_sh(
        config,
        str(config["influx_service"]),
        'INFLUX_TOKEN="$DOCKER_INFLUXDB_INIT_ADMIN_TOKEN" '
        f"influx restore {CONTAINER_TEMP} --full >/dev/null && "
        f"rm -rf {CONTAINER_TEMP}",
    )


def replace_data(
    config: dict[str, Any], root: Path, folder: Path, rows: list[tuple[str, str]]
) -> bool:
    """PostgreSQL, then InfluxDB; stops at the first failure and says what it left."""
    if not run_piped(
        postgres_restore_command(config), root, stdin=folder / "postgres.sql"
    ).ok:
        rows.append(("PostgreSQL restored", "FAILED, rolled back"))
        rows.append(("InfluxDB restored", "not run (PostgreSQL failed)"))
        fail(
            "psql refused the dump; the transaction was rolled back, so PostgreSQL "
            "holds what it held before, and InfluxDB was not touched"
        )
        return False
    rows.append(("PostgreSQL restored", "ok"))
    ok("PostgreSQL restored")

    influx_source = folder / "influxdb"
    influx_ok = (
        influx_source.is_dir()
        and run(influx_copy_command(config, influx_source), root).ok
        and run(influx_restore_command(config), root).ok
    )
    rows.append(("InfluxDB restored", "ok" if influx_ok else "FAILED"))
    if not influx_ok:
        fail(
            "influx restore failed; PostgreSQL is already restored from this backup, "
            "so the two databases now disagree — run restore again once it is fixed"
        )
        return False
    ok("InfluxDB restored")
    return True


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="restore",
        description="Restore PostgreSQL and InfluxDB from a folder written by backup.",
    )
    parser.add_argument(
        "--from",
        dest="source",
        help="the backup folder (default: the newest one under backups/)",
    )
    parser.add_argument(
        "--into", help="where the backup folders live (default: backups)"
    )
    parser.add_argument(
        "--yes", action="store_true", help="do not ask before overwriting the databases"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="restore a folder whose manifest is missing or names another project",
    )
    parser.add_argument(
        "--list", action="store_true", help="only list the backups that can be restored"
    )
    args = parser.parse_args(argv)

    root = repo_root()
    config = load_config(SCRIPT_DIR, "restore.json")
    base = root / str(args.into or config["backup_dir"])

    if args.list:
        heading(f"restore — backups in {display_path(base, root)}")
        folders = backup_folders(base)
        for listed in folders:
            info(listed.name)
        if not folders:
            info("none")
        return 0

    folder = chosen_folder(base, args.source)
    if folder is None:
        fail(
            f"no backup to restore in {base}: run backup first, or name one with --from"
        )
        return 1

    heading(f"restore — {folder.name}")
    problem = manifest_problem(folder, str(config["project_name"]))
    if problem and not args.force:
        fail(
            f"{folder.name} does not look like a backup of this stack: {problem}; "
            "pass --force to restore it anyway"
        )
        return 1
    if problem:
        warn(f"restoring despite the manifest check: {problem}")

    if not confirm(
        "Replace the stack's PostgreSQL and InfluxDB data with this backup?",
        assume_yes=args.yes,
    ):
        info("nothing was restored")
        return 1

    rows: list[tuple[str, str]] = []
    stopping = run(stop_command(config), root)
    if stopping.returncode == NOT_FOUND:
        fail("docker is not on PATH — install Docker Desktop and start it")
        return NOT_FOUND
    restored = False
    if stopping.ok:
        rows.append(("services stopped", "ok"))
        restored = replace_data(config, root, folder, rows)
    else:
        rows.append(("services stopped", "FAILED"))
        rows.append(("databases restored", "not run (services still running)"))
        fail("the services did not stop; neither database was touched")

    started = run(start_command(config), root).ok
    rows.append(("services started", "ok" if started else "FAILED"))
    info("check the stack with: python dev_tools_scripts_runner.py smoke")
    summary("restore", rows)
    return 0 if (restored and started) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
