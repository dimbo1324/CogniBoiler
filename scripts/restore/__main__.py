"""Put a folder written by `backup` back into the running stack's databases.

This overwrites live data, so it asks first. The services that hold connections are stopped
while the databases are replaced and started again afterwards: PostgreSQL refuses to drop
tables another session is using, and InfluxDB should not be written to mid-restore.

Both commands run inside their container, which already holds the credentials, so no
password or token is ever passed on a command line.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from scripts._toolkit.compose import compose_argv, exec_sh
from scripts._toolkit.config import load_config, repo_root
from scripts._toolkit.console import confirm, fail, heading, info, ok, summary
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
    """psql inside the container, reading the dump from this process's stdin."""
    return exec_sh(
        config,
        str(config["postgres_service"]),
        'psql --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" '
        "--quiet --set ON_ERROR_STOP=1",
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
        "--list", action="store_true", help="only list the backups that can be restored"
    )
    args = parser.parse_args(argv)

    root = repo_root()
    config = load_config(SCRIPT_DIR, "restore.json")
    base = root / str(args.into or config["backup_dir"])

    if args.list:
        heading(f"restore — backups in {base.relative_to(root).as_posix()}")
        folders = backup_folders(base)
        for folder in folders:
            info(folder.name)
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
    stopped = stopping.ok
    rows.append(("services stopped", "ok" if stopped else "FAILED"))

    postgres_ok = run_piped(
        postgres_restore_command(config), root, stdin=folder / "postgres.sql"
    ).ok
    rows.append(("PostgreSQL restored", "ok" if postgres_ok else "FAILED"))
    if postgres_ok:
        ok("PostgreSQL restored")
    else:
        fail("psql refused the dump")

    influx_source = folder / "influxdb"
    influx_ok = (
        influx_source.is_dir()
        and run(influx_copy_command(config, influx_source), root).ok
    )
    influx_ok = influx_ok and run(influx_restore_command(config), root).ok
    rows.append(("InfluxDB restored", "ok" if influx_ok else "FAILED"))
    if influx_ok:
        ok("InfluxDB restored")
    else:
        fail("influx restore failed")

    started = run(start_command(config), root).ok
    rows.append(("services started", "ok" if started else "FAILED"))
    info("check the stack with: python dev_tools_scripts_runner.py smoke")
    summary("restore", rows)
    return 0 if (stopped and postgres_ok and influx_ok and started) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
