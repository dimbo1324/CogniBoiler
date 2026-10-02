"""Back up the stack's databases into a timestamped folder.

PostgreSQL (users, roles, sessions, the append-only audit log, scenario runs and the alarm
lifecycle) is dumped with `pg_dump`; InfluxDB (telemetry, KPIs, labels and events) with
`influx backup`. Both run inside their container, which already holds the credentials, so
no password or token is ever passed on a command line.

The folder itself is as sensitive as `.env`: the dump holds password hashes and session
records, and the InfluxDB backup holds its API tokens. On POSIX only its owner may enter
it.

The stack must be up: these are the live databases, read through Compose. `restore` puts a
folder written here back.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from scripts._toolkit.compose import compose_argv, exec_sh
from scripts._toolkit.config import load_config, repo_root
from scripts._toolkit.console import display_path, fail, heading, info, ok, summary
from scripts._toolkit.files import make_private_dir
from scripts._toolkit.processes import NOT_FOUND, run, run_piped

SCRIPT_DIR = Path(__file__).resolve().parent

CONTAINER_TEMP = "/tmp/cogniboiler-backup"  # noqa: S108 — inside the container, not here


def folder_name(moment: datetime) -> str:
    """One folder per backup, named by the UTC instant it started."""
    return moment.strftime("%Y%m%dT%H%M%SZ")


def postgres_dump_command(config: dict[str, Any]) -> list[str]:
    """pg_dump inside the container: the password never leaves it."""
    return exec_sh(
        config,
        str(config["postgres_service"]),
        'pg_dump --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" '
        "--clean --if-exists --no-owner",
    )


def influx_backup_command(config: dict[str, Any]) -> list[str]:
    """influx backup inside the container, with the admin token from its own environment.

    The token reaches the CLI as ``INFLUX_TOKEN`` in its environment: a ``--token`` flag
    would be expanded by the container's shell into the influx process's argv.
    """
    return exec_sh(
        config,
        str(config["influx_service"]),
        f"rm -rf {CONTAINER_TEMP} && "
        'INFLUX_TOKEN="$DOCKER_INFLUXDB_INIT_ADMIN_TOKEN" '
        f"influx backup {CONTAINER_TEMP} >/dev/null",
    )


def influx_copy_command(config: dict[str, Any], destination: Path) -> list[str]:
    return compose_argv(
        config,
        "cp",
        f"{config['influx_service']}:{CONTAINER_TEMP}",
        str(destination),
    )


def influx_cleanup_command(config: dict[str, Any]) -> list[str]:
    return compose_argv(
        config, "exec", "-T", str(config["influx_service"]), "rm", "-rf", CONTAINER_TEMP
    )


def manifest(moment: datetime, config: dict[str, Any], files: dict[str, int]) -> str:
    return (
        json.dumps(
            {
                "taken_at": moment.isoformat(timespec="seconds"),
                "project": config["project_name"],
                "postgres_service": config["postgres_service"],
                "influx_service": config["influx_service"],
                "files": files,
            },
            indent=2,
        )
        + "\n"
    )


def directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="backup",
        description="Back up PostgreSQL and InfluxDB of the running stack.",
    )
    parser.add_argument(
        "--into", help="where the backup folders live (default: backups)"
    )
    args = parser.parse_args(argv)

    root = repo_root()
    config = load_config(SCRIPT_DIR, "backup.json")
    moment = datetime.now(UTC)
    base = root / str(args.into or config["backup_dir"])
    target = base / folder_name(moment)

    heading(f"backup — {display_path(target, root)}")
    try:
        make_private_dir(target)
    except OSError as error:
        fail(f"cannot create {target}: {error}")
        return 1

    rows: list[tuple[str, str]] = []
    dump = target / "postgres.sql"
    dumped = run_piped(postgres_dump_command(config), root, stdout=dump)
    if dumped.returncode == NOT_FOUND:
        shutil.rmtree(target, ignore_errors=True)
        fail("docker is not on PATH — install Docker Desktop and start it")
        return NOT_FOUND
    size = dump.stat().st_size if dump.exists() else 0
    postgres_ok = dumped.ok and size > 0
    rows.append(
        ("PostgreSQL dump", f"{size / 1024:.0f} KiB" if postgres_ok else "FAILED")
    )
    if postgres_ok:
        ok(f"PostgreSQL dumped into {dump.name}")
    else:
        fail("pg_dump failed — is the stack up?")

    influx_dir = target / "influxdb"
    influx_ok = (
        run(influx_backup_command(config), root).ok
        and run(influx_copy_command(config, influx_dir), root).ok
    )
    run(influx_cleanup_command(config), root)
    influx_size = directory_size(influx_dir) if influx_dir.is_dir() else 0
    influx_ok = influx_ok and influx_size > 0
    rows.append(
        ("InfluxDB backup", f"{influx_size / 1024:.0f} KiB" if influx_ok else "FAILED")
    )
    if influx_ok:
        ok(f"InfluxDB backed up into {influx_dir.name}/")
    else:
        fail("influx backup failed — is the stack up?")

    if not (postgres_ok and influx_ok):
        shutil.rmtree(target, ignore_errors=True)
        info("the incomplete folder was removed")
        summary("backup", rows)
        return 1

    (target / "manifest.json").write_text(
        manifest(moment, config, {"postgres.sql": size, "influxdb": influx_size}),
        encoding="utf-8",
    )
    info(
        f"restore it with: python dev_tools_scripts_runner.py restore --from {target.name}"
    )
    summary("backup", rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
