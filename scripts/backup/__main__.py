"""Back up the stack's databases into a timestamped folder.

PostgreSQL (users, roles, sessions, the append-only audit log, scenario runs and the alarm
lifecycle) is dumped with `pg_dump`; InfluxDB (telemetry, KPIs, labels and events) with
`influx backup`. Both run inside their container, which already holds the credentials, so
no password or token is ever passed on a command line or written into the backup folder.

The stack must be up: these are the live databases, read through Compose. `restore` puts a
folder written here back.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from scripts._toolkit.config import load_config, repo_root
from scripts._toolkit.console import fail, heading, info, ok, summary
from scripts._toolkit.processes import NOT_FOUND

SCRIPT_DIR = Path(__file__).resolve().parent

CONTAINER_TEMP = "/tmp/cogniboiler-backup"  # noqa: S108 — inside the container, not here


def compose(config: dict[str, Any], *args: str) -> list[str]:
    return [
        "docker",
        "compose",
        "--project-name",
        str(config["project_name"]),
        "--file",
        str(config["compose_file"]),
        *args,
    ]


def folder_name(moment: datetime) -> str:
    """One folder per backup, named by the UTC instant it started."""
    return moment.strftime("%Y%m%dT%H%M%SZ")


def postgres_dump_command(config: dict[str, Any]) -> list[str]:
    """pg_dump inside the container: the password never leaves it."""
    return compose(
        config,
        "exec",
        "-T",
        str(config["postgres_service"]),
        "sh",
        "-c",
        'pg_dump --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" '
        "--clean --if-exists --no-owner",
    )


def influx_backup_command(config: dict[str, Any]) -> list[str]:
    """influx backup inside the container, with the admin token from its own environment."""
    return compose(
        config,
        "exec",
        "-T",
        str(config["influx_service"]),
        "sh",
        "-c",
        f"rm -rf {CONTAINER_TEMP} && influx backup {CONTAINER_TEMP} "
        '--token "$DOCKER_INFLUXDB_INIT_ADMIN_TOKEN" >/dev/null',
    )


def influx_copy_command(config: dict[str, Any], destination: Path) -> list[str]:
    return compose(
        config,
        "cp",
        f"{config['influx_service']}:{CONTAINER_TEMP}",
        str(destination),
    )


def influx_cleanup_command(config: dict[str, Any]) -> list[str]:
    return compose(
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


def run(command: list[str], root: Path, *, capture_to: Path | None = None) -> bool:
    if capture_to is None:
        return subprocess.run(command, cwd=root).returncode == 0
    with capture_to.open("wb") as sink:
        return subprocess.run(command, cwd=root, stdout=sink).returncode == 0


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

    heading(f"backup — {target.relative_to(root).as_posix()}")
    try:
        target.mkdir(parents=True, exist_ok=False)
    except OSError as error:
        fail(f"cannot create {target}: {error}")
        return 1

    rows: list[tuple[str, str]] = []
    dump = target / "postgres.sql"
    try:
        postgres_ok = run(postgres_dump_command(config), root, capture_to=dump)
    except FileNotFoundError:
        fail("docker is not on PATH — install Docker Desktop and start it")
        return NOT_FOUND
    size = dump.stat().st_size if dump.exists() else 0
    postgres_ok = postgres_ok and size > 0
    rows.append(
        ("PostgreSQL dump", f"{size / 1024:.0f} KiB" if postgres_ok else "FAILED")
    )
    if postgres_ok:
        ok(f"PostgreSQL dumped into {dump.name}")
    else:
        fail("pg_dump failed — is the stack up?")

    influx_dir = target / "influxdb"
    influx_ok = run(influx_backup_command(config), root) and run(
        influx_copy_command(config, influx_dir), root
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
