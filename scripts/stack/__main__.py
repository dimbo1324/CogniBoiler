"""A cross-platform door to docker compose for this repository.

``up`` builds and starts a Compose profile detached and waits for it to become healthy:
``full`` by default, or ``infra`` (broker and databases, for running the Python services
from the host), ``core`` (infra, the services and the console) or ``observability``
(Prometheus, Grafana, InfluxDB). ``--infra-only`` is ``--profile infra``. ``status`` lists
containers with their health, ``logs`` follows one service, ``down`` stops everything.
Deleting volumes wipes the databases, so it asks first. Before ``up`` it creates the log
directory the services write their files into through a bind mount.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

from scripts._toolkit.config import load_config, repo_root
from scripts._toolkit.console import confirm, fail, info, warn
from scripts._toolkit.processes import NOT_FOUND, run

SCRIPT_DIR = Path(__file__).resolve().parent


def compose(config: dict[str, Any], profile: str, *args: str) -> list[str]:
    return [
        "docker",
        "compose",
        "--project-name",
        str(config["project_name"]),
        "--file",
        str(config["compose_file"]),
        "--profile",
        profile,
        *args,
    ]


def prepare_log_dir(directory: Path) -> bool:
    """Create the services' log directory; False when they may be unable to write it."""
    try:
        directory.mkdir(exist_ok=True)
        if os.name == "posix":
            # The services run as uid 10001, and a Linux bind mount keeps the host's owner
            # and mode; Docker Desktop on Windows and macOS maps the rights itself.
            directory.chmod(0o777)
    except OSError as error:
        warn(
            f"{directory} is not writable for the services ({error}); "
            "they will log to standard output only"
        )
        return False
    return True


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stack",
        description="Start, inspect or stop the Docker Compose stack.",
    )
    actions = parser.add_subparsers(dest="action", required=True)

    up = actions.add_parser("up", help="build and start the stack, wait until healthy")
    up.add_argument(
        "--profile",
        choices=["infra", "core", "observability", "full"],
        help="which part of the stack to start (default: full)",
    )
    up.add_argument(
        "--infra-only",
        action="store_true",
        help="the same as --profile infra: only the broker and the databases",
    )
    up.add_argument(
        "--no-build", action="store_true", help="reuse existing service images"
    )

    actions.add_parser("status", help="list containers and their health")

    logs = actions.add_parser("logs", help="follow the logs of one service")
    logs.add_argument("service")
    logs.add_argument("--tail", type=int, default=200)

    down = actions.add_parser("down", help="stop the stack")
    down.add_argument(
        "--volumes",
        action="store_true",
        help="also delete PostgreSQL, InfluxDB, Grafana and broker data",
    )
    down.add_argument(
        "--yes", action="store_true", help="do not ask before deleting volumes"
    )
    return parser


def build_command(args: argparse.Namespace, config: dict[str, Any]) -> list[str]:
    everything = str(config["default_profile"])
    if args.action == "up":
        profile = "infra" if args.infra_only else (args.profile or everything)
        command = compose(
            config,
            profile,
            "up",
            "--detach",
            "--wait",
            "--wait-timeout",
            str(config["wait_timeout_s"]),
        )
        if not args.no_build:
            command.append("--build")
        return command
    if args.action == "status":
        return compose(config, everything, "ps", "--all")
    if args.action == "logs":
        return compose(
            config,
            everything,
            "logs",
            "--follow",
            "--tail",
            str(args.tail),
            args.service,
        )
    command = compose(config, everything, "down")
    if args.volumes:
        command.append("--volumes")
    return command


def main(argv: list[str]) -> int:
    args = _parser().parse_args(argv)
    root = repo_root()
    config = load_config(SCRIPT_DIR, "stack.json")

    if args.action == "up" and not (root / str(config["env_file"])).exists():
        fail(
            f"{config['env_file']} is missing — run: "
            "python dev_tools_scripts_runner.py dev-secrets"
        )
        return 1

    if args.action == "up":
        prepare_log_dir(root / str(config["log_dir"]))

    if args.action == "down" and args.volumes:
        question = (
            "Delete the stack's volumes (PostgreSQL, InfluxDB, Grafana, broker data)?"
        )
        if not confirm(question, assume_yes=args.yes):
            info("nothing deleted")
            return 1

    result = run(build_command(args, config), root)
    if result.returncode == NOT_FOUND:
        fail("docker is not on PATH — install Docker Desktop and start it")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
