"""The docker compose command lines of the Docker-facing scripts, built in one place.

Every script names the project and the Compose file explicitly, so container and volume
names do not depend on the checkout's directory name. The values come from each script's
own config (``project_name``, ``compose_file``).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def compose_argv(
    config: Mapping[str, Any], *args: str, profile: str | None = None
) -> list[str]:
    """``docker compose --project-name … --file … [--profile …] <args>``."""
    argv = [
        "docker",
        "compose",
        "--project-name",
        str(config["project_name"]),
        "--file",
        str(config["compose_file"]),
    ]
    if profile is not None:
        argv.extend(["--profile", profile])
    return [*argv, *args]


def exec_sh(config: Mapping[str, Any], service: str, script: str) -> list[str]:
    """Run ``script`` with ``sh -c`` inside ``service``, without a terminal.

    The script is expanded by the container's shell, so ``"$POSTGRES_USER"`` reads the
    container's own environment and the value never appears on this host's command line.
    """
    return compose_argv(config, "exec", "-T", service, "sh", "-c", script)
