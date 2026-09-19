"""
One log format for every service: a JSON object per line on standard output.

Each line carries `timestamp` (ISO 8601, UTC), `level`, `service`, `logger`, `event` and
`correlation_id` (null when no request is in progress), plus `exception` when there is
one. Existing standard-library calls such as `logger.info("... %s", value)` are rendered
the same way, so no call site changes. `LOG_FORMAT=console` gives one readable line
instead, for a person running a service by hand; `LOG_LEVEL` sets the threshold.

With `LOG_DIR` set, the same lines also go to `<LOG_DIR>/<service>.log`, always as JSON,
rotated by size (`LOG_FILE_MAX_BYTES`, default 10 MiB; `LOG_FILE_BACKUPS` older files,
default 5). A directory that cannot be written leaves the service logging to standard
output only, with a warning: losing the files must never stop the plant.
"""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import MutableMapping
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

import structlog

from cogniboiler_observability.correlation import current_correlation_id

# Libraries that log every connection or request at INFO; their warnings still show.
_QUIET_LOGGERS = ("asyncua", "httpx", "httpcore", "aiosqlite")

DEFAULT_FILE_MAX_BYTES = 10 * 1024 * 1024
DEFAULT_FILE_BACKUPS = 5


class ServiceLogFile(RotatingFileHandler):
    """The handler configure_logging owns, so a second call closes only its own files."""


def _stamp(service: str) -> Any:
    def add_context(
        _: Any, __: str, event: MutableMapping[str, Any]
    ) -> MutableMapping[str, Any]:
        event.setdefault("service", service)
        event.setdefault("correlation_id", current_correlation_id())
        return event

    return add_context


def _formatter(shared: list[Any], renderer: Any) -> logging.Formatter:
    return structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.format_exc_info,
            renderer,
        ],
    )


def _non_negative(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{name} must be a whole number, got {raw!r}") from None
    if value < 0:
        raise ValueError(f"{name} must not be negative, got {value}")
    return value


def _log_file(service: str, directory: Path) -> ServiceLogFile:
    directory.mkdir(parents=True, exist_ok=True)
    return ServiceLogFile(
        directory / f"{service}.log",
        maxBytes=_non_negative("LOG_FILE_MAX_BYTES", DEFAULT_FILE_MAX_BYTES),
        backupCount=_non_negative("LOG_FILE_BACKUPS", DEFAULT_FILE_BACKUPS),
        encoding="utf-8",
    )


def configure_logging(
    service: str,
    *,
    level: str | None = None,
    fmt: str | None = None,
    log_dir: str | Path | None = None,
) -> None:
    """Route every standard-library logger through one structured formatter."""
    level_name = (level or os.environ.get("LOG_LEVEL") or "INFO").upper()
    fmt_name = (fmt or os.environ.get("LOG_FORMAT") or "json").lower()
    directory = str(log_dir if log_dir is not None else os.environ.get("LOG_DIR", ""))

    shared: list[Any] = [
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True, key="timestamp"),
        _stamp(service),
    ]
    json_renderer = structlog.processors.JSONRenderer(ensure_ascii=False)
    renderer: Any = (
        structlog.dev.ConsoleRenderer(colors=False)
        if fmt_name == "console"
        else json_renderer
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_formatter(shared, renderer))
    handlers: list[logging.Handler] = [handler]

    file_problem: OSError | None = None
    if directory.strip():
        try:
            log_file = _log_file(service, Path(directory))
        except OSError as error:
            file_problem = error
        else:
            log_file.setFormatter(_formatter(shared, json_renderer))
            handlers.append(log_file)

    root = logging.getLogger()
    for previous in root.handlers:
        if isinstance(previous, ServiceLogFile):
            previous.close()
    root.handlers[:] = handlers
    root.setLevel(level_name)
    # Uvicorn installs its own handlers; its records go through the root handler instead.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uvicorn_logger = logging.getLogger(name)
        uvicorn_logger.handlers.clear()
        uvicorn_logger.propagate = True
    for name in _QUIET_LOGGERS:
        logging.getLogger(name).setLevel(max(logging.WARNING, root.level))

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            *shared,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    if file_problem is not None:
        logging.getLogger(__name__).warning(
            "Log files are off: cannot write to %s (%s); logging to standard output only",
            directory,
            file_problem,
        )
