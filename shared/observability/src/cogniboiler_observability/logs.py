"""
One log format for every service: a JSON object per line on standard output.

Each line carries `timestamp` (ISO 8601, UTC), `level`, `service`, `logger`, `event` and
`correlation_id` (null when no request is in progress), plus `exception` when there is
one. Existing standard-library calls such as `logger.info("... %s", value)` are rendered
the same way, so no call site changes. `LOG_FORMAT=console` gives one readable line
instead, for a person running a service by hand; `LOG_LEVEL` sets the threshold.
"""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import MutableMapping
from typing import Any

import structlog

from cogniboiler_observability.correlation import current_correlation_id

# Libraries that log every connection or request at INFO; their warnings still show.
_QUIET_LOGGERS = ("asyncua", "httpx", "httpcore", "aiosqlite")


def _stamp(service: str) -> Any:
    def add_context(
        _: Any, __: str, event: MutableMapping[str, Any]
    ) -> MutableMapping[str, Any]:
        event.setdefault("service", service)
        event.setdefault("correlation_id", current_correlation_id())
        return event

    return add_context


def configure_logging(
    service: str, *, level: str | None = None, fmt: str | None = None
) -> None:
    """Route every standard-library logger through one structured formatter."""
    level_name = (level or os.environ.get("LOG_LEVEL") or "INFO").upper()
    fmt_name = (fmt or os.environ.get("LOG_FORMAT") or "json").lower()

    shared: list[Any] = [
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True, key="timestamp"),
        _stamp(service),
    ]
    renderer: Any = (
        structlog.dev.ConsoleRenderer(colors=False)
        if fmt_name == "console"
        else structlog.processors.JSONRenderer(ensure_ascii=False)
    )
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.format_exc_info,
            renderer,
        ],
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers[:] = [handler]
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
