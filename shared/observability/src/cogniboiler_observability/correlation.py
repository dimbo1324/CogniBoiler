"""
The correlation id of the work in progress.

One id follows a request from the gateway's HTTP edge through gRPC metadata into the PLC,
physics and alarm services, so their log lines about one operator action can be found
together. It lives in a context variable: every asyncio task started while it is set
inherits it, and it never leaks into unrelated tasks.
"""

from __future__ import annotations

import re
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

# The HTTP header and the gRPC metadata key (gRPC keys are lowercase).
CORRELATION_HEADER = "X-Correlation-ID"
CORRELATION_METADATA_KEY = "x-correlation-id"

# Ids from outside are accepted only in a safe shape, so a caller cannot inject log
# syntax or unbounded text through a header.
_ACCEPTED = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")

_current: ContextVar[str | None] = ContextVar("correlation_id", default=None)


def new_correlation_id() -> str:
    return uuid.uuid4().hex


def accepted_correlation_id(value: str | None) -> str | None:
    """The given id if it is safe to adopt, otherwise None."""
    if value is None:
        return None
    value = value.strip()
    return value if _ACCEPTED.fullmatch(value) else None


def current_correlation_id() -> str | None:
    return _current.get()


@contextmanager
def correlation_scope(correlation_id: str | None) -> Iterator[str]:
    """Run a block under a correlation id: the given one if acceptable, or a new one."""
    value = accepted_correlation_id(correlation_id) or new_correlation_id()
    token = _current.set(value)
    try:
        yield value
    finally:
        _current.reset(token)
