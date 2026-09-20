"""Historical telemetry endpoint backed by InfluxDB."""

from __future__ import annotations

import asyncio
import re
import time

from cogniboiler_runtime import MILLISECONDS_PER_DAY
from fastapi import APIRouter, Query, Request
from influxdb_client.rest import ApiException
from urllib3.exceptions import HTTPError as Urllib3HTTPError

from api_gateway.auth.rbac import ViewerUser
from api_gateway.clients import HistorianQueryClient, history_window_s
from api_gateway.problems import ProblemError, upstream_unavailable
from api_gateway.schemas.ops import HistoryPointResponse, HistoryResponse

router = APIRouter(prefix="/api/v1", tags=["history"])

MAX_HISTORY_SPAN_MS = 90 * MILLISECONDS_PER_DAY
DEFAULT_HISTORY_SPAN_MS = 15 * 60_000
FIELD_NAME = re.compile("^[a-z][a-z0-9_]{0,63}$")
HISTORY_ERRORS = (ApiException, OSError, Urllib3HTTPError)

_FLUX_METADATA = frozenset(
    {"result", "table", "_start", "_stop", "_time", "_measurement", "timestamp_ms"}
)


def _historian_client(request: Request) -> HistorianQueryClient:
    """Resolve the shared historian query client from app state."""
    return request.app.state.historian_client  # type: ignore[no-any-return]


def resolve_range(start_ms: int | None, end_ms: int | None) -> tuple[int, int]:
    """A bounded time range: the last 15 minutes by default, at most 90 days."""
    current_ms = int(time.time() * 1000)
    end = end_ms if end_ms is not None else current_ms
    start = start_ms if start_ms is not None else end - DEFAULT_HISTORY_SPAN_MS
    if start >= end:
        raise ProblemError(
            422, "request.invalid_range", "start_ms must be before end_ms."
        )
    if end - start > MAX_HISTORY_SPAN_MS:
        raise ProblemError(
            422, "request.range_too_long", "A history range may span at most 90 days."
        )
    return start, end


def parse_fields(fields: str | None) -> tuple[str, ...]:
    if not fields:
        return ()
    names = tuple(name.strip() for name in fields.split(",") if name.strip())
    if len(names) > 32 or not all(FIELD_NAME.fullmatch(name) for name in names):
        raise ProblemError(
            422,
            "request.invalid_fields",
            "fields is a comma-separated list of up to 32 field names.",
        )
    return names


@router.get("/history", response_model=HistoryResponse)
async def get_history(
    request: Request,
    _: ViewerUser,
    measurement: str = Query(
        default="boiler_sensors",
        pattern="^(boiler_sensors|turbine_sensors|plant_status)$",
    ),
    start_ms: int | None = Query(default=None, ge=0, description="[UTC epoch ms]"),
    end_ms: int | None = Query(default=None, ge=0, description="[UTC epoch ms]"),
    limit: int = Query(
        default=200, ge=1, le=2000, description="Maximum number of points."
    ),
    fields: str | None = Query(
        default=None, description="Comma-separated field names; all when omitted."
    ),
) -> HistoryResponse:
    """
    Telemetry over a bounded range at a resolution chosen from its length.

    The range is split into standard windows (1 s … 1 day) so that it fits in `limit`
    points; each point is the mean of its window.
    """
    start, end = resolve_range(start_ms, end_ms)
    names = parse_fields(fields)
    window_s = history_window_s(start, end, limit)
    try:
        raw_points = await asyncio.to_thread(
            _historian_client(request).fetch_history,
            measurement=measurement,
            start_ms=start,
            end_ms=end,
            limit=limit,
            window_s=window_s,
            fields=names,
        )
    except HISTORY_ERRORS as exc:
        raise upstream_unavailable("Historian", exc) from exc

    points = [
        HistoryPointResponse(
            measurement=measurement,
            timestamp_ms=int(item["timestamp_ms"]),
            values={
                key: value for key, value in item.items() if key not in _FLUX_METADATA
            },
        )
        for item in raw_points
    ]
    return HistoryResponse(
        measurement=measurement,
        start_ms=start,
        end_ms=end,
        window_s=window_s,
        points=points,
    )
