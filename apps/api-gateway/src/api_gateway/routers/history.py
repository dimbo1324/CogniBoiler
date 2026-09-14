"""Historical telemetry endpoint backed by InfluxDB."""

from __future__ import annotations

import time
from typing import Annotated

from fastapi import APIRouter, Depends, Query, Request

from api_gateway.auth.jwt_handler import TokenData
from api_gateway.auth.rbac import require_role
from api_gateway.clients import HistorianQueryClient
from api_gateway.schemas.ops import HistoryPointResponse, HistoryResponse

router = APIRouter(prefix="/api/v1", tags=["history"])


def _historian_client(request: Request) -> HistorianQueryClient:
    """Resolve the shared historian query client from app state."""
    return request.app.state.historian_client  # type: ignore[no-any-return]


@router.get("/history", response_model=HistoryResponse)
async def get_history(
    request: Request,
    _: Annotated[TokenData, Depends(require_role("viewer"))],
    measurement: str = Query(
        default="boiler_sensors", pattern="^(boiler_sensors|turbine_sensors)$"
    ),
    start_ms: int | None = Query(default=None, ge=0),
    end_ms: int | None = Query(default=None, ge=0),
    limit: int = Query(default=200, ge=1, le=2000),
) -> HistoryResponse:
    """Fetch recent boiler or turbine history from InfluxDB."""
    current_ms = int(time.time() * 1000)
    start_ms = start_ms if start_ms is not None else current_ms - 15 * 60 * 1000
    end_ms = end_ms if end_ms is not None else current_ms

    raw_points = _historian_client(request).fetch_history(
        measurement=measurement,
        start_ms=start_ms,
        end_ms=end_ms,
        limit=limit,
    )
    points: list[HistoryPointResponse] = []
    for item in raw_points:
        values = {
            key: value
            for key, value in item.items()
            if key
            not in {
                "result",
                "table",
                "_start",
                "_stop",
                "_measurement",
                "timestamp_ms",
            }
        }
        points.append(
            HistoryPointResponse(
                measurement=measurement,
                timestamp_ms=int(item["timestamp_ms"]),
                values=values,
            )
        )
    return HistoryResponse(measurement=measurement, points=points)
