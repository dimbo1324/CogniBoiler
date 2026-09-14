"""Schemas for history, alarms, and audit endpoints."""

from __future__ import annotations

from pydantic import BaseModel, Field


class HistoryPointResponse(BaseModel):
    """One historical telemetry row returned from InfluxDB."""

    measurement: str = Field(..., description="Influx measurement name.")
    timestamp_ms: int = Field(..., description="UTC epoch milliseconds.")
    values: dict[str, float | int | str | None] = Field(
        default_factory=dict,
        description="Field/value map for the historical row.",
    )


class HistoryResponse(BaseModel):
    """Historical telemetry response payload."""

    measurement: str
    points: list[HistoryPointResponse]


class AlarmResponse(BaseModel):
    """Alarm event returned by the API gateway."""

    alarm_id: str
    source_service: str
    severity: str
    parameter: str
    value: float
    threshold: float
    action: str
    message: str
    topic: str
    occurred_at_ms: int
    acknowledged: bool
    acknowledged_at_ms: int | None
    cleared: bool


class AuditResponse(BaseModel):
    """Audit log entry returned by the API gateway."""

    id: int
    user_id: int | None
    ip_address: str
    method: str
    endpoint: str
    request_body_hash: str | None
    response_status: int
    duration_ms: int
    timestamp_ms: int
    detail: str | None
