"""Schemas for history, alarms, and audit endpoints."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

AlarmStateName = Literal["ACTIVE_UNACK", "ACTIVE_ACK", "CLEARED_UNACK", "CLEARED"]


class HistoryPointResponse(BaseModel):
    """One historical telemetry row returned from InfluxDB."""

    measurement: str = Field(..., description="Influx measurement name.")
    timestamp_ms: int = Field(..., description="UTC epoch milliseconds.")
    values: dict[str, float | int | str | None] = Field(
        default_factory=dict,
        description="Field/value map for the historical row.",
    )


class HistoryResponse(BaseModel):
    """Historical telemetry over a bounded range at an automatic resolution."""

    measurement: str
    start_ms: int = Field(..., description="Range start [UTC epoch ms].")
    end_ms: int = Field(..., description="Range end [UTC epoch ms].")
    window_s: int = Field(
        ..., description="Each point is the mean over a window this long [s]."
    )
    points: list[HistoryPointResponse]


class AlarmResponse(BaseModel):
    """
    An alarm with its lifecycle.

    `alarm_id`, `occurred_at_ms`, `acknowledged` and `cleared` keep their earlier
    meaning; `state` is the full lifecycle state.
    """

    alarm_id: str = Field(..., description="The alarm id as a string.")
    id: int
    key: str = Field(..., description="Condition identity; one open alarm per key.")
    source_service: str
    severity: str
    parameter: str
    direction: str
    unit: str
    state: AlarmStateName
    value: float = Field(..., description="Latest value, in `unit`.")
    threshold: float
    action: str
    message: str
    topic: str
    occurred_at_ms: int = Field(..., description="Same as raised_at_ms.")
    raised_at_ms: int
    cleared_at_ms: int | None
    acknowledged: bool
    acknowledged_at_ms: int | None
    acknowledged_by: str | None
    ack_comment: str | None
    cleared: bool = Field(..., description="The condition has ended.")
    occurrence_count: int
    updated_at_ms: int


class AlarmTransitionResponse(BaseModel):
    """One state change of an alarm."""

    id: int
    alarm_id: int
    from_state: AlarmStateName | None
    to_state: AlarmStateName
    at_ms: int
    actor: str = Field(..., description="Source service or the user who acknowledged.")
    comment: str | None
    value: float | None


class AlarmPageResponse(BaseModel):
    """A page of alarm history."""

    items: list[AlarmResponse]
    total: int
    limit: int
    offset: int


class AlarmDetailResponse(BaseModel):
    """An alarm with its transitions, oldest first."""

    alarm: AlarmResponse
    transitions: list[AlarmTransitionResponse]


class AcknowledgeRequest(BaseModel):
    """Request body for POST /api/v1/alarms/{alarm_id}/ack."""

    comment: str = Field(default="", max_length=500)


class AcknowledgeAllRequest(BaseModel):
    """Request body for POST /api/v1/alarms/ack-all."""

    comment: str = Field(default="", max_length=500)
    severity: Literal["warning", "critical"] | None = Field(
        default=None, description="Only alarms of this severity; all when omitted."
    )


class AcknowledgeResponse(BaseModel):
    """Result of an acknowledgement."""

    accepted: bool
    reason: str
    timestamp_ms: int
    alarms: list[AlarmResponse]


class AuditResponse(BaseModel):
    """Audit log entry returned by the API gateway."""

    id: int
    user_id: int | None
    username: str | None = Field(..., description="Acting user.")
    role: str | None = Field(..., description="Role the user held when acting.")
    ip_address: str
    method: str
    endpoint: str
    request_body_hash: str | None
    response_status: int
    duration_ms: int
    timestamp_ms: int
    detail: str | None
    outcome: str | None = Field(
        ..., description="How the action ended, e.g. accepted, refused: <reason>."
    )


class AuditPageResponse(BaseModel):
    """A page of audit entries, newest first."""

    items: list[AuditResponse]
    total: int
    limit: int
    offset: int
