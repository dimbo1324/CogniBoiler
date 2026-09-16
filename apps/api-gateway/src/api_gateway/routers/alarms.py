"""
Alarm endpoints backed by the alert-manager AlarmService.

The gateway owns no alarm state and reads no alarm table: lists, history, details and
acknowledgements all go through AlarmService. Acknowledgements record the user's name,
and the request itself lands in the audit log like every other call.
"""

from __future__ import annotations

from typing import Literal, cast

import cogniboiler_pb2 as pb2
import grpc
import grpc.aio
from fastapi import APIRouter, Path, Query, Request

from api_gateway.audit import set_audit_outcome
from api_gateway.auth.rbac import OperatorUser, ViewerUser
from api_gateway.clients import AlarmGatewayClient
from api_gateway.problems import ProblemError, upstream_unavailable
from api_gateway.schemas.ops import (
    AcknowledgeAllRequest,
    AcknowledgeRequest,
    AcknowledgeResponse,
    AlarmDetailResponse,
    AlarmPageResponse,
    AlarmResponse,
    AlarmStateName,
    AlarmTransitionResponse,
)

router = APIRouter(prefix="/api/v1/alarms", tags=["alarms"])

_STATE_NAMES: dict[int, str] = {
    int(pb2.AlarmState.ALARM_ACTIVE_UNACK): "ACTIVE_UNACK",
    int(pb2.AlarmState.ALARM_ACTIVE_ACK): "ACTIVE_ACK",
    int(pb2.AlarmState.ALARM_CLEARED_UNACK): "CLEARED_UNACK",
    int(pb2.AlarmState.ALARM_CLEARED): "CLEARED",
}
_ACKNOWLEDGED = frozenset({"ACTIVE_ACK", "CLEARED"})
_CLEARED = frozenset({"CLEARED_UNACK", "CLEARED"})


def _alarm_client(request: Request) -> AlarmGatewayClient:
    """Resolve the shared AlarmService client from application state."""
    return request.app.state.alarm_client  # type: ignore[no-any-return]


def _state(value: int) -> AlarmStateName:
    return cast(AlarmStateName, _STATE_NAMES.get(int(value), "CLEARED"))


def alarm_from_proto(message: pb2.AlarmMsg) -> AlarmResponse:
    """Map an AlarmService alarm to the REST schema."""
    state = _state(message.state)
    return AlarmResponse(
        alarm_id=str(message.alarm_id),
        id=message.alarm_id,
        key=message.key,
        source_service=message.source_service,
        severity=message.severity,
        parameter=message.parameter,
        direction=message.direction,
        unit=message.unit,
        state=state,
        value=message.value,
        threshold=message.threshold,
        action=message.action,
        message=message.message,
        topic=message.topic,
        occurred_at_ms=message.raised_at_ms,
        raised_at_ms=message.raised_at_ms,
        cleared_at_ms=message.cleared_at_ms or None,
        acknowledged=state in _ACKNOWLEDGED,
        acknowledged_at_ms=message.acknowledged_at_ms or None,
        acknowledged_by=message.acknowledged_by or None,
        ack_comment=message.ack_comment or None,
        cleared=state in _CLEARED,
        occurrence_count=message.occurrence_count,
        updated_at_ms=message.updated_at_ms,
    )


def _transition_from_proto(message: pb2.AlarmTransitionMsg) -> AlarmTransitionResponse:
    return AlarmTransitionResponse(
        id=message.transition_id,
        alarm_id=message.alarm_id,
        from_state=(
            _state(message.from_state)
            if message.from_state != pb2.AlarmState.ALARM_STATE_UNSPECIFIED
            else None
        ),
        to_state=_state(message.to_state),
        at_ms=message.at_ms,
        actor=message.actor,
        comment=message.comment or None,
        value=message.value,
    )


def _upstream_error(exc: grpc.RpcError) -> ProblemError:
    if (
        isinstance(exc, grpc.aio.AioRpcError)
        and exc.code() == grpc.StatusCode.NOT_FOUND
    ):
        return ProblemError(404, "alarms.not_found", "The alarm does not exist.")
    return upstream_unavailable("AlarmService", exc)


def _ack_outcome(request: Request, result: pb2.AcknowledgeResult) -> None:
    if result.accepted:
        set_audit_outcome(request, f"acknowledged {len(result.alarms)} alarm(s)")
    else:
        set_audit_outcome(request, f"refused: {result.reason}")


@router.get("", response_model=list[AlarmResponse])
async def list_alarms(
    request: Request,
    _: ViewerUser,
    active_only: bool = Query(
        default=False,
        description="Only open alarms: not both cleared and acknowledged.",
    ),
    limit: int = Query(default=100, ge=1, le=1000),
) -> list[AlarmResponse]:
    """Recent alarms; with active_only, open alarms with critical and unacknowledged first."""
    try:
        result = await _alarm_client(request).list_alarms(
            pb2.ListAlarmsRequest(open_only=active_only, limit=limit)
        )
    except grpc.RpcError as exc:
        raise _upstream_error(exc) from exc
    return [alarm_from_proto(alarm) for alarm in result.alarms]


@router.get("/history", response_model=AlarmPageResponse)
async def alarm_history(
    request: Request,
    _: ViewerUser,
    severity: Literal["warning", "critical"] | None = Query(default=None),
    parameter: str | None = Query(default=None, max_length=128),
    from_ms: int = Query(default=0, ge=0, description="Raised at or after [UTC ms]."),
    to_ms: int = Query(default=0, ge=0, description="Raised at or before [UTC ms]."),
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
) -> AlarmPageResponse:
    """Alarm history with filters, newest first, paged."""
    try:
        result = await _alarm_client(request).list_alarms(
            pb2.ListAlarmsRequest(
                severity=severity or "",
                parameter=parameter or "",
                from_ms=from_ms,
                to_ms=to_ms,
                limit=limit,
                offset=offset,
            )
        )
    except grpc.RpcError as exc:
        raise _upstream_error(exc) from exc
    return AlarmPageResponse(
        items=[alarm_from_proto(alarm) for alarm in result.alarms],
        total=result.total,
        limit=limit,
        offset=offset,
    )


@router.post("/ack-all", response_model=AcknowledgeResponse)
async def acknowledge_all(
    request: Request,
    body: AcknowledgeAllRequest,
    user: OperatorUser,
) -> AcknowledgeResponse:
    """Acknowledge every unacknowledged alarm, optionally of one severity."""
    try:
        result = await _alarm_client(request).acknowledge_all(
            user.username, body.comment, body.severity or ""
        )
    except grpc.RpcError as exc:
        raise _upstream_error(exc) from exc
    _ack_outcome(request, result)
    return AcknowledgeResponse(
        accepted=result.accepted,
        reason=result.reason,
        timestamp_ms=result.timestamp_ms,
        alarms=[alarm_from_proto(alarm) for alarm in result.alarms],
    )


@router.get("/{alarm_id}", response_model=AlarmDetailResponse)
async def get_alarm(
    request: Request,
    _: ViewerUser,
    alarm_id: int = Path(..., ge=1),
) -> AlarmDetailResponse:
    """One alarm with every state change it went through."""
    try:
        result = await _alarm_client(request).get_alarm(alarm_id)
    except grpc.RpcError as exc:
        raise _upstream_error(exc) from exc
    return AlarmDetailResponse(
        alarm=alarm_from_proto(result.alarm),
        transitions=[_transition_from_proto(t) for t in result.transitions],
    )


@router.post("/{alarm_id}/ack", response_model=AcknowledgeResponse)
async def acknowledge_alarm(
    request: Request,
    body: AcknowledgeRequest,
    user: OperatorUser,
    alarm_id: int = Path(..., ge=1),
) -> AcknowledgeResponse:
    """Acknowledge one alarm; refused if it is already acknowledged."""
    try:
        result = await _alarm_client(request).acknowledge(
            alarm_id, user.username, body.comment
        )
    except grpc.RpcError as exc:
        raise _upstream_error(exc) from exc
    _ack_outcome(request, result)
    return AcknowledgeResponse(
        accepted=result.accepted,
        reason=result.reason,
        timestamp_ms=result.timestamp_ms,
        alarms=[alarm_from_proto(alarm) for alarm in result.alarms],
    )
