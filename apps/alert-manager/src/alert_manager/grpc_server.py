"""gRPC AlarmService: open alarms, history, details and acknowledgement."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

import cogniboiler_pb2 as pb2
import cogniboiler_pb2_grpc as pb2_grpc
import grpc
import grpc.aio
from cogniboiler_observability import ServerObservability
from cogniboiler_runtime import now_ms

from alert_manager.lifecycle import AlarmState, LifecycleError
from alert_manager.processor import (
    AlarmNotFoundError,
    AlarmProcessor,
    AlarmQuery,
)
from alert_manager.views import AlarmView, TransitionView

logger = logging.getLogger(__name__)

DEFAULT_PORT: int = 50053
VERSION: str = "0.2.0"

_STATES: dict[AlarmState, int] = {
    AlarmState.ACTIVE_UNACK: int(pb2.AlarmState.ALARM_ACTIVE_UNACK),
    AlarmState.ACTIVE_ACK: int(pb2.AlarmState.ALARM_ACTIVE_ACK),
    AlarmState.CLEARED_UNACK: int(pb2.AlarmState.ALARM_CLEARED_UNACK),
    AlarmState.CLEARED: int(pb2.AlarmState.ALARM_CLEARED),
}
_SEVERITIES = frozenset({"", "warning", "critical"})


def alarm_to_proto(alarm: AlarmView) -> pb2.AlarmMsg:
    return pb2.AlarmMsg(
        alarm_id=alarm.id,
        key=alarm.key,
        source_service=alarm.source_service,
        parameter=alarm.parameter,
        severity=alarm.severity,
        direction=alarm.direction,
        unit=alarm.unit,
        state=_STATES[alarm.state],
        message=alarm.message,
        action=alarm.action,
        topic=alarm.topic,
        value=alarm.value,
        threshold=alarm.threshold,
        raised_at_ms=alarm.raised_at_ms,
        cleared_at_ms=alarm.cleared_at_ms or 0,
        acknowledged_at_ms=alarm.acknowledged_at_ms or 0,
        acknowledged_by=alarm.acknowledged_by or "",
        ack_comment=alarm.ack_comment or "",
        occurrence_count=alarm.occurrence_count,
        updated_at_ms=alarm.updated_at_ms,
    )


def transition_to_proto(transition: TransitionView) -> pb2.AlarmTransitionMsg:
    return pb2.AlarmTransitionMsg(
        transition_id=transition.id,
        alarm_id=transition.alarm_id,
        from_state=(
            _STATES[transition.from_state]
            if transition.from_state is not None
            else pb2.AlarmState.ALARM_STATE_UNSPECIFIED
        ),
        to_state=_STATES[transition.to_state],
        at_ms=transition.at_ms,
        actor=transition.actor,
        comment=transition.comment or "",
        value=transition.value or 0.0,
    )


class AlarmServicer(pb2_grpc.AlarmServiceServicer):  # type: ignore[misc]
    """Bridges gRPC calls to the alarm processor."""

    def __init__(
        self, processor: AlarmProcessor, *, is_subscribed: Callable[[], bool]
    ) -> None:
        self._processor = processor
        self._is_subscribed = is_subscribed
        self._started = time.monotonic()

    async def Health(  # noqa: N802
        self, request: pb2.Empty, context: grpc.aio.ServicerContext
    ) -> pb2.HealthStatus:
        try:
            await self._processor.ping()
            status = "running" if self._is_subscribed() else "degraded"
        except Exception as exc:
            logger.warning("AlarmService health: database unreachable: %s", exc)
            status = "degraded"
        return pb2.HealthStatus(
            service="alert-manager",
            status=status,
            version=VERSION,
            uptime_seconds=time.monotonic() - self._started,
        )

    async def ListAlarms(  # noqa: N802
        self, request: pb2.ListAlarmsRequest, context: grpc.aio.ServicerContext
    ) -> pb2.AlarmListMsg:
        if request.severity not in _SEVERITIES:
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"unknown severity {request.severity!r}",
            )
        alarms, total = await self._processor.list_alarms(
            AlarmQuery(
                open_only=request.open_only,
                severity=request.severity,
                parameter=request.parameter,
                from_ms=request.from_ms,
                to_ms=request.to_ms,
                limit=request.limit,
                offset=request.offset,
            )
        )
        return pb2.AlarmListMsg(
            alarms=[alarm_to_proto(alarm) for alarm in alarms], total=total
        )

    async def GetAlarm(  # noqa: N802
        self, request: pb2.AlarmRef, context: grpc.aio.ServicerContext
    ) -> pb2.AlarmDetailMsg:
        try:
            alarm, transitions = await self._processor.get_alarm(request.alarm_id)
        except AlarmNotFoundError as exc:
            await context.abort(grpc.StatusCode.NOT_FOUND, str(exc))
            raise
        return pb2.AlarmDetailMsg(
            alarm=alarm_to_proto(alarm),
            transitions=[transition_to_proto(t) for t in transitions],
        )

    async def AcknowledgeAlarm(  # noqa: N802
        self, request: pb2.AcknowledgeAlarmRequest, context: grpc.aio.ServicerContext
    ) -> pb2.AcknowledgeResult:
        try:
            alarm = await self._processor.acknowledge(
                request.alarm_id, request.operator_id, request.comment
            )
        except AlarmNotFoundError as exc:
            await context.abort(grpc.StatusCode.NOT_FOUND, str(exc))
            raise
        except LifecycleError as exc:
            return pb2.AcknowledgeResult(
                accepted=False, reason=str(exc), timestamp_ms=now_ms()
            )
        return pb2.AcknowledgeResult(
            accepted=True, timestamp_ms=now_ms(), alarms=[alarm_to_proto(alarm)]
        )

    async def AcknowledgeAll(  # noqa: N802
        self, request: pb2.AcknowledgeAllRequest, context: grpc.aio.ServicerContext
    ) -> pb2.AcknowledgeResult:
        if request.severity not in _SEVERITIES:
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"unknown severity {request.severity!r}",
            )
        alarms = await self._processor.acknowledge_all(
            request.operator_id, request.comment, request.severity
        )
        return pb2.AcknowledgeResult(
            accepted=True,
            timestamp_ms=now_ms(),
            alarms=[alarm_to_proto(alarm) for alarm in alarms],
        )


async def start_server(servicer: AlarmServicer, port: int) -> grpc.aio.Server:
    """Start the AlarmService on a port; the caller stops it."""
    server = grpc.aio.server(interceptors=[ServerObservability()])
    pb2_grpc.add_AlarmServiceServicer_to_server(servicer, server)
    server.add_insecure_port(f"[::]:{port}")
    await server.start()
    logger.info("AlarmService gRPC listening on [::]:%d", port)
    return server
