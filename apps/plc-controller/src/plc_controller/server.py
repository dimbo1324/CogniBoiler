"""
gRPC server for PLCService.

Wraps PLCService business logic in gRPC transport layer.
"""

from __future__ import annotations

import asyncio
import logging
import math
from collections.abc import AsyncGenerator

import cogniboiler_pb2 as pb2
import cogniboiler_pb2_grpc as pb2_grpc
import grpc
import grpc.aio
from cogniboiler_observability import (
    ServerObservability,
    serve_until_cancelled,
    start_metrics_server,
)
from cogniboiler_runtime import now_ms

from plc_controller.client import (
    DEFAULT_PHYSICS_TARGET,
    PhysicsClient,
    PhysicsClientConfig,
)
from plc_controller.commands import ValidationResult
from plc_controller.events import DEFAULT_MQTT_HOST, DEFAULT_MQTT_PORT
from plc_controller.metrics import observe_service
from plc_controller.modes import from_proto
from plc_controller.service import PLCService
from plc_controller.status import command_msg, setpoints_msg

logger = logging.getLogger(__name__)

DEFAULT_PORT: int = 50051
# PLCService takes valve commands and E-Stop resets from whoever reaches it, so by
# default it listens on this machine only; the Compose network passes 0.0.0.0.
DEFAULT_HOST: str = "127.0.0.1"

# StreamCommands: a caller gets the latest command every 0.1 s to 60 s, and at most
# this many streams are open at once, so no caller can park unbounded server work.
MIN_COMMAND_STREAM_INTERVAL_S: float = 0.1
MAX_COMMAND_STREAM_INTERVAL_S: float = 60.0
MAX_COMMAND_STREAMS: int = 8


def command_stream_interval(requested_s: float) -> float:
    """The interval a finite request gets, kept between a tenth and a minute."""
    return min(
        max(requested_s, MIN_COMMAND_STREAM_INTERVAL_S), MAX_COMMAND_STREAM_INTERVAL_S
    )


def _ack(result: ValidationResult) -> pb2.CommandAck:
    return pb2.CommandAck(
        accepted=result.accepted,
        reason=result.reason,
        timestamp_ms=now_ms(),
    )


class PLCServicer(pb2_grpc.PLCServiceServicer):  # type: ignore[misc]
    """gRPC servicer: bridges gRPC calls to PLCService business logic."""

    def __init__(self, service: PLCService | None = None) -> None:
        self._svc = service or PLCService()
        self._command_streams = 0

    async def Health(  # noqa: N802
        self,
        request: pb2.Empty,
        context: grpc.aio.ServicerContext,
    ) -> pb2.HealthStatus:
        status = await self._svc.physics_status()
        return pb2.HealthStatus(
            service="plc-controller",
            status=status,
            version=PLCService.VERSION,
            uptime_seconds=self._svc.uptime_seconds,
        )

    async def SendCommand(  # noqa: N802
        self,
        request: pb2.ControlCommandMsg,
        context: grpc.aio.ServicerContext,
    ) -> pb2.CommandAck:
        spray = request.spray_valve if request.HasField("spray_valve") else None
        result = await self._svc.send_command(
            fuel_valve=request.fuel_valve,
            feedwater_valve=request.feedwater_valve,
            steam_valve=request.steam_valve,
            spray_valve=spray,
            source=request.source,
            operator_id=request.operator_id,
        )
        logger.info(
            "Command from %r: fv=%.3f fw=%.3f sv=%.3f spray=%s -> %s",
            request.operator_id,
            request.fuel_valve,
            request.feedwater_valve,
            request.steam_valve,
            f"{spray:.3f}" if spray is not None else "unchanged",
            "accepted" if result.accepted else f"rejected: {result.reason}",
        )
        return _ack(result)

    async def GetSetpoints(  # noqa: N802
        self,
        request: pb2.Empty,
        context: grpc.aio.ServicerContext,
    ) -> pb2.SetpointsMsg:
        return setpoints_msg(self._svc.get_setpoints())

    async def UpdateSetpoints(  # noqa: N802
        self,
        request: pb2.SetpointsMsg,
        context: grpc.aio.ServicerContext,
    ) -> pb2.CommandAck:
        result = self._svc.update_setpoints(
            pressure_pa=request.pressure_pa,
            water_level_m=request.water_level_m,
            steam_temp_k=request.steam_temp_k,
            operator_id=request.operator_id,
        )
        return _ack(result)

    async def SetLoadDemand(  # noqa: N802
        self,
        request: pb2.LoadDemandRequest,
        context: grpc.aio.ServicerContext,
    ) -> pb2.CommandAck:
        result = self._svc.set_load_demand(request.load_w, request.operator_id)
        logger.info(
            "Load demand %.1f MW from %r -> %s",
            request.load_w / 1e6,
            request.operator_id,
            "accepted" if result.accepted else f"rejected: {result.reason}",
        )
        return _ack(result)

    async def SetControlMode(  # noqa: N802
        self,
        request: pb2.ControlModeRequest,
        context: grpc.aio.ServicerContext,
    ) -> pb2.CommandAck:
        mode = from_proto(request.mode)
        if mode is None:
            return _ack(ValidationResult(False, f"unknown control mode {request.mode}"))
        return _ack(await self._svc.set_mode(mode, request.operator_id))

    async def GetControlStatus(  # noqa: N802
        self,
        request: pb2.Empty,
        context: grpc.aio.ServicerContext,
    ) -> pb2.PLCStatusMsg:
        return await self._svc.get_control_status()

    async def ResetEmergencyStop(  # noqa: N802
        self,
        request: pb2.ResetRequest,
        context: grpc.aio.ServicerContext,
    ) -> pb2.CommandAck:
        result = await self._svc.reset_emergency_stop(request.operator_id)
        return _ack(result)

    async def StreamCommands(  # noqa: N802
        self,
        request: pb2.StreamRequest,
        context: grpc.aio.ServicerContext,
    ) -> AsyncGenerator[pb2.ControlCommandMsg]:
        """Stream control commands at the requested interval."""
        if not math.isfinite(request.interval_s):
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT, "interval_s must be a finite number"
            )
        if self._command_streams >= MAX_COMMAND_STREAMS:
            await context.abort(
                grpc.StatusCode.RESOURCE_EXHAUSTED,
                f"at most {MAX_COMMAND_STREAMS} command streams may be open",
            )
        interval = command_stream_interval(request.interval_s)
        self._command_streams += 1
        try:
            while not context.done():
                yield command_msg(self._svc.latest_command())
                await asyncio.sleep(interval)
        finally:
            self._command_streams -= 1


def listen_address(host: str, port: int) -> str:
    """`host:port` for gRPC, with an IPv6 host in brackets."""
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    return f"{host}:{port}"


async def serve(
    port: int = DEFAULT_PORT,
    *,
    host: str = DEFAULT_HOST,
    physics_target: str = DEFAULT_PHYSICS_TARGET,
    mqtt_host: str = DEFAULT_MQTT_HOST,
    mqtt_port: int = DEFAULT_MQTT_PORT,
    mqtt_username: str | None = None,
    mqtt_password: str | None = None,
    metrics_port: int = 0,
    metrics_host: str = "127.0.0.1",
) -> None:
    """Start gRPC server and block until termination."""
    service = PLCService(
        physics_client=PhysicsClient(PhysicsClientConfig(target=physics_target)),
        mqtt_host=mqtt_host,
        mqtt_port=mqtt_port,
        mqtt_username=mqtt_username,
        mqtt_password=mqtt_password,
    )
    await service.start()
    observe_service(service)
    start_metrics_server(metrics_port, metrics_host)
    server = grpc.aio.server(interceptors=[ServerObservability()])
    pb2_grpc.add_PLCServiceServicer_to_server(PLCServicer(service), server)
    listen_addr = listen_address(host, port)
    server.add_insecure_port(listen_addr)
    await server.start()
    logger.info("PLC gRPC server listening on %s", listen_addr)
    try:
        await serve_until_cancelled(server)
    finally:
        await service.close()
