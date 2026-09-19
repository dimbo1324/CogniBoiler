"""
gRPC server for PLCService.

Wraps PLCService business logic in gRPC transport layer.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator

import cogniboiler_pb2 as pb2
import cogniboiler_pb2_grpc as pb2_grpc
import grpc
import grpc.aio
from cogniboiler_observability import ServerObservability, start_metrics_server

from plc_controller.client import PhysicsClient, PhysicsClientConfig
from plc_controller.events import now_ms
from plc_controller.metrics import observe_service
from plc_controller.service import PLCService, RuntimeMode, ValidationResult

logger = logging.getLogger(__name__)

DEFAULT_PORT: int = 50051

_MODES: dict[int, RuntimeMode] = {
    int(pb2.ControlMode.AUTO): RuntimeMode.AUTO,
    int(pb2.ControlMode.MANUAL): RuntimeMode.MANUAL,
    int(pb2.ControlMode.ESTOP): RuntimeMode.ESTOP,
}


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
            "Command from %s: fv=%.3f fw=%.3f sv=%.3f spray=%s -> %s",
            request.operator_id or "unknown",
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
        sp = self._svc.get_setpoints()
        return pb2.SetpointsMsg(
            pressure_pa=sp.pressure_pa,
            water_level_m=sp.water_level_m,
            steam_temp_k=sp.steam_temp_k,
            timestamp_ms=sp.updated_at_ms,
        )

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
            "Load demand %.1f MW from %s -> %s",
            request.load_w / 1e6,
            request.operator_id or "unknown",
            "accepted" if result.accepted else f"rejected: {result.reason}",
        )
        return _ack(result)

    async def SetControlMode(  # noqa: N802
        self,
        request: pb2.ControlModeRequest,
        context: grpc.aio.ServicerContext,
    ) -> pb2.CommandAck:
        mode = _MODES.get(int(request.mode))
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
        result = await self._svc.reset_emergency_stop(request.operator_id or "unknown")
        return _ack(result)

    async def StreamCommands(  # noqa: N802
        self,
        request: pb2.StreamRequest,
        context: grpc.aio.ServicerContext,
    ) -> AsyncGenerator[pb2.ControlCommandMsg]:
        """Stream control commands at the requested interval."""
        interval = max(request.interval_s, 0.1)
        while context.is_active():
            latest = self._svc.latest_command()
            yield pb2.ControlCommandMsg(
                fuel_valve=latest.fuel_valve,
                feedwater_valve=latest.feedwater_valve,
                steam_valve=latest.steam_valve,
                spray_valve=latest.spray_valve,
                timestamp_ms=latest.timestamp_ms,
                source=latest.source,
                operator_id=latest.operator_id,
            )
            await asyncio.sleep(interval)


async def serve(
    port: int = DEFAULT_PORT,
    *,
    physics_target: str = "localhost:50052",
    mqtt_host: str = "localhost",
    mqtt_port: int = 1883,
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
    listen_addr = f"[::]:{port}"
    server.add_insecure_port(listen_addr)
    await server.start()
    logger.info("PLC gRPC server listening on %s", listen_addr)
    try:
        # Cancelling wait_for_termination() cancels the server's own completion
        # future, after which stop() fails; shielded, the shutdown below is graceful.
        await asyncio.shield(server.wait_for_termination())
    finally:
        await server.stop(grace=5)
        await service.close()
