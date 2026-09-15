"""
gRPC server for PLCService.

Wraps PLCService business logic in gRPC transport layer.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncGenerator

import cogniboiler_pb2 as pb2
import cogniboiler_pb2_grpc as pb2_grpc
import grpc
import grpc.aio

from plc_controller.client import PhysicsClient, PhysicsClientConfig
from plc_controller.service import PLCService

logger = logging.getLogger(__name__)

DEFAULT_PORT: int = 50051


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
        result = await self._svc.send_command(
            fuel_valve=request.fuel_valve,
            feedwater_valve=request.feedwater_valve,
            steam_valve=request.steam_valve,
            source=request.source,
            operator_id=request.operator_id,
        )
        logger.info(
            "Command received: fv=%.2f fw=%.2f sv=%.2f -> %s",
            request.fuel_valve,
            request.feedwater_valve,
            request.steam_valve,
            "accepted" if result.accepted else f"rejected: {result.reason}",
        )
        return pb2.CommandAck(
            accepted=result.accepted,
            reason=result.reason,
            timestamp_ms=int(time.time() * 1000),
        )

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
        )
        return pb2.CommandAck(
            accepted=result.accepted,
            reason=result.reason,
            timestamp_ms=int(time.time() * 1000),
        )

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
        return pb2.CommandAck(
            accepted=result.accepted,
            reason=result.reason,
            timestamp_ms=int(time.time() * 1000),
        )

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
) -> None:
    """Start gRPC server and block until KeyboardInterrupt."""
    service = PLCService(
        physics_client=PhysicsClient(PhysicsClientConfig(target=physics_target)),
        mqtt_host=mqtt_host,
        mqtt_port=mqtt_port,
    )
    await service.start()
    server = grpc.aio.server()
    pb2_grpc.add_PLCServiceServicer_to_server(PLCServicer(service), server)
    listen_addr = f"[::]:{port}"
    server.add_insecure_port(listen_addr)
    await server.start()
    logger.info("PLC gRPC server listening on %s", listen_addr)
    try:
        await server.wait_for_termination()
    finally:
        await server.stop(grace=5)
        await service.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(serve())
