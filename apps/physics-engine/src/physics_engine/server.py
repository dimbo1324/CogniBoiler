"""
gRPC transport for the live PhysicsService.
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

from physics_engine import __version__
from physics_engine.models import ControlInputs
from physics_engine.mqtt_publisher import boiler_state_to_proto, turbine_state_to_proto
from physics_engine.runtime import PhysicsRuntime
from physics_engine.system import SystemState

logger = logging.getLogger(__name__)

DEFAULT_PORT: int = 50052


def actuator_state_to_proto(controls: ControlInputs) -> pb2.ActuatorStateMsg:
    """Convert the current control commands and actuator positions to protobuf."""
    return pb2.ActuatorStateMsg(
        fuel_valve_command=controls.fuel_valve_command,
        fuel_valve_position=controls.fuel_valve.position,
        feedwater_valve_command=controls.feedwater_valve_command,
        feedwater_valve_position=controls.feedwater_valve.position,
        steam_valve_command=controls.steam_valve_command,
        steam_valve_position=controls.steam_valve.position,
    )


def system_state_to_proto(
    state: SystemState,
    controls: ControlInputs,
) -> pb2.SystemStateMsg:
    """Convert a runtime SystemState snapshot to protobuf."""
    return pb2.SystemStateMsg(
        boiler=boiler_state_to_proto(state.boiler),
        turbine=turbine_state_to_proto(state.turbine),
        actuators=actuator_state_to_proto(controls),
        simulation_time_s=state.time,
    )


class PhysicsServicer(pb2_grpc.PhysicsServiceServicer):  # type: ignore[misc]
    """gRPC servicer for live physics state and control."""

    def __init__(self, runtime: PhysicsRuntime) -> None:
        self._runtime = runtime

    async def Health(  # noqa: N802
        self,
        request: pb2.Empty,
        context: grpc.aio.ServicerContext,
    ) -> pb2.HealthStatus:
        return pb2.HealthStatus(
            service="physics-engine",
            status=self._runtime.status,
            version=__version__,
            uptime_seconds=self._runtime.uptime_seconds,
        )

    async def GetSystemState(  # noqa: N802
        self,
        request: pb2.Empty,
        context: grpc.aio.ServicerContext,
    ) -> pb2.SystemStateMsg:
        state, controls = await self._runtime.get_snapshot()
        return system_state_to_proto(state, controls)

    async def StreamSystemState(  # noqa: N802
        self,
        request: pb2.StreamRequest,
        context: grpc.aio.ServicerContext,
    ) -> AsyncGenerator[pb2.SystemStateMsg]:
        interval_s = (
            request.interval_s if request.interval_s > 0 else self._runtime.wall_step_s
        )

        if request.interval_s > 0:
            while True:
                state, controls = await self._runtime.get_snapshot()
                yield system_state_to_proto(state, controls)
                await asyncio.sleep(interval_s)

        sequence = -1
        while True:
            sequence, state = await self._runtime.wait_for_update(sequence)
            controls = await self._runtime.get_controls()
            yield system_state_to_proto(state, controls)

    async def ApplyControlCommand(  # noqa: N802
        self,
        request: pb2.ControlCommandMsg,
        context: grpc.aio.ServicerContext,
    ) -> pb2.CommandAck:
        try:
            await self._runtime.apply_command(
                fuel_valve=request.fuel_valve,
                feedwater_valve=request.feedwater_valve,
                steam_valve=request.steam_valve,
            )
        except ValueError as exc:
            return pb2.CommandAck(
                accepted=False,
                reason=str(exc),
                timestamp_ms=int(time.time() * 1000),
            )

        return pb2.CommandAck(
            accepted=True,
            reason="",
            timestamp_ms=int(time.time() * 1000),
        )


async def serve(
    runtime: PhysicsRuntime,
    *,
    port: int = DEFAULT_PORT,
) -> None:
    """Start the PhysicsService gRPC server and block until termination."""
    await runtime.start()
    server = grpc.aio.server()
    pb2_grpc.add_PhysicsServiceServicer_to_server(PhysicsServicer(runtime), server)
    listen_addr = f"[::]:{port}"
    server.add_insecure_port(listen_addr)
    await server.start()
    logger.info("Physics gRPC server listening on %s", listen_addr)
    try:
        await server.wait_for_termination()
    finally:
        await server.stop(grace=5)
        await runtime.stop()
