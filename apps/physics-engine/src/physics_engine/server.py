"""
gRPC transport for the live PhysicsService: plant state, valve commands from the PLC,
and simulation control — pause, stepping, speed, scenarios and faults.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator

import cogniboiler_pb2 as pb2
import cogniboiler_pb2_grpc as pb2_grpc
import grpc
import grpc.aio

from physics_engine import __version__
from physics_engine.faults import FaultError, FaultSpec
from physics_engine.proto_mapping import (
    fault_kind_from_proto,
    fault_to_proto,
    now_ms,
    scenario_to_proto,
    simulation_status_to_proto,
    system_state_to_proto,
)
from physics_engine.runtime import (
    PhysicsRuntime,
    RuntimeCommandError,
    RuntimeUnavailableError,
)
from physics_engine.scenarios import SCENARIOS, ScenarioError

logger = logging.getLogger(__name__)

DEFAULT_PORT: int = 50052


def _operator(operator_id: str) -> str:
    return operator_id or "unknown"


class PhysicsServicer(pb2_grpc.PhysicsServiceServicer):  # type: ignore[misc]
    """gRPC servicer for live physics state, control and simulation management."""

    def __init__(self, runtime: PhysicsRuntime) -> None:
        self._runtime = runtime

    def _simulation_ack(self, accepted: bool, reason: str = "") -> pb2.SimulationAck:
        return pb2.SimulationAck(
            accepted=accepted,
            reason=reason,
            timestamp_ms=now_ms(),
            status=simulation_status_to_proto(self._runtime.simulation_status()),
        )

    def _state(self) -> pb2.SystemStateMsg:
        return system_state_to_proto(
            self._runtime.snapshot, self._runtime.simulation_status()
        )

    # ─── State and commands ──────────────────────────────────────────────────

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
        return self._state()

    async def StreamSystemState(  # noqa: N802
        self,
        request: pb2.StreamRequest,
        context: grpc.aio.ServicerContext,
    ) -> AsyncGenerator[pb2.SystemStateMsg]:
        if request.interval_s > 0:
            while True:
                yield self._state()
                await asyncio.sleep(request.interval_s)

        sequence = -1
        while True:
            try:
                sequence, snapshot = await self._runtime.wait_for_update(sequence)
            except RuntimeUnavailableError as exc:
                await context.abort(grpc.StatusCode.UNAVAILABLE, str(exc))
                return
            yield system_state_to_proto(snapshot, self._runtime.simulation_status())

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
                spray_valve=(
                    request.spray_valve if request.HasField("spray_valve") else None
                ),
            )
        except ValueError as exc:
            return pb2.CommandAck(
                accepted=False, reason=str(exc), timestamp_ms=now_ms()
            )
        return pb2.CommandAck(accepted=True, reason="", timestamp_ms=now_ms())

    # ─── Simulation control ──────────────────────────────────────────────────

    async def GetSimulationStatus(  # noqa: N802
        self,
        request: pb2.Empty,
        context: grpc.aio.ServicerContext,
    ) -> pb2.SimulationStatusMsg:
        return simulation_status_to_proto(self._runtime.simulation_status())

    async def PauseSimulation(  # noqa: N802
        self,
        request: pb2.SimulationControlRequest,
        context: grpc.aio.ServicerContext,
    ) -> pb2.SimulationAck:
        await self._runtime.pause()
        logger.info("Pause requested by %s", _operator(request.operator_id))
        return self._simulation_ack(True)

    async def ResumeSimulation(  # noqa: N802
        self,
        request: pb2.SimulationControlRequest,
        context: grpc.aio.ServicerContext,
    ) -> pb2.SimulationAck:
        await self._runtime.resume()
        logger.info("Resume requested by %s", _operator(request.operator_id))
        return self._simulation_ack(True)

    async def SetSimulationSpeed(  # noqa: N802
        self,
        request: pb2.SimulationSpeedRequest,
        context: grpc.aio.ServicerContext,
    ) -> pb2.SimulationAck:
        try:
            await self._runtime.set_speed(request.speed_factor)
        except RuntimeCommandError as exc:
            return self._simulation_ack(False, str(exc))
        logger.info(
            "Speed %g× requested by %s",
            request.speed_factor,
            _operator(request.operator_id),
        )
        return self._simulation_ack(True)

    async def StepSimulation(  # noqa: N802
        self,
        request: pb2.StepRequest,
        context: grpc.aio.ServicerContext,
    ) -> pb2.SimulationAck:
        try:
            await self._runtime.step(request.steps)
        except RuntimeCommandError as exc:
            return self._simulation_ack(False, str(exc))
        return self._simulation_ack(True)

    async def ListScenarios(  # noqa: N802
        self,
        request: pb2.Empty,
        context: grpc.aio.ServicerContext,
    ) -> pb2.ScenarioListMsg:
        return pb2.ScenarioListMsg(
            scenarios=[
                scenario_to_proto(definition) for definition in SCENARIOS.values()
            ],
            current=self._runtime.simulation_status().scenario.value,
        )

    async def LoadScenario(  # noqa: N802
        self,
        request: pb2.ScenarioRequest,
        context: grpc.aio.ServicerContext,
    ) -> pb2.SimulationAck:
        try:
            await self._runtime.load_scenario(request.name)
        except ScenarioError as exc:
            return self._simulation_ack(False, str(exc))
        logger.warning(
            "Scenario %s loaded by %s", request.name, _operator(request.operator_id)
        )
        return self._simulation_ack(True)

    async def InjectFault(  # noqa: N802
        self,
        request: pb2.FaultRequest,
        context: grpc.aio.ServicerContext,
    ) -> pb2.FaultAck:
        try:
            spec = FaultSpec(
                kind=fault_kind_from_proto(request.kind),
                target=request.target,
                severity=request.severity,
                ramp_s=request.ramp_s,
            )
            fault = await self._runtime.inject_fault(spec)
        except (FaultError, ValueError) as exc:
            return pb2.FaultAck(accepted=False, reason=str(exc), timestamp_ms=now_ms())
        logger.warning(
            "Fault %s injected by %s", fault.label, _operator(request.operator_id)
        )
        now_s = self._runtime.snapshot.simulation_time_s
        return pb2.FaultAck(
            accepted=True,
            timestamp_ms=now_ms(),
            faults=[fault_to_proto(fault, now_s)],
        )

    async def ClearFault(  # noqa: N802
        self,
        request: pb2.FaultClearRequest,
        context: grpc.aio.ServicerContext,
    ) -> pb2.FaultAck:
        try:
            if request.all:
                cleared = await self._runtime.clear_faults()
            else:
                cleared = (await self._runtime.clear_fault(request.fault_id),)
        except FaultError as exc:
            return pb2.FaultAck(accepted=False, reason=str(exc), timestamp_ms=now_ms())
        logger.info(
            "Faults cleared by %s: %s",
            _operator(request.operator_id),
            ", ".join(fault.label for fault in cleared) or "none",
        )
        now_s = self._runtime.snapshot.simulation_time_s
        return pb2.FaultAck(
            accepted=True,
            timestamp_ms=now_ms(),
            faults=[fault_to_proto(fault, now_s) for fault in cleared],
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
