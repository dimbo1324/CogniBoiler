"""
gRPC transport for the live PhysicsService: plant state, valve commands from the PLC,
and simulation control — pause, stepping, speed, scenarios and faults.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from collections.abc import AsyncGenerator, Callable

import cogniboiler_pb2 as pb2
import cogniboiler_pb2_grpc as pb2_grpc
import grpc
import grpc.aio
from cogniboiler_observability import ServerObservability, serve_until_cancelled
from cogniboiler_runtime import now_ms

from physics_engine import __version__
from physics_engine.faults import FaultError, FaultSpec
from physics_engine.metrics import COMMANDS_REFUSED
from physics_engine.operating_point import OperatingPointError
from physics_engine.proto_mapping import (
    fault_kind_from_proto,
    fault_to_proto,
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

DEFAULT_HOST: str = "127.0.0.1"
DEFAULT_PORT: int = 50052

# A timed stream sends a full state message per interval: no faster than the PLC's own
# stream allows, and not so slowly that a client cannot tell it from a dead link.
MIN_STREAM_INTERVAL_S: float = 0.1
MAX_STREAM_INTERVAL_S: float = 60.0

# The PLC calls ApplyControlCommand at its scan rate, so a defect that makes it send
# invalid valves would otherwise log a warning per scan.
REFUSAL_LOG_WINDOW_S: float = 60.0

# The PLC keeps its state stream alive with HTTP/2 pings every 10 s, also while a paused
# plant sends no data; a server with default options answers the third such ping with
# GOAWAY "too_many_pings". Accept pings every 5 s, with or without an active call.
SERVER_OPTIONS: tuple[tuple[str, int], ...] = (
    ("grpc.http2.min_ping_interval_without_data_ms", 5_000),
    ("grpc.keepalive_permit_without_calls", 1),
)


def _operator(operator_id: str) -> str:
    return operator_id or "unknown"


def stream_interval_s(requested: float) -> float | None:
    """The timed-stream interval for a request: None streams every update instead.

    Raises ValueError for an interval that is not a duration.
    """
    if not math.isfinite(requested) or requested < 0.0:
        raise ValueError(f"interval_s must be a finite number >= 0, not {requested!r}")
    if requested == 0.0:
        return None
    return min(max(requested, MIN_STREAM_INTERVAL_S), MAX_STREAM_INTERVAL_S)


class _RefusalLog:
    """Warns about a refusal reason at most once per window; every refusal is counted."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._last: dict[tuple[str, str], float] = {}

    def refused(self, rpc: str, operator_id: str, reason: str, *, limit: bool) -> None:
        COMMANDS_REFUSED.labels(rpc).inc()
        now = self._clock()
        self._last = {
            key: at for key, at in self._last.items() if now - at < REFUSAL_LOG_WINDOW_S
        }
        if limit and (rpc, reason) in self._last:
            logger.debug("%s refused for %s: %s", rpc, operator_id, reason)
            return
        self._last[(rpc, reason)] = now
        logger.warning("%s refused for %s: %s", rpc, operator_id, reason)


class PhysicsServicer(pb2_grpc.PhysicsServiceServicer):  # type: ignore[misc]
    """gRPC servicer for live physics state, control and simulation management."""

    def __init__(self, runtime: PhysicsRuntime) -> None:
        self._runtime = runtime
        self._refusals = _RefusalLog()

    def _simulation_ack(self, accepted: bool, reason: str = "") -> pb2.SimulationAck:
        return pb2.SimulationAck(
            accepted=accepted,
            reason=reason,
            timestamp_ms=now_ms(),
            status=simulation_status_to_proto(self._runtime.simulation_status()),
        )

    def _refused(
        self, rpc: str, operator_id: str, reason: str, *, limit: bool = False
    ) -> None:
        self._refusals.refused(rpc, _operator(operator_id), reason, limit=limit)

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
        try:
            interval = stream_interval_s(request.interval_s)
        except ValueError as exc:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
            return

        if interval is not None:
            while True:
                if self._runtime.status != "running":
                    await context.abort(
                        grpc.StatusCode.UNAVAILABLE, self._runtime.unavailable_reason
                    )
                    return
                yield self._state()
                await asyncio.sleep(interval)

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
            self._refused(
                "ApplyControlCommand", request.operator_id, str(exc), limit=True
            )
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
        logger.info(
            "Pause requested by %s at t=%.1fs",
            _operator(request.operator_id),
            self._runtime.snapshot.simulation_time_s,
        )
        return self._simulation_ack(True)

    async def ResumeSimulation(  # noqa: N802
        self,
        request: pb2.SimulationControlRequest,
        context: grpc.aio.ServicerContext,
    ) -> pb2.SimulationAck:
        await self._runtime.resume()
        logger.info(
            "Resume requested by %s at t=%.1fs",
            _operator(request.operator_id),
            self._runtime.snapshot.simulation_time_s,
        )
        return self._simulation_ack(True)

    async def SetSimulationSpeed(  # noqa: N802
        self,
        request: pb2.SimulationSpeedRequest,
        context: grpc.aio.ServicerContext,
    ) -> pb2.SimulationAck:
        try:
            await self._runtime.set_speed(request.speed_factor)
        except RuntimeCommandError as exc:
            self._refused("SetSimulationSpeed", request.operator_id, str(exc))
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
        operator = _operator(request.operator_id)
        try:
            await self._runtime.step(request.steps)
        except RuntimeCommandError as exc:
            self._refused("StepSimulation", request.operator_id, str(exc))
            return self._simulation_ack(False, str(exc))
        except Exception:
            logger.exception("StepSimulation failed for %s", operator)
            await context.abort(grpc.StatusCode.INTERNAL, "physics step failed")
        logger.info("%d steps requested by %s", request.steps, operator)
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
        operator = _operator(request.operator_id)
        try:
            await self._runtime.load_scenario(request.name)
        except (ScenarioError, OperatingPointError) as exc:
            self._refused("LoadScenario", request.operator_id, str(exc))
            return self._simulation_ack(False, str(exc))
        except Exception:
            logger.exception("LoadScenario %s failed for %s", request.name, operator)
            await context.abort(grpc.StatusCode.INTERNAL, "scenario load failed")
        logger.warning("Scenario %s loaded by %s", request.name, operator)
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
            self._refused("InjectFault", request.operator_id, str(exc))
            return pb2.FaultAck(accepted=False, reason=str(exc), timestamp_ms=now_ms())
        logger.warning(
            "Fault %s (%s) injected by %s",
            fault.label,
            fault.fault_id,
            _operator(request.operator_id),
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
            self._refused("ClearFault", request.operator_id, str(exc))
            return pb2.FaultAck(accepted=False, reason=str(exc), timestamp_ms=now_ms())
        logger.info(
            "Faults cleared by %s: %s",
            _operator(request.operator_id),
            ", ".join(f"{fault.label} ({fault.fault_id})" for fault in cleared)
            or "none",
        )
        now_s = self._runtime.snapshot.simulation_time_s
        return pb2.FaultAck(
            accepted=True,
            timestamp_ms=now_ms(),
            faults=[fault_to_proto(fault, now_s) for fault in cleared],
        )


def create_server(runtime: PhysicsRuntime) -> grpc.aio.Server:
    """The PhysicsService on a gRPC server with observability and keepalive; not bound."""
    server = grpc.aio.server(
        interceptors=[ServerObservability()], options=list(SERVER_OPTIONS)
    )
    pb2_grpc.add_PhysicsServiceServicer_to_server(PhysicsServicer(runtime), server)
    return server


async def serve(
    runtime: PhysicsRuntime,
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
) -> None:
    """Start the PhysicsService gRPC server and block until termination.

    The service authenticates no caller and drives the valves directly, so it listens on
    loopback unless told otherwise; only a private network (Compose) may widen it.
    """
    await runtime.start()
    server = create_server(runtime)
    listen_addr = f"{host}:{port}"
    server.add_insecure_port(listen_addr)
    await server.start()
    logger.info("Physics gRPC server listening on %s", listen_addr)
    try:
        await serve_until_cancelled(server)
    finally:
        await runtime.stop()
