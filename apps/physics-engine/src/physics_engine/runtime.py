"""
Live runtime of the physics engine: paces the deterministic plant against wall time.

The runtime owns one PlantSimulator. While running it advances one step every
`step_s / speed_factor` seconds of wall time; while paused it advances only when a step is
requested, which makes a run independent of machine speed. Every change of the plant — a
step, a scenario load, a fault — publishes a new snapshot to waiting readers.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum

from physics_engine.condenser import COOLING_WATER_TEMP_DESIGN
from physics_engine.faults import ActiveFault, FaultSpec
from physics_engine.metrics import RUNTIME_FAILURES, STEP_SECONDS, STEPS
from physics_engine.models import BoilerParameters, BoilerState, ControlInputs
from physics_engine.plant import PlantConfig, PlantSimulator, PlantSnapshot
from physics_engine.scenarios import ScenarioName
from physics_engine.turbine import TurbineParameters

logger = logging.getLogger(__name__)

# Limits on what an operator may request at run time; code may configure any speed.
SPEED_FACTOR_MIN: float = 0.1
SPEED_FACTOR_MAX: float = 50.0
MAX_STEPS_PER_REQUEST: int = 3600


class RunState(StrEnum):
    """Whether the runtime advances the plant on its own."""

    RUNNING = "running"
    PAUSED = "paused"


class RuntimeCommandError(ValueError):
    """A simulation control request the runtime refuses."""


class RuntimeUnavailableError(RuntimeError):
    """The runtime stopped or failed; no further snapshots will be published."""


@dataclass(frozen=True)
class SimulationStatus:
    """Where the simulation stands."""

    run_state: RunState
    speed_factor: float
    simulation_time_s: float
    step_count: int
    scenario: ScenarioName
    run_id: int
    step_s: float


@dataclass
class PhysicsRuntimeConfig:
    """Configuration for the live simulator runtime."""

    scenario: ScenarioName | str = ScenarioName.STEADY_STATE
    speed_factor: float = 1.0
    dt: float = 1.0
    boiler_params: BoilerParameters = field(default_factory=BoilerParameters)
    turbine_params: TurbineParameters = field(default_factory=TurbineParameters)
    cooling_water_temp_k: float = COOLING_WATER_TEMP_DESIGN
    initial_state: BoilerState | None = None
    initial_controls: ControlInputs | None = None
    start_paused: bool = False


class PhysicsRuntime:
    """Async-safe live simulator with pause, stepping, speed, scenarios and faults."""

    def __init__(self, config: PhysicsRuntimeConfig | None = None) -> None:
        self.config = config or PhysicsRuntimeConfig()
        if self.config.speed_factor <= 0:
            raise ValueError("speed_factor must be > 0")
        self._plant = PlantSimulator(
            PlantConfig(
                step_s=self.config.dt,
                cooling_water_temp_k=self.config.cooling_water_temp_k,
                boiler_params=self.config.boiler_params,
                turbine_params=self.config.turbine_params,
            ),
            self.config.scenario,
            initial_state=self.config.initial_state,
            initial_controls=self.config.initial_controls,
        )
        self._speed_factor = self.config.speed_factor
        self._run_state = (
            RunState.PAUSED if self.config.start_paused else RunState.RUNNING
        )
        self._resumed = asyncio.Event()
        if self._run_state is RunState.RUNNING:
            self._resumed.set()
        self._lock = asyncio.Lock()
        self._update_event = asyncio.Event()
        self._sequence = 0
        self._snapshot = self._plant.snapshot
        self._task: asyncio.Task[None] | None = None
        self._start_time = time.monotonic()
        self._status = "starting"
        self._last_error = ""

    # ─── Introspection ───────────────────────────────────────────────────────

    @property
    def uptime_seconds(self) -> float:
        """Process uptime from runtime construction."""
        return time.monotonic() - self._start_time

    @property
    def status(self) -> str:
        """Liveness: "running" while the loop is healthy, paused or not."""
        return self._status

    @property
    def last_error(self) -> str:
        return self._last_error

    @property
    def wall_step_s(self) -> float:
        """Wall-clock delay between physics steps while running."""
        return self.config.dt / self._speed_factor

    @property
    def snapshot(self) -> PlantSnapshot:
        """The latest published plant snapshot."""
        return self._snapshot

    def simulation_status(self) -> SimulationStatus:
        """Run state, speed, time and scenario of the simulation."""
        snapshot = self._snapshot
        return SimulationStatus(
            run_state=self._run_state,
            speed_factor=self._speed_factor,
            simulation_time_s=snapshot.simulation_time_s,
            step_count=snapshot.step_count,
            scenario=snapshot.scenario,
            run_id=snapshot.run_id,
            step_s=snapshot.step_s,
        )

    # ─── Lifecycle ───────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Start the background stepping task if it is not running yet."""
        if self._task and not self._task.done():
            return
        self._status = "running"
        self._task = asyncio.create_task(self._run_loop(), name="physics-runtime")

    async def stop(self) -> None:
        """Stop the background stepping task."""
        task = self._task
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            self._task = None
        self._status = "stopped"
        self._update_event.set()

    # ─── Readers ─────────────────────────────────────────────────────────────

    async def get_snapshot(self) -> PlantSnapshot:
        """The latest plant snapshot (immutable, safe to keep)."""
        return self._snapshot

    async def wait_for_update(self, last_sequence: int) -> tuple[int, PlantSnapshot]:
        """Wait until a snapshot newer than `last_sequence` is published."""
        while self._sequence <= last_sequence:
            event = self._update_event
            await event.wait()
            if self._sequence <= last_sequence and self._status != "running":
                raise RuntimeUnavailableError(
                    self._last_error or f"physics runtime {self._status}"
                )
        return self._sequence, self._snapshot

    # ─── Commands ────────────────────────────────────────────────────────────

    async def apply_command(
        self,
        *,
        fuel_valve: float,
        feedwater_valve: float,
        steam_valve: float,
        spray_valve: float | None = None,
    ) -> None:
        """Apply a validated valve command to the live plant."""
        async with self._lock:
            self._plant.apply_command(
                fuel_valve=fuel_valve,
                feedwater_valve=feedwater_valve,
                steam_valve=steam_valve,
                spray_valve=spray_valve,
            )

    async def pause(self) -> SimulationStatus:
        """Stop advancing the plant on the wall clock."""
        async with self._lock:
            self._run_state = RunState.PAUSED
            self._resumed.clear()
        logger.info("Simulation paused at t=%.1fs", self._snapshot.simulation_time_s)
        return self.simulation_status()

    async def resume(self) -> SimulationStatus:
        """Advance the plant on the wall clock again."""
        async with self._lock:
            self._run_state = RunState.RUNNING
            self._resumed.set()
        logger.info("Simulation resumed at t=%.1fs", self._snapshot.simulation_time_s)
        return self.simulation_status()

    async def set_speed(self, speed_factor: float) -> SimulationStatus:
        """Change how many simulated seconds pass per wall-clock second."""
        if not SPEED_FACTOR_MIN <= speed_factor <= SPEED_FACTOR_MAX:
            raise RuntimeCommandError(
                f"speed factor {speed_factor:g} outside "
                f"[{SPEED_FACTOR_MIN:g}, {SPEED_FACTOR_MAX:g}]"
            )
        self._speed_factor = speed_factor
        logger.info("Simulation speed set to %g×", speed_factor)
        return self.simulation_status()

    async def step(self, steps: int) -> SimulationStatus:
        """Advance a paused simulation by a number of steps, publishing each one.

        A request that is resumed while it runs ends early: the wall clock paces the
        plant again, and stepping on top of it would run faster than the chosen speed.
        """
        if not 1 <= steps <= MAX_STEPS_PER_REQUEST:
            raise RuntimeCommandError(
                f"steps={steps} outside [1, {MAX_STEPS_PER_REQUEST}]"
            )
        self._refuse_unless_healthy()
        if self._run_state is not RunState.PAUSED:
            raise RuntimeCommandError("pause the simulation before stepping it")
        for _ in range(steps):
            async with self._lock:
                if self._run_state is not RunState.PAUSED or self._status != "running":
                    break
                try:
                    await self._publish_from_worker(self._timed_step)
                except Exception as exc:
                    self._degrade(exc, "Physics step failed")
                    raise
        return self.simulation_status()

    async def load_scenario(self, scenario: ScenarioName | str) -> SimulationStatus:
        """Reset the plant into a scenario; the run id changes."""
        async with self._lock:
            await self._publish_from_worker(lambda: self._plant.load_scenario(scenario))
        return self.simulation_status()

    async def inject_fault(self, spec: FaultSpec) -> ActiveFault:
        """Activate a fault in the live plant."""
        async with self._lock:
            fault = self._plant.inject_fault(spec)
            self._publish(self._plant.snapshot)
        return fault

    async def clear_fault(self, fault_id: str) -> ActiveFault:
        """Deactivate one fault."""
        async with self._lock:
            fault = self._plant.clear_fault(fault_id)
            self._publish(self._plant.snapshot)
        return fault

    async def clear_faults(self) -> tuple[ActiveFault, ...]:
        """Deactivate every fault."""
        async with self._lock:
            cleared = self._plant.clear_faults()
            self._publish(self._plant.snapshot)
        return cleared

    # ─── Loop ────────────────────────────────────────────────────────────────

    def _timed_step(self) -> PlantSnapshot:
        started = time.perf_counter()
        snapshot = self._plant.step(1)
        STEP_SECONDS.observe(time.perf_counter() - started)
        STEPS.inc()
        return snapshot

    async def _publish_from_worker(self, work: Callable[[], PlantSnapshot]) -> None:
        """Run `work` on a worker thread and publish its snapshot; the lock is held.

        A thread cannot be cancelled. If the caller is, the plant is still being changed,
        so the lock stays held until the worker returns, and what it did is published
        before the cancellation goes on.
        """
        worker = asyncio.ensure_future(asyncio.to_thread(work))
        try:
            snapshot = await asyncio.shield(worker)
        except asyncio.CancelledError:
            await self._outlast(worker)
            raise
        self._publish(snapshot)

    async def _outlast(self, worker: asyncio.Future[PlantSnapshot]) -> None:
        while not worker.done():
            try:
                await asyncio.wait({worker})
            except asyncio.CancelledError:
                continue
        if worker.cancelled():
            return
        error = worker.exception()
        if error is None:
            self._publish(worker.result())
        else:
            logger.warning(
                "Physics work failed after its caller was cancelled", exc_info=error
            )

    def _refuse_unless_healthy(self) -> None:
        if self._status != "running":
            detail = f": {self._last_error}" if self._last_error else ""
            raise RuntimeCommandError(f"physics runtime {self._status}{detail}")

    def _degrade(self, error: BaseException, what: str) -> None:
        """A failed step leaves the plant where the model broke: nothing steps it again."""
        self._status = "degraded"
        self._last_error = str(error)
        RUNTIME_FAILURES.inc()
        logger.error("%s: %s", what, error, exc_info=error)
        self._update_event.set()

    def _publish(self, snapshot: PlantSnapshot) -> None:
        self._snapshot = snapshot
        self._sequence += 1
        event, self._update_event = self._update_event, asyncio.Event()
        event.set()

    async def _run_loop(self) -> None:
        """Advance the plant on the wall clock while running."""
        try:
            while True:
                await self._resumed.wait()
                wall_start = time.perf_counter()
                async with self._lock:
                    if self._status != "running":
                        return
                    if self._run_state is not RunState.RUNNING:
                        continue
                    await self._publish_from_worker(self._timed_step)
                elapsed = time.perf_counter() - wall_start
                await asyncio.sleep(max(self.wall_step_s - elapsed, 0.0))
        except Exception as exc:
            self._degrade(exc, "Physics runtime loop failed")
