"""
Async simulator for real-time boiler-turbine telemetry.

Runs the physics simulation in a background OS thread (ThreadPoolExecutor)
so the asyncio event loop stays responsive while heavy ODE integration
executes concurrently.

Architecture:
    ┌─────────────────────────┐      asyncio.Queue      ┌──────────────────┐
    │  ThreadPoolExecutor     │  ─── SystemState ──►    │  Event loop      │
    │  ScenarioRunner._run()  │                         │  MQTT Publisher  │
    └─────────────────────────┘                         └──────────────────┘

Speed factors:
    speed_factor = 1   → real-time (1 sim-second = 1 wall-second)
    speed_factor = 10  → 10× faster (used for testing)
    speed_factor = 100 → 100× faster (used for dataset generation)

Usage:
    sim = AsyncSimulator(SimulatorConfig(scenario="cold_start", speed_factor=1))
    async for state in sim.run():
        await mqtt_publisher.publish_boiler(client, state.boiler)
        await mqtt_publisher.publish_turbine(client, state.turbine)
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncGenerator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum

from physics_engine.models import BoilerParameters
from physics_engine.scenarios import ScenarioResult, ScenarioRunner
from physics_engine.system import BoilerTurbineSystem, SystemState
from physics_engine.turbine import TurbineParameters

logger = logging.getLogger(__name__)


# ─── Scenario names ───────────────────────────────────────────────────────────


class ScenarioName(str, Enum):
    """Available simulation scenarios."""

    STEADY_STATE = "steady_state"
    LOAD_RAMP = "load_ramp"
    FUEL_TRIP = "fuel_trip"
    COLD_START = "cold_start"


# ─── Simulator configuration ─────────────────────────────────────────────────


@dataclass
class SimulatorConfig:
    """
    Configuration for AsyncSimulator.

    Attributes:
        scenario:      Which scenario to run (see ScenarioName).
        speed_factor:  Simulation speed multiplier vs wall clock.
                       1 = real-time, 10 = 10× faster, 100 = batch mode.
        dt:            Physics time step [s]. Smaller = more accurate, slower.
        duration:      Total scenario duration [s].
        queue_maxsize: Maximum buffered SystemState objects in the queue.
                       If the consumer is slow, older states are dropped
                       to prevent unbounded memory growth.
        boiler_params: Optional custom boiler parameters.
        turbine_params: Optional custom turbine parameters.
    """

    scenario: ScenarioName = ScenarioName.STEADY_STATE
    speed_factor: float = 1.0
    dt: float = 1.0
    duration: float = 600.0
    queue_maxsize: int = 100

    boiler_params: BoilerParameters = field(default_factory=BoilerParameters)
    turbine_params: TurbineParameters = field(default_factory=TurbineParameters)


# ─── Simulator ────────────────────────────────────────────────────────────────


class AsyncSimulator:
    """
    Async wrapper around ScenarioRunner for real-time telemetry streaming.

    Runs the blocking physics simulation in a ThreadPoolExecutor and
    streams SystemState updates via an asyncio.Queue.

    The simulator is single-use: call run() once per instance.
    Create a new AsyncSimulator to restart a scenario.

    Usage:
        config = SimulatorConfig(
            scenario=ScenarioName.COLD_START,
            speed_factor=1.0,
            duration=3600.0,
        )
        sim = AsyncSimulator(config)

        async for state in sim.run():
            print(f"t={state.time:.0f}s  P={state.boiler.pressure_bar:.1f} bar"
                  f"  W={state.electrical_power_mw:.1f} MW")
    """

    def __init__(self, config: SimulatorConfig | None = None) -> None:
        self.config = config or SimulatorConfig()
        self._queue: asyncio.Queue[SystemState | None] = asyncio.Queue(
            maxsize=self.config.queue_maxsize
        )
        self._published: int = 0
        self._dropped: int = 0

    # ─── Stats ───────────────────────────────────────────────────────────────

    @property
    def published(self) -> int:
        """Total SystemState objects delivered to the consumer."""
        return self._published

    @property
    def dropped(self) -> int:
        """Total SystemState objects dropped due to full queue."""
        return self._dropped

    # ─── Internal: physics thread ─────────────────────────────────────────────

    def _build_scenario_result(self) -> ScenarioResult:
        """
        Run the selected scenario synchronously in a worker thread.

        This method executes entirely in a ThreadPoolExecutor — it must
        not call any asyncio primitives directly.

        Returns:
            ScenarioResult with the full time series.
        """
        runner = ScenarioRunner(
            boiler_params=self.config.boiler_params,
            turbine_params=self.config.turbine_params,
        )

        cfg = self.config

        if cfg.scenario == ScenarioName.STEADY_STATE:
            return runner.steady_state(duration=cfg.duration, dt=cfg.dt)

        if cfg.scenario == ScenarioName.LOAD_RAMP:
            return runner.load_ramp(duration=cfg.duration, dt=cfg.dt)

        if cfg.scenario == ScenarioName.FUEL_TRIP:
            return runner.fuel_trip(
                t_trip=cfg.duration * 0.4,
                duration=cfg.duration,
                dt=cfg.dt,
            )

        if cfg.scenario == ScenarioName.COLD_START:
            return runner.cold_start(duration=cfg.duration, dt=cfg.dt)

        raise ValueError(f"Unknown scenario: {cfg.scenario}")

    def _stream_to_queue(
        self,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        """
        Run simulation and push each time step into the asyncio queue.

        Runs in a ThreadPoolExecutor thread. Uses run_coroutine_threadsafe
        to safely schedule puts on the event loop from the worker thread.

        Enforces the speed_factor by sleeping between steps when running
        slower than real-time (speed_factor ≤ 1).

        Args:
            loop: The running event loop (captured before entering executor).
        """
        logger.info(
            "Simulator starting scenario=%s duration=%.0fs dt=%.1fs speed=%.0f×",
            self.config.scenario.value,
            self.config.duration,
            self.config.dt,
            self.config.speed_factor,
        )

        result = self._build_scenario_result()

        # Build a BoilerTurbineSystem to convert ScenarioResult rows → SystemState
        system = BoilerTurbineSystem(
            boiler_params=self.config.boiler_params,
            turbine_params=self.config.turbine_params,
        )

        wall_step = self.config.dt / self.config.speed_factor

        for i in range(result.n_steps):
            wall_start = time.perf_counter()

            # Reconstruct SystemState from ScenarioResult arrays
            from physics_engine.models import BoilerState, ControlInputs

            boiler_state = BoilerState(
                internal_energy=float(result.internal_energy[i]),
                pressure=float(result.pressure[i]),
                water_level=float(result.water_level[i]),
                flue_gas_temp=float(result.flue_gas_temp[i]),
                water_temp=float(result.water_temp[i]),
            )

            controls = ControlInputs(
                fuel_valve_command=float(result.fuel_valve[i]),
                feedwater_valve_command=float(result.feedwater_valve[i]),
                steam_valve_command=float(result.steam_valve[i]),
            )

            state = system.evaluate_at(
                boiler_state=boiler_state,
                controls=controls,
                time=float(result.time[i]),
            )

            # Push to queue (non-blocking: drop if full to avoid stalling)
            try:
                future = asyncio.run_coroutine_threadsafe(self._queue.put(state), loop)
                future.result(timeout=0.1)
                self._published += 1
            except Exception:
                self._dropped += 1

            # Pace to speed_factor — sleep remainder of wall_step
            elapsed = time.perf_counter() - wall_start
            sleep_for = wall_step - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)

        # Signal end of stream
        asyncio.run_coroutine_threadsafe(self._queue.put(None), loop).result(
            timeout=1.0
        )

        logger.info(
            "Simulator finished: published=%d dropped=%d",
            self._published,
            self._dropped,
        )

    # ─── Public: async generator ──────────────────────────────────────────────

    async def run(self) -> AsyncGenerator[SystemState, None]:
        """
        Start the simulation and yield SystemState objects in real time.

        Launches the physics thread and yields states from the queue
        as they arrive. The generator exits when the scenario ends.

        Yields:
            SystemState for each simulation time step.

        Example:
            sim = AsyncSimulator(SimulatorConfig(scenario=ScenarioName.COLD_START))
            async for state in sim.run():
                print(state.boiler.pressure_bar, "bar")
        """
        loop = asyncio.get_running_loop()

        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="simulator") as pool:
            # Launch physics in background thread
            future = loop.run_in_executor(pool, self._stream_to_queue, loop)

            # Yield states as they arrive
            while True:
                state = await self._queue.get()
                if state is None:
                    break
                yield state

            # Ensure the worker thread completed without exception
            await future
