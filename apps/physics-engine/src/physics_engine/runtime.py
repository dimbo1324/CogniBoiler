"""
Live process runtime for the Physics Engine.

Runs the coupled boiler-turbine model as a continuously stepping simulator,
keeps the latest system state in memory, and accepts control commands from
upstream services such as the PLC controller.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field, replace

from physics_engine import steam_tables
from physics_engine.async_simulator import ScenarioName
from physics_engine.constants import DRUM_CROSS_SECTION, DRUM_HEIGHT, TEMP_AMBIENT
from physics_engine.models import BoilerParameters, BoilerState, ControlInputs
from physics_engine.system import BoilerTurbineSystem, SystemState
from physics_engine.turbine import TurbineParameters

logger = logging.getLogger(__name__)


@dataclass
class PhysicsRuntimeConfig:
    """Configuration for the live simulator runtime."""

    scenario: ScenarioName = ScenarioName.STEADY_STATE
    speed_factor: float = 1.0
    dt: float = 1.0
    boiler_params: BoilerParameters = field(default_factory=BoilerParameters)
    turbine_params: TurbineParameters = field(default_factory=TurbineParameters)
    initial_state: BoilerState | None = None
    initial_controls: ControlInputs | None = None


class PhysicsRuntime:
    """
    In-memory live simulator with async-safe command/state access.

    The runtime is command-driven: it starts from a selected startup profile
    and continuously advances the physical model. PLC commands update the
    control inputs that are applied on subsequent time steps.
    """

    def __init__(self, config: PhysicsRuntimeConfig | None = None) -> None:
        self.config = config or PhysicsRuntimeConfig()
        if self.config.speed_factor <= 0:
            raise ValueError("speed_factor must be > 0")
        self._system = BoilerTurbineSystem(
            boiler_params=self.config.boiler_params,
            turbine_params=self.config.turbine_params,
        )
        self._boiler_state = self._initial_boiler_state(self.config.scenario)
        self._controls = self._initial_controls(self.config.scenario)
        self._system_state = self._system.evaluate_at(
            self._boiler_state,
            self._controls,
            time=0.0,
        )
        self._sim_time = 0.0
        self._state_sequence = 0
        self._lock = asyncio.Lock()
        self._update_event = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._start_time = time.monotonic()
        self._status = "starting"
        self._last_error = ""

    @property
    def uptime_seconds(self) -> float:
        """Process uptime from runtime construction."""
        return time.monotonic() - self._start_time

    @property
    def status(self) -> str:
        """Current runtime status string."""
        return self._status

    @property
    def wall_step_s(self) -> float:
        """Wall-clock delay between physics steps."""
        return self.config.dt / self.config.speed_factor

    async def start(self) -> None:
        """Start the background stepping task if it is not running yet."""
        if self._task and not self._task.done():
            return
        self._status = "running"
        self._task = asyncio.create_task(self._run_loop(), name="physics-runtime")

    async def stop(self) -> None:
        """Stop the background stepping task."""
        task = self._task
        if task is None:
            self._status = "stopped"
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        self._task = None
        self._status = "stopped"
        self._update_event.set()

    async def get_system_state(self) -> SystemState:
        """Return a copy of the latest simulator snapshot."""
        async with self._lock:
            return self._clone_system_state(self._system_state)

    async def get_controls(self) -> ControlInputs:
        """Return a copy of the latest command/actuator state."""
        async with self._lock:
            return self._clone_controls(self._controls)

    async def get_snapshot(self) -> tuple[SystemState, ControlInputs]:
        """Return a consistent copy of state and actuator snapshot."""
        async with self._lock:
            return (
                self._clone_system_state(self._system_state),
                self._clone_controls(self._controls),
            )

    async def wait_for_update(self, last_sequence: int) -> tuple[int, SystemState]:
        """Wait until a newer state snapshot becomes available."""
        while True:
            async with self._lock:
                if self._state_sequence > last_sequence:
                    return self._state_sequence, self._clone_system_state(
                        self._system_state
                    )
                event = self._update_event
            await event.wait()

    async def apply_command(
        self,
        *,
        fuel_valve: float,
        feedwater_valve: float,
        steam_valve: float,
    ) -> None:
        """Apply a validated command to the live simulator."""
        for name, value in (
            ("fuel_valve", fuel_valve),
            ("feedwater_valve", feedwater_valve),
            ("steam_valve", steam_valve),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name}={value:.3f} outside [0.0, 1.0]")

        async with self._lock:
            self._controls.fuel_valve_command = fuel_valve
            self._controls.feedwater_valve_command = feedwater_valve
            self._controls.steam_valve_command = steam_valve

    async def _run_loop(self) -> None:
        """Continuously advance the physical model in the background."""
        try:
            while True:
                wall_start = time.perf_counter()

                async with self._lock:
                    boiler_state = replace(self._boiler_state)
                    controls = self._clone_controls(self._controls)
                    sim_time = self._sim_time

                next_state, stepped_controls = await asyncio.to_thread(
                    self._step_sync,
                    boiler_state,
                    controls,
                    sim_time,
                )

                async with self._lock:
                    self._boiler_state = next_state
                    self._sim_time = sim_time + self.config.dt
                    self._system_state = self._system.evaluate_at(
                        next_state,
                        stepped_controls,
                        time=self._sim_time,
                    )
                    self._controls.fuel_valve.position = (
                        stepped_controls.fuel_valve.position
                    )
                    self._controls.feedwater_valve.position = (
                        stepped_controls.feedwater_valve.position
                    )
                    self._controls.steam_valve.position = (
                        stepped_controls.steam_valve.position
                    )
                    event = self._update_event
                    self._update_event = asyncio.Event()
                    self._state_sequence += 1
                    self._status = "running"
                    self._last_error = ""

                event.set()

                elapsed = time.perf_counter() - wall_start
                sleep_for = self.wall_step_s - elapsed
                if sleep_for > 0:
                    await asyncio.sleep(sleep_for)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._status = "degraded"
            self._last_error = str(exc)
            logger.exception("Physics runtime loop failed: %s", exc)
            self._update_event.set()

    def _step_sync(
        self,
        state: BoilerState,
        controls: ControlInputs,
        sim_time: float,
    ) -> tuple[BoilerState, ControlInputs]:
        """Advance the boiler model by one fixed step."""
        result = self._system.boiler.simulate(
            state,
            controls,
            t_span=(sim_time, sim_time + self.config.dt + 1e-9),
            dt=self.config.dt,
        )
        if result.y.shape[1] == 0:
            raise RuntimeError("Physics runtime produced no state samples.")
        next_state = self._system.boiler.get_state_at(result, -1)
        return next_state, controls

    def _clone_controls(self, controls: ControlInputs) -> ControlInputs:
        """Copy commands and valve positions for off-thread stepping."""
        return controls.copy()

    def _clone_system_state(self, state: SystemState) -> SystemState:
        """Deep-copy the current state snapshot for callers."""
        return SystemState(
            boiler=replace(state.boiler),
            turbine=replace(state.turbine),
            time=state.time,
        )

    def _initial_boiler_state(self, scenario: ScenarioName) -> BoilerState:
        """Choose the startup state for the live runtime."""
        if self.config.initial_state is not None:
            return replace(self.config.initial_state)
        if scenario != ScenarioName.COLD_START:
            return self.config.boiler_params.nominal_initial_state()

        cold_pressure = 2.0e5
        cold_water_temp = 373.15
        cold_water_level = DRUM_HEIGHT * 0.6
        cold_water_density = steam_tables.water_density(
            temp_k=cold_water_temp,
            pressure_pa=cold_pressure,
        )
        cold_water_mass = cold_water_density * DRUM_CROSS_SECTION * cold_water_level
        cold_internal_energy = cold_water_mass * steam_tables.water_enthalpy(
            cold_water_temp,
            cold_pressure,
        )
        return BoilerState(
            internal_energy=cold_internal_energy,
            pressure=cold_pressure,
            water_level=cold_water_level,
            flue_gas_temp=TEMP_AMBIENT,
            water_temp=cold_water_temp,
        )

    def _initial_controls(self, scenario: ScenarioName) -> ControlInputs:
        """Choose the startup valve commands for the live runtime."""
        if self.config.initial_controls is not None:
            return self.config.initial_controls.copy()
        if scenario == ScenarioName.FUEL_TRIP:
            return ControlInputs(
                fuel_valve_command=0.0,
                feedwater_valve_command=0.5,
                steam_valve_command=0.5,
            )
        if scenario == ScenarioName.COLD_START:
            return ControlInputs(
                fuel_valve_command=0.0,
                feedwater_valve_command=0.2,
                steam_valve_command=0.0,
            )
        return ControlInputs(
            fuel_valve_command=0.5,
            feedwater_valve_command=0.5,
            steam_valve_command=0.5,
        )
