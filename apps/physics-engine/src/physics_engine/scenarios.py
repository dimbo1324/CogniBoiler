"""
Pre-built simulation scenarios for the boiler-turbine system.

Each scenario runs a complete closed-loop simulation:
    BoilerModel + BoilerController + TurbineModel

Scenarios are used for:
    - Verifying controller behavior under known conditions
    - Generating training data for Phase 6 (PyTorch LSTM)
    - Demonstrating system response for portfolio

Available scenarios:
    steady_state   — hold nominal operating point for N seconds
    load_ramp      — increase steam output from 50% to 80% over ramp_time
    fuel_trip      — sudden fuel cutoff (emergency shutdown simulation)
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from physics_engine.boiler import BoilerModel
from physics_engine.combustion import CombustionModel
from physics_engine.controller import BoilerController, BoilerSetpoints
from physics_engine.models import BoilerParameters, BoilerState, ControlInputs
from physics_engine.turbine import TurbineModel, TurbineParameters

# ─── Type aliases (for readability in _run signature) ─────────────────────────

ScenarioSetpointFn = Callable[[float], BoilerSetpoints]
SteamValveFn = Callable[[float], float | None]

# ─── Scenario result ──────────────────────────────────────────────────────────


@dataclass
class ScenarioResult:
    """
    Time-series output of a closed-loop scenario simulation.

    All arrays have the same length N (number of time steps).
    Used directly for visualization and ML training data export.
    """

    # Time axis
    time: np.ndarray  # s      — simulation time

    # Boiler state
    pressure: np.ndarray  # Pa     — drum pressure
    water_level: np.ndarray  # m      — drum water level
    water_temp: np.ndarray  # K      — drum water temperature
    flue_gas_temp: np.ndarray  # K      — furnace flue gas temperature
    internal_energy: np.ndarray  # J      — drum internal energy

    # Turbine output
    electrical_power: np.ndarray  # W   — generator electrical output

    # Controller outputs (valve commands)
    fuel_valve: np.ndarray  # —   — fuel valve position
    feedwater_valve: np.ndarray  # —   — feedwater valve position
    steam_valve: np.ndarray  # —   — steam valve position

    # Controller internals (for diagnostics)
    pressure_error: np.ndarray  # Pa  — pressure setpoint error
    level_error: np.ndarray  # m   — level setpoint error

    # Scenario metadata
    scenario_name: str = ""
    dt: float = 1.0  # s   — time step used

    @property
    def n_steps(self) -> int:
        """Number of time steps in the result."""
        return len(self.time)

    @property
    def electrical_power_mw(self) -> np.ndarray:
        """Electrical power in MW (convenience property)."""
        return self.electrical_power / 1.0e6

    @property
    def pressure_bar(self) -> np.ndarray:
        """Pressure in bar (convenience property)."""
        return self.pressure / 1.0e5


# ─── Scenario runner ──────────────────────────────────────────────────────────


class ScenarioRunner:
    """
    Closed-loop simulation runner.

    Integrates BoilerModel step-by-step with BoilerController feedback,
    recording the full time series for analysis and export.

    Unlike BoilerModel.simulate() which uses solve_ivp (adaptive step),
    ScenarioRunner steps the ODE one fixed dt at a time so the controller
    can update its output at each step — proper discrete-time control loop.

    Usage:
        runner = ScenarioRunner()
        result = runner.steady_state(duration=600)
        result = runner.load_ramp(steam_valve_start=0.4, steam_valve_end=0.7)
        result = runner.fuel_trip(t_trip=120, duration=300)
    """

    def __init__(
        self,
        boiler_params: BoilerParameters | None = None,
        turbine_params: TurbineParameters | None = None,
    ) -> None:
        self.boiler_params = boiler_params or BoilerParameters()
        self.turbine_params = turbine_params or TurbineParameters()
        self.boiler = BoilerModel(self.boiler_params)
        self.turbine = TurbineModel(self.turbine_params)

    def _run(
        self,
        scenario_name: str,
        initial_state: BoilerState,
        controller: BoilerController,
        setpoints_fn: ScenarioSetpointFn,
        steam_valve_fn: SteamValveFn,
        duration: float,
        dt: float = 1.0,
    ) -> ScenarioResult:
        """
        Core simulation loop.

        Steps the boiler ODE one dt at a time, calling the controller
        at each step to update valve commands.

        Args:
            scenario_name:   Label for the result.
            initial_state:   Starting boiler state.
            controller:      BoilerController instance (pre-configured).
            setpoints_fn:    Callable(t) -> BoilerSetpoints — time-varying setpoints.
            steam_valve_fn:  Callable(t) -> float | None — steam valve override or None.
            duration:        Total simulation time [s].
            dt:              Fixed time step [s].

        Returns:
            ScenarioResult with full time series.
        """
        n = int(duration / dt)
        times = np.zeros(n)
        pressures = np.zeros(n)
        levels = np.zeros(n)
        water_temps = np.zeros(n)
        flue_temps = np.zeros(n)
        energies = np.zeros(n)
        powers = np.zeros(n)
        fuel_valves = np.zeros(n)
        feedwater_valves = np.zeros(n)
        steam_valves = np.zeros(n)
        pressure_errors = np.zeros(n)
        level_errors = np.zeros(n)

        state = initial_state

        # Combustion model for fuel flow measurement (inner loop feedback)
        combustion = CombustionModel(
            max_fuel_flow=self.boiler_params.max_fuel_flow,
        )

        # Initial valve positions
        fuel_cmd = 0.5
        feedwater_cmd = 0.5
        steam_cmd = 0.5

        for i in range(n):
            t = i * dt
            times[i] = t

            # ── Record current state ──────────────────────────────────────────
            pressures[i] = state.pressure
            levels[i] = state.water_level
            water_temps[i] = state.water_temp
            flue_temps[i] = state.flue_gas_temp
            energies[i] = state.internal_energy
            fuel_valves[i] = fuel_cmd
            feedwater_valves[i] = feedwater_cmd
            steam_valves[i] = steam_cmd

            # ── Turbine power at current state ────────────────────────────────
            turbine_state = self.turbine.calculate(
                steam_temp_in=state.water_temp,
                steam_pressure_in=state.pressure,
                steam_flow=self.boiler._steam_flow(state.pressure, steam_cmd),
            )
            powers[i] = turbine_state.electrical_power

            # ── Controller step ───────────────────────────────────────────────
            setpoints = setpoints_fn(t)
            feedwater_flow = self.boiler._feedwater_flow(feedwater_cmd)
            comb = combustion.calculate(fuel_valve=fuel_cmd)

            ctrl_output = controller.step(
                setpoints=setpoints,
                pressure=state.pressure,
                water_level=state.water_level,
                steam_temp=state.water_temp,
                fuel_flow=comb.fuel_flow,
                feedwater_flow=feedwater_flow,
                dt=dt,
            )

            pressure_errors[i] = ctrl_output.pressure_error
            level_errors[i] = ctrl_output.level_error

            # ── Apply steam valve override if provided ────────────────────────
            sv = steam_valve_fn(t)
            fuel_cmd = ctrl_output.fuel_valve
            feedwater_cmd = ctrl_output.feedwater_valve
            steam_cmd = sv if sv is not None else ctrl_output.steam_valve

            # ── Advance boiler ODE by one dt ──────────────────────────────────
            controls = ControlInputs(
                fuel_valve_command=fuel_cmd,
                feedwater_valve_command=feedwater_cmd,
                steam_valve_command=steam_cmd,
            )
            # t_span end slightly past t+dt so t_eval = [t, t+dt] (two points, not one)
            sim_result = self.boiler.simulate(
                state, controls, t_span=(t, t + dt + 1e-9), dt=dt
            )
            if sim_result.y.shape[1] > 0:
                state = self.boiler.get_state_at(sim_result, -1)

        return ScenarioResult(
            time=times,
            pressure=pressures,
            water_level=levels,
            water_temp=water_temps,
            flue_gas_temp=flue_temps,
            internal_energy=energies,
            electrical_power=powers,
            fuel_valve=fuel_valves,
            feedwater_valve=feedwater_valves,
            steam_valve=steam_valves,
            pressure_error=pressure_errors,
            level_error=level_errors,
            scenario_name=scenario_name,
            dt=dt,
        )

    # ─── Public scenarios ─────────────────────────────────────────────────────

    def steady_state(
        self,
        duration: float = 600.0,
        dt: float = 1.0,
        pressure_sp: float = 140.0e5,
        level_sp: float = 4.8,
    ) -> ScenarioResult:
        """
        Hold nominal operating point with closed-loop control.

        Controller drives boiler to setpoints and maintains them.
        Useful for verifying controller stability and steady-state error.

        Args:
            duration:    Simulation duration [s]. Default 600 s (10 min).
            dt:          Time step [s].
            pressure_sp: Pressure setpoint [Pa].
            level_sp:    Water level setpoint [m].
        """
        initial_state = self.boiler_params.nominal_initial_state()
        controller = BoilerController()

        def setpoints_fn(t: float) -> BoilerSetpoints:
            return BoilerSetpoints(pressure=pressure_sp, water_level=level_sp)

        def steam_valve_fn(t: float) -> float | None:
            return 0.5  # fixed steam output for steady-state test

        return self._run(
            scenario_name="steady_state",
            initial_state=initial_state,
            controller=controller,
            setpoints_fn=setpoints_fn,
            steam_valve_fn=steam_valve_fn,
            duration=duration,
            dt=dt,
        )

    def load_ramp(
        self,
        steam_valve_start: float = 0.4,
        steam_valve_end: float = 0.7,
        ramp_start: float = 60.0,
        ramp_duration: float = 120.0,
        duration: float = 600.0,
        dt: float = 1.0,
    ) -> ScenarioResult:
        """
        Ramp steam output from start to end position over ramp_duration seconds.

        Controller maintains pressure and level while steam demand increases.
        Simulates a load increase request from the grid dispatcher.

        Args:
            steam_valve_start: Initial steam valve position [0, 1].
            steam_valve_end:   Final steam valve position [0, 1].
            ramp_start:        Time when ramp begins [s].
            ramp_duration:     Duration of the ramp [s].
            duration:          Total simulation time [s].
            dt:                Time step [s].
        """
        initial_state = self.boiler_params.nominal_initial_state()
        controller = BoilerController()

        def setpoints_fn(t: float) -> BoilerSetpoints:
            return BoilerSetpoints(pressure=140.0e5, water_level=4.8)

        def steam_valve_fn(t: float) -> float | None:
            if t < ramp_start:
                return steam_valve_start
            elif t < ramp_start + ramp_duration:
                alpha = (t - ramp_start) / ramp_duration
                return steam_valve_start + alpha * (steam_valve_end - steam_valve_start)
            else:
                return steam_valve_end

        return self._run(
            scenario_name="load_ramp",
            initial_state=initial_state,
            controller=controller,
            setpoints_fn=setpoints_fn,
            steam_valve_fn=steam_valve_fn,
            duration=duration,
            dt=dt,
        )

    def fuel_trip(
        self,
        t_trip: float = 120.0,
        duration: float = 300.0,
        dt: float = 1.0,
    ) -> ScenarioResult:
        """
        Sudden fuel cutoff at t_trip seconds.

        Simulates emergency fuel trip: boiler loses heat source,
        pressure and temperature drop, controller responds by
        closing steam valve to preserve pressure.

        The fuel valve is forced to zero after t_trip regardless of
        controller output — all other loops (feedwater, steam) remain
        under automatic control.

        Args:
            t_trip:   Time of fuel trip [s].
            duration: Total simulation time [s].
            dt:       Time step [s].
        """
        initial_state = self.boiler_params.nominal_initial_state()
        controller_trip = BoilerController()

        n = int(duration / dt)
        times = np.zeros(n)
        pressures = np.zeros(n)
        levels = np.zeros(n)
        water_temps = np.zeros(n)
        flue_temps = np.zeros(n)
        energies = np.zeros(n)
        powers = np.zeros(n)
        fuel_valves = np.zeros(n)
        feedwater_valves = np.zeros(n)
        steam_valves = np.zeros(n)
        pressure_errors = np.zeros(n)
        level_errors = np.zeros(n)

        state = initial_state
        combustion = CombustionModel(
            max_fuel_flow=self.boiler_params.max_fuel_flow,
        )
        fuel_cmd = 0.5
        feedwater_cmd = 0.5
        steam_cmd = 0.5

        for i in range(n):
            t = i * dt
            times[i] = t

            pressures[i] = state.pressure
            levels[i] = state.water_level
            water_temps[i] = state.water_temp
            flue_temps[i] = state.flue_gas_temp
            energies[i] = state.internal_energy
            fuel_valves[i] = fuel_cmd
            feedwater_valves[i] = feedwater_cmd
            steam_valves[i] = steam_cmd

            turbine_state = self.turbine.calculate(
                steam_temp_in=state.water_temp,
                steam_pressure_in=state.pressure,
                steam_flow=self.boiler._steam_flow(state.pressure, steam_cmd),
            )
            powers[i] = turbine_state.electrical_power

            setpoints = BoilerSetpoints(pressure=140.0e5, water_level=4.8)
            feedwater_flow = self.boiler._feedwater_flow(feedwater_cmd)
            comb = combustion.calculate(fuel_valve=fuel_cmd)

            ctrl_output = controller_trip.step(
                setpoints=setpoints,
                pressure=state.pressure,
                water_level=state.water_level,
                steam_temp=state.water_temp,
                fuel_flow=comb.fuel_flow,
                feedwater_flow=feedwater_flow,
                dt=dt,
            )

            pressure_errors[i] = ctrl_output.pressure_error
            level_errors[i] = ctrl_output.level_error

            # ── Fuel trip: override fuel valve to zero after t_trip ───────────
            fuel_cmd = 0.0 if t >= t_trip else ctrl_output.fuel_valve
            feedwater_cmd = ctrl_output.feedwater_valve
            steam_cmd = ctrl_output.steam_valve

            controls = ControlInputs(
                fuel_valve_command=fuel_cmd,
                feedwater_valve_command=feedwater_cmd,
                steam_valve_command=steam_cmd,
            )
            # t_span end slightly past t+dt so t_eval = [t, t+dt] (two points, not one)
            sim_result = self.boiler.simulate(
                state, controls, t_span=(t, t + dt + 1e-9), dt=dt
            )
            if sim_result.y.shape[1] > 0:
                state = self.boiler.get_state_at(sim_result, -1)

        return ScenarioResult(
            time=times,
            pressure=pressures,
            water_level=levels,
            water_temp=water_temps,
            flue_gas_temp=flue_temps,
            internal_energy=energies,
            electrical_power=powers,
            fuel_valve=fuel_valves,
            feedwater_valve=feedwater_valves,
            steam_valve=steam_valves,
            pressure_error=pressure_errors,
            level_error=level_errors,
            scenario_name="fuel_trip",
            dt=dt,
        )

    def cold_start(
        self,
        duration: float = 3600.0,
        dt: float = 1.0,
    ) -> ScenarioResult:
        """
        Cold start from ambient conditions to nominal operating point.

        Three phases:
            Phase 1 — Warmup    (0 → 20% of duration):
                Fuel valve ramps 0 → 0.3. Steam valve closed.
                Pressure builds from 2 bar toward 40 bar.

            Phase 2 — Pressure buildup (20% → 60% of duration):
                Fuel valve ramps 0.3 → 0.7. Steam valve opens slightly.
                Controller takes over pressure and level loops.

            Phase 3 — Load acceptance (60% → 100% of duration):
                Controller maintains 140 bar / 4.8 m.
                Steam valve opens to nominal 0.5.

        Starting conditions:
            - Pressure:    2 bar  (steam-tight drum, not vacuum)
            - Water level: nominal (drum pre-filled before ignition)
            - Water temp:  100°C  (just above ambient — cold metal)
            - Flue gas:    ambient temperature (furnace not ignited yet)
            - All valves:  closed / minimum

        Args:
            duration: Total cold start duration [s]. Default 3600 s (1 hour).
            dt:       Time step [s].
        """
        from physics_engine import steam_tables
        from physics_engine.constants import (
            DRUM_CROSS_SECTION,
            DRUM_HEIGHT,
            TEMP_AMBIENT,
        )

        # ── Cold initial state ────────────────────────────────────────────────
        # Pressure: 2 bar — drum is sealed but not yet at steam pressure
        # Water level: 60% of drum height (pre-filled with cold water)
        # Water temp: 100°C — just above ambient, cold metal
        # Flue gas: ambient — furnace not yet ignited

        cold_pressure = 2.0e5  # Pa — 2 bar
        cold_water_temp = 373.15  # K — 100°C
        cold_water_level = DRUM_HEIGHT * 0.6

        cold_water_density = steam_tables.water_density(
            temp_k=cold_water_temp,
            pressure_pa=cold_pressure,
        )
        cold_water_mass = cold_water_density * DRUM_CROSS_SECTION * cold_water_level
        cold_internal_energy = cold_water_mass * steam_tables.water_enthalpy(
            cold_water_temp, cold_pressure
        )

        from physics_engine.models import BoilerState

        cold_state = BoilerState(
            internal_energy=cold_internal_energy,
            pressure=cold_pressure,
            water_level=cold_water_level,
            flue_gas_temp=TEMP_AMBIENT,
            water_temp=cold_water_temp,
        )

        controller = BoilerController()

        # ── Phase boundaries ──────────────────────────────────────────────────
        t_phase2 = duration * 0.20  # warmup ends
        t_phase3 = duration * 0.60  # pressure buildup ends

        def setpoints_fn(t: float) -> BoilerSetpoints:
            """
            Gradually increase pressure setpoint as boiler warms up.

            Low setpoint in early phases prevents controller from
            demanding maximum fuel before the furnace is hot.
            """
            if t < t_phase2:
                # Phase 1: target low pressure — let the boiler warm up
                return BoilerSetpoints(pressure=30.0e5, water_level=4.8)
            elif t < t_phase3:
                # Phase 2: ramp pressure setpoint toward nominal
                alpha = (t - t_phase2) / (t_phase3 - t_phase2)
                target_pressure = 30.0e5 + alpha * (140.0e5 - 30.0e5)
                return BoilerSetpoints(pressure=target_pressure, water_level=4.8)
            else:
                # Phase 3: hold nominal operating point
                return BoilerSetpoints(pressure=140.0e5, water_level=4.8)

        def steam_valve_fn(t: float) -> float | None:
            """
            Keep steam valve closed until pressure is sufficient.

            Opening the steam valve too early collapses pressure before
            the furnace has enough heat output to sustain steam generation.
            """
            if t < t_phase2:
                return 0.0  # Phase 1: fully closed — build pressure
            elif t < t_phase3:
                # Phase 2: crack open proportionally to phase progress
                alpha = (t - t_phase2) / (t_phase3 - t_phase2)
                return alpha * 0.3  # open up to 30%
            else:
                return None  # Phase 3: controller manages steam valve

        return self._run(
            scenario_name="cold_start",
            initial_state=cold_state,
            controller=controller,
            setpoints_fn=setpoints_fn,
            steam_valve_fn=steam_valve_fn,
            duration=duration,
            dt=dt,
        )
