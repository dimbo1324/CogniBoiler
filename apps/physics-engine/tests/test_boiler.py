"""
Unit tests for the CogniBoiler thermodynamic boiler model.

Test categories:
    TestPhysics   — physical behavior correctness (monotonicity, energy balance)
    TestEvents    — alarm event detection (pressure, level, temperature limits)
    TestNumerics  — solver stability and result validity
"""

from dataclasses import replace

import pytest
from physics_engine import offline
from physics_engine.boiler import (
    MIN_MODEL_PRESSURE,
    WATER_TEMP_MAX,
    WATER_TEMP_MIN,
    BoilerModel,
    PlantDivergedError,
)
from physics_engine.condenser import COOLING_WATER_TEMP_DESIGN
from physics_engine.models import BoilerParameters, BoilerState, ControlInputs
from physics_engine.scenarios import InitialConditions, get_scenario
from physics_engine.turbine import TurbineParameters

# ─── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def params() -> BoilerParameters:
    """Default boiler design parameters."""
    return BoilerParameters()


@pytest.fixture
def model(params: BoilerParameters) -> BoilerModel:
    """BoilerModel with default parameters."""
    return BoilerModel(params)


@pytest.fixture
def initial_state(params: BoilerParameters) -> BoilerState:
    """Physically consistent initial state at nominal operating point."""
    return params.nominal_initial_state()


@pytest.fixture
def nominal_controls() -> ControlInputs:
    """Balanced operating point: moderate fuel, feedwater matches steam output."""
    return ControlInputs(
        fuel_valve_command=0.6,
        feedwater_valve_command=0.5,
        steam_valve_command=0.5,
    )


# ─── Physical behavior tests ──────────────────────────────────────────────────


class TestPhysics:
    """
    Verify that simulation output matches expected physical behavior.

    Each test runs a short simulation (120–300 s) with extreme control
    inputs designed to make the physical effect clearly observable.
    """

    def test_pressure_rises_with_full_fuel(
        self,
        model: BoilerModel,
        initial_state: BoilerState,
    ) -> None:
        """
        With maximum fuel and steam valve closed, pressure must rise monotonically.

        Physics: Q_combustion >> Q_loss -> dU/dt > 0 -> T rises -> P_sat rises -> dP/dt > 0
        """
        controls = ControlInputs(
            fuel_valve_command=1.0,
            feedwater_valve_command=1.0,
            steam_valve_command=0.0,  # closed — no steam leaving
        )
        result = offline.simulate(
            model, initial_state, controls, t_span=(0, 120), dt=1.0
        )

        assert result.y.shape[1] > 10, "Solver produced too few points"

        pressure = result.y[1]  # Pa
        # Pressure at end must be higher than at start
        assert pressure[-1] > pressure[0], (
            f"Pressure did not rise with full fuel: "
            f"P_start={pressure[0] / 1e5:.1f} bar, P_end={pressure[-1] / 1e5:.1f} bar"
        )

    def test_water_level_drops_with_zero_feedwater(
        self,
        model: BoilerModel,
        initial_state: BoilerState,
    ) -> None:
        """
        With steam valve open and no feedwater, drum level must fall.

        Physics: m_steam > 0, m_feed = 0 -> dh/dt < 0
        """
        controls = ControlInputs(
            fuel_valve_command=0.6,
            feedwater_valve_command=0.0,  # no feedwater
            steam_valve_command=0.8,  # steam leaving
        )
        result = offline.simulate(
            model, initial_state, controls, t_span=(0, 120), dt=1.0
        )

        assert result.y.shape[1] > 10, "Solver produced too few points"

        level = result.y[2]  # m
        assert level[-1] < level[0], (
            f"Water level did not drop with zero feedwater: "
            f"h_start={level[0]:.3f} m, h_end={level[-1]:.3f} m"
        )

    def test_water_level_rises_with_excess_feedwater(
        self,
        model: BoilerModel,
        initial_state: BoilerState,
    ) -> None:
        """
        With maximum feedwater and steam valve closed, drum level must rise.

        Physics: m_feed >> m_steam -> dh/dt > 0
        """
        controls = ControlInputs(
            fuel_valve_command=0.5,
            feedwater_valve_command=1.0,  # maximum feedwater
            steam_valve_command=0.0,  # closed
        )
        result = offline.simulate(
            model, initial_state, controls, t_span=(0, 120), dt=1.0
        )

        assert result.y.shape[1] > 10, "Solver produced too few points"

        level = result.y[2]  # m
        assert level[-1] > level[0], (
            f"Water level did not rise with excess feedwater: "
            f"h_start={level[0]:.3f} m, h_end={level[-1]:.3f} m"
        )

    def test_temperature_drops_with_zero_fuel(
        self,
        model: BoilerModel,
        initial_state: BoilerState,
    ) -> None:
        """
        With zero fuel, water temperature must decrease over time (cooling).

        Physics: Q_combustion = 0, Q_loss > 0 -> dU/dt < 0 -> dT/dt < 0
        """
        controls = ControlInputs(
            fuel_valve_command=0.0,  # no fuel
            feedwater_valve_command=0.3,
            steam_valve_command=0.3,
        )
        result = offline.simulate(
            model, initial_state, controls, t_span=(0, 300), dt=1.0
        )

        assert result.y.shape[1] > 10, "Solver produced too few points"

        temp = result.y[4]  # T_water [K]
        assert temp[-1] < temp[0], (
            f"Temperature did not drop with zero fuel: "
            f"T_start={temp[0] - 273.15:.1f}°C, T_end={temp[-1] - 273.15:.1f}°C"
        )

    def test_internal_energy_increases_with_full_fuel_no_steam(
        self,
        model: BoilerModel,
        initial_state: BoilerState,
    ) -> None:
        """
        With full fuel and no steam output, internal energy must increase.

        Physics: Q_in > Q_loss -> dU/dt > 0
        """
        controls = ControlInputs(
            fuel_valve_command=1.0,
            feedwater_valve_command=0.5,
            steam_valve_command=0.0,
        )
        result = offline.simulate(
            model, initial_state, controls, t_span=(0, 60), dt=1.0
        )

        energy = result.y[0]  # U [J]
        assert energy[-1] > energy[0], (
            f"Internal energy did not increase with full fuel: "
            f"U_start={energy[0]:.3e} J, U_end={energy[-1]:.3e} J"
        )


# ─── Event detection tests ────────────────────────────────────────────────────


class TestEvents:
    """
    Verify that solve_ivp terminal events fire correctly under alarm conditions.

    Each test drives the boiler into a known alarm condition and checks
    that the simulation stops with status=1 and the correct event name.
    """

    def test_pressure_high_event_fires(
        self,
        model: BoilerModel,
        initial_state: BoilerState,
    ) -> None:
        """
        Full fuel + closed steam valve must eventually trigger PRESSURE HIGH.
        """
        controls = ControlInputs(
            fuel_valve_command=1.0,
            feedwater_valve_command=0.3,
            steam_valve_command=0.0,  # no steam release — pressure builds
        )
        result = offline.simulate(
            model, initial_state, controls, t_span=(0, 3600), dt=1.0
        )

        # status=1 means a terminal event was triggered
        assert result.status == 1, (
            f"Expected terminal event, got status={result.status}: {result.message}"
        )
        termination = offline.check_result(result)
        assert "PRESSURE HIGH" in termination, (
            f"Expected PRESSURE HIGH alarm, got: {termination}"
        )

    def test_drum_dry_event_fires(
        self,
        model: BoilerModel,
        initial_state: BoilerState,
    ) -> None:
        """
        Zero feedwater + open steam valve must eventually trigger DRUM DRY.
        """
        controls = ControlInputs(
            fuel_valve_command=0.5,
            feedwater_valve_command=0.0,  # no water input
            steam_valve_command=1.0,  # maximum steam output
        )
        result = offline.simulate(
            model, initial_state, controls, t_span=(0, 3600), dt=1.0
        )

        assert result.status == 1, (
            f"Expected terminal event, got status={result.status}: {result.message}"
        )
        termination = offline.check_result(result)
        assert "DRUM DRY" in termination, f"Expected DRUM DRY alarm, got: {termination}"

    def test_drum_overflow_event_fires(
        self,
        model: BoilerModel,
        initial_state: BoilerState,
    ) -> None:
        """
        Maximum feedwater + zero steam output must eventually trigger DRUM OVERFLOW.
        """
        controls = ControlInputs(
            fuel_valve_command=0.3,
            feedwater_valve_command=1.0,  # maximum feedwater
            steam_valve_command=0.0,  # no steam release
        )
        result = offline.simulate(
            model, initial_state, controls, t_span=(0, 3600), dt=1.0
        )

        assert result.status == 1, (
            f"Expected terminal event, got status={result.status}: {result.message}"
        )
        termination = offline.check_result(result)
        assert "DRUM OVERFLOW" in termination, (
            f"Expected DRUM OVERFLOW alarm, got: {termination}"
        )


# ─── Numerical stability tests ────────────────────────────────────────────────


class TestNumerics:
    """
    Verify solver stability, result shape, and physical plausibility of outputs.
    """

    def test_simulation_completes_normally(
        self,
        model: BoilerModel,
        initial_state: BoilerState,
        nominal_controls: ControlInputs,
    ) -> None:
        """
        Balanced operating point must complete full 300 s without solver failure.
        """
        result = offline.simulate(
            model, initial_state, nominal_controls, t_span=(0, 300), dt=1.0
        )

        assert result.status != -1, f"Solver failed: {result.message}"
        assert result.y.shape[0] == 5, "Result must have 5 state variables"
        assert result.y.shape[1] > 0, "Result must contain at least one time point"

    def test_result_has_correct_shape(
        self,
        model: BoilerModel,
        initial_state: BoilerState,
        nominal_controls: ControlInputs,
    ) -> None:
        """
        ODE result must have shape (5, N) matching the 5D state vector.
        """
        result = offline.simulate(
            model, initial_state, nominal_controls, t_span=(0, 60), dt=1.0
        )

        assert result.y.ndim == 2, "result.y must be 2D array"
        assert result.y.shape[0] == 5, (
            f"Expected 5 state variables, got {result.y.shape[0]}"
        )

    def test_state_values_are_physically_plausible(
        self,
        model: BoilerModel,
        initial_state: BoilerState,
        nominal_controls: ControlInputs,
    ) -> None:
        """
        All state variables must remain within physically meaningful ranges
        throughout the simulation.

        Ranges checked:
            U  > 0              — internal energy always positive
            P  in [1, 250] bar  — pressure within realistic bounds
            h  in [0, 8] m      — water level within drum geometry
            T_gas > 273 K       — flue gas above freezing
            T_w   > 273 K       — water above freezing
        """
        result = offline.simulate(
            model, initial_state, nominal_controls, t_span=(0, 300), dt=1.0
        )

        u = result.y[0]  # J
        p = result.y[1]  # Pa
        h = result.y[2]  # m
        t_gas = result.y[3]  # K
        t_w = result.y[4]  # K

        assert (u > 0).all(), "Internal energy went negative"
        assert (p > 1.0e5).all(), "Pressure dropped below 1 bar"
        assert (p < 250.0e5).all(), "Pressure exceeded 250 bar"
        assert (h >= 0).all(), "Water level went negative"
        assert (h <= 8.5).all(), "Water level exceeded drum height"
        assert (t_gas > 273.15).all(), "Flue gas temperature below freezing"
        assert (t_w > 273.15).all(), "Water temperature below freezing"

    def test_get_state_at_returns_correct_type(
        self,
        model: BoilerModel,
        initial_state: BoilerState,
        nominal_controls: ControlInputs,
    ) -> None:
        """
        get_state_at() must return a valid BoilerState at any time index.
        """
        result = offline.simulate(
            model, initial_state, nominal_controls, t_span=(0, 60), dt=1.0
        )

        state_mid = offline.state_at(result, index=result.y.shape[1] // 2)

        assert isinstance(state_mid, BoilerState)
        assert state_mid.pressure > 0
        assert state_mid.water_level > 0
        assert state_mid.water_temp > 273.15

    def test_check_result_returns_string(
        self,
        model: BoilerModel,
        initial_state: BoilerState,
        nominal_controls: ControlInputs,
    ) -> None:
        """
        check_result() must always return a non-empty string.
        """
        result = offline.simulate(
            model, initial_state, nominal_controls, t_span=(0, 60), dt=1.0
        )

        message = offline.check_result(result)
        assert isinstance(message, str)
        assert len(message) > 0


class TestRk4Step:
    """The fixed-step integration the live plant runs."""

    @pytest.fixture
    def operating_point(self, params: BoilerParameters) -> InitialConditions:
        return get_scenario("steady_state").initial_conditions(
            params, TurbineParameters(), COOLING_WATER_TEMP_DESIGN
        )

    def test_the_nominal_operating_point_holds_with_its_valves_left_alone(
        self, model: BoilerModel, operating_point: InitialConditions
    ) -> None:
        start = operating_point.boiler_state
        controls = operating_point.controls.copy()
        state = start
        for _ in range(1800):
            state = model.step(state, controls, dt=1.0)
        assert state.pressure == pytest.approx(start.pressure, rel=1e-6)
        assert state.water_level == pytest.approx(start.water_level, rel=1e-6)
        assert state.flue_gas_temp == pytest.approx(start.flue_gas_temp, rel=1e-6)
        assert state.water_temp == pytest.approx(start.water_temp, rel=1e-6)

    def test_closing_the_steam_valve_raises_drum_pressure(
        self, model: BoilerModel, operating_point: InitialConditions
    ) -> None:
        controls = operating_point.controls.copy()
        controls.steam_valve_command = 0.0
        pressures = [operating_point.boiler_state.pressure]
        state = operating_point.boiler_state
        for _ in range(60):
            state = model.step(state, controls, dt=1.0)
            pressures.append(state.pressure)
        assert pressures == sorted(pressures)
        assert pressures[-1] > pressures[0] + 20.0e5

    def test_an_integrated_state_is_kept_inside_the_model_domain(
        self, model: BoilerModel, params: BoilerParameters
    ) -> None:
        bounded = model._bounded(
            BoilerState(
                internal_energy=-1.0,
                pressure=-5.0,
                water_level=params.drum_height + 3.0,
                flue_gas_temp=params.ambient_temp - 50.0,
                water_temp=WATER_TEMP_MAX + 100.0,
            )
        )
        assert bounded == BoilerState(
            internal_energy=0.0,
            pressure=MIN_MODEL_PRESSURE,
            water_level=params.drum_height,
            flue_gas_temp=params.ambient_temp,
            water_temp=WATER_TEMP_MAX,
        )
        assert model._bounded(replace(bounded, water_level=-1.0)).water_level == 0.0
        assert (
            model._bounded(replace(bounded, water_temp=0.0)).water_temp
            == WATER_TEMP_MIN
        )

    def test_a_non_finite_state_raises_instead_of_being_clamped(
        self,
        model: BoilerModel,
        initial_state: BoilerState,
        nominal_controls: ControlInputs,
    ) -> None:
        broken = replace(initial_state, flue_gas_temp=float("nan"))
        with pytest.raises(PlantDivergedError, match="non-finite"):
            model.step(broken, nominal_controls, dt=1.0)
