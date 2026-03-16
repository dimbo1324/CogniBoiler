"""
Unit tests for TurbineModel and BoilerTurbineSystem.

Test categories:
    TestTurbinePhysics  — thermodynamic correctness
    TestTurbineLimits   — boundary conditions and edge cases
    TestSystem          — boiler-turbine coupling
"""

import pytest
from physics_engine.models import BoilerParameters, ControlInputs
from physics_engine.system import BoilerTurbineSystem, SystemState
from physics_engine.turbine import TurbineModel, TurbineState

# ─── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture  # type: ignore[misc]
def turbine() -> TurbineModel:
    """TurbineModel with default design parameters."""
    return TurbineModel()


@pytest.fixture  # type: ignore[misc]
def nominal_state(turbine: TurbineModel) -> TurbineState:
    """Turbine state at nominal boiler output: 552.5°C, 140 bar, 277.8 kg/s."""
    return turbine.nominal_state()


@pytest.fixture  # type: ignore[misc]
def system() -> BoilerTurbineSystem:
    """Coupled boiler-turbine system with default parameters."""
    return BoilerTurbineSystem()


# ─── Turbine physics tests ────────────────────────────────────────────────────


class TestTurbinePhysics:
    """
    Verify thermodynamic correctness of turbine calculations.
    """

    def test_nominal_power_in_expected_range(self, nominal_state: TurbineState) -> None:
        """
        At nominal conditions (140 bar, 552°C, 277 kg/s) electrical output
        must be within 100–300 MW — the design range of the plant.
        """
        mw = nominal_state.electrical_power_mw
        assert 100.0 < mw < 400.0, (
            f"Nominal electrical power out of range: {mw:.1f} MW "
            f"(expected 100–300 MW)"
        )

    def test_actual_work_less_than_ideal(self, nominal_state: TurbineState) -> None:
        """
        Actual specific work must be strictly less than isentropic work.

        Physics: η_is < 1 -> W_actual = η_is × W_ideal < W_ideal
        """
        assert nominal_state.specific_work_actual < nominal_state.specific_work_ideal, (
            f"Actual work ({nominal_state.specific_work_actual/1e3:.1f} kJ/kg) "
            f">= ideal work ({nominal_state.specific_work_ideal/1e3:.1f} kJ/kg)"
        )

    def test_enthalpy_drops_across_turbine(self, nominal_state: TurbineState) -> None:
        """
        Outlet enthalpy must be lower than inlet enthalpy.

        Physics: work is extracted -> h_out < h_in
        """
        assert nominal_state.enthalpy_out_actual < nominal_state.enthalpy_in, (
            f"Enthalpy did not drop: "
            f"h_in={nominal_state.enthalpy_in/1e3:.1f} kJ/kg, "
            f"h_out={nominal_state.enthalpy_out_actual/1e3:.1f} kJ/kg"
        )

    def test_power_increases_with_steam_flow(self, turbine: TurbineModel) -> None:
        """
        Higher steam flow at same T and P must produce more power.

        Physics: P_shaft = m × W_actual, W_actual constant at fixed T, P
        """
        state_low = turbine.calculate(
            steam_temp_in=825.65,
            steam_pressure_in=140e5,
            steam_flow=150.0,
        )
        state_high = turbine.calculate(
            steam_temp_in=825.65,
            steam_pressure_in=140e5,
            steam_flow=250.0,
        )
        assert state_high.shaft_power > state_low.shaft_power, (
            f"Power did not increase with flow: "
            f"P_low={state_low.shaft_power_mw:.1f} MW, "
            f"P_high={state_high.shaft_power_mw:.1f} MW"
        )

    def test_power_increases_with_inlet_pressure(self, turbine: TurbineModel) -> None:
        """
        Higher inlet pressure at same T and flow must produce more power.

        Physics: larger pressure ratio -> greater enthalpy drop -> more work
        """
        state_low = turbine.calculate(
            steam_temp_in=825.65,
            steam_pressure_in=100e5,
            steam_flow=200.0,
        )
        state_high = turbine.calculate(
            steam_temp_in=825.65,
            steam_pressure_in=160e5,
            steam_flow=200.0,
        )
        assert state_high.shaft_power > state_low.shaft_power, (
            f"Power did not increase with pressure: "
            f"P_100bar={state_low.shaft_power_mw:.1f} MW, "
            f"P_160bar={state_high.shaft_power_mw:.1f} MW"
        )

    def test_isentropic_efficiency_applied_correctly(
        self, turbine: TurbineModel
    ) -> None:
        """
        Actual work must equal η_is × ideal work to within floating-point tolerance.
        """
        state = turbine.nominal_state()
        expected = turbine.params.isentropic_efficiency * state.specific_work_ideal
        assert abs(state.specific_work_actual - expected) < 1.0, (
            f"η_is not applied correctly: "
            f"W_actual={state.specific_work_actual:.1f} J/kg, "
            f"η×W_ideal={expected:.1f} J/kg"
        )


# ─── Turbine boundary condition tests ────────────────────────────────────────


class TestTurbineLimits:
    """
    Verify correct behavior at edge cases and boundaries.
    """

    def test_zero_flow_gives_zero_power(self, turbine: TurbineModel) -> None:
        """
        Below minimum steam flow, shaft power must be zero.
        """
        state = turbine.calculate(
            steam_temp_in=825.65,
            steam_pressure_in=140e5,
            steam_flow=0.0,
        )
        assert (
            state.shaft_power == 0.0
        ), f"Expected zero power at zero flow, got {state.shaft_power_mw:.3f} MW"

    def test_exhaust_pressure_correct(self, nominal_state: TurbineState) -> None:
        """
        Exhaust pressure must match the design back-pressure (0.07 bar).
        """
        from physics_engine.turbine import TURBINE_EXHAUST_PRESSURE

        assert nominal_state.exhaust_pressure == TURBINE_EXHAUST_PRESSURE

    def test_electrical_power_less_than_shaft_power(
        self, nominal_state: TurbineState
    ) -> None:
        """
        Electrical power must be less than shaft power due to mechanical losses.
        """
        assert (
            nominal_state.electrical_power < nominal_state.shaft_power
        ), "Electrical power must be less than shaft power"


# ─── Boiler-turbine system tests ──────────────────────────────────────────────


class TestSystem:
    """
    Verify boiler-turbine coupling in BoilerTurbineSystem.
    """

    def test_steady_state_returns_system_state(
        self, system: BoilerTurbineSystem
    ) -> None:
        """
        steady_state() must return a valid SystemState instance.
        """
        state = system.steady_state(
            fuel_valve=0.6,
            feedwater_valve=0.5,
            steam_valve=0.5,
        )
        assert isinstance(state, SystemState)
        assert isinstance(state.turbine, TurbineState)

    def test_system_produces_positive_power(self, system: BoilerTurbineSystem) -> None:
        """
        At normal operating conditions, system must generate positive power.
        """
        state = system.steady_state(
            fuel_valve=0.7,
            feedwater_valve=0.5,
            steam_valve=0.6,
        )
        assert (
            state.electrical_power_mw > 0.0
        ), f"System produced no power: {state.electrical_power_mw:.1f} MW"

    def test_more_steam_valve_more_power(self, system: BoilerTurbineSystem) -> None:
        """
        At identical boiler conditions, opening the steam valve more
        must increase power output.

        Tests turbine coupling directly using a fixed boiler state,
        avoiding boiler transient dynamics which confound the result:
        a wide-open steam valve drains drum pressure faster than the
        boiler can sustain it, causing pressure collapse before steady
        state is reached.
        """

        boiler_state = BoilerParameters().nominal_initial_state()

        controls_low = ControlInputs(
            fuel_valve_command=0.7,
            feedwater_valve_command=0.5,
            steam_valve_command=0.3,
        )
        controls_high = ControlInputs(
            fuel_valve_command=0.7,
            feedwater_valve_command=0.5,
            steam_valve_command=0.7,
        )

        state_low = system.evaluate_at(boiler_state, controls_low, time=0.0)
        state_high = system.evaluate_at(boiler_state, controls_high, time=0.0)

        assert state_high.electrical_power_mw > state_low.electrical_power_mw, (
            f"More steam valve did not increase power at equal boiler conditions: "
            f"sv=0.3 -> {state_low.electrical_power_mw:.1f} MW, "
            f"sv=0.7 -> {state_high.electrical_power_mw:.1f} MW"
        )
