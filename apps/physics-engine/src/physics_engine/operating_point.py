"""
Steady operating points: the state and valve positions at which the plant does not move.

For a requested electrical load at nominal pressure, drum level and turbine inlet
temperature, the solver finds the fuel valve, turbine admission valve, spray valve and
furnace gas temperature that zero the furnace and drum energy balances, deliver the load
and bring the steam to its temperature setpoint; the feedwater valve follows from the drum
mass balance and the condenser back-pressure is iterated to its own fixed point. When the
superheater cannot reach the setpoint the spray stays shut and the steam runs cooler.

A plant started from such a point with its valves left alone stays there — the open-loop
equilibrium the runtime relies on.
"""

from __future__ import annotations

from dataclasses import dataclass

from scipy.optimize import least_squares

from physics_engine import properties, steam_tables
from physics_engine.boiler import BoilerBalance
from physics_engine.condenser import COOLING_WATER_TEMP_DESIGN, CondenserModel
from physics_engine.constants import (
    NOMINAL_WATER_LEVEL,
    PRESSURE_NOMINAL,
    RATED_POWER,
    TEMP_STEAM_RATED,
)
from physics_engine.models import BoilerParameters, BoilerState, ControlInputs
from physics_engine.system import BoilerTurbineSystem
from physics_engine.turbine import TURBINE_EXHAUST_PRESSURE, TurbineParameters

MIN_LOAD_FRACTION: float = 0.2
MAX_LOAD_FRACTION: float = 1.05

# Converged when the drum water temperature moves less than this per hour and the
# furnace gas less than this per second — invisible over an 8-hour run.
WATER_TEMP_RATE_TOLERANCE: float = 0.05 / 3600.0  # K/s
GAS_TEMP_RATE_TOLERANCE: float = 1.0e-4  # K/s
POWER_TOLERANCE: float = 0.05e6  # W
BACKPRESSURE_TOLERANCE: float = 1.0  # Pa
MAX_CONDENSER_ITERATIONS: int = 12
SPRAY_BOUND_EPSILON: float = 1.0e-6


class OperatingPointError(ValueError):
    """The requested load cannot be held steadily by this plant."""


@dataclass(frozen=True)
class OperatingPoint:
    """A steady state of the plant and the valve positions that hold it."""

    load_w: float
    boiler_state: BoilerState
    controls: ControlInputs
    exhaust_pressure_pa: float
    balance: BoilerBalance


@dataclass(frozen=True)
class _Trial:
    boiler_state: BoilerState
    controls: ControlInputs
    balance: BoilerBalance
    power_w: float
    exhaust_enthalpy: float
    turbine_flow: float


class _Solver:
    def __init__(
        self,
        load_w: float,
        pressure_pa: float,
        water_level_m: float,
        steam_temp_k: float,
        boiler_params: BoilerParameters,
        turbine_params: TurbineParameters,
        cooling_water_temp_k: float,
    ) -> None:
        self.load_w = load_w
        self.pressure_pa = pressure_pa
        self.water_level_m = water_level_m
        self.system = BoilerTurbineSystem(boiler_params, turbine_params)
        self.condenser = CondenserModel(cooling_water_temp=cooling_water_temp_k)
        self.water_temp_k = properties.saturation_temperature(pressure_pa)
        # The spray target is expressed as enthalpy: unlike temperature it is a
        # continuous function of the valve positions, which the solver needs.
        self.target_enthalpy = steam_tables.steam_enthalpy(steam_temp_k, pressure_pa)
        params = self.system.boiler_params
        water_mass = (
            properties.liquid_density(self.water_temp_k)
            * params.drum_cross_section
            * water_level_m
        )
        self.internal_energy = water_mass * properties.liquid_enthalpy(
            self.water_temp_k
        )

    def trial(self, x: list[float], exhaust_pressure_pa: float) -> _Trial:
        fuel_valve, gas_temp_k, steam_valve, spray_valve = x
        boiler = self.system.boiler
        state = BoilerState(
            internal_energy=self.internal_energy,
            pressure=self.pressure_pa,
            water_level=self.water_level_m,
            flue_gas_temp=gas_temp_k,
            water_temp=self.water_temp_k,
        )
        probe = boiler.balance(
            state,
            ControlInputs(
                fuel_valve_command=fuel_valve,
                feedwater_valve_command=0.0,
                steam_valve_command=steam_valve,
                spray_valve_command=spray_valve,
            ),
        )
        controls = ControlInputs(
            fuel_valve_command=fuel_valve,
            feedwater_valve_command=(probe.drum_steam_flow + probe.leak_flow)
            / boiler.params.max_feedwater_flow,
            steam_valve_command=steam_valve,
            spray_valve_command=spray_valve,
        )
        balance = boiler.balance(state, controls)
        turbine = self.system.turbine_at(state, balance, exhaust_pressure_pa)
        return _Trial(
            boiler_state=state,
            controls=controls,
            balance=balance,
            power_w=turbine.electrical_power,
            exhaust_enthalpy=turbine.enthalpy_out_actual,
            turbine_flow=turbine.steam_flow,
        )

    def _balance_residuals(self, candidate: _Trial) -> list[float]:
        _, _, _, dt_gas, dt_water = candidate.balance.derivatives
        return [
            dt_water * 1.0e3,
            dt_gas * 1.0e2,
            (candidate.power_w - self.load_w) / RATED_POWER * 1.0e2,
        ]

    def solve(
        self,
        x0: list[float],
        exhaust_pressure_pa: float,
        spray_valve: float | None,
    ) -> tuple[_Trial, list[float]]:
        """Solve with the spray free (None) or held at a fixed opening."""
        lower = [0.0, self.water_temp_k + 1.0, 0.0]
        upper = [1.0, 2500.0, 1.0]

        if spray_valve is None:

            def residuals(x: list[float]) -> list[float]:
                candidate = self.trial(list(x), exhaust_pressure_pa)
                return [
                    *self._balance_residuals(candidate),
                    (candidate.balance.turbine_inlet_enthalpy - self.target_enthalpy)
                    / 1.0e3,
                ]

            start = list(x0)
            bounds = ([*lower, 0.0], [*upper, 1.0])
        else:

            def residuals(x: list[float]) -> list[float]:
                candidate = self.trial([*x, spray_valve], exhaust_pressure_pa)
                return self._balance_residuals(candidate)

            start = list(x0[:3])
            bounds = (lower, upper)

        solution = least_squares(
            residuals,
            x0=start,
            bounds=bounds,
            xtol=1.0e-12,
            ftol=1.0e-12,
            gtol=1.0e-12,
            max_nfev=500,
        )
        x = [float(v) for v in solution.x]
        if spray_valve is not None:
            x.append(spray_valve)
        return self.trial(x, exhaust_pressure_pa), x

    def hold(
        self, x0: list[float], exhaust_pressure_pa: float
    ) -> tuple[_Trial, list[float]]:
        """The load at steam temperature setpoint, or unsprayed if it runs cooler."""
        unsprayed, x = self.solve(x0, exhaust_pressure_pa, spray_valve=0.0)
        if unsprayed.balance.turbine_inlet_enthalpy <= self.target_enthalpy:
            return unsprayed, x
        sprayed, x = self.solve([*x[:3], max(x0[3], 0.05)], exhaust_pressure_pa, None)
        if x[3] >= 1.0 - SPRAY_BOUND_EPSILON:
            return self.solve(x, exhaust_pressure_pa, spray_valve=1.0)
        return sprayed, x


def solve_operating_point(
    load_w: float,
    *,
    pressure_pa: float = PRESSURE_NOMINAL,
    water_level_m: float = NOMINAL_WATER_LEVEL,
    steam_temp_k: float = TEMP_STEAM_RATED,
    boiler_params: BoilerParameters | None = None,
    turbine_params: TurbineParameters | None = None,
    cooling_water_temp_k: float = COOLING_WATER_TEMP_DESIGN,
) -> OperatingPoint:
    """Find the steady operating point for an electrical load [W]."""
    if not MIN_LOAD_FRACTION * RATED_POWER <= load_w <= MAX_LOAD_FRACTION * RATED_POWER:
        raise OperatingPointError(
            f"load {load_w / 1e6:.1f} MW outside "
            f"[{MIN_LOAD_FRACTION * RATED_POWER / 1e6:.0f}, "
            f"{MAX_LOAD_FRACTION * RATED_POWER / 1e6:.0f}] MW"
        )

    solver = _Solver(
        load_w,
        pressure_pa,
        water_level_m,
        steam_temp_k,
        boiler_params or BoilerParameters(),
        turbine_params or TurbineParameters(),
        cooling_water_temp_k,
    )

    fraction = load_w / RATED_POWER
    x = [0.78 * fraction, 1400.0, min(0.85 * fraction, 1.0), 0.2]
    exhaust = TURBINE_EXHAUST_PRESSURE
    point, x = solver.hold(x, exhaust)
    for _ in range(MAX_CONDENSER_ITERATIONS):
        condenser = solver.condenser.calculate(
            steam_flow=point.turbine_flow,
            steam_enthalpy_in=point.exhaust_enthalpy,
        )
        converged = abs(condenser.backpressure_pa - exhaust) < BACKPRESSURE_TOLERANCE
        exhaust = condenser.backpressure_pa
        point, x = solver.hold(x, exhaust)
        if converged:
            break

    _, _, _, dt_gas, dt_water = point.balance.derivatives
    if (
        abs(dt_water) > WATER_TEMP_RATE_TOLERANCE
        or abs(dt_gas) > GAS_TEMP_RATE_TOLERANCE
        or abs(point.power_w - load_w) > POWER_TOLERANCE
    ):
        raise OperatingPointError(
            f"no steady point for {load_w / 1e6:.1f} MW: "
            f"dT_water/dt={dt_water:.2e} K/s, dT_gas/dt={dt_gas:.2e} K/s, "
            f"power {point.power_w / 1e6:.2f} MW"
        )

    return OperatingPoint(
        load_w=load_w,
        boiler_state=point.boiler_state,
        controls=point.controls,
        exhaust_pressure_pa=exhaust,
        balance=point.balance,
    )
