"""
Offline analysis of the boiler: adaptive integration with alarm events.

The live plant steps the boiler with a fixed-step RK4 (`BoilerModel.step`), which is
deterministic and fast. This module integrates the same balances with scipy's implicit
Radau solver instead, stopping at the first alarm condition, and settles the coupled
boiler and turbine at fixed valve positions. Nothing in the runtime uses it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from scipy.integrate._ivp.ivp import OdeResult

from physics_engine.boiler import BoilerModel
from physics_engine.constants import PRESSURE_MAX, PRESSURE_MIN, TEMP_STEAM_MAX
from physics_engine.faults import NO_DISTURBANCES, PlantDisturbances
from physics_engine.models import BoilerState, ControlInputs
from physics_engine.system import BoilerTurbineSystem
from physics_engine.turbine import TurbineState

ODE_METHOD: str = "Radau"
ODE_RTOL: float = 1e-4
ODE_ATOL: float = 1e-6
ODE_MAX_STEP: float = 5.0  # s

# The drum counts as dry below this level and as overflowing within this margin of the
# top; both end an offline run.
DRY_DRUM_LEVEL: float = 0.05  # m
OVERFLOW_MARGIN: float = 0.1  # m

RISING: float = 1.0
FALLING: float = -1.0

ALARM_NAMES: tuple[str, ...] = (
    "PRESSURE HIGH",
    "PRESSURE LOW",
    "DRUM DRY",
    "DRUM OVERFLOW",
    "STEAM TEMP HIGH",
)


@dataclass(frozen=True)
class AlarmEvent:
    """A terminal solve_ivp event: fires when `crossing(y)` changes sign in `direction`."""

    name: str
    crossing: Callable[[list[float]], float]
    direction: float
    terminal: bool = True

    def __call__(self, t: float, y: list[float], *args: object) -> float:
        return self.crossing(y)


def alarm_events(model: BoilerModel) -> tuple[AlarmEvent, ...]:
    """The alarm conditions that end an offline run, in ALARM_NAMES order.

    The steam-temperature event is a numerical safety net: the water temperature the
    balances use is clamped below the critical point, so it never fires in operation.
    """
    drum_top = model.params.drum_height - OVERFLOW_MARGIN
    high, low, dry, overflow, hot = ALARM_NAMES
    return (
        AlarmEvent(high, lambda y: y[1] - PRESSURE_MAX, RISING),
        AlarmEvent(low, lambda y: y[1] - PRESSURE_MIN, FALLING),
        AlarmEvent(dry, lambda y: y[2] - DRY_DRUM_LEVEL, FALLING),
        AlarmEvent(overflow, lambda y: y[2] - drum_top, RISING),
        AlarmEvent(hot, lambda y: y[4] - TEMP_STEAM_MAX, RISING),
    )


def simulate(
    model: BoilerModel,
    initial_state: BoilerState,
    controls: ControlInputs,
    t_span: tuple[float, float],
    dt: float = 1.0,
    disturbances: PlantDisturbances = NO_DISTURBANCES,
) -> OdeResult:
    """
    Integrate the boiler over a time span, stopping at the first alarm event.

    Valves move once, by one `dt`, before the integration; a caller that wants actuator
    lag across runs reuses the same ControlInputs.

    Returns the scipy OdeResult: status 0 reached t_end, -1 the solver failed, 1 an
    alarm event ended the run (see `check_result`).
    """
    controls.update_valves(dt)

    def rate(t: float, y: list[float]) -> list[float]:
        state = BoilerState.from_vector([float(v) for v in y])
        return list(model.balance(state, controls, disturbances).derivatives)

    result: OdeResult = solve_ivp(
        fun=rate,
        t_span=t_span,
        y0=initial_state.to_vector(),
        method=ODE_METHOD,
        t_eval=np.arange(t_span[0], t_span[1], dt),
        events=list(alarm_events(model)),
        rtol=ODE_RTOL,
        atol=ODE_ATOL,
        max_step=ODE_MAX_STEP,
    )
    return result


def state_at(result: OdeResult, index: int) -> BoilerState:
    """The boiler state at one output time of an offline run."""
    return BoilerState.from_vector([float(result.y[i, index]) for i in range(5)])


def check_result(result: OdeResult) -> str:
    """Why an offline run ended, in words."""
    if result.status == 0:
        return "Simulation completed normally."
    if result.status == -1:
        return f"Solver failed: {result.message}"
    for name, times in zip(ALARM_NAMES, result.t_events, strict=False):
        if len(times) > 0:
            return f"ALARM [{name}] at t={times[0]:.1f}s"
    return "Terminated by unknown event."


@dataclass(frozen=True)
class SystemState:
    """The boiler state and the turbine performance it gives, at one time."""

    boiler: BoilerState
    turbine: TurbineState
    time: float  # s

    @property
    def electrical_power_mw(self) -> float:
        return self.turbine.electrical_power_mw

    @property
    def steam_flow(self) -> float:
        """Steam mass flow from boiler to turbine [kg/s]."""
        return self.turbine.steam_flow


def evaluate_at(
    system: BoilerTurbineSystem,
    boiler_state: BoilerState,
    controls: ControlInputs,
    time: float = 0.0,
    disturbances: PlantDisturbances = NO_DISTURBANCES,
    exhaust_pressure: float | None = None,
) -> SystemState:
    """Turbine performance for a boiler state; the design back-pressure if None."""
    balance = system.boiler.balance(boiler_state, controls, disturbances)
    return SystemState(
        boiler=boiler_state,
        turbine=system.turbine_at(boiler_state, balance, exhaust_pressure),
        time=time,
    )


def steady_state(
    system: BoilerTurbineSystem,
    fuel_valve: float = 0.7,
    feedwater_valve: float = 0.5,
    steam_valve: float = 0.6,
    t_settle: float = 300.0,
) -> SystemState:
    """Run the boiler from its nominal state at fixed valves, then evaluate the system."""
    controls = ControlInputs(
        fuel_valve_command=fuel_valve,
        feedwater_valve_command=feedwater_valve,
        steam_valve_command=steam_valve,
    )
    initial_state = system.boiler_params.nominal_initial_state()
    result = simulate(system.boiler, initial_state, controls, t_span=(0.0, t_settle))
    final_index = result.y.shape[1] - 1
    return evaluate_at(
        system,
        state_at(result, final_index),
        controls,
        time=float(result.t[final_index]),
    )
