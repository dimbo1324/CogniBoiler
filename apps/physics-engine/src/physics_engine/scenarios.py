"""
Scenarios of the live plant: where a run starts and what is scheduled to go wrong.

A scenario is plant-side only — an initial state, the valve positions that come with it,
and optionally faults that strike at a given simulation time. It never contains control
policy: how the unit is then loaded, started or shut down is decided by the PLC and the
operator, as on a real training simulator.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache

from physics_engine import properties
from physics_engine.condenser import COOLING_WATER_TEMP_DESIGN, MIN_BACKPRESSURE
from physics_engine.constants import (
    NOMINAL_LOAD,
    NOMINAL_WATER_LEVEL,
    RATED_POWER,
    TEMP_AMBIENT,
)
from physics_engine.faults import FaultKind, FaultSpec
from physics_engine.models import BoilerParameters, BoilerState, ControlInputs
from physics_engine.operating_point import OperatingPoint, solve_operating_point
from physics_engine.turbine import TurbineParameters

HOT_STANDBY_PRESSURE: float = 100.0e5  # Pa
COLD_WATER_TEMP: float = 373.15  # K
PART_LOAD_FRACTION: float = 0.6


class ScenarioName(StrEnum):
    """Scenarios the physics engine can start or be reset into."""

    STEADY_STATE = "steady_state"
    PART_LOAD = "part_load"
    FULL_LOAD = "full_load"
    HOT_START = "hot_start"
    COLD_START = "cold_start"
    FEEDWATER_PUMP_DRILL = "feedwater_pump_drill"


class StartKind(StrEnum):
    """How the initial state of a scenario is obtained."""

    OPERATING_POINT = "operating_point"
    HOT_STANDBY = "hot_standby"
    COLD = "cold"


class ScenarioError(ValueError):
    """An unknown scenario name."""


@dataclass(frozen=True)
class ScheduledFault:
    """A fault the scenario injects once simulation time reaches `at_s`."""

    at_s: float
    spec: FaultSpec


@dataclass(frozen=True)
class InitialConditions:
    """The plant at simulation time zero."""

    boiler_state: BoilerState
    controls: ControlInputs
    exhaust_pressure_pa: float


@dataclass(frozen=True)
class ScenarioDefinition:
    """One named scenario."""

    name: ScenarioName
    title: str
    description: str
    start: StartKind
    load_w: float = 0.0
    scheduled_faults: tuple[ScheduledFault, ...] = ()

    def initial_conditions(
        self,
        boiler_params: BoilerParameters,
        turbine_params: TurbineParameters,
        cooling_water_temp_k: float,
    ) -> InitialConditions:
        """Build the initial state and valve positions of this scenario."""
        match self.start:
            case StartKind.OPERATING_POINT:
                point = _operating_point(
                    self.load_w, boiler_params, turbine_params, cooling_water_temp_k
                )
                return InitialConditions(
                    boiler_state=point.boiler_state,
                    controls=point.controls.copy(),
                    exhaust_pressure_pa=point.exhaust_pressure_pa,
                )
            case StartKind.HOT_STANDBY:
                water_temp = properties.saturation_temperature(HOT_STANDBY_PRESSURE)
                return _idle_conditions(
                    boiler_params,
                    pressure=HOT_STANDBY_PRESSURE,
                    water_temp=water_temp,
                    gas_temp=water_temp,
                )
            case StartKind.COLD:
                return _idle_conditions(
                    boiler_params,
                    pressure=properties.saturation_pressure(COLD_WATER_TEMP),
                    water_temp=COLD_WATER_TEMP,
                    gas_temp=TEMP_AMBIENT,
                )


def _idle_conditions(
    params: BoilerParameters,
    *,
    pressure: float,
    water_temp: float,
    gas_temp: float,
) -> InitialConditions:
    water_mass = (
        properties.liquid_density(water_temp)
        * params.drum_cross_section
        * NOMINAL_WATER_LEVEL
    )
    return InitialConditions(
        boiler_state=BoilerState(
            internal_energy=water_mass * properties.liquid_enthalpy(water_temp),
            pressure=pressure,
            water_level=NOMINAL_WATER_LEVEL,
            flue_gas_temp=gas_temp,
            water_temp=water_temp,
        ),
        controls=ControlInputs(
            fuel_valve_command=0.0,
            feedwater_valve_command=0.0,
            steam_valve_command=0.0,
            spray_valve_command=0.0,
        ),
        exhaust_pressure_pa=MIN_BACKPRESSURE,
    )


@lru_cache(maxsize=16)
def _default_operating_point(
    load_w: float, cooling_water_temp_k: float
) -> OperatingPoint:
    return solve_operating_point(load_w, cooling_water_temp_k=cooling_water_temp_k)


def _operating_point(
    load_w: float,
    boiler_params: BoilerParameters,
    turbine_params: TurbineParameters,
    cooling_water_temp_k: float,
) -> OperatingPoint:
    if boiler_params == BoilerParameters() and turbine_params == TurbineParameters():
        return _default_operating_point(load_w, cooling_water_temp_k)
    return solve_operating_point(
        load_w,
        boiler_params=boiler_params,
        turbine_params=turbine_params,
        cooling_water_temp_k=cooling_water_temp_k,
    )


SCENARIOS: dict[ScenarioName, ScenarioDefinition] = {
    definition.name: definition
    for definition in (
        ScenarioDefinition(
            name=ScenarioName.STEADY_STATE,
            title="Nominal load",
            description="Unit steady at 250 MW, 140 bar, drum level 4.8 m.",
            start=StartKind.OPERATING_POINT,
            load_w=NOMINAL_LOAD,
        ),
        ScenarioDefinition(
            name=ScenarioName.PART_LOAD,
            title="Part load",
            description="Unit steady at 60 % load (180 MW), ready for a load increase.",
            start=StartKind.OPERATING_POINT,
            load_w=PART_LOAD_FRACTION * RATED_POWER,
        ),
        ScenarioDefinition(
            name=ScenarioName.FULL_LOAD,
            title="Full load",
            description="Unit steady at its 300 MW rating.",
            start=StartKind.OPERATING_POINT,
            load_w=RATED_POWER,
        ),
        ScenarioDefinition(
            name=ScenarioName.HOT_START,
            title="Hot start",
            description=(
                "Burners off, turbine isolated, drum hot at 100 bar after a short stop."
            ),
            start=StartKind.HOT_STANDBY,
        ),
        ScenarioDefinition(
            name=ScenarioName.COLD_START,
            title="Cold start",
            description="Drum filled with water at 100 °C and atmospheric pressure.",
            start=StartKind.COLD,
        ),
        ScenarioDefinition(
            name=ScenarioName.FEEDWATER_PUMP_DRILL,
            title="Feedwater pump failure drill",
            description="Nominal load; the feedwater pump trips two minutes in.",
            start=StartKind.OPERATING_POINT,
            load_w=NOMINAL_LOAD,
            scheduled_faults=(
                ScheduledFault(
                    at_s=120.0,
                    spec=FaultSpec(kind=FaultKind.FEEDWATER_PUMP_FAILURE, severity=1.0),
                ),
            ),
        ),
    )
}


def get_scenario(name: str) -> ScenarioDefinition:
    """Resolve a scenario by name, raising ScenarioError for an unknown one."""
    try:
        return SCENARIOS[ScenarioName(name)]
    except ValueError as exc:
        raise ScenarioError(
            f"unknown scenario {name!r}; known: {', '.join(ScenarioName)}"
        ) from exc


__all__ = [
    "COOLING_WATER_TEMP_DESIGN",
    "SCENARIOS",
    "InitialConditions",
    "ScenarioDefinition",
    "ScenarioError",
    "ScenarioName",
    "ScheduledFault",
    "StartKind",
    "get_scenario",
]
