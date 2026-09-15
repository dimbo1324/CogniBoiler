"""
Process measurements as the PLC sees them: instrument readings with their quality.

The PLC acts only on what the instruments report through PhysicsService, never on the
true plant state it cannot know. This module is the boundary between the protobuf
contract and the control and protection logic.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import IntEnum

import cogniboiler_pb2 as pb2


class SignalQuality(IntEnum):
    """Instrument quality; values match `SensorQuality` in the protobuf contract."""

    GOOD = 0
    UNCERTAIN = 1
    BAD = 2


SENSOR_DRUM_PRESSURE: str = "drum_pressure"
SENSOR_DRUM_LEVEL: str = "drum_level"
SENSOR_DRUM_WATER_TEMP: str = "drum_water_temp"
SENSOR_FURNACE_GAS_TEMP: str = "furnace_gas_temp"
SENSOR_STEAM_TEMP: str = "steam_temp"
SENSOR_STEAM_FLOW: str = "steam_flow"
SENSOR_FEEDWATER_FLOW: str = "feedwater_flow"
SENSOR_FUEL_FLOW: str = "fuel_flow"
SENSOR_ELECTRICAL_POWER: str = "electrical_power"


def _quality(value: int) -> SignalQuality:
    try:
        return SignalQuality(value)
    except ValueError:
        return SignalQuality.BAD


@dataclass(frozen=True)
class ValveSet:
    """One value per actuator, normalized to [0, 1]."""

    fuel: float
    feedwater: float
    steam: float
    spray: float


@dataclass(frozen=True)
class ProcessMeasurements:
    """One scan's worth of readings from the plant."""

    simulation_time_s: float
    step_s: float
    run_id: int
    pressure_pa: float
    water_level_m: float
    water_temp_k: float
    flue_gas_temp_k: float
    steam_temp_k: float
    steam_flow_kg_s: float
    spray_flow_kg_s: float
    feedwater_flow_kg_s: float
    fuel_flow_kg_s: float
    electrical_power_w: float
    commands: ValveSet
    positions: ValveSet
    qualities: Mapping[str, SignalQuality] = field(default_factory=dict)

    def quality(self, sensor_id: str) -> SignalQuality:
        """Quality of one instrument; GOOD when the plant does not report it."""
        return self.qualities.get(sensor_id, SignalQuality.GOOD)

    def is_bad(self, sensor_id: str) -> bool:
        return self.quality(sensor_id) is SignalQuality.BAD

    @property
    def drum_steam_flow_kg_s(self) -> float:
        """Steam leaving the drum: turbine flow less the spray water added to it."""
        return max(self.steam_flow_kg_s - self.spray_flow_kg_s, 0.0)

    @classmethod
    def from_proto(cls, state: pb2.SystemStateMsg) -> ProcessMeasurements:
        """Read measurements out of a PhysicsService state message."""
        boiler = state.boiler
        turbine = state.turbine
        actuators = state.actuators
        return cls(
            simulation_time_s=state.simulation_time_s,
            step_s=state.simulation.step_s if state.simulation.step_s > 0 else 1.0,
            run_id=int(state.simulation.run_id),
            pressure_pa=boiler.pressure_pa,
            water_level_m=boiler.water_level_m,
            water_temp_k=boiler.water_temp_k,
            flue_gas_temp_k=boiler.flue_gas_temp_k,
            steam_temp_k=turbine.steam_temp_in_k,
            steam_flow_kg_s=turbine.steam_flow_kg_s,
            spray_flow_kg_s=boiler.spray_flow_kg_s,
            feedwater_flow_kg_s=boiler.feedwater_flow_kg_s,
            fuel_flow_kg_s=boiler.fuel_flow_kg_s,
            electrical_power_w=turbine.electrical_power_w,
            commands=ValveSet(
                fuel=actuators.fuel_valve_command,
                feedwater=actuators.feedwater_valve_command,
                steam=actuators.steam_valve_command,
                spray=actuators.spray_valve_command,
            ),
            positions=ValveSet(
                fuel=actuators.fuel_valve_position,
                feedwater=actuators.feedwater_valve_position,
                steam=actuators.steam_valve_position,
                spray=actuators.spray_valve_position,
            ),
            qualities={
                sensor.sensor_id: _quality(sensor.quality) for sensor in state.sensors
            },
        )
