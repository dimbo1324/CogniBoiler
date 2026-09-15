"""
Measurement layer of the plant: what instruments report instead of the true state.

The plant publishes measured values. A healthy sensor reports the true value with GOOD
quality. A drifting sensor adds a bias that grows with time; once the bias exceeds the
validation tolerance the reading is flagged UNCERTAIN, as a plausibility check against
redundant instruments would do. A failed sensor holds its last value and reports BAD.
Control and protection in the PLC act on these readings, never on the true state.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import IntEnum, StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from physics_engine.faults import ActiveFault


class SensorId(StrEnum):
    """Instruments of the unit that faults can affect."""

    DRUM_PRESSURE = "drum_pressure"
    DRUM_LEVEL = "drum_level"
    DRUM_WATER_TEMP = "drum_water_temp"
    FURNACE_GAS_TEMP = "furnace_gas_temp"
    STEAM_TEMP = "steam_temp"
    STEAM_FLOW = "steam_flow"
    FEEDWATER_FLOW = "feedwater_flow"
    FUEL_FLOW = "fuel_flow"
    ELECTRICAL_POWER = "electrical_power"


class Quality(IntEnum):
    """Signal quality; values match `SensorQuality` in the protobuf contract."""

    GOOD = 0
    UNCERTAIN = 1
    BAD = 2


# Full-scale span of each instrument, in SI units. Drift rates and the validation
# tolerance are expressed as fractions of it.
SENSOR_SPANS: dict[SensorId, float] = {
    SensorId.DRUM_PRESSURE: 200.0e5,
    SensorId.DRUM_LEVEL: 8.0,
    SensorId.DRUM_WATER_TEMP: 400.0,
    SensorId.FURNACE_GAS_TEMP: 1500.0,
    SensorId.STEAM_TEMP: 600.0,
    SensorId.STEAM_FLOW: 350.0,
    SensorId.FEEDWATER_FLOW: 400.0,
    SensorId.FUEL_FLOW: 30.0,
    SensorId.ELECTRICAL_POWER: 350.0e6,
}

# A bias beyond this fraction of span is caught by plausibility checks.
VALIDATION_TOLERANCE: float = 0.02


@dataclass(frozen=True)
class SensorReading:
    """One instrument reading."""

    sensor_id: SensorId
    measured_value: float
    quality: Quality


class SensorBank:
    """Applies drift and failure faults to true process values."""

    def __init__(self) -> None:
        self._held: dict[str, float] = {}

    def reset(self) -> None:
        """Forget values held by failed sensors (a new scenario run)."""
        self._held.clear()

    def read(
        self,
        true_values: Mapping[SensorId, float],
        sensor_faults: Iterable[ActiveFault],
        simulation_time_s: float,
    ) -> dict[SensorId, SensorReading]:
        """Return the reading of every instrument for the current true values."""
        from physics_engine.faults import FaultKind

        readings = {
            sensor: SensorReading(sensor, value, Quality.GOOD)
            for sensor, value in true_values.items()
        }
        live_faults: set[str] = set()
        for fault in sensor_faults:
            sensor = SensorId(fault.spec.target)
            if sensor not in readings:
                continue
            live_faults.add(fault.fault_id)
            current = readings[sensor]
            if fault.spec.kind is FaultKind.SENSOR_FAILURE:
                held = self._held.setdefault(fault.fault_id, current.measured_value)
                readings[sensor] = SensorReading(sensor, held, Quality.BAD)
                continue
            if current.quality is Quality.BAD:
                continue
            span = SENSOR_SPANS[sensor]
            elapsed_min = max(simulation_time_s - fault.started_at_s, 0.0) / 60.0
            bias = fault.spec.severity * span * elapsed_min
            quality = (
                Quality.UNCERTAIN
                if abs(bias) > VALIDATION_TOLERANCE * span
                else current.quality
            )
            readings[sensor] = SensorReading(
                sensor, current.measured_value + bias, quality
            )
        for fault_id in list(self._held):
            if fault_id not in live_faults:
                del self._held[fault_id]
        return readings


def worst_quality(readings: Iterable[SensorReading]) -> Quality:
    """The worst quality among readings; GOOD for none."""
    return max((reading.quality for reading in readings), default=Quality.GOOD)
