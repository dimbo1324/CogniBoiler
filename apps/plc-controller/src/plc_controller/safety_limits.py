"""What the protection watches, and where its thresholds are.

Levels, actions, the limits of every protected parameter and the arming rules that decide
when a protection is meaningful at all. The machinery that applies them — the rate
limiter, the E-Stop latch and the interlock itself — is in `safety.py`; splitting them
keeps each file readable and lets the alarm rules import the limits without pulling in the
latch.

Protection levels per parameter:
    WARN_LOW  / WARN_HIGH  — advisory, operator notification only
    TRIP_LOW  / TRIP_HIGH  — immediate emergency stop, no delay

Some protections are meaningful only in an operating state and are armed by it, as in a
burner management system — this is protection logic, not a way to switch it off:
    low drum pressure        armed while the turbine is on line
    low furnace temperature  armed once the flame is proven (firing for 10 s)
    high steam temperature   armed while the turbine is on line
A failed drum pressure or level instrument trips the unit: it can no longer be protected.

What a trip does depends on its cause. Fuel and spray always shut. The turbine valve
opens fully only to relieve high drum pressure; otherwise it closes to keep the water
and heat in the boiler. Feedwater stops only for a high drum level; otherwise level
control keeps the drum wet.

Thresholds are code constants covered by tests; no configuration changes them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

# ─── Enums ────────────────────────────────────────────────────────────────────


class SafetyLevel(StrEnum):
    """Severity level of a safety event."""

    NORMAL = "normal"
    WARNING = "warning"
    TRIP = "trip"


class SafetyAction(StrEnum):
    """Action taken in response to a safety event."""

    NONE = "none"
    WARN = "warn"
    EMERGENCY_STOP = "emergency_stop"


# ─── Data classes ─────────────────────────────────────────────────────────────


@dataclass
class ParameterLimits:
    """
    Four-level protection limits for a single process parameter.

    All values must be in SI units (Pa, K, m, etc.) — same units
    as the physics engine outputs. An infinite limit leaves that side unprotected.

    Attributes:
        warn_low:  Advisory low limit. Below this -> WARNING.
        warn_high: Advisory high limit. Above this -> WARNING.
        trip_low:  Emergency low limit. Below this -> TRIP.
        trip_high: Emergency high limit. Above this -> TRIP.
    """

    warn_low: float
    warn_high: float
    trip_low: float
    trip_high: float

    def __post_init__(self) -> None:
        if not (self.trip_low <= self.warn_low <= self.warn_high <= self.trip_high):
            raise ValueError(
                f"Limits must satisfy trip_low ≤ warn_low ≤ warn_high ≤ trip_high. "
                f"Got: {self.trip_low} ≤ {self.warn_low} ≤ {self.warn_high} ≤ {self.trip_high}"
            )

    def check(self, value: float) -> SafetyLevel:
        """
        Evaluate a measured value against the limits.

        Args:
            value: Current measured value (SI units).

        Returns:
            SafetyLevel: NORMAL, WARNING, or TRIP.
        """
        if value <= self.trip_low or value >= self.trip_high:
            return SafetyLevel.TRIP
        if value <= self.warn_low or value >= self.warn_high:
            return SafetyLevel.WARNING
        return SafetyLevel.NORMAL


@dataclass
class SafetyEvent:
    """
    Record of a single safety event for structured logging.

    Attributes:
        timestamp_ms: UTC epoch milliseconds when the event occurred.
        parameter:    Name of the parameter that triggered the event.
        value:        Measured value at the time of the event (SI).
        threshold:    The limit that was crossed.
        level:        WARNING or TRIP.
        action:       Action taken (WARN or EMERGENCY_STOP).
    """

    timestamp_ms: int
    parameter: str
    value: float
    threshold: float
    level: SafetyLevel
    action: SafetyAction

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for structured logging."""
        return {
            "timestamp_ms": self.timestamp_ms,
            "parameter": self.parameter,
            "value": self.value,
            "threshold": self.threshold,
            "level": self.level.value,
            "action": self.action.value,
        }


@dataclass
class SafetyStatus:
    """
    Result of a single safety check cycle.

    Returned by SafetyInterlock.check() on every time step.

    Attributes:
        safe:   True if all parameters are within normal limits.
        level:  Worst SafetyLevel seen across all parameters.
        events: List of all events triggered this cycle.
        fuel_valve_override:      If not None, force the fuel valve to this value.
        steam_valve_override:     If not None, force the turbine valve to this value.
        feedwater_valve_override: If not None, force the feedwater valve to this value.
        spray_valve_override:     If not None, force the spray valve to this value.
    """

    safe: bool
    level: SafetyLevel
    events: list[SafetyEvent] = field(default_factory=list)
    fuel_valve_override: float | None = None
    steam_valve_override: float | None = None
    feedwater_valve_override: float | None = None
    spray_valve_override: float | None = None


@dataclass(frozen=True)
class ArmingState:
    """Operating state that arms state-dependent protections."""

    on_line: bool = True
    firing_proven: bool = True


ALL_ARMED = ArmingState()


@dataclass(frozen=True)
class TripOverrides:
    """Valve positions a latched trip imposes; None leaves the valve to control."""

    fuel: float = 0.0
    steam: float = 0.0
    feedwater: float | None = None
    spray: float = 0.0


# ─── Default safety limits ────────────────────────────────────────────────────

# Pressure limits [Pa]
PRESSURE_LIMITS = ParameterLimits(
    warn_low=50.0e5,  # 50 bar  — low pressure warning
    warn_high=160.0e5,  # 160 bar — high pressure warning
    trip_low=20.0e5,  # 20 bar  — low pressure trip
    trip_high=185.0e5,  # 185 bar — high pressure trip (design limit)
)

# Water level limits [m]
WATER_LEVEL_LIMITS = ParameterLimits(
    warn_low=2.0,  # 2 m  — low level warning
    warn_high=7.0,  # 7 m  — high level warning
    trip_low=0.5,  # 0.5 m — drum nearly dry -> trip
    trip_high=7.8,  # 7.8 m — drum nearly full -> trip
)

# Water temperature limits [K]
WATER_TEMP_LIMITS = ParameterLimits(
    warn_low=373.0,  # 100°C — abnormally cold water
    warn_high=630.0,  # 357°C — approaching critical temp
    trip_low=320.0,  # 47°C  — critically cold (sensor fault likely)
    trip_high=648.0,  # 375°C — above critical point -> trip
)

# Flue gas temperature limits [K]
FLUE_GAS_TEMP_LIMITS = ParameterLimits(
    warn_low=500.0,  # 227°C — furnace too cold (poor combustion)
    warn_high=1500.0,  # 1227°C — furnace very hot
    trip_low=300.0,  # 27°C  — furnace cold (flame out)
    trip_high=1700.0,  # 1427°C — furnace critically hot
)

# Turbine inlet steam temperature limits [K]: 565 °C warns, 580 °C trips — beyond it
# superheater tubes and the turbine inlet creep far faster than designed.
STEAM_TEMP_LIMITS = ParameterLimits(
    warn_low=-math.inf,
    warn_high=838.15,
    trip_low=-math.inf,
    trip_high=853.15,
)

# Rate of change limits [Pa/s] for pressure
PRESSURE_RATE_WARN = 5.0e5  # 5 bar/s — warning
PRESSURE_RATE_TRIP = 10.0e5  # 10 bar/s — trip
PRESSURE_RATE_LIMITS = ParameterLimits(
    warn_low=-math.inf,
    warn_high=PRESSURE_RATE_WARN,
    trip_low=-math.inf,
    trip_high=PRESSURE_RATE_TRIP,
)

# Arming thresholds
ON_LINE_STEAM_FLOW_KG_S: float = 24.5  # 10 % of rated turbine steam flow
FLAME_FUEL_FLOW_KG_S: float = 1.0
FLAME_PROVING_S: float = 10.0

# Instruments whose failure trips the unit, and their quality codes (SensorQuality).
TRIP_SENSORS: tuple[str, ...] = ("drum_pressure", "drum_level")
QUALITY_UNCERTAIN: int = 1
QUALITY_BAD: int = 2


def trip_overrides(event: SafetyEvent | None) -> TripOverrides:
    """Valve positions a latched trip imposes, by its cause."""
    if event is None:
        return TripOverrides()
    high_side = event.value >= event.threshold
    vent = event.parameter == "pressure_pa" and high_side
    feedwater = 0.0 if event.parameter == "water_level_m" and high_side else None
    return TripOverrides(steam=1.0 if vent else 0.0, feedwater=feedwater)
