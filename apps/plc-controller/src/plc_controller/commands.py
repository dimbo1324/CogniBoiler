"""What an operator may ask the PLC for, and the bounds an ask has to fall inside.

The PLC is the only path from a person to an actuator, so this is where a request stops
being text and becomes a number the plant will act on. Every bound here is a hard one:
the physics runtime validates the valve positions again, and the interlocks and the
E-Stop latch stand behind both — a setpoint inside these bounds is not a promise that the
unit will accept it, only that the request itself is well formed.

The setpoint ranges stay inside the alarm warning bands, so an accepted setpoint never
parks the unit in a standing alarm: pressure below the 160 bar warning, level inside
2–7 m, steam below the 565 °C warning.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cogniboiler_pb2 as pb2

from plc_controller.control import RATED_POWER_W
from plc_controller.events import now_ms

# Valve position limits.
VALVE_MIN: float = 0.0
VALVE_MAX: float = 1.0

PRESSURE_SETPOINT_MIN_PA: float = 60.0e5
PRESSURE_SETPOINT_MAX_PA: float = 155.0e5
LEVEL_SETPOINT_MIN_M: float = 2.5
LEVEL_SETPOINT_MAX_M: float = 6.5
TEMP_SETPOINT_MIN_K: float = 700.0
TEMP_SETPOINT_MAX_K: float = 835.0

LOAD_DEMAND_MIN_W: float = 0.0
LOAD_DEMAND_MAX_W: float = RATED_POWER_W

PASCALS_PER_BAR: float = 1.0e5
WATTS_PER_MEGAWATT: float = 1.0e6

# The valves an operator can command by name, in the order a refusal names them.
VALVE_NAMES: tuple[str, ...] = (
    "fuel_valve",
    "feedwater_valve",
    "steam_valve",
    "spray_valve",
)


@dataclass
class Setpoints:
    """Engineer's targets for the process loops."""

    pressure_pa: float = 140.0e5  # 140 bar nominal
    water_level_m: float = 4.8
    steam_temp_k: float = 811.0  # turbine inlet design temperature
    updated_at_ms: int = field(default_factory=now_ms)

    def copy(self) -> Setpoints:
        return Setpoints(
            pressure_pa=self.pressure_pa,
            water_level_m=self.water_level_m,
            steam_temp_k=self.steam_temp_k,
            updated_at_ms=self.updated_at_ms,
        )


@dataclass
class ValidationResult:
    """Result of a command or setpoint validation check."""

    accepted: bool
    reason: str = ""


@dataclass
class CommandSnapshot:
    """A valve command the PLC sent or is about to send."""

    fuel_valve: float = 0.5
    feedwater_valve: float = 0.5
    steam_valve: float = 0.5
    spray_valve: float = 0.0
    source: int = pb2.CommandSource.PID
    operator_id: str = ""
    timestamp_ms: int = field(default_factory=now_ms)

    def copy(self) -> CommandSnapshot:
        return CommandSnapshot(
            fuel_valve=self.fuel_valve,
            feedwater_valve=self.feedwater_valve,
            steam_valve=self.steam_valve,
            spray_valve=self.spray_valve,
            source=self.source,
            operator_id=self.operator_id,
            timestamp_ms=self.timestamp_ms,
        )


def _refuse(reason: str) -> ValidationResult:
    return ValidationResult(accepted=False, reason=reason)


def check_setpoints(
    pressure_pa: float, water_level_m: float, steam_temp_k: float
) -> ValidationResult:
    """Are these targets inside the ranges an engineer may choose?"""
    if not (PRESSURE_SETPOINT_MIN_PA <= pressure_pa <= PRESSURE_SETPOINT_MAX_PA):
        return _refuse(
            f"Pressure setpoint {pressure_pa / PASCALS_PER_BAR:.1f} bar "
            f"outside [{PRESSURE_SETPOINT_MIN_PA / PASCALS_PER_BAR:.0f}, "
            f"{PRESSURE_SETPOINT_MAX_PA / PASCALS_PER_BAR:.0f}] bar"
        )
    if not (LEVEL_SETPOINT_MIN_M <= water_level_m <= LEVEL_SETPOINT_MAX_M):
        return _refuse(
            f"Level setpoint {water_level_m:.2f} m "
            f"outside [{LEVEL_SETPOINT_MIN_M}, {LEVEL_SETPOINT_MAX_M}] m"
        )
    if not (TEMP_SETPOINT_MIN_K <= steam_temp_k <= TEMP_SETPOINT_MAX_K):
        return _refuse(
            f"Steam temp setpoint {steam_temp_k:.1f} K "
            f"outside [{TEMP_SETPOINT_MIN_K}, {TEMP_SETPOINT_MAX_K}] K"
        )
    return ValidationResult(accepted=True)


def check_load_demand(load_w: float) -> ValidationResult:
    """Is this load target one the unit is rated for?"""
    if not LOAD_DEMAND_MIN_W <= load_w <= LOAD_DEMAND_MAX_W:
        return _refuse(
            f"Load demand {load_w / WATTS_PER_MEGAWATT:.1f} MW outside "
            f"[{LOAD_DEMAND_MIN_W / WATTS_PER_MEGAWATT:.0f}, "
            f"{LOAD_DEMAND_MAX_W / WATTS_PER_MEGAWATT:.0f}] MW"
        )
    return ValidationResult(accepted=True)


def check_valves(
    fuel_valve: float,
    feedwater_valve: float,
    steam_valve: float,
    spray_valve: float | None = None,
) -> ValidationResult:
    """Are all four positions fractions of a fully open valve?

    A value that is not a number at all — NaN through a proto field — fails every
    comparison, so it is refused here rather than reaching the plant as a position.
    """
    positions = [fuel_valve, feedwater_valve, steam_valve]
    if spray_valve is not None:
        positions.append(spray_valve)
    for name, value in zip(VALVE_NAMES, positions, strict=False):
        if not (VALVE_MIN <= value <= VALVE_MAX):
            return _refuse(f"{name}={value:.3f} outside [{VALVE_MIN}, {VALVE_MAX}]")
    return ValidationResult(accepted=True)
