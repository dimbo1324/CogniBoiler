"""
Alarm conditions detected by the PLC.

Every protected parameter is watched against the same limits the interlocks use, and each
condition is reported when it begins and when it ends. A condition ends only once the
value is back inside its limit by a deadband, so a signal hovering on a limit does not
chatter. A condition that is not armed — low drum pressure while the turbine is off line,
a cold furnace before the flame is proven — is not an alarm. Acknowledgement, history and
notification of alarms belong to the alert manager.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from enum import StrEnum

from plc_controller.measurements import (
    SENSOR_DRUM_LEVEL,
    SENSOR_DRUM_PRESSURE,
    SENSOR_DRUM_WATER_TEMP,
    SENSOR_ELECTRICAL_POWER,
    SENSOR_FEEDWATER_FLOW,
    SENSOR_FUEL_FLOW,
    SENSOR_FURNACE_GAS_TEMP,
    SENSOR_STEAM_FLOW,
    SENSOR_STEAM_TEMP,
    SignalQuality,
)
from plc_controller.safety import (
    FLUE_GAS_TEMP_LIMITS,
    PRESSURE_LIMITS,
    PRESSURE_RATE_LIMITS,
    STEAM_TEMP_LIMITS,
    WATER_LEVEL_LIMITS,
    WATER_TEMP_LIMITS,
    ArmingState,
    ParameterLimits,
)

SOURCE_SERVICE: str = "plc-controller"


class Severity(StrEnum):
    WARNING = "warning"
    CRITICAL = "critical"


class Direction(StrEnum):
    HIGH = "high"
    LOW = "low"


class Arming(StrEnum):
    """When a condition is meaningful."""

    ALWAYS = "always"
    ON_LINE = "on_line"  # the turbine takes steam
    FIRING = "firing"  # the flame is proven


@dataclass(frozen=True)
class ConditionRule:
    """One limit on one parameter."""

    parameter: str
    unit: str
    severity: Severity
    direction: Direction
    threshold: float
    deadband: float
    arming: Arming = Arming.ALWAYS

    @property
    def key(self) -> str:
        """Identity of the condition, stable across occurrences."""
        return (
            f"{SOURCE_SERVICE}:{self.parameter}:"
            f"{self.direction.value}:{self.severity.value}"
        )

    @property
    def action(self) -> str:
        return "emergency_stop" if self.severity is Severity.CRITICAL else "warn"

    def breached(self, value: float) -> bool:
        if self.direction is Direction.HIGH:
            return value >= self.threshold
        return value <= self.threshold

    def restored(self, value: float) -> bool:
        if self.direction is Direction.HIGH:
            return value < self.threshold - self.deadband
        return value > self.threshold + self.deadband


@dataclass(frozen=True)
class AlarmCondition:
    """A condition that is currently active."""

    rule: ConditionRule
    value: float
    since_ms: int

    @property
    def key(self) -> str:
        return self.rule.key

    @property
    def message(self) -> str:
        rule = self.rule
        return (
            f"{rule.parameter} {rule.direction.value} {rule.severity.value}: "
            f"{self.value:.4g} {rule.unit} against limit {rule.threshold:.4g} {rule.unit}"
        )


@dataclass(frozen=True)
class AlarmTransition:
    """A condition that became active or ended."""

    condition: AlarmCondition
    active: bool
    timestamp_ms: int


def limit_rules(
    parameter: str,
    unit: str,
    limits: ParameterLimits,
    deadband: float,
    *,
    low_arming: Arming = Arming.ALWAYS,
    high_arming: Arming = Arming.ALWAYS,
) -> tuple[ConditionRule, ...]:
    """Warning and critical rules on both sides of a parameter's limits."""
    rules: list[ConditionRule] = []
    for severity, low, high in (
        (Severity.WARNING, limits.warn_low, limits.warn_high),
        (Severity.CRITICAL, limits.trip_low, limits.trip_high),
    ):
        if math.isfinite(high):
            rules.append(
                ConditionRule(
                    parameter,
                    unit,
                    severity,
                    Direction.HIGH,
                    high,
                    deadband,
                    high_arming,
                )
            )
        if math.isfinite(low):
            rules.append(
                ConditionRule(
                    parameter, unit, severity, Direction.LOW, low, deadband, low_arming
                )
            )
    return tuple(rules)


def quality_rules(sensor_id: str, *, trips: bool) -> tuple[ConditionRule, ...]:
    """A warning for a doubtful instrument; critical if a failed one trips the unit."""
    parameter = f"{sensor_id}_quality"
    warning = ConditionRule(
        parameter,
        "quality",
        Severity.WARNING,
        Direction.HIGH,
        float(SignalQuality.UNCERTAIN),
        0.5,
    )
    if not trips:
        return (warning,)
    return (
        warning,
        ConditionRule(
            parameter,
            "quality",
            Severity.CRITICAL,
            Direction.HIGH,
            float(SignalQuality.BAD),
            0.5,
        ),
    )


DEFAULT_RULES: tuple[ConditionRule, ...] = (
    *limit_rules(
        "pressure_pa", "Pa", PRESSURE_LIMITS, 2.0e5, low_arming=Arming.ON_LINE
    ),
    *limit_rules("water_level_m", "m", WATER_LEVEL_LIMITS, 0.1),
    *limit_rules("water_temp_k", "K", WATER_TEMP_LIMITS, 2.0),
    *limit_rules(
        "flue_gas_temp_k", "K", FLUE_GAS_TEMP_LIMITS, 20.0, low_arming=Arming.FIRING
    ),
    *limit_rules(
        "steam_temp_k", "K", STEAM_TEMP_LIMITS, 3.0, high_arming=Arming.ON_LINE
    ),
    *limit_rules("pressure_rate_pa_s", "Pa/s", PRESSURE_RATE_LIMITS, 1.0e5),
    *quality_rules(SENSOR_DRUM_PRESSURE, trips=True),
    *quality_rules(SENSOR_DRUM_LEVEL, trips=True),
    *(
        rule
        for sensor in (
            SENSOR_DRUM_WATER_TEMP,
            SENSOR_FURNACE_GAS_TEMP,
            SENSOR_STEAM_TEMP,
            SENSOR_STEAM_FLOW,
            SENSOR_FEEDWATER_FLOW,
            SENSOR_FUEL_FLOW,
            SENSOR_ELECTRICAL_POWER,
        )
        for rule in quality_rules(sensor, trips=False)
    ),
)


class AlarmConditionMonitor:
    """Tracks which conditions are active and reports their transitions."""

    def __init__(self, rules: Iterable[ConditionRule] = DEFAULT_RULES) -> None:
        self._rules = tuple(rules)
        keys = [rule.key for rule in self._rules]
        if len(keys) != len(set(keys)):
            raise ValueError("alarm rules must have unique keys")
        self._active: dict[str, AlarmCondition] = {}

    def evaluate(
        self,
        values: Mapping[str, float],
        arming: ArmingState,
        now_ms: int,
    ) -> list[AlarmTransition]:
        """Update every condition from the scan's values; return what changed."""
        transitions: list[AlarmTransition] = []
        for rule in self._rules:
            value = values.get(rule.parameter)
            if value is None or not math.isfinite(value):
                continue
            armed = self._armed(rule.arming, arming)
            current = self._active.get(rule.key)
            if current is None:
                if armed and rule.breached(value):
                    condition = AlarmCondition(rule, value, now_ms)
                    self._active[rule.key] = condition
                    transitions.append(AlarmTransition(condition, True, now_ms))
            elif not armed or rule.restored(value):
                del self._active[rule.key]
                transitions.append(
                    AlarmTransition(replace(current, value=value), False, now_ms)
                )
            else:
                self._active[rule.key] = replace(current, value=value)
        return transitions

    def clear_all(self, now_ms: int) -> list[AlarmTransition]:
        """End every active condition, e.g. when the plant is reset into a new run."""
        transitions = [
            AlarmTransition(condition, False, now_ms)
            for condition in self._active.values()
        ]
        self._active.clear()
        return transitions

    def active(self) -> tuple[AlarmCondition, ...]:
        """Active conditions, critical first, oldest first within a severity."""
        return tuple(
            sorted(
                self._active.values(),
                key=lambda c: (c.rule.severity is not Severity.CRITICAL, c.since_ms),
            )
        )

    def active_critical(self) -> tuple[AlarmCondition, ...]:
        return tuple(c for c in self.active() if c.rule.severity is Severity.CRITICAL)

    @staticmethod
    def _armed(arming: Arming, state: ArmingState) -> bool:
        match arming:
            case Arming.ON_LINE:
                return state.on_line
            case Arming.FIRING:
                return state.firing_proven
            case _:
                return True
