"""
Safety Interlock Layer for the CogniBoiler simulation.

Implements three independent protection classes that operate on top of
the PID controller. The safety layer has higher authority than the
controller — it can override any control output.

Architecture:
    Operator command
        ↓
    BoilerController (PID)
        ↓
    SafetyInterlock.check()   ← checks all parameters
        ↓
    RateOfChangeLimiter.check() ← checks rate of change
        ↓
    EmergencyStop (if trip)   ← overrides all outputs to safe state
        ↓
    Physics Engine

Protection levels (per parameter):
    WARN_LOW  / WARN_HIGH  — advisory, operator notification only
    TRIP_LOW  / TRIP_HIGH  — immediate emergency stop, no delay

Logging:
    All safety events are logged as structured JSON via structlog.
    The log contains: timestamp, parameter, value, threshold, action.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# ─── Enums ────────────────────────────────────────────────────────────────────


class SafetyLevel(str, Enum):
    """Severity level of a safety event."""

    NORMAL = "normal"
    WARNING = "warning"
    TRIP = "trip"


class SafetyAction(str, Enum):
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
    as the physics engine outputs.

    Attributes:
        warn_low:  Advisory low limit. Below this → WARNING.
        warn_high: Advisory high limit. Above this → WARNING.
        trip_low:  Emergency low limit. Below this → TRIP.
        trip_high: Emergency high limit. Above this → TRIP.
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
        fuel_valve_override: If not None, override fuel valve to this value.
        steam_valve_override: If not None, override steam valve to this value.
    """

    safe: bool
    level: SafetyLevel
    events: list[SafetyEvent] = field(default_factory=list)
    fuel_valve_override: float | None = None
    steam_valve_override: float | None = None


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
    trip_low=0.5,  # 0.5 m — drum nearly dry → trip
    trip_high=7.8,  # 7.8 m — drum nearly full → trip
)

# Water temperature limits [K]
WATER_TEMP_LIMITS = ParameterLimits(
    warn_low=373.0,  # 100°C — abnormally cold water
    warn_high=630.0,  # 357°C — approaching critical temp
    trip_low=320.0,  # 47°C  — critically cold (sensor fault likely)
    trip_high=648.0,  # 375°C — above critical point → trip
)

# Flue gas temperature limits [K]
FLUE_GAS_TEMP_LIMITS = ParameterLimits(
    warn_low=500.0,  # 227°C — furnace too cold (poor combustion)
    warn_high=1500.0,  # 1227°C — furnace very hot
    trip_low=300.0,  # 27°C  — furnace cold (flame out)
    trip_high=1700.0,  # 1427°C — furnace critically hot
)

# Rate of change limits [Pa/s] for pressure
PRESSURE_RATE_WARN = 5.0e5  # 5 bar/s — warning
PRESSURE_RATE_TRIP = 10.0e5  # 10 bar/s — trip


# ─── Rate-of-change limiter ───────────────────────────────────────────────────


class RateOfChangeLimiter:
    """
    Monitors the rate of change of a process variable.

    Real boilers have maximum safe ramp rates. A pressure spike of
    10 bar/s indicates either a sensor fault or a catastrophic event.

    Keeps a rolling history of the last measurement to compute
    the first-order derivative: rate = (value_now - value_prev) / dt

    Usage:
        limiter = RateOfChangeLimiter("pressure_pa", warn=5e5, trip=10e5)
        level = limiter.check(pressure, dt=1.0)
    """

    def __init__(
        self,
        parameter: str,
        warn_rate: float,
        trip_rate: float,
    ) -> None:
        """
        Args:
            parameter:  Parameter name (for logging).
            warn_rate:  Absolute rate threshold for WARNING [units/s].
            trip_rate:  Absolute rate threshold for TRIP [units/s].
        """
        self.parameter = parameter
        self.warn_rate = warn_rate
        self.trip_rate = trip_rate
        self._prev_value: float | None = None

    def check(self, value: float, dt: float) -> SafetyLevel:
        """
        Compute rate of change and check against thresholds.

        On the first call (no history) always returns NORMAL —
        no rate can be computed without a previous value.

        Args:
            value: Current measurement (SI units).
            dt:    Time step [s] since last call.

        Returns:
            SafetyLevel: NORMAL, WARNING, or TRIP.
        """
        if self._prev_value is None or dt <= 0.0:
            self._prev_value = value
            return SafetyLevel.NORMAL

        rate = abs(value - self._prev_value) / dt
        self._prev_value = value

        if rate >= self.trip_rate:
            return SafetyLevel.TRIP
        if rate >= self.warn_rate:
            return SafetyLevel.WARNING
        return SafetyLevel.NORMAL

    def reset(self) -> None:
        """Clear history (call after emergency stop or restart)."""
        self._prev_value = None


# ─── Emergency stop ───────────────────────────────────────────────────────────


class EmergencyStop:
    """
    Emergency stop state machine.

    When triggered, the emergency stop:
      1. Forces fuel valve to 0.0 (cuts all heat input immediately)
      2. Forces steam valve to 1.0 (dumps steam to reduce pressure)
      3. Locks the system — restart requires explicit operator reset()

    The system cannot be restarted without calling reset() — this
    prevents automatic restart after a trip, which is standard in
    industrial safety systems (IEC 61511).

    Usage:
        estop = EmergencyStop()
        estop.trigger("pressure_pa", value=190e5, threshold=185e5)
        assert estop.is_active
        # ... operator investigates and clears the trip cause ...
        estop.reset(operator_id="operator_1")
        assert not estop.is_active
    """

    def __init__(self) -> None:
        self._active: bool = False
        self._trigger_event: SafetyEvent | None = None
        self._reset_count: int = 0

    @property
    def is_active(self) -> bool:
        """True if the emergency stop has been triggered and not reset."""
        return self._active

    @property
    def trigger_event(self) -> SafetyEvent | None:
        """The event that caused the last trip, or None if never tripped."""
        return self._trigger_event

    @property
    def reset_count(self) -> int:
        """Number of times this instance has been reset by an operator."""
        return self._reset_count

    def trigger(
        self,
        parameter: str,
        value: float,
        threshold: float,
    ) -> SafetyEvent:
        """
        Activate the emergency stop.

        Idempotent — calling trigger() when already active records
        the new event but does not change the locked state.

        Args:
            parameter:  Name of the parameter that caused the trip.
            value:      Measured value that exceeded the threshold.
            threshold:  The limit that was crossed.

        Returns:
            SafetyEvent describing this trip.
        """
        event = SafetyEvent(
            timestamp_ms=int(time.time() * 1000),
            parameter=parameter,
            value=value,
            threshold=threshold,
            level=SafetyLevel.TRIP,
            action=SafetyAction.EMERGENCY_STOP,
        )
        self._active = True
        self._trigger_event = event

        logger.error(
            "EMERGENCY STOP triggered: %s",
            event.to_dict(),
        )
        return event

    def reset(self, operator_id: str = "unknown") -> None:
        """
        Clear the emergency stop latch (operator action required).

        Args:
            operator_id: ID of the operator performing the reset.
                         Logged for audit trail.
        """
        if not self._active:
            return

        self._active = False
        self._reset_count += 1
        logger.warning(
            "Emergency stop RESET by operator=%s  reset_count=%d",
            operator_id,
            self._reset_count,
        )


# ─── Main safety interlock ────────────────────────────────────────────────────


class SafetyInterlock:
    """
    Main safety interlock: checks all process parameters every time step.

    Integrates ParameterLimits, RateOfChangeLimiter, and EmergencyStop
    into a single check() call that returns SafetyStatus.

    Permissive logic:
        The fuel valve can only be opened if water level > trip_low.
        This prevents firing the furnace with an empty drum.

    Usage:
        interlock = SafetyInterlock()

        # In the control loop, after controller.step():
        status = interlock.check(
            pressure=state.pressure,
            water_level=state.water_level,
            water_temp=state.water_temp,
            flue_gas_temp=state.flue_gas_temp,
            dt=1.0,
        )

        if not status.safe:
            fuel_valve = status.fuel_valve_override   # 0.0
            steam_valve = status.steam_valve_override  # 1.0
    """

    def __init__(
        self,
        pressure_limits: ParameterLimits = PRESSURE_LIMITS,
        water_level_limits: ParameterLimits = WATER_LEVEL_LIMITS,
        water_temp_limits: ParameterLimits = WATER_TEMP_LIMITS,
        flue_gas_temp_limits: ParameterLimits = FLUE_GAS_TEMP_LIMITS,
    ) -> None:
        self._pressure_limits = pressure_limits
        self._water_level_limits = water_level_limits
        self._water_temp_limits = water_temp_limits
        self._flue_gas_temp_limits = flue_gas_temp_limits

        self._pressure_rate = RateOfChangeLimiter(
            parameter="pressure_pa",
            warn_rate=PRESSURE_RATE_WARN,
            trip_rate=PRESSURE_RATE_TRIP,
        )
        self.emergency_stop = EmergencyStop()

        self._check_count: int = 0
        self._warning_count: int = 0
        self._trip_count: int = 0

    # ─── Stats ───────────────────────────────────────────────────────────────

    @property
    def check_count(self) -> int:
        """Total number of check() calls since creation."""
        return self._check_count

    @property
    def warning_count(self) -> int:
        """Total number of checks that resulted in WARNING."""
        return self._warning_count

    @property
    def trip_count(self) -> int:
        """Total number of checks that resulted in TRIP."""
        return self._trip_count

    # ─── Permissive check ─────────────────────────────────────────────────────

    def fuel_permitted(self, water_level: float) -> bool:
        """
        Check permissive logic: can the fuel valve be opened?

        Fuel is only permitted if the drum has enough water to absorb heat.
        Opening the furnace with an empty drum would cause a dry-fire
        explosion — this is a hard interlock, not advisory.

        Args:
            water_level: Current water level in drum [m].

        Returns:
            True if fuel valve is permitted to open.
        """
        return water_level > self._water_level_limits.trip_low

    # ─── Main check ───────────────────────────────────────────────────────────

    def check(
        self,
        pressure: float,
        water_level: float,
        water_temp: float,
        flue_gas_temp: float,
        dt: float = 1.0,
    ) -> SafetyStatus:
        """
        Run one safety check cycle against all parameters.

        If the emergency stop is already active, immediately returns
        a TRIP status with overrides — no further evaluation needed.

        Args:
            pressure:      Drum pressure [Pa].
            water_level:   Water level in drum [m].
            water_temp:    Bulk water temperature [K].
            flue_gas_temp: Flue gas temperature [K].
            dt:            Time step since last call [s].

        Returns:
            SafetyStatus with safe flag, worst level, events, and overrides.
        """
        self._check_count += 1

        # If E-stop already active — return immediately
        if self.emergency_stop.is_active:
            return SafetyStatus(
                safe=False,
                level=SafetyLevel.TRIP,
                fuel_valve_override=0.0,
                steam_valve_override=1.0,
            )

        events: list[SafetyEvent] = []
        worst_level = SafetyLevel.NORMAL

        # ── Helper: evaluate one parameter ────────────────────────────────────
        def _evaluate(
            param_name: str,
            value: float,
            limits: ParameterLimits,
            level: SafetyLevel,
        ) -> None:
            nonlocal worst_level
            if level == SafetyLevel.NORMAL:
                return

            # Determine which threshold was crossed
            if level == SafetyLevel.TRIP:
                threshold = (
                    limits.trip_low if value < limits.warn_low else limits.trip_high
                )
                action = SafetyAction.EMERGENCY_STOP
            else:
                threshold = (
                    limits.warn_low if value < limits.warn_low else limits.warn_high
                )
                action = SafetyAction.WARN

            event = SafetyEvent(
                timestamp_ms=int(time.time() * 1000),
                parameter=param_name,
                value=value,
                threshold=threshold,
                level=level,
                action=action,
            )
            events.append(event)

            if level.value > worst_level.value:
                worst_level = level

            logger.warning("Safety event: %s", event.to_dict())

        # ── Check each parameter ──────────────────────────────────────────────
        _evaluate(
            "pressure_pa",
            pressure,
            self._pressure_limits,
            self._pressure_limits.check(pressure),
        )
        _evaluate(
            "water_level_m",
            water_level,
            self._water_level_limits,
            self._water_level_limits.check(water_level),
        )
        _evaluate(
            "water_temp_k",
            water_temp,
            self._water_temp_limits,
            self._water_temp_limits.check(water_temp),
        )
        _evaluate(
            "flue_gas_temp_k",
            flue_gas_temp,
            self._flue_gas_temp_limits,
            self._flue_gas_temp_limits.check(flue_gas_temp),
        )

        # ── Rate-of-change check for pressure ─────────────────────────────────
        rate_level = self._pressure_rate.check(pressure, dt)
        if rate_level != SafetyLevel.NORMAL:
            rate = abs(pressure - (self._pressure_rate._prev_value or pressure))
            threshold = (
                PRESSURE_RATE_TRIP
                if rate_level == SafetyLevel.TRIP
                else PRESSURE_RATE_WARN
            )
            action = (
                SafetyAction.EMERGENCY_STOP
                if rate_level == SafetyLevel.TRIP
                else SafetyAction.WARN
            )
            event = SafetyEvent(
                timestamp_ms=int(time.time() * 1000),
                parameter="pressure_rate_pa_s",
                value=rate,
                threshold=threshold,
                level=rate_level,
                action=action,
            )
            events.append(event)
            if rate_level.value > worst_level.value:
                worst_level = rate_level
            logger.warning("Rate-of-change event: %s", event.to_dict())

        # ── Permissive: block fuel if drum is dry ─────────────────────────────
        if not self.fuel_permitted(water_level):
            event = SafetyEvent(
                timestamp_ms=int(time.time() * 1000),
                parameter="fuel_permissive",
                value=water_level,
                threshold=self._water_level_limits.trip_low,
                level=SafetyLevel.TRIP,
                action=SafetyAction.EMERGENCY_STOP,
            )
            events.append(event)
            worst_level = SafetyLevel.TRIP
            logger.error(
                "Fuel permissive DENIED — drum level too low: %s", event.to_dict()
            )

        # ── Update counters and trigger E-stop if needed ──────────────────────
        if worst_level == SafetyLevel.WARNING:
            self._warning_count += 1

        if worst_level == SafetyLevel.TRIP:
            self._trip_count += 1
            # Find the worst event to pass to emergency stop
            trip_events = [e for e in events if e.level == SafetyLevel.TRIP]
            if trip_events:
                worst = trip_events[0]
                self.emergency_stop.trigger(
                    parameter=worst.parameter,
                    value=worst.value,
                    threshold=worst.threshold,
                )
            return SafetyStatus(
                safe=False,
                level=SafetyLevel.TRIP,
                events=events,
                fuel_valve_override=0.0,
                steam_valve_override=1.0,
            )

        return SafetyStatus(
            safe=worst_level == SafetyLevel.NORMAL,
            level=worst_level,
            events=events,
            fuel_valve_override=None,
            steam_valve_override=None,
        )

    def reset(self, operator_id: str = "unknown") -> None:
        """
        Reset the safety interlock after an emergency stop.

        Clears the E-stop latch and rate-of-change history.
        Must be called explicitly by an operator before restart.

        Args:
            operator_id: Operator performing the reset (for audit log).
        """
        self.emergency_stop.reset(operator_id=operator_id)
        self._pressure_rate.reset()
