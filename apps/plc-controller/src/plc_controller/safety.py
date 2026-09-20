"""Safety interlocks of the virtual PLC.

The interlock layer has more authority than control: whatever the controller or an
operator asks for, a trip shuts the fuel and latches until an explicit reset. What it
watches and where the thresholds are is in `safety_limits.py`; this is what applies them.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping

from plc_controller.safety_limits import (
    ALL_ARMED,
    FLAME_FUEL_FLOW_KG_S,
    FLAME_PROVING_S,
    FLUE_GAS_TEMP_LIMITS,
    ON_LINE_STEAM_FLOW_KG_S,
    PRESSURE_LIMITS,
    PRESSURE_RATE_TRIP,
    PRESSURE_RATE_WARN,
    QUALITY_BAD,
    QUALITY_UNCERTAIN,
    STEAM_TEMP_LIMITS,
    TRIP_SENSORS,
    WATER_LEVEL_LIMITS,
    WATER_TEMP_LIMITS,
    ArmingState,
    ParameterLimits,
    SafetyAction,
    SafetyEvent,
    SafetyLevel,
    SafetyStatus,
    trip_overrides,
)

logger = logging.getLogger(__name__)


# ─── Arming ───────────────────────────────────────────────────────────────────


class ArmingTracker:
    """Derives the arming state from steam and fuel flow over time."""

    def __init__(self) -> None:
        self._firing_s = 0.0
        self._state = ArmingState(on_line=False, firing_proven=False)

    @property
    def state(self) -> ArmingState:
        return self._state

    def update(
        self, steam_flow_kg_s: float, fuel_flow_kg_s: float, dt: float
    ) -> ArmingState:
        if fuel_flow_kg_s >= FLAME_FUEL_FLOW_KG_S:
            self._firing_s += max(dt, 0.0)
        else:
            self._firing_s = 0.0
        self._state = ArmingState(
            on_line=steam_flow_kg_s >= ON_LINE_STEAM_FLOW_KG_S,
            firing_proven=self._firing_s >= FLAME_PROVING_S,
        )
        return self._state

    def reset(self) -> None:
        self._firing_s = 0.0
        self._state = ArmingState(on_line=False, firing_proven=False)


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
        self._last_rate: float | None = None

    @property
    def last_rate(self) -> float | None:
        """Absolute rate computed by the latest check, None without history."""
        return self._last_rate

    def check(self, value: float, dt: float) -> SafetyLevel:
        """
        Compute rate of change and check against thresholds.

        On the first call (no history) always returns NORMAL —
        no rate can be computed without a previous value. A zero interval keeps the
        previous rate and reports NORMAL.

        Args:
            value: Current measurement (SI units).
            dt:    Time step [s] since last call.

        Returns:
            SafetyLevel: NORMAL, WARNING, or TRIP.
        """
        if self._prev_value is None or dt <= 0.0:
            if self._prev_value is None:
                self._prev_value = value
            return SafetyLevel.NORMAL

        rate = abs(value - self._prev_value) / dt
        self._prev_value = value
        self._last_rate = rate

        if rate >= self.trip_rate:
            return SafetyLevel.TRIP
        if rate >= self.warn_rate:
            return SafetyLevel.WARNING
        return SafetyLevel.NORMAL

    def reset(self) -> None:
        """Clear history (call after emergency stop or restart)."""
        self._prev_value = None
        self._last_rate = None


# ─── Emergency stop ───────────────────────────────────────────────────────────


class EmergencyStop:
    """
    Emergency stop state machine.

    When triggered, the emergency stop latches: the unit is held in its trip state
    and cannot restart until an explicit reset() — standard for industrial safety
    systems (IEC 61511).

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

        # A trip is the protection doing its job, not a platform failure: `error` stays
        # reserved for faults of the software itself (owner decision 2026-09-19).
        logger.warning(
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

        status = interlock.check(
            pressure=state.pressure,
            water_level=state.water_level,
            water_temp=state.water_temp,
            flue_gas_temp=state.flue_gas_temp,
            dt=1.0,
        )

        if not status.safe:
            fuel_valve = status.fuel_valve_override   # 0.0 on a trip
    """

    def __init__(
        self,
        pressure_limits: ParameterLimits = PRESSURE_LIMITS,
        water_level_limits: ParameterLimits = WATER_LEVEL_LIMITS,
        water_temp_limits: ParameterLimits = WATER_TEMP_LIMITS,
        flue_gas_temp_limits: ParameterLimits = FLUE_GAS_TEMP_LIMITS,
        steam_temp_limits: ParameterLimits = STEAM_TEMP_LIMITS,
    ) -> None:
        self._pressure_limits = pressure_limits
        self._water_level_limits = water_level_limits
        self._water_temp_limits = water_temp_limits
        self._flue_gas_temp_limits = flue_gas_temp_limits
        self._steam_temp_limits = steam_temp_limits

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
        """Total number of trips, automatic or manual."""
        return self._trip_count

    @property
    def last_pressure_rate(self) -> float | None:
        """Absolute drum pressure rate of change [Pa/s] from the latest check."""
        return self._pressure_rate.last_rate

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
        *,
        steam_temp: float | None = None,
        arming: ArmingState = ALL_ARMED,
        sensor_qualities: Mapping[str, int] | None = None,
    ) -> SafetyStatus:
        """
        Run one safety check cycle against all parameters.

        If the emergency stop is already active, returns a TRIP status with the
        overrides of the latched cause — no further evaluation is needed.

        Args:
            pressure:         Drum pressure [Pa].
            water_level:      Water level in drum [m].
            water_temp:       Bulk water temperature [K].
            flue_gas_temp:    Flue gas temperature [K].
            dt:               Time step since last call [s].
            steam_temp:       Turbine inlet steam temperature [K], if measured.
            arming:           Operating state arming state-dependent protections.
            sensor_qualities: Instrument quality codes by sensor id.

        Returns:
            SafetyStatus with safe flag, worst level, events, and overrides.
        """
        self._check_count += 1

        if self.emergency_stop.is_active:
            self._pressure_rate.check(pressure, dt)
            overrides = trip_overrides(self.emergency_stop.trigger_event)
            return SafetyStatus(
                safe=False,
                level=SafetyLevel.TRIP,
                fuel_valve_override=overrides.fuel,
                steam_valve_override=overrides.steam,
                feedwater_valve_override=overrides.feedwater,
                spray_valve_override=overrides.spray,
            )

        events: list[SafetyEvent] = []

        def record(
            parameter: str,
            value: float,
            threshold: float,
            level: SafetyLevel,
        ) -> None:
            action = (
                SafetyAction.EMERGENCY_STOP
                if level is SafetyLevel.TRIP
                else SafetyAction.WARN
            )
            event = SafetyEvent(
                timestamp_ms=int(time.time() * 1000),
                parameter=parameter,
                value=value,
                threshold=threshold,
                level=level,
                action=action,
            )
            events.append(event)
            logger.warning("Safety event: %s", event.to_dict())

        def evaluate(
            parameter: str,
            value: float,
            limits: ParameterLimits,
            *,
            low_armed: bool = True,
            high_armed: bool = True,
        ) -> None:
            level = limits.check(value)
            if level is SafetyLevel.NORMAL:
                return
            low_side = value <= limits.warn_low
            if not (low_armed if low_side else high_armed):
                return
            if level is SafetyLevel.TRIP:
                threshold = limits.trip_low if low_side else limits.trip_high
            else:
                threshold = limits.warn_low if low_side else limits.warn_high
            record(parameter, value, threshold, level)

        # ── Check each parameter ──────────────────────────────────────────────
        evaluate(
            "pressure_pa", pressure, self._pressure_limits, low_armed=arming.on_line
        )
        evaluate("water_level_m", water_level, self._water_level_limits)
        evaluate("water_temp_k", water_temp, self._water_temp_limits)
        evaluate(
            "flue_gas_temp_k",
            flue_gas_temp,
            self._flue_gas_temp_limits,
            low_armed=arming.firing_proven,
        )
        if steam_temp is not None:
            evaluate(
                "steam_temp_k",
                steam_temp,
                self._steam_temp_limits,
                high_armed=arming.on_line,
            )

        # ── Rate-of-change check for pressure ─────────────────────────────────
        rate_level = self._pressure_rate.check(pressure, dt)
        if rate_level is not SafetyLevel.NORMAL:
            record(
                "pressure_rate_pa_s",
                self._pressure_rate.last_rate or 0.0,
                PRESSURE_RATE_TRIP
                if rate_level is SafetyLevel.TRIP
                else PRESSURE_RATE_WARN,
                rate_level,
            )

        # ── Instruments the unit cannot be protected without ──────────────────
        for sensor in TRIP_SENSORS:
            quality = (sensor_qualities or {}).get(sensor, 0)
            if quality >= QUALITY_BAD:
                record(f"{sensor}_quality", quality, QUALITY_BAD, SafetyLevel.TRIP)
            elif quality >= QUALITY_UNCERTAIN:
                record(
                    f"{sensor}_quality", quality, QUALITY_UNCERTAIN, SafetyLevel.WARNING
                )

        # ── Permissive: block fuel if drum is dry ─────────────────────────────
        if not self.fuel_permitted(water_level):
            record(
                "fuel_permissive",
                water_level,
                self._water_level_limits.trip_low,
                SafetyLevel.TRIP,
            )

        # ── Update counters and trigger E-stop if needed ──────────────────────
        trips = [e for e in events if e.level is SafetyLevel.TRIP]
        if trips:
            self._trip_count += 1
            worst = trips[0]
            event = self.emergency_stop.trigger(
                parameter=worst.parameter,
                value=worst.value,
                threshold=worst.threshold,
            )
            overrides = trip_overrides(event)
            return SafetyStatus(
                safe=False,
                level=SafetyLevel.TRIP,
                events=events,
                fuel_valve_override=overrides.fuel,
                steam_valve_override=overrides.steam,
                feedwater_valve_override=overrides.feedwater,
                spray_valve_override=overrides.spray,
            )

        if events:
            self._warning_count += 1
            return SafetyStatus(safe=False, level=SafetyLevel.WARNING, events=events)

        return SafetyStatus(safe=True, level=SafetyLevel.NORMAL)

    def trip(self, parameter: str, value: float, threshold: float) -> SafetyEvent:
        """Latch the E-Stop on request (a manual trip)."""
        self._trip_count += 1
        return self.emergency_stop.trigger(parameter, value, threshold)

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

    def reset_rate_history(self) -> None:
        """Forget the previous pressure, e.g. when the plant jumps to a new run."""
        self._pressure_rate.reset()
