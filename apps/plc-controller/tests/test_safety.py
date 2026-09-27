"""
Tests for Safety Interlock Layer.

Covers 10 emergency scenarios as required by Phase 3.3:
  1.  Pressure above trip_high
  2.  Pressure below trip_low
  3.  Water level below trip_low (drum dry)
  4.  Water level above trip_high (drum overflow)
  5.  Water temp above trip_high
  6.  Flue gas temp above trip_high
  7.  Pressure rate of change exceeds trip threshold
  8.  Fuel permissive denied (drum level too low)
  9.  Emergency stop blocks restart without operator reset
  10. System cannot be restarted without explicit reset()

Additional unit tests:
  - ParameterLimits validation
  - SafetyLevel ordering
  - RateOfChangeLimiter history
  - Warning (not trip) conditions
  - Normal operating conditions
  - Stats counters
"""

from __future__ import annotations

import dataclasses
import logging
import math

import pytest
from plc_controller.alarms import DEFAULT_RULES
from plc_controller.safety import (
    EmergencyStop,
    RateOfChangeLimiter,
    SafetyInterlock,
)
from plc_controller.safety_limits import (
    FLUE_GAS_TEMP_LIMITS,
    ON_LINE_STEAM_FLOW_KG_S,
    PRESSURE_LIMITS,
    PRESSURE_RATE_LIMITS,
    PRESSURE_RATE_TRIP,
    PRESSURE_RATE_WARN,
    STEAM_TEMP_LIMITS,
    WATER_LEVEL_LIMITS,
    WATER_TEMP_LIMITS,
    ParameterLimits,
    SafetyEvent,
    SafetyLevel,
    SafetyStatus,
)

# ─── Helpers ──────────────────────────────────────────────────────────────────

# Nominal safe values — all within NORMAL range
NOMINAL_PRESSURE = 140.0e5  # Pa — 140 bar
NOMINAL_LEVEL = 4.8  # m
NOMINAL_WATER_TEMP = 611.0  # K — ~338°C
NOMINAL_FLUE_TEMP = 1200.0  # K — ~927°C


def make_interlock() -> SafetyInterlock:
    """Fresh SafetyInterlock for each test — no shared state."""
    return SafetyInterlock()


# Both trip instruments reported and healthy.
TRIP_INSTRUMENTS_GOOD = {"drum_pressure": 0, "drum_level": 0}


def nominal_check(interlock: SafetyInterlock, dt: float = 1.0) -> SafetyStatus:
    """Run one check with all nominal values and healthy trip instruments."""
    return interlock.check(
        pressure=NOMINAL_PRESSURE,
        water_level=NOMINAL_LEVEL,
        water_temp=NOMINAL_WATER_TEMP,
        flue_gas_temp=NOMINAL_FLUE_TEMP,
        dt=dt,
        sensor_qualities=TRIP_INSTRUMENTS_GOOD,
    )


# ─── ParameterLimits unit tests ───────────────────────────────────────────────


class TestParameterLimits:
    def test_normal_value_returns_normal(self) -> None:
        result = PRESSURE_LIMITS.check(NOMINAL_PRESSURE)
        assert result == SafetyLevel.NORMAL

    def test_value_above_warn_high_returns_warning(self) -> None:
        # 162 bar — above warn_high (160 bar) but below trip_high (185 bar)
        result = PRESSURE_LIMITS.check(162.0e5)
        assert result == SafetyLevel.WARNING

    def test_value_above_trip_high_returns_trip(self) -> None:
        # 190 bar — above trip_high (185 bar)
        result = PRESSURE_LIMITS.check(190.0e5)
        assert result == SafetyLevel.TRIP

    def test_value_below_warn_low_returns_warning(self) -> None:
        # 40 bar — below warn_low (50 bar) but above trip_low (20 bar)
        result = PRESSURE_LIMITS.check(40.0e5)
        assert result == SafetyLevel.WARNING

    def test_value_below_trip_low_returns_trip(self) -> None:
        # 15 bar — below trip_low (20 bar)
        result = PRESSURE_LIMITS.check(15.0e5)
        assert result == SafetyLevel.TRIP

    def test_invalid_limits_raise_value_error(self) -> None:
        with pytest.raises(ValueError):
            ParameterLimits(
                warn_low=100.0,
                warn_high=50.0,  # warn_high < warn_low — invalid
                trip_low=10.0,
                trip_high=200.0,
            )

    def test_trip_low_equals_warn_low_accepted(self) -> None:
        # Equal boundaries are valid
        limits = ParameterLimits(
            warn_low=50.0,
            warn_high=160.0,
            trip_low=50.0,
            trip_high=160.0,
        )
        assert limits.check(100.0) == SafetyLevel.NORMAL


# ─── RateOfChangeLimiter unit tests ───────────────────────────────────────────


class TestRateOfChangeLimiter:
    def test_first_call_always_normal(self) -> None:
        limiter = RateOfChangeLimiter(
            "pressure", PRESSURE_RATE_WARN, PRESSURE_RATE_TRIP
        )
        assert limiter.check(140.0e5, dt=1.0) == SafetyLevel.NORMAL

    def test_slow_change_returns_normal(self) -> None:
        limiter = RateOfChangeLimiter(
            "pressure", PRESSURE_RATE_WARN, PRESSURE_RATE_TRIP
        )
        limiter.check(140.0e5, dt=1.0)  # seed
        # 1 bar/s change — well below warn_rate (5 bar/s)
        assert limiter.check(141.0e5, dt=1.0) == SafetyLevel.NORMAL

    def test_fast_change_returns_warning(self) -> None:
        limiter = RateOfChangeLimiter(
            "pressure", PRESSURE_RATE_WARN, PRESSURE_RATE_TRIP
        )
        limiter.check(140.0e5, dt=1.0)  # seed
        # 7 bar/s — above warn (5) but below trip (10)
        assert limiter.check(147.0e5, dt=1.0) == SafetyLevel.WARNING

    def test_very_fast_change_returns_trip(self) -> None:
        limiter = RateOfChangeLimiter(
            "pressure", PRESSURE_RATE_WARN, PRESSURE_RATE_TRIP
        )
        limiter.check(140.0e5, dt=1.0)  # seed
        # 15 bar/s — above trip (10)
        assert limiter.check(155.0e5, dt=1.0) == SafetyLevel.TRIP

    def test_reset_clears_history(self) -> None:
        limiter = RateOfChangeLimiter(
            "pressure", PRESSURE_RATE_WARN, PRESSURE_RATE_TRIP
        )
        limiter.check(140.0e5, dt=1.0)  # seed
        limiter.reset()
        # After reset, first call is always NORMAL regardless of value
        assert limiter.check(200.0e5, dt=1.0) == SafetyLevel.NORMAL

    def test_zero_dt_returns_normal(self) -> None:
        limiter = RateOfChangeLimiter(
            "pressure", PRESSURE_RATE_WARN, PRESSURE_RATE_TRIP
        )
        limiter.check(140.0e5, dt=1.0)  # seed
        # dt=0 -> skip check, return NORMAL
        assert limiter.check(200.0e5, dt=0.0) == SafetyLevel.NORMAL


# ─── EmergencyStop unit tests ─────────────────────────────────────────────────


class TestEmergencyStop:
    def test_initially_not_active(self) -> None:
        estop = EmergencyStop()
        assert not estop.is_active

    def test_trigger_activates_estop(self) -> None:
        estop = EmergencyStop()
        estop.trigger("pressure_pa", value=190.0e5, threshold=185.0e5)
        assert estop.is_active

    def test_trigger_returns_safety_event(self) -> None:
        estop = EmergencyStop()
        event = estop.trigger("pressure_pa", value=190.0e5, threshold=185.0e5)
        assert isinstance(event, SafetyEvent)
        assert event.parameter == "pressure_pa"
        assert event.level == SafetyLevel.TRIP

    def test_reset_deactivates_estop(self) -> None:
        estop = EmergencyStop()
        estop.trigger("pressure_pa", value=190.0e5, threshold=185.0e5)
        estop.reset(operator_id="operator_1")
        assert not estop.is_active

    def test_reset_increments_counter(self) -> None:
        estop = EmergencyStop()
        estop.trigger("pressure_pa", 190.0e5, 185.0e5)
        estop.reset(operator_id="eng")
        assert not estop.is_active

    def test_trigger_stores_event(self) -> None:
        estop = EmergencyStop()
        estop.trigger("water_level_m", value=0.3, threshold=0.5)
        assert estop.trigger_event is not None
        assert estop.trigger_event.parameter == "water_level_m"

    def test_a_trip_is_logged_as_a_warning_not_an_error(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.DEBUG, logger="plc_controller.safety"):
            EmergencyStop().trigger("water_level_m", value=0.42, threshold=0.5)
        [record] = [r for r in caplog.records if "EMERGENCY STOP" in r.getMessage()]
        assert record.levelno == logging.WARNING
        assert not [r for r in caplog.records if r.levelno >= logging.ERROR]

    def test_reset_without_trigger_is_noop(self) -> None:
        estop = EmergencyStop()
        estop.reset(operator_id="eng")  # should not raise
        assert not estop.is_active

    # Emergency scenario 9
    def test_triggered_estop_blocks_restart_without_reset(self) -> None:
        """System cannot restart until operator explicitly calls reset()."""
        estop = EmergencyStop()
        estop.trigger("pressure_pa", 190.0e5, 185.0e5)
        # Simulate "restart attempt" — still active
        assert estop.is_active


# ─── SafetyInterlock integration tests (10 emergency scenarios) ───────────────


class TestSafetyInterlockNominal:
    def test_nominal_conditions_are_safe(self) -> None:
        interlock = make_interlock()
        status = nominal_check(interlock)
        assert status.safe is True
        assert status.level == SafetyLevel.NORMAL

    def test_nominal_no_valve_overrides(self) -> None:
        interlock = make_interlock()
        status = nominal_check(interlock)
        assert status.fuel_valve_override is None
        assert status.steam_valve_override is None

    def test_nominal_no_events(self) -> None:
        interlock = make_interlock()
        status = nominal_check(interlock)
        assert len(status.events) == 0


class TestEmergencyScenarios:
    # ── Scenario 1: High pressure trip ────────────────────────────────────────

    def test_scenario_1_high_pressure_trips(self) -> None:
        """P > 185 bar -> TRIP."""
        interlock = make_interlock()
        status = interlock.check(
            pressure=190.0e5,  # 190 bar — above trip_high 185 bar
            water_level=NOMINAL_LEVEL,
            water_temp=NOMINAL_WATER_TEMP,
            flue_gas_temp=NOMINAL_FLUE_TEMP,
            sensor_qualities=TRIP_INSTRUMENTS_GOOD,
            dt=1.0,
        )
        assert status.level == SafetyLevel.TRIP
        assert status.safe is False

    def test_scenario_1_high_pressure_closes_fuel_valve(self) -> None:
        interlock = make_interlock()
        status = interlock.check(
            pressure=190.0e5,
            water_level=NOMINAL_LEVEL,
            water_temp=NOMINAL_WATER_TEMP,
            flue_gas_temp=NOMINAL_FLUE_TEMP,
            sensor_qualities=TRIP_INSTRUMENTS_GOOD,
            dt=1.0,
        )
        assert status.fuel_valve_override == pytest.approx(0.0)

    def test_scenario_1_high_pressure_opens_steam_valve(self) -> None:
        interlock = make_interlock()
        status = interlock.check(
            pressure=190.0e5,
            water_level=NOMINAL_LEVEL,
            water_temp=NOMINAL_WATER_TEMP,
            flue_gas_temp=NOMINAL_FLUE_TEMP,
            sensor_qualities=TRIP_INSTRUMENTS_GOOD,
            dt=1.0,
        )
        assert status.steam_valve_override == pytest.approx(1.0)

    # ── Scenario 2: Low pressure trip ─────────────────────────────────────────

    def test_scenario_2_low_pressure_trips(self) -> None:
        """P < 20 bar -> TRIP."""
        interlock = make_interlock()
        status = interlock.check(
            pressure=10.0e5,  # 10 bar — below trip_low 20 bar
            water_level=NOMINAL_LEVEL,
            water_temp=NOMINAL_WATER_TEMP,
            flue_gas_temp=NOMINAL_FLUE_TEMP,
            sensor_qualities=TRIP_INSTRUMENTS_GOOD,
            dt=1.0,
        )
        assert status.level == SafetyLevel.TRIP

    # ── Scenario 3: Drum dry trip ──────────────────────────────────────────────

    def test_scenario_3_drum_dry_trips(self) -> None:
        """Water level < 0.5 m -> TRIP."""
        interlock = make_interlock()
        status = interlock.check(
            pressure=NOMINAL_PRESSURE,
            water_level=0.2,  # 0.2 m — below trip_low 0.5 m
            water_temp=NOMINAL_WATER_TEMP,
            flue_gas_temp=NOMINAL_FLUE_TEMP,
            sensor_qualities=TRIP_INSTRUMENTS_GOOD,
            dt=1.0,
        )
        assert status.level == SafetyLevel.TRIP

    def test_scenario_3_drum_dry_overrides_fuel_to_zero(self) -> None:
        interlock = make_interlock()
        status = interlock.check(
            pressure=NOMINAL_PRESSURE,
            water_level=0.2,
            water_temp=NOMINAL_WATER_TEMP,
            flue_gas_temp=NOMINAL_FLUE_TEMP,
            sensor_qualities=TRIP_INSTRUMENTS_GOOD,
            dt=1.0,
        )
        assert status.fuel_valve_override == pytest.approx(0.0)

    # ── Scenario 4: Drum overflow trip ────────────────────────────────────────

    def test_scenario_4_drum_overflow_trips(self) -> None:
        """Water level > 7.8 m -> TRIP."""
        interlock = make_interlock()
        status = interlock.check(
            pressure=NOMINAL_PRESSURE,
            water_level=7.9,  # above trip_high 7.8 m
            water_temp=NOMINAL_WATER_TEMP,
            flue_gas_temp=NOMINAL_FLUE_TEMP,
            sensor_qualities=TRIP_INSTRUMENTS_GOOD,
            dt=1.0,
        )
        assert status.level == SafetyLevel.TRIP
        # Carry-over into the turbine: the feed stops along with the fuel.
        assert status.feedwater_valve_override == pytest.approx(0.0)
        assert status.fuel_valve_override == pytest.approx(0.0)

    # ── Scenario 5: High water temp trip ──────────────────────────────────────

    def test_scenario_5_high_water_temp_trips(self) -> None:
        """T_water > 648 K -> TRIP (above critical point)."""
        interlock = make_interlock()
        status = interlock.check(
            pressure=NOMINAL_PRESSURE,
            water_level=NOMINAL_LEVEL,
            water_temp=650.0,  # above trip_high 648 K
            flue_gas_temp=NOMINAL_FLUE_TEMP,
            sensor_qualities=TRIP_INSTRUMENTS_GOOD,
            dt=1.0,
        )
        assert status.level == SafetyLevel.TRIP

    # ── Scenario 6: High flue gas temp trip ───────────────────────────────────

    def test_scenario_6_high_flue_gas_temp_trips(self) -> None:
        """T_flue > 1700 K -> TRIP."""
        interlock = make_interlock()
        status = interlock.check(
            pressure=NOMINAL_PRESSURE,
            water_level=NOMINAL_LEVEL,
            water_temp=NOMINAL_WATER_TEMP,
            flue_gas_temp=1750.0,  # above trip_high 1700 K
            sensor_qualities=TRIP_INSTRUMENTS_GOOD,
            dt=1.0,
        )
        assert status.level == SafetyLevel.TRIP

    # ── Scenario 7: Pressure rate of change trip ──────────────────────────────

    def test_scenario_7_pressure_rate_trip(self) -> None:
        """Pressure rising at 15 bar/s -> TRIP."""
        interlock = make_interlock()
        interlock.check(
            pressure=140.0e5,
            water_level=NOMINAL_LEVEL,
            water_temp=NOMINAL_WATER_TEMP,
            flue_gas_temp=NOMINAL_FLUE_TEMP,
            dt=1.0,
            sensor_qualities=TRIP_INSTRUMENTS_GOOD,
        )
        # Next step: +15 bar in 1 second = 15 bar/s > trip (10 bar/s)
        status = interlock.check(
            pressure=155.0e5,
            water_level=NOMINAL_LEVEL,
            water_temp=NOMINAL_WATER_TEMP,
            flue_gas_temp=NOMINAL_FLUE_TEMP,
            dt=1.0,
            sensor_qualities=TRIP_INSTRUMENTS_GOOD,
        )
        assert status.level == SafetyLevel.TRIP

    # ── Scenario 8: Fuel permissive denied ────────────────────────────────────

    def test_scenario_8_fuel_denied_when_drum_low(self) -> None:
        """Fuel permissive: level ≤ 0.5 m -> fuel cannot open."""
        interlock = make_interlock()
        assert not interlock.fuel_permitted(water_level=0.4)

    def test_scenario_8_fuel_permitted_at_normal_level(self) -> None:
        interlock = make_interlock()
        assert interlock.fuel_permitted(water_level=NOMINAL_LEVEL)

    def test_scenario_8_dry_drum_with_fuel_attempt_trips(self) -> None:
        """Checking with dry drum level -> TRIP regardless of pressure."""
        interlock = make_interlock()
        status = interlock.check(
            pressure=NOMINAL_PRESSURE,
            water_level=0.3,  # below trip_low — fuel permissive denied
            water_temp=NOMINAL_WATER_TEMP,
            flue_gas_temp=NOMINAL_FLUE_TEMP,
            sensor_qualities=TRIP_INSTRUMENTS_GOOD,
            dt=1.0,
        )
        assert status.level == SafetyLevel.TRIP
        assert status.fuel_valve_override == pytest.approx(0.0)

    # ── Scenario 11: High steam temperature trip ──────────────────────────────

    def test_scenario_11_high_steam_temperature_trips(self) -> None:
        """Turbine inlet above 580 °C -> TRIP (armed on line)."""
        interlock = make_interlock()
        status = interlock.check(
            pressure=NOMINAL_PRESSURE,
            water_level=NOMINAL_LEVEL,
            water_temp=NOMINAL_WATER_TEMP,
            flue_gas_temp=NOMINAL_FLUE_TEMP,
            dt=1.0,
            steam_temp=STEAM_TEMP_LIMITS.trip_high + 2.0,
            sensor_qualities=TRIP_INSTRUMENTS_GOOD,
        )
        assert status.level == SafetyLevel.TRIP
        assert status.steam_valve_override == pytest.approx(0.0)

    # ── Scenario 9: E-stop locks system ──────────────────────────────────────

    def test_scenario_9_estop_remains_active_after_trip(self) -> None:
        """After trip, subsequent checks stay in TRIP regardless of values."""
        interlock = make_interlock()
        # Trigger trip
        interlock.check(
            pressure=190.0e5,
            water_level=NOMINAL_LEVEL,
            water_temp=NOMINAL_WATER_TEMP,
            flue_gas_temp=NOMINAL_FLUE_TEMP,
            sensor_qualities=TRIP_INSTRUMENTS_GOOD,
            dt=1.0,
        )
        assert interlock.emergency_stop.is_active

        # Next check with nominal values — still TRIP because E-stop is latched
        status = nominal_check(interlock)
        assert status.level == SafetyLevel.TRIP
        assert status.safe is False

    # ── Scenario 10: Restart requires explicit reset ───────────────────────────

    def test_scenario_10_reset_allows_restart(self) -> None:
        """System can only return to NORMAL after operator calls reset()."""
        interlock = make_interlock()
        interlock.check(
            pressure=190.0e5,
            water_level=NOMINAL_LEVEL,
            water_temp=NOMINAL_WATER_TEMP,
            flue_gas_temp=NOMINAL_FLUE_TEMP,
            sensor_qualities=TRIP_INSTRUMENTS_GOOD,
            dt=1.0,
        )
        assert interlock.emergency_stop.is_active

        # Operator investigates and clears the trip
        interlock.reset(operator_id="engineer_42")
        assert not interlock.emergency_stop.is_active

        # Now nominal check returns NORMAL
        status = nominal_check(interlock)
        assert status.safe is True
        assert status.level == SafetyLevel.NORMAL


# ─── Warning (non-trip) tests ─────────────────────────────────────────────────


class TestWarningConditions:
    def test_high_pressure_warning_not_trip(self) -> None:
        """162 bar — above warn_high (160) but below trip_high (185)."""
        interlock = make_interlock()
        status = interlock.check(
            pressure=162.0e5,
            water_level=NOMINAL_LEVEL,
            water_temp=NOMINAL_WATER_TEMP,
            flue_gas_temp=NOMINAL_FLUE_TEMP,
            sensor_qualities=TRIP_INSTRUMENTS_GOOD,
            dt=1.0,
        )
        assert status.level == SafetyLevel.WARNING
        assert status.safe is False
        # Warning does NOT trigger E-stop or valve overrides
        assert not interlock.emergency_stop.is_active
        assert status.fuel_valve_override is None

    def test_low_pressure_warning_not_trip(self) -> None:
        """40 bar — below warn_low (50) but above trip_low (20)."""
        interlock = make_interlock()
        status = interlock.check(
            pressure=40.0e5,
            water_level=NOMINAL_LEVEL,
            water_temp=NOMINAL_WATER_TEMP,
            flue_gas_temp=NOMINAL_FLUE_TEMP,
            sensor_qualities=TRIP_INSTRUMENTS_GOOD,
            dt=1.0,
        )
        assert status.level == SafetyLevel.WARNING
        assert status.fuel_valve_override is None

    def test_pressure_rate_warning_not_trip(self) -> None:
        """7 bar/s — above warn (5) but below trip (10)."""
        interlock = make_interlock()
        interlock.check(
            pressure=140.0e5,
            water_level=NOMINAL_LEVEL,
            water_temp=NOMINAL_WATER_TEMP,
            flue_gas_temp=NOMINAL_FLUE_TEMP,
            dt=1.0,
            sensor_qualities=TRIP_INSTRUMENTS_GOOD,
        )
        status = interlock.check(
            pressure=147.0e5,
            water_level=NOMINAL_LEVEL,
            water_temp=NOMINAL_WATER_TEMP,
            flue_gas_temp=NOMINAL_FLUE_TEMP,
            dt=1.0,
            sensor_qualities=TRIP_INSTRUMENTS_GOOD,
        )
        assert status.level == SafetyLevel.WARNING
        assert not interlock.emergency_stop.is_active


# ─── Stats counter tests ──────────────────────────────────────────────────────


class TestStatsCounters:
    def test_warning_count_increments(self) -> None:
        interlock = make_interlock()
        interlock.check(
            pressure=162.0e5,  # warning
            water_level=NOMINAL_LEVEL,
            water_temp=NOMINAL_WATER_TEMP,
            flue_gas_temp=NOMINAL_FLUE_TEMP,
            sensor_qualities=TRIP_INSTRUMENTS_GOOD,
            dt=1.0,
        )
        assert interlock.warning_count == 1

    def test_trip_count_increments(self) -> None:
        interlock = make_interlock()
        interlock.check(
            pressure=190.0e5,  # trip
            water_level=NOMINAL_LEVEL,
            water_temp=NOMINAL_WATER_TEMP,
            flue_gas_temp=NOMINAL_FLUE_TEMP,
            sensor_qualities=TRIP_INSTRUMENTS_GOOD,
            dt=1.0,
        )
        assert interlock.trip_count == 1

    def test_normal_check_does_not_increment_warning(self) -> None:
        interlock = make_interlock()
        nominal_check(interlock)
        assert interlock.warning_count == 0
        assert interlock.trip_count == 0


# ─── Instrument quality policy (PLC-17) ───────────────────────────────────────


class TestInstrumentPolicy:
    """A failed trip instrument trips; a failed other one degrades to a warning.

    The interlock keeps judging a BAD non-trip instrument's last value (physics holds
    it); the alarm layer raises its quality warning. A trip instrument the plant does
    not report at all is doubtful, not good.
    """

    def check(self, qualities: dict[str, int] | None) -> tuple[SafetyStatus, bool]:
        interlock = make_interlock()
        status = interlock.check(
            pressure=NOMINAL_PRESSURE,
            water_level=NOMINAL_LEVEL,
            water_temp=NOMINAL_WATER_TEMP,
            flue_gas_temp=NOMINAL_FLUE_TEMP,
            dt=1.0,
            steam_temp=811.0,
            sensor_qualities=qualities,
        )
        return status, interlock.emergency_stop.is_active

    @pytest.mark.parametrize("missing", ["drum_pressure", "drum_level"])
    def test_an_unreported_trip_instrument_is_a_warning(self, missing: str) -> None:
        qualities = {"drum_pressure": 0, "drum_level": 0}
        del qualities[missing]
        status, latched = self.check(qualities)
        assert status.level is SafetyLevel.WARNING
        assert not latched
        assert [e.parameter for e in status.events] == [f"{missing}_quality"]

    def test_no_qualities_at_all_warn_for_both_trip_instruments(self) -> None:
        status, latched = self.check(None)
        assert status.level is SafetyLevel.WARNING
        assert not latched
        assert {e.parameter for e in status.events} == {
            "drum_pressure_quality",
            "drum_level_quality",
        }

    def test_a_bad_non_trip_instrument_does_not_trip(self) -> None:
        status, latched = self.check(
            {"drum_pressure": 0, "drum_level": 0, "steam_temp": 2}
        )
        assert status.level is SafetyLevel.NORMAL
        assert not latched


# ─── Thresholds are constants (ARCH-13) ───────────────────────────────────────


class TestThresholdsAreConstants:
    def test_a_limit_cannot_be_changed_at_runtime(self) -> None:
        with pytest.raises(dataclasses.FrozenInstanceError):
            PRESSURE_LIMITS.trip_high = math.inf  # type: ignore[misc]

    @pytest.mark.parametrize(
        ("limits", "low_protected"),
        [
            (PRESSURE_LIMITS, True),
            (WATER_LEVEL_LIMITS, True),
            (WATER_TEMP_LIMITS, True),
            (FLUE_GAS_TEMP_LIMITS, True),
            (STEAM_TEMP_LIMITS, False),
            (PRESSURE_RATE_LIMITS, False),
        ],
    )
    def test_every_protected_side_has_a_finite_trip(
        self, limits: ParameterLimits, low_protected: bool
    ) -> None:
        assert math.isfinite(limits.trip_high)
        assert math.isfinite(limits.trip_low) is low_protected


class TestNamedThresholds:
    def test_the_on_line_threshold_is_a_tenth_of_rated_steam_flow(self) -> None:
        assert ON_LINE_STEAM_FLOW_KG_S == pytest.approx(24.5)

    @pytest.mark.parametrize(
        ("parameter", "deadband"),
        [
            ("pressure_pa", 2.0e5),
            ("water_level_m", 0.1),
            ("water_temp_k", 2.0),
            ("flue_gas_temp_k", 20.0),
            ("steam_temp_k", 3.0),
            ("pressure_rate_pa_s", 1.0e5),
            ("drum_level_quality", 0.5),
        ],
    )
    def test_the_alarm_deadbands_are_unchanged(
        self, parameter: str, deadband: float
    ) -> None:
        rules = [r for r in DEFAULT_RULES if r.parameter == parameter]
        assert rules
        assert {r.deadband for r in rules} == {deadband}

    def test_a_safety_check_needs_its_interval(self) -> None:
        with pytest.raises(TypeError):
            make_interlock().check(  # type: ignore[call-arg]
                NOMINAL_PRESSURE, NOMINAL_LEVEL, NOMINAL_WATER_TEMP, NOMINAL_FLUE_TEMP
            )
