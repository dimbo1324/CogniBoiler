"""What an operator may ask for, what the PLC reports back, and the awkward cases.

These are the two pure layers of the PLC: the bounds a request has to clear before it can
reach a valve, and the projection of the PLC's state onto the contract. Neither needs a
plant, a broker or a gRPC server, so they can be pushed much harder than the scan loop —
NaN through a proto field, a value exactly on a bound, a trip with no cause recorded.
"""

from __future__ import annotations

import math

import cogniboiler_pb2 as pb2
import pytest
from plc_controller.alarms import (
    AlarmCondition,
    Arming,
    ConditionRule,
    Direction,
    Severity,
    alarm_values,
    blocking_conditions,
)
from plc_controller.commands import (
    LEVEL_SETPOINT_MAX_M,
    LEVEL_SETPOINT_MIN_M,
    LOAD_DEMAND_MAX_W,
    PRESSURE_SETPOINT_MAX_PA,
    PRESSURE_SETPOINT_MIN_PA,
    TEMP_SETPOINT_MAX_K,
    TEMP_SETPOINT_MIN_K,
    VALVE_MAX,
    VALVE_MIN,
    CommandSnapshot,
    Setpoints,
    check_load_demand,
    check_setpoints,
    check_valves,
)
from plc_controller.control import ControlTargets, LoopStatus
from plc_controller.measurements import (
    SENSOR_DRUM_LEVEL,
    SENSOR_DRUM_PRESSURE,
    SENSOR_STEAM_TEMP,
    ProcessMeasurements,
    SignalQuality,
    ValveSet,
)
from plc_controller.status import SafetySnapshot, control_status

NOMINAL = Setpoints(pressure_pa=140.0e5, water_level_m=4.8, steam_temp_k=811.0)


def measurements(**overrides: object) -> ProcessMeasurements:
    values: dict[str, object] = {
        "simulation_time_s": 10.0,
        "step_s": 1.0,
        "run_id": 7,
        "pressure_pa": 140.0e5,
        "water_level_m": 4.8,
        "water_temp_k": 611.0,
        "flue_gas_temp_k": 1200.0,
        "steam_temp_k": 811.0,
        "steam_flow_kg_s": 245.0,
        "spray_flow_kg_s": 2.0,
        "feedwater_flow_kg_s": 243.0,
        "fuel_flow_kg_s": 17.0,
        "electrical_power_w": 300.0e6,
        "commands": ValveSet(0.6, 0.6, 0.7, 0.05),
        "positions": ValveSet(0.6, 0.6, 0.7, 0.05),
        "qualities": {},
    }
    values.update(overrides)
    return ProcessMeasurements(**values)  # type: ignore[arg-type]


class TestValveBounds:
    def test_both_ends_of_the_range_are_valid_positions(self) -> None:
        assert check_valves(VALVE_MIN, VALVE_MIN, VALVE_MIN, VALVE_MIN).accepted
        assert check_valves(VALVE_MAX, VALVE_MAX, VALVE_MAX, VALVE_MAX).accepted

    def test_the_spray_valve_may_be_left_out(self) -> None:
        assert check_valves(0.5, 0.5, 0.5).accepted
        assert not check_valves(0.5, 0.5, 0.5, 1.5).accepted

    @pytest.mark.parametrize(
        ("value", "named"),
        [(-0.001, "fuel_valve"), (1.001, "fuel_valve")],
    )
    def test_a_position_outside_the_range_is_refused_by_name(
        self, value: float, named: str
    ) -> None:
        result = check_valves(value, 0.5, 0.5)
        assert not result.accepted
        assert named in result.reason
        assert "[0.0, 1.0]" in result.reason

    def test_the_first_offending_valve_is_the_one_named(self) -> None:
        result = check_valves(0.5, 2.0, 3.0)
        assert not result.accepted
        assert result.reason.startswith("feedwater_valve=")

    @pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
    def test_a_position_that_is_not_a_real_number_never_reaches_the_plant(
        self, value: float
    ) -> None:
        # A float field of the contract accepts these; a comparison with them is False,
        # so the refusal has to come from the bound check, not from a later surprise.
        assert not check_valves(value, 0.5, 0.5).accepted
        assert not check_valves(0.5, 0.5, 0.5, value).accepted

    def test_a_negative_zero_is_still_shut(self) -> None:
        assert check_valves(-0.0, 0.5, 0.5).accepted


class TestSetpointBounds:
    def test_the_nominal_targets_are_inside_the_range(self) -> None:
        assert check_setpoints(140.0e5, 4.8, 811.0).accepted

    def test_every_bound_is_inclusive(self) -> None:
        assert check_setpoints(
            PRESSURE_SETPOINT_MIN_PA, LEVEL_SETPOINT_MIN_M, TEMP_SETPOINT_MIN_K
        ).accepted
        assert check_setpoints(
            PRESSURE_SETPOINT_MAX_PA, LEVEL_SETPOINT_MAX_M, TEMP_SETPOINT_MAX_K
        ).accepted

    def test_a_pressure_above_the_range_is_refused_in_bar(self) -> None:
        result = check_setpoints(PRESSURE_SETPOINT_MAX_PA + 1.0e5, 4.8, 811.0)
        assert not result.accepted
        assert "bar" in result.reason and "Pressure setpoint" in result.reason

    def test_the_level_is_checked_after_the_pressure_and_before_the_temperature(
        self,
    ) -> None:
        # One refusal, naming the first thing that is wrong, not a list of everything.
        result = check_setpoints(140.0e5, LEVEL_SETPOINT_MAX_M + 1.0, 9_000.0)
        assert not result.accepted
        assert result.reason.startswith("Level setpoint")

    @pytest.mark.parametrize("value", [math.nan, math.inf])
    def test_a_target_that_is_not_a_real_number_is_refused(self, value: float) -> None:
        assert not check_setpoints(value, 4.8, 811.0).accepted
        assert not check_setpoints(140.0e5, value, 811.0).accepted
        assert not check_setpoints(140.0e5, 4.8, value).accepted


class TestLoadDemand:
    def test_from_nothing_to_the_rated_output(self) -> None:
        assert check_load_demand(0.0).accepted
        assert check_load_demand(LOAD_DEMAND_MAX_W).accepted

    def test_above_the_rating_or_below_zero_is_refused_in_megawatts(self) -> None:
        over = check_load_demand(LOAD_DEMAND_MAX_W + 1.0e6)
        assert not over.accepted
        assert "MW" in over.reason
        assert not check_load_demand(-1.0).accepted

    def test_a_demand_that_is_not_a_real_number_is_refused(self) -> None:
        assert not check_load_demand(math.nan).accepted


class TestCopies:
    def test_a_copy_of_the_setpoints_does_not_follow_later_changes(self) -> None:
        setpoints = Setpoints()
        taken = setpoints.copy()
        setpoints.pressure_pa = 100.0e5
        assert taken.pressure_pa != setpoints.pressure_pa

    def test_a_copy_of_a_command_keeps_who_sent_it_and_when(self) -> None:
        command = CommandSnapshot(
            fuel_valve=0.4, source=int(pb2.CommandSource.OPERATOR), operator_id="anna"
        )
        taken = command.copy()
        command.fuel_valve = 0.9
        assert (taken.fuel_valve, taken.operator_id) == (0.4, "anna")
        assert taken.source == int(pb2.CommandSource.OPERATOR)
        assert taken.timestamp_ms == command.timestamp_ms


def status(**overrides: object) -> pb2.PLCStatusMsg:
    values: dict[str, object] = {
        "mode": int(pb2.ControlMode.AUTO),
        "emergency_stop_active": False,
        "setpoints": NOMINAL,
        "latest_command": CommandSnapshot(),
        "warning_count": 0,
        "trip_count": 0,
        "trip_cause": None,
        "load_demand_w": 300.0e6,
        "working": None,
        "reset_blockers": [],
        "conditions": (),
        "loops": (),
        "run_id": 7,
    }
    values.update(overrides)
    return control_status(**values)  # type: ignore[arg-type]


class TestStatusProjection:
    def test_before_the_controller_is_primed_the_setpoint_reads_as_the_demand(
        self,
    ) -> None:
        message = status(working=None, load_demand_w=250.0e6)
        assert message.load_setpoint_w == pytest.approx(250.0e6)
        assert message.active_setpoints.pressure_pa == 0.0

    def test_the_working_setpoints_replace_it_once_there_are_any(self) -> None:
        working = ControlTargets(
            load_w=180.0e6, pressure_pa=139.0e5, water_level_m=4.7, steam_temp_k=805.0
        )
        message = status(working=working, load_demand_w=250.0e6)
        assert message.load_setpoint_w == pytest.approx(180.0e6)
        assert message.active_setpoints.steam_temp_k == pytest.approx(805.0)

    def test_no_trip_leaves_the_event_empty_rather_than_half_filled(self) -> None:
        message = status(trip_cause=None)
        assert message.active_trip.parameter == ""
        assert message.active_trip.timestamp_ms == 0

    def test_a_trip_carries_its_cause_and_the_moment_it_latched(self) -> None:
        cause = SafetySnapshot(
            timestamp_ms=1_700_000_000_123,
            parameter="water_level_m",
            value=0.9,
            threshold=1.0,
            level="trip_low",
            action="emergency_stop",
        )
        message = status(emergency_stop_active=True, trip_cause=cause)
        assert message.active_trip.parameter == "water_level_m"
        assert message.active_trip.timestamp_ms == 1_700_000_000_123

    def test_a_reset_is_permitted_only_while_tripped_and_unblocked(self) -> None:
        assert status(emergency_stop_active=True, reset_blockers=[]).reset_permitted
        blocked = status(emergency_stop_active=True, reset_blockers=["drum level low"])
        assert not blocked.reset_permitted
        assert list(blocked.reset_blockers) == ["drum level low"]
        # Not tripped at all: there is nothing to permit.
        assert not status(
            emergency_stop_active=False, reset_blockers=[]
        ).reset_permitted

    def test_conditions_and_loops_are_projected_in_order(self) -> None:
        rule = ConditionRule(
            "water_level_m", "m", Severity.CRITICAL, Direction.LOW, 1.0, 0.2
        )
        condition = AlarmCondition(rule=rule, value=0.9, since_ms=1_000)
        loops = (
            LoopStatus("pressure", 140.0e5, 139.0e5, 0.6, "Pa"),
            LoopStatus("level", 4.8, 4.7, 0.5, "m"),
        )
        message = status(conditions=(condition,), loops=loops)
        assert [loop.name for loop in message.loops] == ["pressure", "level"]
        assert message.active_conditions[0].parameter == "water_level_m"
        assert message.active_conditions[0].severity == "critical"
        assert message.active_conditions[0].since_ms == 1_000


class TestAlarmValues:
    def test_a_reading_from_a_bad_instrument_is_left_out_entirely(self) -> None:
        values = alarm_values(
            measurements(qualities={SENSOR_DRUM_LEVEL: SignalQuality.BAD}), None
        )
        assert "water_level_m" not in values
        assert "pressure_pa" in values
        # The instrument's own quality is still judged: that is what raises the alarm.
        assert values[f"{SENSOR_DRUM_LEVEL}_quality"] == float(SignalQuality.BAD)

    def test_an_uncertain_instrument_is_still_read(self) -> None:
        values = alarm_values(
            measurements(qualities={SENSOR_STEAM_TEMP: SignalQuality.UNCERTAIN}), None
        )
        assert values["steam_temp_k"] == pytest.approx(811.0)
        assert values[f"{SENSOR_STEAM_TEMP}_quality"] == float(SignalQuality.UNCERTAIN)

    def test_the_pressure_rate_needs_a_pressure_worth_differentiating(self) -> None:
        assert "pressure_rate_pa_s" in alarm_values(measurements(), 1.0e5)
        assert "pressure_rate_pa_s" not in alarm_values(measurements(), None)
        assert "pressure_rate_pa_s" not in alarm_values(
            measurements(qualities={SENSOR_DRUM_PRESSURE: SignalQuality.BAD}), 1.0e5
        )

    def test_a_rate_of_zero_is_a_value_like_any_other(self) -> None:
        # `if rate:` would have dropped a steady pressure; `is not None` keeps it.
        assert alarm_values(measurements(), 0.0)["pressure_rate_pa_s"] == 0.0


class TestResetBlockers:
    def critical(self) -> AlarmCondition:
        return AlarmCondition(
            rule=ConditionRule(
                "water_level_m", "m", Severity.CRITICAL, Direction.LOW, 1.0, 0.2
            ),
            value=0.9,
            since_ms=0,
        )

    def warning(self, parameter: str) -> AlarmCondition:
        return AlarmCondition(
            rule=ConditionRule(
                parameter, "m", Severity.WARNING, Direction.LOW, 2.0, 0.2, Arming.ALWAYS
            ),
            value=1.9,
            since_ms=0,
        )

    def test_every_critical_condition_blocks_whatever_caused_the_trip(self) -> None:
        assert blocking_conditions([self.critical()], "steam_temp_k")

    def test_a_warning_on_the_trip_cause_blocks_too(self) -> None:
        assert blocking_conditions([self.warning("water_level_m")], "water_level_m")

    def test_a_warning_elsewhere_does_not_block(self) -> None:
        assert (
            blocking_conditions([self.warning("steam_temp_k")], "water_level_m") == []
        )

    def test_an_interlock_cause_is_read_as_the_parameter_it_depends_on(self) -> None:
        # The fuel permissive trips on the drum level; the reset waits for that warning.
        assert blocking_conditions([self.warning("water_level_m")], "fuel_permissive")

    def test_nothing_active_means_nothing_blocking(self) -> None:
        assert blocking_conditions([], "water_level_m") == []
