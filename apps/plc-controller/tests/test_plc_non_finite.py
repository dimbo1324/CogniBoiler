"""A reading that is not a number must fail safe, never read as normal.

Every comparison with NaN is false, so without an explicit check a NaN pressure passes
every limit, a NaN in a PID becomes a fully open valve, and a NaN interval disarms the
flame proving for good. These tests pin the fail-safe answer at each layer: the
measurement boundary marks the instrument BAD, the interlock trips, the controllers keep
their state, and the scan holds the valves rather than computing them from garbage.
"""

from __future__ import annotations

import json
import math

import cogniboiler_pb2 as pb2
import pytest
from plc_controller.events import event_payload
from plc_controller.measurements import ProcessMeasurements, SignalQuality
from plc_controller.numeric import clamp
from plc_controller.pid import PIDController, PIDParameters
from plc_controller.ramps import RampedSetpoint
from plc_controller.safety import ArmingTracker, RateOfChangeLimiter, SafetyInterlock
from plc_controller.safety_limits import (
    PRESSURE_LIMITS,
    PRESSURE_RATE_TRIP,
    PRESSURE_RATE_WARN,
    STEAM_TEMP_LIMITS,
    ArmingState,
    SafetyLevel,
)
from plc_controller.service import RuntimeMode
from plc_fakes import RecordingPublisher, plc, state

NON_FINITE = [math.nan, math.inf, -math.inf]
GOOD = {"drum_pressure": 0, "drum_level": 0}
NOMINAL_READINGS: dict[str, float] = {
    "pressure": 140.0e5,
    "water_level": 4.8,
    "water_temp": 611.0,
    "flue_gas_temp": 1200.0,
}


class TestLimits:
    @pytest.mark.parametrize("value", NON_FINITE)
    def test_a_non_finite_value_trips_a_protected_parameter(self, value: float) -> None:
        assert PRESSURE_LIMITS.check(value) is SafetyLevel.TRIP

    def test_minus_infinity_on_an_unprotected_side_is_not_a_trip(self) -> None:
        # Steam temperature has no low limits at all: -inf is below nothing.
        assert STEAM_TEMP_LIMITS.check(-math.inf) is SafetyLevel.NORMAL
        assert STEAM_TEMP_LIMITS.check(math.inf) is SafetyLevel.TRIP
        assert STEAM_TEMP_LIMITS.check(math.nan) is SafetyLevel.TRIP


class TestInterlock:
    @pytest.mark.parametrize("value", NON_FINITE)
    @pytest.mark.parametrize(
        ("reading", "sensor"),
        [
            ("pressure", "drum_pressure"),
            ("water_level", "drum_level"),
            ("water_temp", "drum_water_temp"),
            ("flue_gas_temp", "furnace_gas_temp"),
        ],
    )
    def test_a_non_finite_reading_latches_the_e_stop(
        self, reading: str, sensor: str, value: float
    ) -> None:
        interlock = SafetyInterlock()
        status = interlock.check(
            **(NOMINAL_READINGS | {reading: value}), dt=1.0, sensor_qualities=GOOD
        )
        assert status.level is SafetyLevel.TRIP
        assert interlock.emergency_stop.is_active
        trigger = interlock.emergency_stop.trigger_event
        assert trigger is not None
        assert trigger.parameter == f"{sensor}_quality"
        assert math.isfinite(trigger.value)

    @pytest.mark.parametrize("value", NON_FINITE)
    def test_a_non_finite_steam_temperature_trips_whatever_the_arming(
        self, value: float
    ) -> None:
        interlock = SafetyInterlock()
        interlock.check(
            **NOMINAL_READINGS,
            dt=1.0,
            steam_temp=value,
            arming=ArmingState(on_line=False, firing_proven=False),
            sensor_qualities=GOOD,
        )
        assert interlock.emergency_stop.is_active

    def test_a_non_finite_pressure_does_not_open_the_turbine_valve(self) -> None:
        # Venting is the answer to a high pressure, not to a pressure nobody can read.
        status = SafetyInterlock().check(
            **(NOMINAL_READINGS | {"pressure": math.nan}),
            dt=1.0,
            sensor_qualities=GOOD,
        )
        assert status.steam_valve_override == 0.0

    def test_finite_readings_in_range_do_not_trip(self) -> None:
        interlock = SafetyInterlock()
        status = interlock.check(**NOMINAL_READINGS, dt=1.0, sensor_qualities=GOOD)
        assert status.level is SafetyLevel.NORMAL
        assert not interlock.emergency_stop.is_active


class TestRateAndArming:
    def test_the_rate_limiter_ignores_a_non_finite_value(self) -> None:
        limiter = RateOfChangeLimiter("p", PRESSURE_RATE_WARN, PRESSURE_RATE_TRIP)
        assert limiter.check(math.nan, dt=1.0) is SafetyLevel.NORMAL
        limiter.check(140.0e5, dt=1.0)
        assert limiter.check(math.nan, dt=1.0) is SafetyLevel.NORMAL
        # The next rate is measured against the last real pressure, 1 bar/s.
        assert limiter.check(141.0e5, dt=1.0) is SafetyLevel.NORMAL
        assert limiter.last_rate == pytest.approx(1.0e5)

    def test_a_non_finite_interval_does_not_disarm_the_flame_proving(self) -> None:
        tracker = ArmingTracker()
        tracker.update(245.0, 17.0, math.nan)
        for _ in range(10):
            state_ = tracker.update(245.0, 17.0, 1.0)
        assert state_.firing_proven

    def test_a_non_finite_flow_holds_the_arming_it_had(self) -> None:
        tracker = ArmingTracker()
        for _ in range(10):
            tracker.update(245.0, 17.0, 1.0)
        held = tracker.update(math.nan, math.nan, 1.0)
        assert held == ArmingState(on_line=True, firing_proven=True)


class TestControllers:
    def pid(self) -> PIDController:
        return PIDController(
            PIDParameters(kp=1.0, ki=0.5, kd=0.0, output_min=0.0, output_max=10.0)
        )

    @pytest.mark.parametrize(
        ("setpoint", "measurement", "dt"),
        [
            (5.0, math.nan, 1.0),
            (5.0, math.inf, 1.0),
            (math.nan, 4.0, 1.0),
            (5.0, 4.0, math.nan),
        ],
    )
    def test_a_non_finite_input_does_not_poison_the_integrator(
        self, setpoint: float, measurement: float, dt: float
    ) -> None:
        pid, reference = self.pid(), self.pid()
        first = pid.step(5.0, 4.0, 1.0)
        reference.step(5.0, 4.0, 1.0)
        assert pid.step(setpoint, measurement, dt) == first
        assert pid.step(5.0, 4.0, 1.0) == reference.step(5.0, 4.0, 1.0)
        assert math.isfinite(pid.state.integral)

    def test_a_reset_or_a_limit_that_is_not_a_number_is_ignored(self) -> None:
        pid = self.pid()
        pid.reset(initial_output=2.0)
        pid.reset(initial_output=math.nan)
        pid.constrain(math.nan)
        assert (pid.state.integral, pid.state.prev_output) == (2.0, 2.0)

    def test_a_ramp_ignores_a_target_or_value_that_is_not_a_number(self) -> None:
        ramp = RampedSetpoint(1.0, value=10.0)
        ramp.set_target(math.nan)
        ramp.track(math.inf)
        assert ramp.step(math.nan) == 10.0
        assert ramp.step(1.0) == 10.0

    def test_clamp_refuses_nan_instead_of_returning_the_upper_bound(self) -> None:
        assert clamp(2.0, 0.0, 1.0) == 1.0
        with pytest.raises(ValueError):
            clamp(math.nan, 0.0, 1.0)


class TestMeasurements:
    @pytest.mark.parametrize(
        ("field", "sensor"),
        [
            ("pressure_pa", "drum_pressure"),
            ("water_level_m", "drum_level"),
            ("water_temp_k", "drum_water_temp"),
            ("flue_gas_temp_k", "furnace_gas_temp"),
            ("steam_temp_k", "steam_temp"),
            ("steam_flow_kg_s", "steam_flow"),
            ("feedwater_flow_kg_s", "feedwater_flow"),
            ("fuel_flow_kg_s", "fuel_flow"),
            ("electrical_power_w", "electrical_power"),
        ],
    )
    def test_a_non_finite_reading_is_marked_bad(self, field: str, sensor: str) -> None:
        m = ProcessMeasurements.from_proto(state(**{field: math.nan}))
        assert m.quality(sensor) is SignalQuality.BAD
        assert not m.finite

    def test_finite_readings_keep_the_quality_the_plant_reported(self) -> None:
        m = ProcessMeasurements.from_proto(state(qualities={"steam_temp": 1}))
        assert m.quality("steam_temp") is SignalQuality.UNCERTAIN
        assert m.quality("drum_pressure") is SignalQuality.GOOD
        assert m.finite


class TestScan:
    async def test_a_nan_pressure_trips_the_unit_and_shuts_the_fuel(self) -> None:
        svc, physics = plc()
        await svc.process_state(state(step=0))
        await svc.process_state(state(step=1))
        await svc.process_state(state(step=2, pressure_pa=math.nan))
        assert svc.mode is RuntimeMode.ESTOP
        last = physics.commands[-1]
        assert last.source == pb2.CommandSource.SAFETY
        assert (last.fuel_valve, last.steam_valve) == (0.0, 0.0)
        assert all(math.isfinite(c.feedwater_valve) for c in physics.commands)

    async def test_a_nan_flow_in_auto_holds_the_valves_and_keeps_the_loops(
        self,
    ) -> None:
        svc, physics = plc()
        await svc.process_state(state(step=0))
        await svc.process_state(state(step=1))
        sent = len(physics.commands)
        await svc.process_state(state(step=2, fuel_flow_kg_s=math.nan))
        await svc.process_state(state(step=3, steam_flow_kg_s=math.inf))
        assert len(physics.commands) == sent
        assert svc.mode is RuntimeMode.AUTO
        await svc.process_state(state(step=4))
        last = physics.commands[-1]
        valves = (
            last.fuel_valve,
            last.feedwater_valve,
            last.steam_valve,
            last.spray_valve,
        )
        assert all(0.0 <= v <= 1.0 for v in valves)
        assert last.fuel_valve < 1.0

    async def test_a_nan_simulation_time_is_no_interval(self) -> None:
        svc, physics = plc()
        await svc.process_state(state(step=0))
        await svc.process_state(state(step=1, simulation_time_s=math.nan))
        await svc.process_state(state(step=2))
        await svc.process_state(state(step=3))
        assert svc.mode is RuntimeMode.AUTO
        assert all(math.isfinite(c.fuel_valve) for c in physics.commands)

    async def test_a_nan_power_on_a_new_run_does_not_become_the_load_demand(
        self,
    ) -> None:
        svc, _ = plc()
        await svc.process_state(state(step=0, run_id=1))
        await svc.process_state(state(step=0, run_id=2, electrical_power_w=math.nan))
        assert svc.load_demand_w == pytest.approx(300.0e6)

    async def test_the_trip_event_carries_no_value_json_cannot_encode(self) -> None:
        svc, _ = plc()
        publisher = RecordingPublisher()
        svc._publisher = publisher  # type: ignore[assignment]
        await svc.process_state(state(step=0, water_temp_k=math.nan))
        assert svc.mode is RuntimeMode.ESTOP
        for event in publisher.events:
            json.loads(event_payload(event), parse_constant=_refuse_constant)


def _refuse_constant(name: str) -> None:
    raise ValueError(f"{name} in a published payload")
