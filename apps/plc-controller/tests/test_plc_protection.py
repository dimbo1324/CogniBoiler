"""Protection and control branches that the live plant rarely reaches.

The arming gates, the instrument-quality trip, the trip overrides, the held loop outputs
on a failed transmitter and the service's own refusals are the paths most likely to be
broken by a refactor, and a lockstep run of the healthy plant never visits most of them.
Each is pinned here directly.
"""

from __future__ import annotations

import cogniboiler_pb2 as pb2
import pytest
from plc_controller.control import ControlTargets, UnitController
from plc_controller.events import PlcEventKind
from plc_controller.measurements import ProcessMeasurements, SignalQuality
from plc_controller.pid import PIDController, PIDParameters
from plc_controller.safety import ArmingTracker, SafetyInterlock
from plc_controller.safety_limits import (
    ALL_ARMED,
    FLAME_FUEL_FLOW_KG_S,
    FLAME_PROVING_S,
    FLUE_GAS_TEMP_LIMITS,
    PRESSURE_LIMITS,
    STEAM_TEMP_LIMITS,
    ArmingState,
    SafetyAction,
    SafetyEvent,
    SafetyLevel,
    trip_overrides,
)
from plc_controller.service import RuntimeMode
from plc_fakes import RecordingPublisher, plc, state

GOOD = {"drum_pressure": 0, "drum_level": 0}
OFF_LINE = ArmingState(on_line=False, firing_proven=False)
ON_LINE_UNPROVEN = ArmingState(on_line=True, firing_proven=False)
TARGETS = ControlTargets(
    load_w=300.0e6, pressure_pa=140.0e5, water_level_m=4.8, steam_temp_k=811.0
)


def measurements(**overrides: object) -> ProcessMeasurements:
    return ProcessMeasurements.from_proto(state(step=10, **overrides))  # type: ignore[arg-type]


def check(
    interlock: SafetyInterlock | None = None,
    *,
    arming: ArmingState = ALL_ARMED,
    qualities: dict[str, int] | None = None,
    steam_temp: float | None = 811.0,
    **readings: float,
) -> tuple[SafetyInterlock, SafetyLevel]:
    interlock = interlock or SafetyInterlock()
    values = {
        "pressure": 140.0e5,
        "water_level": 4.8,
        "water_temp": 611.0,
        "flue_gas_temp": 1200.0,
    } | readings
    status = interlock.check(
        **values,
        dt=1.0,
        steam_temp=steam_temp,
        arming=arming,
        sensor_qualities=GOOD if qualities is None else qualities,
    )
    return interlock, status.level


class TestInstrumentQualityTrip:
    @pytest.mark.parametrize("sensor", ["drum_pressure", "drum_level"])
    def test_a_bad_trip_instrument_trips(self, sensor: str) -> None:
        interlock, level = check(qualities=GOOD | {sensor: 2})
        assert level is SafetyLevel.TRIP
        assert interlock.emergency_stop.is_active
        trigger = interlock.emergency_stop.trigger_event
        assert trigger is not None and trigger.parameter == f"{sensor}_quality"

    def test_an_uncertain_trip_instrument_warns_only(self) -> None:
        interlock, level = check(qualities=GOOD | {"drum_level": 1})
        assert level is SafetyLevel.WARNING
        assert not interlock.emergency_stop.is_active

    def test_an_unknown_quality_code_reads_as_bad(self) -> None:
        m = measurements(qualities={"drum_level": 99})
        assert m.is_bad("drum_level")
        assert m.quality("drum_level") is SignalQuality.BAD


class TestArming:
    def test_low_pressure_does_not_trip_while_off_line(self) -> None:
        _, level = check(arming=OFF_LINE, pressure=10.0e5)
        assert level is SafetyLevel.NORMAL

    def test_low_pressure_trips_once_on_line(self) -> None:
        _, level = check(arming=ON_LINE_UNPROVEN, pressure=10.0e5)
        assert level is SafetyLevel.TRIP

    def test_high_pressure_trips_even_while_off_line(self) -> None:
        _, level = check(arming=OFF_LINE, pressure=PRESSURE_LIMITS.trip_high + 1.0e5)
        assert level is SafetyLevel.TRIP

    def test_high_steam_temperature_is_armed_only_on_line(self) -> None:
        hot = STEAM_TEMP_LIMITS.trip_high + 5.0
        assert check(arming=OFF_LINE, steam_temp=hot)[1] is SafetyLevel.NORMAL
        assert check(arming=ON_LINE_UNPROVEN, steam_temp=hot)[1] is SafetyLevel.TRIP

    def test_low_flue_gas_temperature_needs_proven_firing(self) -> None:
        cold = FLUE_GAS_TEMP_LIMITS.trip_low - 10.0
        unproven = check(arming=ON_LINE_UNPROVEN, flue_gas_temp=cold)[1]
        proven = check(arming=ALL_ARMED, flue_gas_temp=cold)[1]
        assert (unproven, proven) == (SafetyLevel.NORMAL, SafetyLevel.TRIP)

    def test_the_tracker_proves_the_flame_and_resets_on_flame_loss(self) -> None:
        tracker = ArmingTracker()
        firing = FLAME_FUEL_FLOW_KG_S
        for _ in range(int(FLAME_PROVING_S) - 1):
            assert not tracker.update(245.0, firing, 1.0).firing_proven
        assert tracker.update(245.0, firing, 1.0).firing_proven
        assert tracker.state == ArmingState(on_line=True, firing_proven=True)
        assert not tracker.update(245.0, 0.0, 1.0).firing_proven
        assert not tracker.update(245.0, firing, 1.0).firing_proven
        assert not tracker.update(10.0, firing, 1.0).on_line


class TestTripResponse:
    def event(self, parameter: str, value: float, threshold: float) -> SafetyEvent:
        return SafetyEvent(
            0,
            parameter,
            value,
            threshold,
            SafetyLevel.TRIP,
            SafetyAction.EMERGENCY_STOP,
        )

    def test_high_pressure_vents_through_the_turbine(self) -> None:
        overrides = trip_overrides(self.event("pressure_pa", 190.0e5, 185.0e5))
        assert (overrides.fuel, overrides.steam, overrides.feedwater) == (
            0.0,
            1.0,
            None,
        )

    def test_low_pressure_closes_the_turbine_valve(self) -> None:
        assert trip_overrides(self.event("pressure_pa", 10.0e5, 20.0e5)).steam == 0.0

    def test_a_high_level_stops_the_feed(self) -> None:
        overrides = trip_overrides(self.event("water_level_m", 7.9, 7.8))
        assert (overrides.fuel, overrides.feedwater, overrides.steam) == (
            0.0,
            0.0,
            0.0,
        )

    def test_a_low_level_leaves_the_feed_to_the_level_hold(self) -> None:
        status = SafetyInterlock().check(
            140.0e5, 0.2, 611.0, 1200.0, dt=1.0, sensor_qualities=GOOD
        )
        assert status.level is SafetyLevel.TRIP
        assert status.feedwater_valve_override is None
        assert status.fuel_valve_override == 0.0


class TestHeldLoopsOnAFailedTransmitter:
    def primed(self) -> tuple[UnitController, float]:
        controller = UnitController()
        controller.scan(measurements(), TARGETS, 1.0)
        return controller, 1.0

    def loop(self, controller: UnitController, name: str) -> float:
        output = controller.last_output
        assert output is not None
        [found] = [loop.output for loop in output.loops if loop.name == name]
        return found

    def test_a_bad_level_transmitter_freezes_the_level_trim(self) -> None:
        controller, dt = self.primed()
        before = self.loop(controller, "drum_level")
        controller.scan(
            measurements(water_level_m=0.1, qualities={"drum_level": 2}), TARGETS, dt
        )
        assert self.loop(controller, "drum_level") == pytest.approx(before)
        # The same reading from a good transmitter moves the loop: the hold is real.
        good, _ = self.primed()
        good.scan(measurements(water_level_m=0.1), TARGETS, dt)
        assert self.loop(good, "drum_level") != pytest.approx(before)

    def test_a_bad_pressure_transmitter_freezes_the_pressure_trim(self) -> None:
        controller, dt = self.primed()
        before = self.loop(controller, "pressure")
        controller.scan(
            measurements(pressure_pa=100.0e5, qualities={"drum_pressure": 2}),
            TARGETS,
            dt,
        )
        assert self.loop(controller, "pressure") == pytest.approx(before)

    def test_a_bad_steam_temperature_holds_the_spray_command(self) -> None:
        controller, dt = self.primed()
        output = controller.scan(
            measurements(
                steam_temp_k=900.0, spray_command=0.3, qualities={"steam_temp": 2}
            ),
            TARGETS,
            dt,
        )
        assert output.valves.spray == pytest.approx(0.3)


class TestTurbinePressureGuard:
    def test_a_large_pressure_deficit_closes_the_turbine_valve(self) -> None:
        controller = UnitController()
        controller.scan(measurements(), TARGETS, 1.0)
        output = controller.scan(measurements(pressure_pa=120.0e5), TARGETS, 1.0)
        assert output.valves.steam < 0.7

    def test_a_moderate_deficit_stops_it_opening(self) -> None:
        controller = UnitController()
        controller.scan(measurements(), TARGETS, 1.0)
        more_load = ControlTargets(
            load_w=300.0e6, pressure_pa=140.0e5, water_level_m=4.8, steam_temp_k=811.0
        )
        output = controller.scan(
            measurements(pressure_pa=130.0e5, electrical_power_w=200.0e6),
            more_load,
            1.0,
        )
        assert output.valves.steam <= 0.7


class TestBackCalculation:
    def test_constrain_moves_the_integral_by_the_limited_delta(self) -> None:
        pid = PIDController(
            PIDParameters(kp=1.0, ki=0.1, kd=0.0, output_min=-10.0, output_max=10.0)
        )
        pid.reset(initial_output=2.0)
        pid.constrain(1.5)
        assert (pid.state.integral, pid.state.prev_output) == (1.5, 1.5)
        pid.constrain(1.5)
        assert pid.state.integral == 1.5


class TestServiceRefusals:
    async def test_fuel_is_refused_while_the_last_level_is_below_trip_low(
        self,
    ) -> None:
        svc, physics = plc()
        await svc.process_state(state(step=0))
        svc._latest_measurements = measurements(water_level_m=0.4)
        result = await svc.send_command(
            fuel_valve=0.2,
            feedwater_valve=0.5,
            steam_valve=0.5,
            source=pb2.CommandSource.OPERATOR,
            operator_id="op",
        )
        assert not result.accepted
        assert "Fuel not permitted" in result.reason
        assert physics.commands == []

    async def test_zero_fuel_is_accepted_with_a_low_level(self) -> None:
        svc, physics = plc()
        await svc.process_state(state(step=0))
        svc._latest_measurements = measurements(water_level_m=0.4)
        result = await svc.send_command(
            fuel_valve=0.0,
            feedwater_valve=0.8,
            steam_valve=0.0,
            source=pb2.CommandSource.OPERATOR,
            operator_id="op",
        )
        assert result.accepted
        assert physics.commands[-1].fuel_valve == 0.0

    async def test_manual_is_refused_while_latched(self) -> None:
        svc, _ = plc()
        await svc.process_state(state(step=0))
        await svc.set_mode(RuntimeMode.ESTOP, "eng")
        result = await svc.set_mode(RuntimeMode.MANUAL, "op")
        assert not result.accepted and "Reset it first" in result.reason
        assert svc.mode is RuntimeMode.ESTOP

    async def test_an_interlock_trip_in_manual_forwards_the_trip_command(
        self,
    ) -> None:
        svc, physics = plc()
        await svc.process_state(state(step=0))
        await svc.send_command(
            fuel_valve=0.6,
            feedwater_valve=0.6,
            steam_valve=0.7,
            source=pb2.CommandSource.OPERATOR,
            operator_id="op",
        )
        assert svc.mode is RuntimeMode.MANUAL
        await svc.process_state(state(step=1, pressure_pa=190.0e5))
        assert svc.mode is RuntimeMode.ESTOP
        last = physics.commands[-1]
        assert last.source == pb2.CommandSource.SAFETY
        assert (last.fuel_valve, last.steam_valve) == (0.0, 1.0)


class TestRunChange:
    async def test_a_new_run_clears_active_alarms_and_announces_itself(self) -> None:
        svc, _ = plc()
        publisher = RecordingPublisher()
        svc._publisher = publisher  # type: ignore[assignment]
        await svc.process_state(state(step=0, run_id=1, water_level_m=1.9))
        assert svc.active_condition_count == 1
        await svc.process_state(state(step=0, run_id=2))
        assert svc.active_condition_count == 0
        cleared = [t for t in publisher.alarms if not t.active]
        assert [t.condition.rule.parameter for t in cleared] == ["water_level_m"]
        [changed] = [e for e in publisher.events if e.kind is PlcEventKind.RUN_CHANGED]
        assert changed.detail == {"run_id": 2}
