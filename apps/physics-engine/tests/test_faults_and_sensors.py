"""Fault specifications, their development and combination, and the instrument layer."""

from __future__ import annotations

import pytest
from physics_engine.faults import (
    MAX_RAMP_S,
    NO_DISTURBANCES,
    ActiveFault,
    FaultError,
    FaultKind,
    FaultRegistry,
    FaultSpec,
    ValveId,
)
from physics_engine.sensors import (
    SENSOR_SPANS,
    VALIDATION_TOLERANCE,
    Quality,
    SensorBank,
    SensorId,
    SensorReading,
    worst_quality,
)

K = FaultKind


class TestSpecifications:
    @pytest.mark.parametrize(
        "spec",
        [
            FaultSpec(K.VALVE_STUCK, " Feedwater "),
            FaultSpec(K.SENSOR_DRIFT, "DRUM_LEVEL", severity=-0.05),
            FaultSpec(K.SENSOR_FAILURE, "steam_flow"),
            FaultSpec(K.BURNER_FOULING, severity=0.5, ramp_s=MAX_RAMP_S),
            FaultSpec(K.STEAM_LEAK, severity=0.3),
            FaultSpec(K.FEEDWATER_PUMP_FAILURE, severity=1.0),
        ],
    )
    def test_valid_specifications_are_normalized(self, spec: FaultSpec) -> None:
        normalized = spec.validated()
        assert normalized.target == spec.target.strip().lower()
        assert (normalized.kind, normalized.severity) == (spec.kind, spec.severity)

    @pytest.mark.parametrize(
        ("spec", "message"),
        [
            (FaultSpec(K.STEAM_LEAK, severity=0.1, ramp_s=-1.0), "ramp_s"),
            (FaultSpec(K.STEAM_LEAK, severity=0.1, ramp_s=MAX_RAMP_S + 1), "ramp_s"),
            (FaultSpec(K.VALVE_STUCK, "boiler"), "valve target"),
            (FaultSpec(K.VALVE_STUCK), "valve target"),
            (FaultSpec(K.SENSOR_DRIFT, "thermometer", severity=0.1), "sensor target"),
            (FaultSpec(K.SENSOR_FAILURE), "sensor target"),
            (FaultSpec(K.STEAM_LEAK, "drum", severity=0.1), "does not take a target"),
            (FaultSpec(K.BURNER_FOULING, severity=0.6), "outside"),
            (FaultSpec(K.STEAM_LEAK, severity=0.0), "or zero"),
            (FaultSpec(K.SENSOR_DRIFT, "drum_level", severity=0.3), "outside"),
            (FaultSpec(K.FEEDWATER_PUMP_FAILURE, severity=-0.1), "outside"),
        ],
    )
    def test_invalid_specifications_are_refused(
        self, spec: FaultSpec, message: str
    ) -> None:
        with pytest.raises(FaultError, match=message):
            spec.validated()

    def test_severity_is_ignored_where_it_means_nothing(self) -> None:
        assert (
            FaultSpec(K.VALVE_STUCK, "spray", severity=7.0).validated().severity == 7.0
        )


class TestDevelopment:
    def test_a_sudden_fault_is_fully_developed_at_once(self) -> None:
        fault = ActiveFault("F1", FaultSpec(K.STEAM_LEAK, severity=0.1), 100.0)
        assert fault.intensity(100.0) == 1.0

    @pytest.mark.parametrize(
        ("time_s", "intensity"), [(90.0, 0.0), (100.0, 0.0), (130.0, 0.5), (500.0, 1.0)]
    )
    def test_a_ramped_fault_grows_linearly(
        self, time_s: float, intensity: float
    ) -> None:
        fault = ActiveFault(
            "F1", FaultSpec(K.STEAM_LEAK, severity=0.1, ramp_s=60.0), 100.0
        )
        assert fault.intensity(time_s) == pytest.approx(intensity)

    def test_labels_name_the_target_when_there_is_one(self) -> None:
        assert ActiveFault(
            "F1", FaultSpec(K.SENSOR_FAILURE, "drum_level"), 0
        ).label == ("sensor_failure:drum_level")
        assert ActiveFault("F2", FaultSpec(K.STEAM_LEAK, severity=0.1), 0).label == (
            "steam_leak"
        )


class TestRegistry:
    def test_faults_get_ids_in_order_and_duplicates_are_refused(self) -> None:
        registry = FaultRegistry()
        first = registry.inject(FaultSpec(K.VALVE_STUCK, "fuel"), 10.0)
        second = registry.inject(FaultSpec(K.VALVE_STUCK, "steam"), 20.0)
        assert (first.fault_id, second.fault_id) == ("F0001", "F0002")
        assert first.started_at_s == 10.0
        with pytest.raises(FaultError, match="already active as F0001"):
            registry.inject(FaultSpec(K.VALVE_STUCK, " FUEL "), 30.0)
        assert [fault.fault_id for fault in registry.active()] == ["F0001", "F0002"]

    def test_clearing_one_and_all(self) -> None:
        registry = FaultRegistry()
        fault = registry.inject(FaultSpec(K.STEAM_LEAK, severity=0.1), 0.0)
        registry.inject(FaultSpec(K.BURNER_FOULING, severity=0.1), 0.0)
        assert registry.clear(fault.fault_id) is fault
        with pytest.raises(FaultError, match="not active"):
            registry.clear(fault.fault_id)
        assert [f.label for f in registry.clear_all()] == ["burner_fouling"]
        assert registry.active() == ()

    def test_a_fault_id_is_never_reused(self) -> None:
        registry = FaultRegistry()
        first = registry.inject(FaultSpec(K.STEAM_LEAK, severity=0.1), 0.0)
        registry.clear(first.fault_id)
        again = registry.inject(FaultSpec(K.STEAM_LEAK, severity=0.1), 0.0)
        assert again.fault_id == "F0002"

    def test_no_faults_disturb_nothing(self) -> None:
        assert FaultRegistry().disturbances(0.0) == NO_DISTURBANCES

    def test_physical_faults_combine(self) -> None:
        registry = FaultRegistry()
        registry.inject(FaultSpec(K.BURNER_FOULING, severity=0.2), 0.0)
        registry.inject(FaultSpec(K.STEAM_LEAK, severity=0.1), 0.0)
        registry.inject(
            FaultSpec(K.FEEDWATER_PUMP_FAILURE, severity=0.5, ramp_s=100.0), 0.0
        )
        registry.inject(FaultSpec(K.SENSOR_DRIFT, "drum_level", severity=0.1), 0.0)
        found = registry.disturbances(50.0)
        assert found.combustion_efficiency_factor == pytest.approx(0.8)
        assert found.steam_leak_fraction == pytest.approx(0.1)
        assert found.feedwater_capacity_factor == pytest.approx(0.75)

    def test_a_tripped_pump_has_no_capacity(self) -> None:
        registry = FaultRegistry()
        registry.inject(FaultSpec(K.FEEDWATER_PUMP_FAILURE, severity=1.0), 0.0)
        assert registry.disturbances(0.0).feedwater_capacity_factor == 0.0

    def test_a_valve_sticks_only_once_the_fault_has_developed(self) -> None:
        registry = FaultRegistry()
        registry.inject(FaultSpec(K.VALVE_STUCK, "spray", ramp_s=30.0), 0.0)
        assert registry.stuck_valves(10.0) == frozenset()
        assert registry.stuck_valves(30.0) == {ValveId.SPRAY}

    def test_sensor_faults_are_listed_apart(self) -> None:
        registry = FaultRegistry()
        registry.inject(FaultSpec(K.STEAM_LEAK, severity=0.1), 0.0)
        drift = registry.inject(
            FaultSpec(K.SENSOR_DRIFT, "drum_level", severity=0.1), 0.0
        )
        failure = registry.inject(FaultSpec(K.SENSOR_FAILURE, "fuel_flow"), 0.0)
        assert registry.sensor_faults() == (drift, failure)


TRUE = {SensorId.DRUM_LEVEL: 4.8, SensorId.FUEL_FLOW: 20.0}


def reading(
    readings: dict[SensorId, SensorReading], sensor: SensorId
) -> tuple[float, Quality]:
    found = readings[sensor]
    return found.measured_value, found.quality


class TestInstruments:
    def test_healthy_instruments_report_the_truth(self) -> None:
        readings = SensorBank().read(TRUE, (), 0.0)
        assert reading(readings, SensorId.DRUM_LEVEL) == (4.8, Quality.GOOD)

    def test_a_drift_grows_and_is_caught_past_the_tolerance(self) -> None:
        drift = ActiveFault(
            "F1", FaultSpec(K.SENSOR_DRIFT, "drum_level", severity=0.01), 0.0
        )
        bank = SensorBank()
        early = bank.read(TRUE, (drift,), 60.0)
        span = SENSOR_SPANS[SensorId.DRUM_LEVEL]
        assert reading(early, SensorId.DRUM_LEVEL) == (
            pytest.approx(4.8 + 0.01 * span),
            Quality.GOOD,
        )
        late = bank.read(TRUE, (drift,), 180.0)
        value, quality = reading(late, SensorId.DRUM_LEVEL)
        assert value - 4.8 > VALIDATION_TOLERANCE * span
        assert quality is Quality.UNCERTAIN
        assert reading(late, SensorId.FUEL_FLOW) == (20.0, Quality.GOOD)

    def test_a_downward_drift(self) -> None:
        drift = ActiveFault(
            "F1", FaultSpec(K.SENSOR_DRIFT, "drum_level", severity=-0.05), 0.0
        )
        value, quality = reading(
            SensorBank().read(TRUE, (drift,), 60.0), SensorId.DRUM_LEVEL
        )
        assert value == pytest.approx(4.8 - 0.4)
        assert quality is Quality.UNCERTAIN

    def test_a_failed_instrument_holds_its_last_value(self) -> None:
        failure = ActiveFault("F1", FaultSpec(K.SENSOR_FAILURE, "fuel_flow"), 0.0)
        bank = SensorBank()
        first = bank.read(TRUE, (failure,), 0.0)
        later = bank.read({**TRUE, SensorId.FUEL_FLOW: 25.0}, (failure,), 10.0)
        assert reading(first, SensorId.FUEL_FLOW) == (20.0, Quality.BAD)
        assert reading(later, SensorId.FUEL_FLOW) == (20.0, Quality.BAD)

    def test_a_drift_does_not_move_a_failed_instrument(self) -> None:
        failure = ActiveFault("F1", FaultSpec(K.SENSOR_FAILURE, "drum_level"), 0.0)
        drift = ActiveFault(
            "F2", FaultSpec(K.SENSOR_DRIFT, "drum_level", severity=0.1), 0.0
        )
        readings = SensorBank().read(TRUE, (failure, drift), 600.0)
        assert reading(readings, SensorId.DRUM_LEVEL) == (4.8, Quality.BAD)

    def test_a_cleared_failure_releases_the_held_value(self) -> None:
        failure = ActiveFault("F1", FaultSpec(K.SENSOR_FAILURE, "fuel_flow"), 0.0)
        bank = SensorBank()
        bank.read(TRUE, (failure,), 0.0)
        bank.read({**TRUE, SensorId.FUEL_FLOW: 25.0}, (), 1.0)
        again = bank.read({**TRUE, SensorId.FUEL_FLOW: 26.0}, (failure,), 2.0)
        assert reading(again, SensorId.FUEL_FLOW) == (26.0, Quality.BAD)

    def test_a_new_run_forgets_held_values(self) -> None:
        failure = ActiveFault("F1", FaultSpec(K.SENSOR_FAILURE, "fuel_flow"), 0.0)
        bank = SensorBank()
        bank.read(TRUE, (failure,), 0.0)
        bank.reset()
        again = bank.read({**TRUE, SensorId.FUEL_FLOW: 21.0}, (failure,), 0.0)
        assert reading(again, SensorId.FUEL_FLOW) == (21.0, Quality.BAD)

    def test_a_fault_on_an_instrument_not_read_is_ignored(self) -> None:
        failure = ActiveFault("F1", FaultSpec(K.SENSOR_FAILURE, "steam_temp"), 0.0)
        readings = SensorBank().read(TRUE, (failure,), 0.0)
        assert SensorId.STEAM_TEMP not in readings

    def test_the_worst_quality_wins(self) -> None:
        readings = [
            SensorReading(SensorId.DRUM_LEVEL, 1.0, Quality.GOOD),
            SensorReading(SensorId.FUEL_FLOW, 1.0, Quality.UNCERTAIN),
        ]
        assert worst_quality(readings) is Quality.UNCERTAIN
        assert worst_quality([]) is Quality.GOOD
