"""Tests for Miner's-rule equipment wear tracking."""

from __future__ import annotations

import pytest
from physics_engine.equipment_health import (
    BOILER_TUBE_DESIGN_HOURS,
    COLD_START_DAMAGE,
    HOT_START_DAMAGE,
    TUBE_DESIGN_TEMP,
    TURBINE_DESIGN_HOURS,
    WARM_START_DAMAGE,
    HealthTracker,
)

ONE_HOUR = 3600.0


class TestFreshEquipment:
    def test_new_tracker_is_fully_healthy(self) -> None:
        health = HealthTracker().current_health
        assert health.turbine_damage == 0.0
        assert health.boiler_tube_damage == 0.0
        assert health.overall_health_pct == 100.0
        assert not health.maintenance_alarm

    def test_initial_counters_are_independent(self) -> None:
        tracker = HealthTracker(
            initial_turbine_hours=10.0,
            initial_tube_hours=20.0,
            initial_pump_hours=30.0,
        )
        health = tracker.current_health
        assert health.turbine_hours == 10.0
        assert health.boiler_tube_hours == 20.0
        assert health.pump_hours == 30.0


class TestRunningWear:
    def test_steady_running_consumes_design_life_per_hour(self) -> None:
        tracker = HealthTracker()
        tracker.update(dt=ONE_HOUR, power_mw=250.0, tube_temp_k=0.0, is_running=True)
        before = tracker.current_health.turbine_damage
        health = tracker.update(
            dt=ONE_HOUR, power_mw=250.0, tube_temp_k=0.0, is_running=True
        )
        assert health.turbine_damage - before == pytest.approx(
            1.0 / TURBINE_DESIGN_HOURS
        )
        assert health.turbine_hours == pytest.approx(2.0)

    def test_load_swings_add_turbine_damage(self) -> None:
        steady = HealthTracker()
        swinging = HealthTracker()
        for power in (250.0, 250.0):
            steady.update(dt=ONE_HOUR, power_mw=power, tube_temp_k=0.0, is_running=True)
        for power in (250.0, 150.0):
            swinging.update(
                dt=ONE_HOUR, power_mw=power, tube_temp_k=0.0, is_running=True
            )
        assert swinging.current_health.turbine_damage > (
            steady.current_health.turbine_damage
        )

    def test_idle_turbine_does_not_age(self) -> None:
        health = HealthTracker().update(
            dt=ONE_HOUR, power_mw=0.0, tube_temp_k=0.0, is_running=False
        )
        assert health.turbine_hours == 0.0
        assert health.turbine_damage == 0.0


class TestTubeCreep:
    def test_design_temperature_consumes_nominal_life(self) -> None:
        health = HealthTracker().update(
            dt=ONE_HOUR, power_mw=0.0, tube_temp_k=TUBE_DESIGN_TEMP, is_running=False
        )
        assert health.boiler_tube_damage == pytest.approx(
            1.0 / BOILER_TUBE_DESIGN_HOURS
        )

    def test_overheating_accelerates_creep(self) -> None:
        nominal = HealthTracker().update(
            dt=ONE_HOUR, power_mw=0.0, tube_temp_k=TUBE_DESIGN_TEMP, is_running=False
        )
        hot = HealthTracker().update(
            dt=ONE_HOUR,
            power_mw=0.0,
            tube_temp_k=TUBE_DESIGN_TEMP * 1.1,
            is_running=False,
        )
        assert hot.boiler_tube_damage == pytest.approx(
            nominal.boiler_tube_damage * 1.1**5
        )

    def test_feed_pump_runs_only_with_hot_water(self) -> None:
        cold = HealthTracker().update(
            dt=ONE_HOUR, power_mw=0.0, tube_temp_k=300.0, is_running=False
        )
        hot = HealthTracker().update(
            dt=ONE_HOUR, power_mw=0.0, tube_temp_k=500.0, is_running=False
        )
        assert cold.pump_hours == 0.0
        assert hot.pump_hours == pytest.approx(1.0)


class TestStartsAndAlarms:
    @pytest.mark.parametrize(
        ("startup_type", "damage"),
        [
            ("cold", COLD_START_DAMAGE),
            ("warm", WARM_START_DAMAGE),
            ("hot", HOT_START_DAMAGE),
        ],
    )
    def test_each_start_counts_and_costs_fatigue(
        self, startup_type: str, damage: float
    ) -> None:
        health = HealthTracker().update(
            dt=0.0,
            power_mw=0.0,
            tube_temp_k=0.0,
            is_running=False,
            startup_type=startup_type,
        )
        assert health.turbine_starts == 1.0
        assert health.turbine_damage == pytest.approx(damage)

    def test_cold_start_is_the_most_damaging(self) -> None:
        assert COLD_START_DAMAGE > WARM_START_DAMAGE > HOT_START_DAMAGE

    def test_warning_and_critical_thresholds(self) -> None:
        worn = HealthTracker(initial_turbine_damage=0.85).current_health
        failed = HealthTracker(initial_tube_damage=1.0).current_health
        assert worn.maintenance_alarm and not worn.maintenance_critical
        assert failed.maintenance_alarm and failed.maintenance_critical
        assert failed.overall_health_pct == 0.0

    def test_reset_returns_to_new_equipment(self) -> None:
        tracker = HealthTracker(initial_turbine_damage=0.5, initial_tube_hours=100.0)
        tracker.reset()
        health = tracker.current_health
        assert health.turbine_damage == 0.0
        assert health.boiler_tube_hours == 0.0
