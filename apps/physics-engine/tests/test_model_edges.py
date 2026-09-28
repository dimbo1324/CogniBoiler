"""Branches of the plant model a healthy run at nominal load never reaches."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from iapws import IAPWS97
from physics_engine import properties
from physics_engine.combustion import CombustionModel
from physics_engine.constants import COMBUSTION_EFFICIENCY, RATED_POWER
from physics_engine.equipment_health import (
    COLD_START_DAMAGE,
    HOT_START_DAMAGE,
    WARM_START_DAMAGE,
)
from physics_engine.heat_exchanger import counterflow_effectiveness
from physics_engine.operating_point import (
    MAX_LOAD_FRACTION,
    MIN_LOAD_FRACTION,
    OperatingPointError,
    solve_operating_point,
)
from physics_engine.plant import (
    HOT_START_STEAM_TEMP_K,
    TURBINE_ONLINE_POWER_W,
    WARM_START_STEAM_TEMP_K,
    PlantSimulator,
)
from physics_engine.proto_mapping import (
    emissions_to_proto,
    fault_kind_from_proto,
    performance_to_proto,
)
from physics_engine.scenarios import ScenarioName


class TestCombustion:
    def test_fuel_rich_combustion_loses_efficiency(self) -> None:
        state = CombustionModel().calculate(fuel_valve=0.5, excess_air_ratio=0.9)
        assert state.eta_combustion == pytest.approx(COMBUSTION_EFFICIENCY * 0.95)

    def test_the_optimal_band_is_flat(self) -> None:
        state = CombustionModel().calculate(fuel_valve=0.5, excess_air_ratio=1.03)
        assert state.eta_combustion == pytest.approx(COMBUSTION_EFFICIENCY)

    def test_excess_air_costs_a_little_efficiency(self) -> None:
        state = CombustionModel().calculate(fuel_valve=0.5, excess_air_ratio=1.15)
        assert state.eta_combustion == pytest.approx(COMBUSTION_EFFICIENCY - 0.005)

    def test_the_air_ratio_is_held_to_its_physical_limits(self) -> None:
        model = CombustionModel()
        assert model.calculate(0.5, excess_air_ratio=0.1).excess_air_ratio == 0.5
        assert model.calculate(0.5, excess_air_ratio=9.0).excess_air_ratio == 2.0

    @pytest.mark.parametrize(("factor", "applied"), [(1.5, 1.0), (-0.2, 0.0)])
    def test_the_fouling_factor_is_a_fraction(
        self, factor: float, applied: float
    ) -> None:
        model = CombustionModel()
        clean = model.calculate(fuel_valve=0.5)
        fouled = model.calculate(fuel_valve=0.5, efficiency_factor=factor)
        assert fouled.eta_combustion == pytest.approx(clean.eta_combustion * applied)

    def test_no_fuel_gives_an_ambient_flame(self) -> None:
        state = CombustionModel().calculate(fuel_valve=0.0)
        assert (state.heat_released, state.flue_gas_flow) == (0.0, 0.0)
        assert state.flue_gas_temp_exit == pytest.approx(293.15)


class TestHeatExchanger:
    def test_no_capacity_on_one_side_transfers_nothing(self) -> None:
        assert counterflow_effectiveness(1.0e5, 0.0, 1.0e4) == 0.0

    def test_balanced_streams_use_the_equal_capacity_limit(self) -> None:
        # With equal capacity rates the general formula divides 0 by 0.
        assert counterflow_effectiveness(2.0e4, 1.0e4, 1.0e4) == pytest.approx(2 / 3)

    def test_unbalanced_streams_follow_the_counterflow_formula(self) -> None:
        ntu, ratio = 2.0, 0.5
        decay = np.exp(-ntu * (1.0 - ratio))
        expected = (1.0 - decay) / (1.0 - ratio * decay)
        assert counterflow_effectiveness(2.0e4, 1.0e4, 2.0e4) == pytest.approx(expected)


class TestPropertyTables:
    @pytest.mark.parametrize("temp_k", [300.0, 450.0, 609.8, 640.0])
    def test_the_tables_agree_with_iapws(self, temp_k: float) -> None:
        liquid = IAPWS97(T=temp_k, x=0.0)
        vapour = IAPWS97(T=temp_k, x=1.0)
        assert properties.saturation_pressure(temp_k) == pytest.approx(
            liquid.P * 1.0e6, rel=1e-3
        )
        assert properties.liquid_density(temp_k) == pytest.approx(liquid.rho, rel=1e-3)
        assert properties.liquid_enthalpy(temp_k) == pytest.approx(
            liquid.h * 1000.0, rel=1e-3
        )
        assert properties.vapor_enthalpy_at_pressure(liquid.P * 1.0e6) == pytest.approx(
            vapour.h * 1000.0, rel=1e-3
        )
        assert properties.saturation_temperature(liquid.P * 1.0e6) == pytest.approx(
            temp_k, abs=0.05
        )

    def test_temperatures_beyond_the_table_are_clamped_to_its_ends(self) -> None:
        low, high = properties.T_TABLE_MIN_K, properties.T_TABLE_MAX_K
        assert properties.liquid_density(100.0) == properties.liquid_density(low)
        assert properties.liquid_cp(900.0) == properties.liquid_cp(high)
        assert properties.saturation_pressure(900.0) == pytest.approx(
            properties.saturation_pressure(high)
        )


class TestOperatingPoint:
    @pytest.mark.parametrize(
        "fraction", [MIN_LOAD_FRACTION - 0.01, MAX_LOAD_FRACTION + 0.01]
    )
    def test_a_load_outside_the_range_is_refused(self, fraction: float) -> None:
        with pytest.raises(OperatingPointError, match="outside"):
            solve_operating_point(fraction * RATED_POWER)


class TestStartClassification:
    @pytest.mark.parametrize(
        ("steam_temp_k", "damage"),
        [
            (HOT_START_STEAM_TEMP_K + 10.0, HOT_START_DAMAGE),
            (HOT_START_STEAM_TEMP_K - 10.0, WARM_START_DAMAGE),
            (WARM_START_STEAM_TEMP_K - 10.0, COLD_START_DAMAGE),
        ],
    )
    def test_a_start_is_classified_by_its_steam_temperature(
        self, steam_temp_k: float, damage: float
    ) -> None:
        plant = PlantSimulator(scenario=ScenarioName.HOT_START)
        before = plant.snapshot.health
        assert not plant._turbine_online
        balance = plant._system.boiler.balance(plant._state, plant._controls)
        online = replace(
            plant.snapshot.turbine,
            electrical_power=TURBINE_ONLINE_POWER_W * 2.0,
            steam_temp_in=steam_temp_k,
        )
        health = plant._update_health(1.0, plant._state, balance, online)
        assert health.turbine_starts == before.turbine_starts + 1
        assert health.turbine_damage - before.turbine_damage == pytest.approx(
            damage, rel=0.05
        )
        again = plant._update_health(1.0, plant._state, balance, online)
        assert again.turbine_starts == health.turbine_starts


class TestProtoEdges:
    def test_a_plant_that_does_not_generate_reports_no_rates(self) -> None:
        snapshot = PlantSimulator(scenario=ScenarioName.HOT_START).snapshot
        performance = performance_to_proto(snapshot)
        assert performance.electrical_power_w == 0.0
        assert (performance.net_efficiency, performance.plant_heat_rate_j_per_j) == (
            0.0,
            0.0,
        )
        assert emissions_to_proto(snapshot).co2_intensity_kg_per_mwh == 0.0

    def test_an_unknown_fault_kind_is_refused(self) -> None:
        with pytest.raises(ValueError, match="99"):
            fault_kind_from_proto(99)
