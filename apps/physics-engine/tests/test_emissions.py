"""Tests for the natural-gas stack emissions model."""

from __future__ import annotations

import pytest
from physics_engine.emissions import CO2_EMISSION_FACTOR, EmissionsCalculator


@pytest.fixture
def calc() -> EmissionsCalculator:
    return EmissionsCalculator()


class TestCarbonDioxide:
    def test_zero_fuel_emits_nothing(self, calc: EmissionsCalculator) -> None:
        state = calc.calculate(fuel_flow=0.0, flame_temp=1650.0, excess_air_ratio=1.1)
        assert state.co2_rate == 0.0
        assert state.nox_rate == 0.0
        assert state.co_rate == 0.0

    def test_co2_follows_stoichiometry(self, calc: EmissionsCalculator) -> None:
        state = calc.calculate(fuel_flow=5.0, flame_temp=1650.0, excess_air_ratio=1.1)
        assert state.co2_rate == pytest.approx(5.0 * CO2_EMISSION_FACTOR)

    def test_co2_intensity_matches_natural_gas_on_lhv_basis(
        self, calc: EmissionsCalculator
    ) -> None:
        state = calc.calculate(fuel_flow=5.0, flame_temp=1650.0, excess_air_ratio=1.1)
        expected = CO2_EMISSION_FACTOR / (42.0 / 3600.0)
        assert state.co2_intensity_kg_per_mwh == pytest.approx(expected, rel=1e-9)
        assert 200.0 < state.co2_intensity_kg_per_mwh < 260.0

    def test_co2_intensity_is_zero_without_fuel(
        self, calc: EmissionsCalculator
    ) -> None:
        state = calc.calculate(fuel_flow=0.0, flame_temp=1650.0, excess_air_ratio=1.1)
        assert state.co2_intensity_kg_per_mwh == 0.0


class TestNitrogenOxides:
    def test_nox_rises_with_flame_temperature(self, calc: EmissionsCalculator) -> None:
        rates = [
            calc.calculate(fuel_flow=5.0, flame_temp=t, excess_air_ratio=1.1).nox_rate
            for t in (1500.0, 1700.0, 2000.0)
        ]
        assert rates[0] < rates[1] < rates[2]

    def test_excess_air_dilution_lowers_nox(self, calc: EmissionsCalculator) -> None:
        optimal = calc.calculate(fuel_flow=5.0, flame_temp=1700.0, excess_air_ratio=1.1)
        diluted = calc.calculate(fuel_flow=5.0, flame_temp=1700.0, excess_air_ratio=1.5)
        assert diluted.nox_rate < optimal.nox_rate

    def test_ppmv_is_independent_of_fuel_flow(self, calc: EmissionsCalculator) -> None:
        low = calc.calculate(fuel_flow=1.0, flame_temp=1700.0, excess_air_ratio=1.1)
        high = calc.calculate(fuel_flow=8.0, flame_temp=1700.0, excess_air_ratio=1.1)
        assert low.nox_ppmv == pytest.approx(high.nox_ppmv)


class TestCarbonMonoxide:
    def test_co_rises_as_mixture_gets_rich(self, calc: EmissionsCalculator) -> None:
        rates = [
            calc.calculate(
                fuel_flow=5.0, flame_temp=1650.0, excess_air_ratio=lam
            ).co_rate
            for lam in (1.10, 1.02, 0.95)
        ]
        assert rates[0] < rates[1] < rates[2]

    def test_co_is_flat_above_design_excess_air(
        self, calc: EmissionsCalculator
    ) -> None:
        design = calc.calculate(fuel_flow=5.0, flame_temp=1650.0, excess_air_ratio=1.05)
        lean = calc.calculate(fuel_flow=5.0, flame_temp=1650.0, excess_air_ratio=1.4)
        assert design.co_rate == pytest.approx(lean.co_rate)
