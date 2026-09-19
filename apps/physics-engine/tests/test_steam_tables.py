"""Steam and water properties (IAPWS-IF97) at reference points and at the edges."""

from __future__ import annotations

import numpy as np
import pytest
from physics_engine import steam_tables as st

ATMOSPHERE_PA = 101_325.0
DRUM_PA = 14.0e6


class TestSaturation:
    def test_water_boils_at_one_atmosphere_at_373_12_k(self) -> None:
        assert st.saturation_temp(ATMOSPHERE_PA) == pytest.approx(373.124, abs=0.01)

    def test_saturation_temperature_and_pressure_are_inverse(self) -> None:
        for pressure in (5_000.0, ATMOSPHERE_PA, 1.0e6, DRUM_PA, 20.0e6):
            back = st.saturation_pressure(st.saturation_temp(pressure))
            assert back == pytest.approx(pressure, rel=1e-4)

    def test_the_drum_saturates_near_609_k(self) -> None:
        assert st.saturation_temp(DRUM_PA) == pytest.approx(609.4, abs=0.5)

    def test_inputs_are_clamped_to_the_saturation_line(self) -> None:
        assert st.saturation_temp(1.0e9) == pytest.approx(st.saturation_temp(22.064e6))
        assert st.saturation_temp(1.0) == pytest.approx(st.saturation_temp(611.7))
        assert st.saturation_pressure(200.0) == pytest.approx(
            st.saturation_pressure(273.16)
        )
        assert st.saturation_pressure(900.0) == pytest.approx(
            st.saturation_pressure(647.0)
        )

    def test_numpy_scalars_are_accepted(self) -> None:
        assert st.saturation_temp(np.float64(ATMOSPHERE_PA)) == pytest.approx(
            373.124, abs=0.01
        )
        assert st.saturation_temp(np.array([ATMOSPHERE_PA])) == pytest.approx(
            373.124, abs=0.01
        )


class TestLatentHeat:
    def test_about_2257_kj_per_kg_at_one_atmosphere(self) -> None:
        assert st.latent_heat(ATMOSPHERE_PA) == pytest.approx(2256.5e3, rel=2e-3)

    def test_it_is_the_gap_between_vapour_and_liquid(self) -> None:
        gap = st.saturated_vapor_enthalpy(DRUM_PA) - st.saturated_liquid_enthalpy(
            DRUM_PA
        )
        assert st.latent_heat(DRUM_PA) == pytest.approx(gap)

    def test_it_shrinks_toward_the_critical_point(self) -> None:
        heats = [st.latent_heat(p) for p in (1.0e5, 1.0e6, 1.0e7, 2.0e7)]
        assert heats == sorted(heats, reverse=True)
        assert st.latent_heat(1.0e9) < 0.2e6


class TestLiquid:
    def test_density_near_997_at_room_temperature(self) -> None:
        assert st.water_density(300.0, ATMOSPHERE_PA) == pytest.approx(996.5, abs=0.5)

    def test_specific_heat_near_4180(self) -> None:
        assert st.water_specific_heat(300.0, ATMOSPHERE_PA) == pytest.approx(
            4180.0, rel=5e-3
        )

    def test_enthalpy_rises_with_temperature(self) -> None:
        cold = st.water_enthalpy(300.0, DRUM_PA)
        warm = st.water_enthalpy(500.0, DRUM_PA)
        assert warm > cold > 0

    def test_a_vapour_state_gives_the_saturated_liquid_density(self) -> None:
        assert st.water_density(450.0, ATMOSPHERE_PA) == pytest.approx(958.4, abs=0.1)

    def test_outside_the_formulation_the_saturated_liquid_is_used(self) -> None:
        assert st.water_enthalpy(200.0, DRUM_PA) == pytest.approx(
            st.saturated_liquid_enthalpy(DRUM_PA)
        )
        assert st.water_specific_heat(200.0, DRUM_PA) == 4186.0


class TestSteam:
    def test_superheated_steam_at_the_turbine_inlet(self) -> None:
        enthalpy = st.steam_enthalpy(813.15, DRUM_PA)
        assert enthalpy == pytest.approx(3430e3, rel=0.01)
        assert 35.0 < st.steam_density(813.15, DRUM_PA) < 50.0
        assert st.steam_specific_heat(813.15, DRUM_PA) > 2000.0

    def test_outside_the_formulation_the_saturated_vapour_is_used(self) -> None:
        assert st.steam_enthalpy(200.0, DRUM_PA) == pytest.approx(
            st.saturated_vapor_enthalpy(DRUM_PA)
        )
        assert st.steam_density(200.0, DRUM_PA) == pytest.approx(87.04, abs=0.01)
        assert st.steam_specific_heat(200.0, DRUM_PA) == 2010.0

    def test_an_isentropic_expansion_releases_enthalpy(self) -> None:
        inlet = st.steam_enthalpy(813.15, DRUM_PA)
        entropy = st.steam_entropy(813.15, DRUM_PA)
        outlet = st.isentropic_enthalpy(entropy, 7_000.0)
        assert 0.3 * inlet < inlet - outlet < 0.45 * inlet

    def test_the_state_is_recovered_from_enthalpy(self) -> None:
        enthalpy = st.steam_enthalpy(813.15, DRUM_PA)
        temperature, entropy = st.steam_state_from_enthalpy(enthalpy, DRUM_PA)
        assert temperature == pytest.approx(813.15, abs=0.5)
        assert entropy == pytest.approx(st.steam_entropy(813.15, DRUM_PA), rel=1e-3)

    def test_wet_exhaust_is_at_the_condenser_saturation_temperature(self) -> None:
        wet = st.saturated_liquid_enthalpy(7_000.0) + 0.9 * st.latent_heat(7_000.0)
        assert st.exhaust_temp(wet, 7_000.0) == pytest.approx(
            st.saturation_temp(7_000.0), abs=0.01
        )

    def test_superheated_exhaust_is_hotter_than_saturation(self) -> None:
        dry = st.saturated_vapor_enthalpy(7_000.0) + 200e3
        assert st.exhaust_temp(dry, 7_000.0) > st.saturation_temp(7_000.0) + 50.0

    def test_entropy_below_saturation_falls_back(self) -> None:
        assert st.steam_entropy(300.0, DRUM_PA) > 0
