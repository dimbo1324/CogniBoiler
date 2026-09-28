"""Steam and water properties (IAPWS-IF97) at reference points and at the edges."""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np
import pytest
from iapws import IAPWS97
from physics_engine import steam_tables as st
from prometheus_client import REGISTRY

ATMOSPHERE_PA = 101_325.0
DRUM_PA = 14.0e6


def saturated(pressure_pa: float, quality: float) -> IAPWS97:
    return IAPWS97(P=pressure_pa / 1.0e6, x=quality)


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


class TestLiquid:
    def test_density_near_997_at_room_temperature(self) -> None:
        assert st.water_density(300.0, ATMOSPHERE_PA) == pytest.approx(996.5, abs=0.5)

    def test_enthalpy_rises_with_temperature(self) -> None:
        cold = st.water_enthalpy(300.0, DRUM_PA)
        warm = st.water_enthalpy(500.0, DRUM_PA)
        assert warm > cold > 0

    def test_a_vapour_state_gives_the_saturated_liquid_density(self) -> None:
        assert st.water_density(450.0, ATMOSPHERE_PA) == pytest.approx(958.4, abs=0.1)

    def test_outside_the_formulation_the_saturated_liquid_is_used(self) -> None:
        assert st.water_enthalpy(200.0, DRUM_PA) == pytest.approx(
            float(saturated(DRUM_PA, 0.0).h) * 1000.0
        )


class TestSteam:
    def test_superheated_steam_at_the_turbine_inlet(self) -> None:
        enthalpy = st.steam_enthalpy(813.15, DRUM_PA)
        assert enthalpy == pytest.approx(3430e3, rel=0.01)

    def test_outside_the_formulation_the_saturated_vapour_is_used(self) -> None:
        assert st.steam_enthalpy(200.0, DRUM_PA) == pytest.approx(
            float(saturated(DRUM_PA, 1.0).h) * 1000.0
        )

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
        wet = float(saturated(7_000.0, 0.9).h) * 1000.0
        assert st.exhaust_temp(wet, 7_000.0) == pytest.approx(
            st.saturation_temp(7_000.0), abs=0.01
        )

    def test_superheated_exhaust_is_hotter_than_saturation(self) -> None:
        dry = float(saturated(7_000.0, 1.0).h) * 1000.0 + 200e3
        assert st.exhaust_temp(dry, 7_000.0) > st.saturation_temp(7_000.0) + 50.0

    def test_entropy_below_saturation_falls_back(self) -> None:
        assert st.steam_entropy(300.0, DRUM_PA) > 0


FALLBACKS = [
    ("water_density", st.water_density, (500.0, DRUM_PA), 0.0, "rho", 1.0),
    ("water_enthalpy", st.water_enthalpy, (500.0, DRUM_PA), 0.0, "h", 1000.0),
    ("steam_enthalpy", st.steam_enthalpy, (813.15, DRUM_PA), 1.0, "h", 1000.0),
    ("steam_entropy", st.steam_entropy, (813.15, DRUM_PA), 1.0, "s", 1000.0),
    ("isentropic_enthalpy", st.isentropic_enthalpy, (6_500.0, 7_000.0), 1.0, "h", 1e3),
    ("exhaust_temp", st.exhaust_temp, (2_300e3, 7_000.0), 1.0, "T", 1.0),
]


def failing_off_the_saturation_line(real: type[IAPWS97]) -> object:
    def build(**inputs: float) -> IAPWS97:
        if "x" not in inputs:
            raise NotImplementedError("Incoming out of bound")
        return real(**inputs)

    return build


class TestFallbacks:
    @pytest.mark.parametrize(
        ("name", "function", "args", "quality", "attribute", "scale"), FALLBACKS
    )
    def test_an_iapws_failure_falls_back_to_saturation_and_is_counted(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
        name: str,
        function: Callable[[float, float], float],
        args: tuple[float, float],
        quality: float,
        attribute: str,
        scale: float,
    ) -> None:
        monkeypatch.setattr(st, "_warned", set())
        monkeypatch.setattr(st, "IAPWS97", failing_off_the_saturation_line(IAPWS97))
        before = fallbacks(name)
        pressure = args[1]
        expected = float(getattr(saturated(pressure, quality), attribute)) * scale
        with caplog.at_level(logging.DEBUG, logger="physics_engine.steam_tables"):
            assert function(*args) == pytest.approx(expected)
            assert function(*args) == pytest.approx(expected)
        assert fallbacks(name) == before + 2
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert name in warnings[0].getMessage()

    @pytest.mark.parametrize(
        ("function", "args"),
        [
            (st.saturation_temp, (float("nan"),)),
            (st.saturation_pressure, (float("inf"),)),
            (st.water_density, (float("nan"), DRUM_PA)),
            (st.steam_enthalpy, (float("nan"), DRUM_PA)),
            (st.steam_enthalpy, (813.15, float("-inf"))),
            (st.isentropic_enthalpy, (float("nan"), 7_000.0)),
            (st.steam_state_from_enthalpy, (float("nan"), DRUM_PA)),
            (st.exhaust_temp, (2_300e3, float("nan"))),
        ],
    )
    def test_a_non_finite_input_is_refused_not_replaced(
        self, function: Callable[..., object], args: tuple[float, ...]
    ) -> None:
        with pytest.raises(ValueError, match="non-finite"):
            function(*args)

    def test_a_defect_in_the_call_is_not_mistaken_for_an_out_of_range_state(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def broken(**inputs: float) -> IAPWS97:
            if "x" not in inputs:
                raise TypeError("unexpected keyword")
            return IAPWS97(**inputs)

        monkeypatch.setattr(st, "IAPWS97", broken)
        with pytest.raises(TypeError):
            st.steam_enthalpy(813.15, DRUM_PA)


def fallbacks(function: str) -> float:
    value = REGISTRY.get_sample_value(
        "physics_property_fallbacks_total", {"function": function}
    )
    return value or 0.0
