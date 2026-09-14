"""Tests for the surface condenser and condensate return model."""

from __future__ import annotations

import pytest
from physics_engine.condenser import (
    DEAERATOR_TEMP,
    DESIGN_BACKPRESSURE,
    MAX_BACKPRESSURE,
    MIN_BACKPRESSURE,
    CondenserModel,
    CondenserState,
)

EXHAUST_ENTHALPY = 2_350e3


def _state(
    steam_flow: float,
    cooling_water_temp: float = 288.15,
    fouling_resistance: float = 0.0,
) -> CondenserState:
    model = CondenserModel(fouling_resistance=fouling_resistance)
    return model.calculate(
        steam_flow=steam_flow,
        steam_enthalpy_in=EXHAUST_ENTHALPY,
        cooling_water_temp=cooling_water_temp,
    )


class TestIdleCondenser:
    def test_idle_condenser_holds_minimum_backpressure(self) -> None:
        state = _state(steam_flow=0.0)
        assert state.backpressure_pa == MIN_BACKPRESSURE
        assert state.heat_rejected_w == 0.0
        assert state.condenser_loading == 0.0

    def test_deaerator_fixes_feedwater_temperature(self) -> None:
        assert _state(steam_flow=0.0).feedwater_temp == DEAERATOR_TEMP
        assert _state(steam_flow=150.0).feedwater_temp == DEAERATOR_TEMP


class TestLoadedCondenser:
    def test_backpressure_stays_within_physical_limits(self) -> None:
        for flow in (50.0, 150.0, 400.0):
            state = _state(steam_flow=flow)
            assert MIN_BACKPRESSURE <= state.backpressure_pa <= MAX_BACKPRESSURE

    def test_heat_flows_from_steam_to_cooling_water(self) -> None:
        state = _state(steam_flow=150.0)
        assert state.heat_rejected_w > 0.0
        assert state.condensate_temp > 288.15
        assert state.cooling_water_temp_out > 288.15

    def test_more_steam_raises_backpressure(self) -> None:
        assert _state(steam_flow=100.0).backpressure_pa < _state(150.0).backpressure_pa

    def test_warmer_cooling_water_raises_backpressure(self) -> None:
        cold = _state(steam_flow=150.0, cooling_water_temp=288.15)
        warm = _state(steam_flow=150.0, cooling_water_temp=298.15)
        assert warm.backpressure_pa > cold.backpressure_pa

    def test_fouling_raises_backpressure(self) -> None:
        clean = _state(steam_flow=150.0)
        fouled = _state(steam_flow=150.0, fouling_resistance=1e-4)
        assert fouled.backpressure_pa > clean.backpressure_pa

    def test_cooling_water_balance_matches_rejected_heat(self) -> None:
        model = CondenserModel()
        state = model.calculate(steam_flow=150.0, steam_enthalpy_in=EXHAUST_ENTHALPY)
        rise = state.heat_rejected_w / (model.cooling_water_flow * 4_186.0)
        assert state.cooling_water_temp_out - 288.15 == pytest.approx(rise)


class TestDerivedKpis:
    def test_no_efficiency_loss_at_design_backpressure(self) -> None:
        state = CondenserState(
            backpressure_pa=DESIGN_BACKPRESSURE,
            condensate_temp=312.0,
            feedwater_temp=DEAERATOR_TEMP,
            cooling_water_temp_out=300.0,
            heat_rejected_w=1.0,
            condenser_loading=0.5,
        )
        assert state.cycle_efficiency_loss_pct == 0.0
        assert state.backpressure_kpa == pytest.approx(7.0)

    def test_efficiency_loss_grows_above_design(self) -> None:
        state = CondenserState(
            backpressure_pa=DESIGN_BACKPRESSURE + 2_000.0,
            condensate_temp=318.0,
            feedwater_temp=DEAERATOR_TEMP,
            cooling_water_temp_out=300.0,
            heat_rejected_w=1.0,
            condenser_loading=0.5,
        )
        assert state.cycle_efficiency_loss_pct == pytest.approx(0.6)
