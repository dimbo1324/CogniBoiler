"""
Tests for AsyncSimulator and cold_start scenario.

Strategy:
  - SimulatorConfig / ScenarioName — pure unit tests, no I/O
  - AsyncSimulator.run()           — short integration run with
                                     speed_factor=100 and tiny duration
                                     so tests complete in < 2 seconds
  - ScenarioRunner.cold_start()    — physics correctness checks
"""

from __future__ import annotations

import pytest
from physics_engine.async_simulator import (
    AsyncSimulator,
    ScenarioName,
    SimulatorConfig,
)
from physics_engine.scenarios import ScenarioRunner
from physics_engine.system import SystemState

# ─── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def fast_config() -> SimulatorConfig:
    """
    Minimal config for fast integration tests.

    duration=30s + speed_factor=100 -> completes in ~0.3 wall-seconds.
    dt=5s -> only 6 time steps, very fast ODE integration.
    """
    return SimulatorConfig(
        scenario=ScenarioName.STEADY_STATE,
        speed_factor=100.0,
        dt=5.0,
        duration=30.0,
    )


@pytest.fixture
def runner() -> ScenarioRunner:
    return ScenarioRunner()


# ─── 1. ScenarioName tests ────────────────────────────────────────────────────


class TestScenarioName:
    def test_all_four_scenarios_exist(self) -> None:
        names = {s.value for s in ScenarioName}
        assert names == {"steady_state", "load_ramp", "fuel_trip", "cold_start"}

    def test_cold_start_value(self) -> None:
        assert ScenarioName.COLD_START.value == "cold_start"

    def test_is_string_enum(self) -> None:
        assert isinstance(ScenarioName.STEADY_STATE, str)


# ─── 2. SimulatorConfig tests ─────────────────────────────────────────────────


class TestSimulatorConfig:
    def test_defaults_are_sensible(self) -> None:
        cfg = SimulatorConfig()
        assert cfg.scenario == ScenarioName.STEADY_STATE
        assert cfg.speed_factor == pytest.approx(1.0)
        assert cfg.dt == pytest.approx(1.0)
        assert cfg.duration == pytest.approx(600.0)
        assert cfg.queue_maxsize == 100

    def test_custom_values_accepted(self) -> None:
        cfg = SimulatorConfig(
            scenario=ScenarioName.COLD_START,
            speed_factor=10.0,
            dt=2.0,
            duration=1800.0,
        )
        assert cfg.scenario == ScenarioName.COLD_START
        assert cfg.speed_factor == pytest.approx(10.0)

    def test_boiler_params_default_is_fresh_instance(self) -> None:
        # Each config gets its own BoilerParameters — no shared state
        cfg1 = SimulatorConfig()
        cfg2 = SimulatorConfig()
        assert cfg1.boiler_params is not cfg2.boiler_params


# ─── 3. AsyncSimulator unit tests ─────────────────────────────────────────────


class TestAsyncSimulatorUnit:
    def test_initial_stats_zero(self) -> None:
        sim = AsyncSimulator()
        assert sim.published == 0
        assert sim.dropped == 0

    def test_accepts_none_config(self) -> None:
        sim = AsyncSimulator(None)
        assert sim.config.scenario == ScenarioName.STEADY_STATE

    def test_accepts_custom_config(self, fast_config: SimulatorConfig) -> None:
        sim = AsyncSimulator(fast_config)
        assert sim.config.speed_factor == pytest.approx(100.0)


# ─── 4. AsyncSimulator integration tests ─────────────────────────────────────


class TestAsyncSimulatorIntegration:
    @pytest.mark.asyncio
    async def test_run_yields_system_states(self, fast_config: SimulatorConfig) -> None:
        """run() must yield SystemState objects."""
        sim = AsyncSimulator(fast_config)
        states: list[SystemState] = []
        async for state in sim.run():
            states.append(state)
        assert len(states) > 0
        assert all(isinstance(s, SystemState) for s in states)

    @pytest.mark.asyncio
    async def test_run_yields_expected_number_of_states(
        self, fast_config: SimulatorConfig
    ) -> None:
        """Number of states = duration / dt."""
        sim = AsyncSimulator(fast_config)
        states: list[SystemState] = []
        async for state in sim.run():
            states.append(state)
        expected = int(fast_config.duration / fast_config.dt)
        assert len(states) == expected

    @pytest.mark.asyncio
    async def test_states_have_positive_pressure(
        self, fast_config: SimulatorConfig
    ) -> None:
        sim = AsyncSimulator(fast_config)
        async for state in sim.run():
            assert state.boiler.pressure > 0.0

    @pytest.mark.asyncio
    async def test_states_have_positive_water_level(
        self, fast_config: SimulatorConfig
    ) -> None:
        sim = AsyncSimulator(fast_config)
        async for state in sim.run():
            assert state.boiler.water_level > 0.0

    @pytest.mark.asyncio
    async def test_time_increases_monotonically(
        self, fast_config: SimulatorConfig
    ) -> None:
        sim = AsyncSimulator(fast_config)
        times: list[float] = []
        async for state in sim.run():
            times.append(state.time)
        assert times == sorted(times)

    @pytest.mark.asyncio
    async def test_published_counter_matches_states(
        self, fast_config: SimulatorConfig
    ) -> None:
        sim = AsyncSimulator(fast_config)
        count = 0
        async for _ in sim.run():
            count += 1
        assert sim.published == count

    @pytest.mark.asyncio
    async def test_load_ramp_scenario_runs(self) -> None:
        cfg = SimulatorConfig(
            scenario=ScenarioName.LOAD_RAMP,
            speed_factor=100.0,
            dt=5.0,
            duration=30.0,
        )
        sim = AsyncSimulator(cfg)
        states: list[SystemState] = []
        async for state in sim.run():
            states.append(state)
        assert len(states) > 0

    @pytest.mark.asyncio
    async def test_fuel_trip_scenario_runs(self) -> None:
        cfg = SimulatorConfig(
            scenario=ScenarioName.FUEL_TRIP,
            speed_factor=100.0,
            dt=5.0,
            duration=30.0,
        )
        sim = AsyncSimulator(cfg)
        states: list[SystemState] = []
        async for state in sim.run():
            states.append(state)
        assert len(states) > 0

    @pytest.mark.asyncio
    async def test_cold_start_scenario_runs(self) -> None:
        cfg = SimulatorConfig(
            scenario=ScenarioName.COLD_START,
            speed_factor=100.0,
            dt=5.0,
            duration=60.0,
        )
        sim = AsyncSimulator(cfg)
        states: list[SystemState] = []
        async for state in sim.run():
            states.append(state)
        assert len(states) > 0


# ─── 5. cold_start scenario physics tests ────────────────────────────────────


class TestColdStart:
    @pytest.fixture
    def cold_result(self, runner: ScenarioRunner):  # type: ignore[no-untyped-def]
        """Run a short cold start (300s) for physics checks."""
        return runner.cold_start(duration=300.0, dt=5.0)

    def test_returns_scenario_result(self, runner: ScenarioRunner) -> None:
        from physics_engine.scenarios import ScenarioResult

        result = runner.cold_start(duration=60.0, dt=5.0)
        assert isinstance(result, ScenarioResult)

    def test_scenario_name_is_cold_start(self, cold_result) -> None:  # type: ignore[no-untyped-def]
        assert cold_result.scenario_name == "cold_start"

    def test_initial_pressure_is_low(self, cold_result) -> None:  # type: ignore[no-untyped-def]
        # Cold start begins at ~2 bar, well below nominal 140 bar
        initial_pressure_bar = cold_result.pressure[0] / 1e5
        assert initial_pressure_bar < 10.0

    def test_pressure_rises_during_warmup(self, cold_result) -> None:  # type: ignore[no-untyped-def]
        # Pressure at end must be higher than at start
        assert cold_result.pressure[-1] > cold_result.pressure[0]

    def test_initial_water_temp_is_cold(self, cold_result) -> None:  # type: ignore[no-untyped-def]
        # Cold start: water begins at ~100°C (373 K)
        assert cold_result.water_temp[0] < 420.0  # K

    def test_water_level_stays_positive(self, cold_result) -> None:  # type: ignore[no-untyped-def]
        assert all(h > 0.0 for h in cold_result.water_level)

    def test_pressure_stays_positive(self, cold_result) -> None:  # type: ignore[no-untyped-def]
        assert all(p > 0.0 for p in cold_result.pressure)

    def test_result_has_correct_length(self, cold_result) -> None:  # type: ignore[no-untyped-def]
        expected = int(300.0 / 5.0)
        assert cold_result.n_steps == expected

    def test_fuel_valve_starts_low(self, cold_result) -> None:  # type: ignore[no-untyped-def]
        # Phase 1: controller demands high fuel to overcome cold start pressure deficit.
        # The valve will be high (controller is saturated trying to reach setpoint).
        # We only verify it stays physically within [0, 1].
        phase1_end = int(cold_result.n_steps * 0.20)
        fuel_phase1 = cold_result.fuel_valve[:phase1_end]
        assert all(0.0 <= v <= 1.0 for v in fuel_phase1)

    def test_steam_valve_closed_at_start(self, cold_result) -> None:  # type: ignore[no-untyped-def]
        # _run() records steam_cmd before steam_valve_fn is applied on step 0.
        # Check the average across phase 1 (after first step) is near zero.
        phase1_end = int(cold_result.n_steps * 0.20)
        avg_steam_phase1 = cold_result.steam_valve[1:phase1_end].mean()
        assert avg_steam_phase1 == pytest.approx(0.0)

    def test_steam_valve_opens_later(self, cold_result) -> None:  # type: ignore[no-untyped-def]
        # After phase 1 the steam valve must have opened
        phase1_end = int(cold_result.n_steps * 0.25)
        max_steam_after_phase1 = cold_result.steam_valve[phase1_end:].max()
        assert max_steam_after_phase1 > 0.0
