"""The PhysicsService over an in-process gRPC server, on a paused runtime stepped by hand."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator

import cogniboiler_pb2 as pb2
import cogniboiler_pb2_grpc as pb2_grpc
import grpc
import grpc.aio
import pytest
import pytest_asyncio
from physics_engine.operating_point import OperatingPointError
from physics_engine.runtime import PhysicsRuntime, PhysicsRuntimeConfig
from physics_engine.scenarios import ScenarioName
from physics_engine.server import (
    MAX_STREAM_INTERVAL_S,
    MIN_STREAM_INTERVAL_S,
    create_server,
    stream_interval_s,
)
from prometheus_client import REGISTRY

# The PLC pings its plant link this often, so that a plant that vanished without
# closing the connection is noticed while a paused plant sends nothing.
PLC_KEEPALIVE_MS = 10_000


def refusals(rpc: str) -> float:
    value = REGISTRY.get_sample_value("physics_commands_refused_total", {"rpc": rpc})
    return value or 0.0


class Plant:
    def __init__(self, runtime: PhysicsRuntime, port: int) -> None:
        self.runtime = runtime
        self.port = port
        self.channel = grpc.aio.insecure_channel(f"127.0.0.1:{port}")
        self.stub = pb2_grpc.PhysicsServiceStub(self.channel)


@pytest_asyncio.fixture
async def plant() -> AsyncIterator[Plant]:
    runtime = PhysicsRuntime(PhysicsRuntimeConfig(start_paused=True, dt=1.0))
    await runtime.start()
    server = create_server(runtime)
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    connected = Plant(runtime, port)
    try:
        yield connected
    finally:
        await connected.channel.close()
        await server.stop(grace=None)
        await runtime.stop()


class TestState:
    async def test_health_returns_running(self, plant: Plant) -> None:
        response = await plant.stub.Health(pb2.Empty())
        assert (response.status, response.service) == ("running", "physics-engine")

    async def test_get_system_state_is_the_published_snapshot(
        self, plant: Plant
    ) -> None:
        response = await plant.stub.GetSystemState(pb2.Empty())
        snapshot = plant.runtime.snapshot
        assert response.boiler.pressure_pa == pytest.approx(snapshot.boiler.pressure)
        assert response.turbine.electrical_power_w == pytest.approx(
            snapshot.turbine.electrical_power
        )
        assert response.simulation.step_count == snapshot.step_count

    async def test_the_event_stream_sends_each_step_once(self, plant: Plant) -> None:
        stream = plant.stub.StreamSystemState(pb2.StreamRequest(interval_s=0.0))
        first = await stream.read()
        await plant.runtime.step(1)
        second = await stream.read()
        stream.cancel()
        assert second.simulation.step_count == first.simulation.step_count + 1


class TestCommands:
    async def test_a_valve_command_changes_the_plant_on_the_next_steps(
        self, plant: Plant
    ) -> None:
        before = await plant.stub.GetSystemState(pb2.Empty())
        ack = await plant.stub.ApplyControlCommand(
            pb2.ControlCommandMsg(
                fuel_valve=0.1,
                feedwater_valve=0.4,
                steam_valve=0.0,
                source=pb2.CommandSource.OPERATOR,
                operator_id="test-operator",
            )
        )
        assert ack.accepted
        await plant.runtime.step(20)
        after = await plant.stub.GetSystemState(pb2.Empty())
        assert after.actuators.steam_valve_command == 0.0
        assert after.turbine.electrical_power_w < before.turbine.electrical_power_w

    async def test_the_spray_valve_command_reaches_the_plant(
        self, plant: Plant
    ) -> None:
        sent = await plant.stub.ApplyControlCommand(
            pb2.ControlCommandMsg(
                fuel_valve=0.5, feedwater_valve=0.5, steam_valve=0.5, spray_valve=0.25
            )
        )
        assert sent.accepted
        await plant.runtime.step(1)
        state = await plant.stub.GetSystemState(pb2.Empty())
        assert state.actuators.spray_valve_command == pytest.approx(0.25)

        kept = await plant.stub.ApplyControlCommand(
            pb2.ControlCommandMsg(fuel_valve=0.5, feedwater_valve=0.5, steam_valve=0.5)
        )
        assert kept.accepted
        await plant.runtime.step(1)
        state = await plant.stub.GetSystemState(pb2.Empty())
        assert state.actuators.spray_valve_command == pytest.approx(0.25)

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1])
    async def test_a_valve_command_that_is_not_a_fraction_is_refused(
        self, plant: Plant, value: float
    ) -> None:
        before = plant.runtime.snapshot.controls
        ack = await plant.stub.ApplyControlCommand(
            pb2.ControlCommandMsg(
                fuel_valve=0.5, feedwater_valve=value, steam_valve=0.5
            )
        )
        assert not ack.accepted and "feedwater_valve" in ack.reason
        await plant.runtime.step(1)
        after = plant.runtime.snapshot.controls
        assert after.feedwater_valve_command == before.feedwater_valve_command


class TestRefusals:
    async def test_a_refusal_is_logged_with_its_operator_and_counted(
        self, plant: Plant, caplog: pytest.LogCaptureFixture
    ) -> None:
        speed, scenario = refusals("SetSimulationSpeed"), refusals("LoadScenario")
        with caplog.at_level(logging.WARNING, logger="physics_engine.server"):
            refused_speed = await plant.stub.SetSimulationSpeed(
                pb2.SimulationSpeedRequest(speed_factor=500.0, operator_id="eng")
            )
            refused_scenario = await plant.stub.LoadScenario(
                pb2.ScenarioRequest(name="moon", operator_id="eng")
            )
        assert not refused_speed.accepted and not refused_scenario.accepted
        assert refusals("SetSimulationSpeed") == speed + 1
        assert refusals("LoadScenario") == scenario + 1
        assert "SetSimulationSpeed refused for eng: speed factor 500" in caplog.text
        assert "LoadScenario refused for eng: unknown scenario 'moon'" in caplog.text

    async def test_repeated_valve_refusals_warn_once_but_all_count(
        self, plant: Plant, caplog: pytest.LogCaptureFixture
    ) -> None:
        before = refusals("ApplyControlCommand")
        bad = pb2.ControlCommandMsg(
            fuel_valve=float("nan"), feedwater_valve=0.5, steam_valve=0.5
        )
        with caplog.at_level(logging.WARNING, logger="physics_engine.server"):
            for _ in range(5):
                assert not (await plant.stub.ApplyControlCommand(bad)).accepted
        assert refusals("ApplyControlCommand") == before + 5
        assert caplog.text.count("ApplyControlCommand refused") == 1

    async def test_an_accepted_step_request_is_logged_with_its_operator(
        self, plant: Plant, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.INFO, logger="physics_engine.server"):
            ack = await plant.stub.StepSimulation(
                pb2.StepRequest(steps=2, operator_id="eng")
            )
        assert ack.accepted
        assert "2 steps requested by eng" in caplog.text

    async def test_a_failing_step_is_an_internal_error_without_its_details(
        self,
        plant: Plant,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        def broken() -> None:
            raise ArithmeticError("secret model internals")

        monkeypatch.setattr(plant.runtime, "_timed_step", broken)
        with caplog.at_level(logging.ERROR, logger="physics_engine.server"):
            with pytest.raises(grpc.aio.AioRpcError) as failed:
                await plant.stub.StepSimulation(
                    pb2.StepRequest(steps=1, operator_id="eng")
                )
        assert failed.value.code() == grpc.StatusCode.INTERNAL
        assert "secret" not in (failed.value.details() or "")
        assert "StepSimulation failed for eng" in caplog.text

    async def test_an_unsolvable_operating_point_is_refused_not_leaked(
        self, plant: Plant, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def unsolvable(scenario: ScenarioName | str) -> None:
            raise OperatingPointError("no steady point for 250.0 MW")

        monkeypatch.setattr(plant.runtime._plant, "load_scenario", unsolvable)
        ack = await plant.stub.LoadScenario(pb2.ScenarioRequest(name="part_load"))
        assert not ack.accepted and "no steady point" in ack.reason

    async def test_a_failing_scenario_load_is_an_internal_error(
        self, plant: Plant, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def broken(scenario: ScenarioName | str) -> None:
            raise RuntimeError("secret solver state")

        monkeypatch.setattr(plant.runtime._plant, "load_scenario", broken)
        with pytest.raises(grpc.aio.AioRpcError) as failed:
            await plant.stub.LoadScenario(pb2.ScenarioRequest(name="part_load"))
        assert failed.value.code() == grpc.StatusCode.INTERNAL
        assert "secret" not in (failed.value.details() or "")


class TestTimedStream:
    @pytest.mark.parametrize(
        ("requested", "interval"),
        [
            (0.0, None),
            (1e-9, MIN_STREAM_INTERVAL_S),
            (0.5, 0.5),
            (1e9, MAX_STREAM_INTERVAL_S),
        ],
    )
    def test_the_interval_is_held_to_its_bounds(
        self, requested: float, interval: float | None
    ) -> None:
        assert stream_interval_s(requested) == interval

    @pytest.mark.parametrize("requested", [float("nan"), float("inf"), -1.0])
    async def test_an_interval_that_is_not_a_duration_is_refused(
        self, plant: Plant, requested: float
    ) -> None:
        stream = plant.stub.StreamSystemState(pb2.StreamRequest(interval_s=requested))
        with pytest.raises(grpc.aio.AioRpcError) as refused:
            await stream.read()
        assert refused.value.code() == grpc.StatusCode.INVALID_ARGUMENT

    async def test_a_timed_stream_ends_when_the_runtime_stops(
        self, plant: Plant
    ) -> None:
        stream = plant.stub.StreamSystemState(
            pb2.StreamRequest(interval_s=MIN_STREAM_INTERVAL_S)
        )
        first = await stream.read()
        assert first.simulation.run_id == plant.runtime.snapshot.run_id
        await plant.runtime.stop()
        with pytest.raises(grpc.aio.AioRpcError) as ended:
            while True:
                await stream.read()
        assert ended.value.code() == grpc.StatusCode.UNAVAILABLE
        assert "physics runtime stopped" in (ended.value.details() or "")


class TestKeepalive:
    async def test_a_client_pinging_every_ten_seconds_keeps_an_idle_stream(
        self, plant: Plant
    ) -> None:
        async with grpc.aio.insecure_channel(
            f"127.0.0.1:{plant.port}",
            options=[
                ("grpc.keepalive_time_ms", PLC_KEEPALIVE_MS),
                ("grpc.keepalive_timeout_ms", 5_000),
                ("grpc.keepalive_permit_without_calls", 1),
                ("grpc.http2.max_pings_without_data", 0),
            ],
        ) as channel:
            stub = pb2_grpc.PhysicsServiceStub(channel)
            stream = stub.StreamSystemState(pb2.StreamRequest(interval_s=0.0))
            first = await stream.read()
            # A server that refuses these pings sends GOAWAY "too_many_pings" at the
            # third one, 30 s in; the paused plant sends nothing until it is stepped.
            await asyncio.sleep(3.5 * PLC_KEEPALIVE_MS / 1000.0)
            await plant.runtime.step(1)
            second = await asyncio.wait_for(stream.read(), timeout=10.0)
        assert second.simulation.step_count == first.simulation.step_count + 1
