"""The live runtime and the PhysicsService around it, stepped rather than timed."""

from __future__ import annotations

import asyncio
import logging
import socket
from collections.abc import AsyncIterator

import cogniboiler_pb2 as pb2
import cogniboiler_pb2_grpc as pb2_grpc
import grpc
import grpc.aio
import pytest
import pytest_asyncio
from physics_engine.faults import FaultError, FaultKind, FaultSpec
from physics_engine.runtime import (
    MAX_STEPS_PER_REQUEST,
    PhysicsRuntime,
    PhysicsRuntimeConfig,
    RunState,
    RuntimeCommandError,
    RuntimeUnavailableError,
)
from physics_engine.scenarios import ScenarioError, ScenarioName
from physics_engine.server import PhysicsServicer, serve


@pytest_asyncio.fixture
async def paused() -> AsyncIterator[PhysicsRuntime]:
    runtime = PhysicsRuntime(PhysicsRuntimeConfig(start_paused=True, dt=1.0))
    await runtime.start()
    try:
        yield runtime
    finally:
        await runtime.stop()


async def until(predicate: object, timeout_s: float = 10.0) -> None:
    async with asyncio.timeout(timeout_s):
        while not predicate():  # type: ignore[operator]
            await asyncio.sleep(0.005)


class TestStepping:
    async def test_a_paused_runtime_advances_only_when_stepped(
        self, paused: PhysicsRuntime
    ) -> None:
        before = paused.simulation_status()
        assert before.run_state is RunState.PAUSED
        await asyncio.sleep(0.05)
        assert paused.simulation_status().step_count == before.step_count
        status = await paused.step(3)
        assert status.step_count == before.step_count + 3
        assert status.simulation_time_s == pytest.approx(before.simulation_time_s + 3.0)

    async def test_every_step_is_published(self, paused: PhysicsRuntime) -> None:
        sequence, _ = await asyncio.wait_for(
            asyncio.shield(paused.wait_for_update(-1)), timeout=1.0
        )
        waiter = asyncio.create_task(paused.wait_for_update(sequence))
        await paused.step(1)
        newer, snapshot = await asyncio.wait_for(waiter, timeout=5.0)
        assert newer == sequence + 1
        assert snapshot is await paused.get_snapshot()

    @pytest.mark.parametrize("steps", [0, MAX_STEPS_PER_REQUEST + 1])
    async def test_a_step_count_outside_the_limits_is_refused(
        self, paused: PhysicsRuntime, steps: int
    ) -> None:
        with pytest.raises(RuntimeCommandError, match="outside"):
            await paused.step(steps)

    async def test_a_running_simulation_cannot_be_stepped(
        self, paused: PhysicsRuntime
    ) -> None:
        await paused.resume()
        with pytest.raises(RuntimeCommandError, match="pause the simulation"):
            await paused.step(1)

    async def test_resuming_advances_on_the_wall_clock_and_pausing_stops_it(
        self, paused: PhysicsRuntime
    ) -> None:
        await paused.set_speed(50.0)
        start = paused.simulation_status().step_count
        await paused.resume()
        await until(lambda: paused.simulation_status().step_count >= start + 2)
        status = await paused.pause()
        assert status.run_state is RunState.PAUSED
        await asyncio.sleep(0.1)
        settled = paused.simulation_status().step_count
        await asyncio.sleep(0.1)
        assert paused.simulation_status().step_count == settled


class TestSpeed:
    async def test_the_speed_sets_the_wall_time_per_step(
        self, paused: PhysicsRuntime
    ) -> None:
        status = await paused.set_speed(10.0)
        assert status.speed_factor == 10.0
        assert paused.wall_step_s == pytest.approx(0.1)

    @pytest.mark.parametrize("speed", [0.05, 50.5])
    async def test_a_speed_outside_the_limits_is_refused(
        self, paused: PhysicsRuntime, speed: float
    ) -> None:
        with pytest.raises(RuntimeCommandError, match="speed factor"):
            await paused.set_speed(speed)

    def test_a_runtime_needs_a_positive_speed(self) -> None:
        with pytest.raises(ValueError, match="speed_factor"):
            PhysicsRuntime(PhysicsRuntimeConfig(speed_factor=0.0))


class TestScenariosAndFaults:
    async def test_loading_a_scenario_starts_a_new_run(
        self, paused: PhysicsRuntime
    ) -> None:
        before = paused.simulation_status()
        status = await paused.load_scenario(ScenarioName.PART_LOAD)
        assert status.scenario is ScenarioName.PART_LOAD
        assert status.run_id == before.run_id + 1
        assert status.step_count == 0

    async def test_an_unknown_scenario_is_refused(self, paused: PhysicsRuntime) -> None:
        with pytest.raises(ScenarioError, match="unknown scenario"):
            await paused.load_scenario("lunar_eclipse")

    async def test_faults_are_published_with_the_plant(
        self, paused: PhysicsRuntime
    ) -> None:
        leak = await paused.inject_fault(FaultSpec(FaultKind.STEAM_LEAK, severity=0.1))
        drift = await paused.inject_fault(
            FaultSpec(FaultKind.SENSOR_DRIFT, "drum_level", severity=0.05)
        )
        assert {f.fault_id for f in paused.snapshot.faults} == {
            leak.fault_id,
            drift.fault_id,
        }
        assert (await paused.clear_fault(leak.fault_id)).label == "steam_leak"
        assert [f.label for f in await paused.clear_faults()] == [
            "sensor_drift:drum_level"
        ]
        assert paused.snapshot.faults == ()
        with pytest.raises(FaultError):
            await paused.clear_fault(leak.fault_id)

    async def test_the_runtime_validates_valves_again(
        self, paused: PhysicsRuntime
    ) -> None:
        with pytest.raises(ValueError, match="feedwater_valve"):
            await paused.apply_command(
                fuel_valve=0.5, feedwater_valve=1.2, steam_valve=0.5
            )
        with pytest.raises(ValueError, match="spray_valve"):
            await paused.apply_command(
                fuel_valve=0.5, feedwater_valve=0.5, steam_valve=0.5, spray_valve=-0.1
            )


class TestLifecycle:
    async def test_starting_twice_keeps_one_loop(self, paused: PhysicsRuntime) -> None:
        await paused.start()
        assert paused.status == "running"

    async def test_a_stopped_runtime_releases_its_readers(self) -> None:
        runtime = PhysicsRuntime(PhysicsRuntimeConfig(start_paused=True))
        await runtime.start()
        sequence, _ = await runtime.wait_for_update(-1)
        waiter = asyncio.create_task(runtime.wait_for_update(sequence))
        await asyncio.sleep(0)
        await runtime.stop()
        assert runtime.status == "stopped"
        with pytest.raises(RuntimeUnavailableError, match="stopped"):
            await asyncio.wait_for(waiter, timeout=5.0)

    async def test_a_failing_step_degrades_the_runtime(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        runtime = PhysicsRuntime(PhysicsRuntimeConfig(speed_factor=50.0))

        def broken() -> None:
            raise ArithmeticError("the model diverged")

        monkeypatch.setattr(runtime, "_timed_step", broken)
        with caplog.at_level(logging.ERROR, logger="physics_engine.runtime"):
            await runtime.start()
            await until(lambda: runtime.status == "degraded")
        assert runtime.last_error == "the model diverged"
        assert "Physics runtime loop failed" in caplog.text
        with pytest.raises(RuntimeUnavailableError, match="diverged"):
            await runtime.wait_for_update(10**9)
        await runtime.stop()


@pytest_asyncio.fixture
async def service(
    paused: PhysicsRuntime,
) -> AsyncIterator[pb2_grpc.PhysicsServiceStub]:
    server = grpc.aio.server()
    pb2_grpc.add_PhysicsServiceServicer_to_server(PhysicsServicer(paused), server)
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    channel = grpc.aio.insecure_channel(f"127.0.0.1:{port}")
    try:
        yield pb2_grpc.PhysicsServiceStub(channel)
    finally:
        await channel.close()
        await server.stop(grace=None)


class TestPhysicsService:
    async def test_simulation_control_over_grpc(
        self, service: pb2_grpc.PhysicsServiceStub
    ) -> None:
        status = await service.GetSimulationStatus(pb2.Empty())
        assert status.run_state == pb2.SimulationRunState.SIMULATION_PAUSED
        stepped = await service.StepSimulation(
            pb2.StepRequest(steps=2, operator_id="e")
        )
        assert stepped.accepted
        assert stepped.status.step_count == status.step_count + 2
        faster = await service.SetSimulationSpeed(
            pb2.SimulationSpeedRequest(speed_factor=5.0)
        )
        assert faster.accepted and faster.status.speed_factor == 5.0
        resumed = await service.ResumeSimulation(pb2.SimulationControlRequest())
        assert resumed.status.run_state == pb2.SimulationRunState.SIMULATION_RUNNING
        refused = await service.StepSimulation(pb2.StepRequest(steps=1))
        assert (refused.accepted, refused.reason) == (
            False,
            "pause the simulation before stepping it",
        )
        paused = await service.PauseSimulation(pb2.SimulationControlRequest())
        assert paused.status.run_state == pb2.SimulationRunState.SIMULATION_PAUSED

    async def test_refused_requests_carry_their_reason(
        self, service: pb2_grpc.PhysicsServiceStub
    ) -> None:
        speed = await service.SetSimulationSpeed(
            pb2.SimulationSpeedRequest(speed_factor=500.0)
        )
        assert not speed.accepted and "speed factor" in speed.reason
        scenario = await service.LoadScenario(pb2.ScenarioRequest(name="moon"))
        assert not scenario.accepted and "unknown scenario" in scenario.reason
        valve = await service.ApplyControlCommand(
            pb2.ControlCommandMsg(fuel_valve=2.0, feedwater_valve=0.5, steam_valve=0.5)
        )
        assert not valve.accepted and "fuel_valve" in valve.reason

    async def test_scenarios_are_listed_and_loaded(
        self, service: pb2_grpc.PhysicsServiceStub
    ) -> None:
        listed = await service.ListScenarios(pb2.Empty())
        assert listed.current == "steady_state"
        assert {item.name for item in listed.scenarios} == {
            s.value for s in ScenarioName
        }
        loaded = await service.LoadScenario(pb2.ScenarioRequest(name="hot_start"))
        assert loaded.accepted and loaded.status.scenario == "hot_start"

    async def test_faults_are_injected_and_cleared(
        self, service: pb2_grpc.PhysicsServiceStub
    ) -> None:
        injected = await service.InjectFault(
            pb2.FaultRequest(
                kind=pb2.FaultKind.FAULT_VALVE_STUCK, target="spray", operator_id="e"
            )
        )
        assert injected.accepted
        fault = injected.faults[0]
        assert (fault.kind, fault.label) == (
            pb2.FaultKind.FAULT_VALVE_STUCK,
            "valve_stuck:spray",
        )
        state = await service.GetSystemState(pb2.Empty())
        assert [f.label for f in state.active_faults] == ["valve_stuck:spray"]
        cleared = await service.ClearFault(
            pb2.FaultClearRequest(fault_id=fault.fault_id)
        )
        assert cleared.accepted and cleared.faults[0].fault_id == fault.fault_id
        again = await service.ClearFault(pb2.FaultClearRequest(fault_id=fault.fault_id))
        assert not again.accepted and "not active" in again.reason
        everything = await service.ClearFault(pb2.FaultClearRequest(all=True))
        assert everything.accepted and list(everything.faults) == []

    @pytest.mark.parametrize(
        "request_",
        [
            pb2.FaultRequest(kind=pb2.FaultKind.FAULT_KIND_UNSPECIFIED),
            pb2.FaultRequest(kind=pb2.FaultKind.FAULT_STEAM_LEAK, severity=0.9),
        ],
    )
    async def test_invalid_faults_are_refused(
        self, service: pb2_grpc.PhysicsServiceStub, request_: pb2.FaultRequest
    ) -> None:
        refused = await service.InjectFault(request_)
        assert not refused.accepted and refused.reason

    async def test_a_timed_stream_repeats_the_state(
        self, service: pb2_grpc.PhysicsServiceStub
    ) -> None:
        stream = service.StreamSystemState(pb2.StreamRequest(interval_s=0.01))
        first = await stream.read()
        second = await stream.read()
        stream.cancel()
        assert first.boiler.pressure_pa > 0
        assert second.simulation.run_id == first.simulation.run_id

    async def test_the_event_stream_follows_steps_and_ends_when_stopped(
        self, service: pb2_grpc.PhysicsServiceStub, paused: PhysicsRuntime
    ) -> None:
        stream = service.StreamSystemState(pb2.StreamRequest(interval_s=0.0))
        first = await stream.read()
        await paused.step(1)
        second = await stream.read()
        assert second.simulation.step_count == first.simulation.step_count + 1
        await paused.stop()
        with pytest.raises(grpc.aio.AioRpcError) as ended:
            await stream.read()
        assert ended.value.code() == grpc.StatusCode.UNAVAILABLE


async def test_serve_runs_until_cancelled_and_stops_the_runtime() -> None:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    runtime = PhysicsRuntime(PhysicsRuntimeConfig(start_paused=True))
    task = asyncio.create_task(serve(runtime, port=port))
    try:
        async with asyncio.timeout(10.0):
            while True:
                try:
                    async with grpc.aio.insecure_channel(
                        f"127.0.0.1:{port}"
                    ) as channel:
                        health = await pb2_grpc.PhysicsServiceStub(channel).Health(
                            pb2.Empty(), timeout=1.0
                        )
                    break
                except grpc.aio.AioRpcError:
                    await asyncio.sleep(0.05)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert health.status == "running"
    assert runtime.status == "stopped"
