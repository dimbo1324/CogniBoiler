"""The plant simulator's scenarios and faults, its metrics, the MQTT mirror, the entry point."""

from __future__ import annotations

import argparse
import asyncio
import logging
from typing import Any

import cogniboiler_pb2 as pb
import pytest
from aiomqtt import MqttError
from physics_engine import __main__ as entry
from physics_engine import mqtt_publisher
from physics_engine.faults import FaultKind, FaultSpec
from physics_engine.metrics import observe_runtime
from physics_engine.mqtt_publisher import (
    TOPIC_AVAILABILITY,
    TOPIC_BOILER,
    TOPIC_HEARTBEAT,
    TOPIC_PLANT,
    TOPIC_TURBINE,
    MQTTConfig,
    MQTTPublisher,
)
from physics_engine.plant import PlantConfig, PlantSimulator
from physics_engine.runtime import PhysicsRuntime, PhysicsRuntimeConfig
from physics_engine.scenarios import ScenarioName
from physics_engine.sensors import SensorId
from prometheus_client import REGISTRY

MW = 1.0e6
BAR = 1.0e5


class TestScenarios:
    @pytest.mark.parametrize(
        ("scenario", "power_mw"),
        [
            (ScenarioName.STEADY_STATE, 250.0),
            (ScenarioName.PART_LOAD, 180.0),
            (ScenarioName.FULL_LOAD, 300.0),
        ],
    )
    def test_operating_points_start_at_their_load(
        self, scenario: ScenarioName, power_mw: float
    ) -> None:
        snapshot = PlantSimulator(scenario=scenario).snapshot
        assert snapshot.turbine.electrical_power / MW == pytest.approx(
            power_mw, rel=0.03
        )
        assert snapshot.scenario is scenario
        assert snapshot.step_count == 0

    def test_a_hot_start_has_a_hot_drum_and_no_output(self) -> None:
        snapshot = PlantSimulator(scenario=ScenarioName.HOT_START).snapshot
        assert snapshot.boiler.pressure / BAR == pytest.approx(100.0, rel=0.05)
        assert snapshot.turbine.electrical_power < 1.0 * MW

    def test_a_cold_start_is_at_atmospheric_pressure(self) -> None:
        snapshot = PlantSimulator(scenario=ScenarioName.COLD_START).snapshot
        assert snapshot.boiler.pressure / BAR == pytest.approx(1.0, abs=0.1)
        assert snapshot.turbine.electrical_power < 1.0 * MW

    def test_the_drill_trips_the_feedwater_pump_after_two_minutes(self) -> None:
        plant = PlantSimulator(scenario=ScenarioName.FEEDWATER_PUMP_DRILL)
        # Due faults are applied at the start of the step that begins at 120 s.
        assert plant.step(120).faults == ()
        (fault,) = plant.step(1).faults
        assert fault.label == "feedwater_pump_failure"
        assert fault.started_at_s == pytest.approx(120.0)

    def test_a_scheduled_fault_that_is_already_active_is_skipped(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        plant = PlantSimulator(scenario=ScenarioName.FEEDWATER_PUMP_DRILL)
        plant.inject_fault(FaultSpec(FaultKind.FEEDWATER_PUMP_FAILURE, severity=0.5))
        with caplog.at_level(logging.WARNING, logger="physics_engine.plant"):
            snapshot = plant.step(121)
        assert len(snapshot.faults) == 1
        assert snapshot.faults[0].spec.severity == 0.5
        assert "Scheduled fault skipped" in caplog.text

    def test_a_new_scenario_starts_a_new_run_without_faults(self) -> None:
        plant = PlantSimulator()
        plant.inject_fault(FaultSpec(FaultKind.STEAM_LEAK, severity=0.1))
        plant.step(5)
        first_run = plant.snapshot.run_id
        snapshot = plant.load_scenario(ScenarioName.PART_LOAD)
        assert snapshot.run_id == first_run + 1
        assert (snapshot.faults, snapshot.step_count) == ((), 0)

    def test_the_step_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="step_s"):
            PlantSimulator(PlantConfig(step_s=0.0))
        with pytest.raises(ValueError, match="steps"):
            PlantSimulator().step(0)


class TestValvesAndInstruments:
    def test_a_stuck_valve_ignores_its_command_until_cleared(self) -> None:
        plant = PlantSimulator()
        held = plant.snapshot.controls.fuel_valve.position
        fault = plant.inject_fault(FaultSpec(FaultKind.VALVE_STUCK, "fuel"))
        plant.apply_command(fuel_valve=0.2, feedwater_valve=0.5, steam_valve=0.5)
        stuck = plant.step(30).controls
        assert stuck.fuel_valve.command == 0.2
        assert stuck.fuel_valve.position == pytest.approx(held)
        plant.clear_fault(fault.fault_id)
        moved = plant.step(30).controls.fuel_valve.position
        assert moved < held

    def test_every_instrument_can_be_read(self) -> None:
        snapshot = PlantSimulator().snapshot
        for sensor in SensorId:
            assert snapshot.measured(sensor) == snapshot.reading(sensor).measured_value

    def test_a_steam_leak_costs_output(self) -> None:
        healthy = PlantSimulator()
        leaking = PlantSimulator()
        leaking.inject_fault(FaultSpec(FaultKind.STEAM_LEAK, severity=0.2))
        assert (
            leaking.step(120).turbine.electrical_power
            < healthy.step(120).turbine.electrical_power
        )


class TestMetrics:
    async def test_the_gauges_read_the_runtime_when_scraped(self) -> None:
        runtime = PhysicsRuntime(
            PhysicsRuntimeConfig(start_paused=True, speed_factor=5)
        )
        observe_runtime(runtime)
        await runtime.start()
        try:
            await runtime.step(3)
            await runtime.inject_fault(FaultSpec(FaultKind.STEAM_LEAK, severity=0.1))
            assert REGISTRY.get_sample_value("physics_simulation_time_seconds") == 3.0
            assert REGISTRY.get_sample_value("physics_simulation_speed_factor") == 5.0
            assert REGISTRY.get_sample_value("physics_simulation_running") == 0.0
            assert REGISTRY.get_sample_value("physics_active_faults") == 1.0
            await runtime.resume()
            assert REGISTRY.get_sample_value("physics_simulation_running") == 1.0
        finally:
            await runtime.stop()


class FakeBroker:
    published: list[tuple[str, Any, int, bool]] = []
    connections: list[dict[str, Any]] = []
    fail_connections = 0
    drop_after_publishes: int | None = None

    def __init__(self, **options: Any) -> None:
        FakeBroker.connections.append(options)

    async def __aenter__(self) -> FakeBroker:
        if FakeBroker.fail_connections:
            FakeBroker.fail_connections -= 1
            raise MqttError("broker unreachable")
        return self

    async def __aexit__(self, *_: object) -> None:
        return None

    async def publish(
        self, topic: str, payload: Any = None, qos: int = 0, retain: bool = False
    ) -> None:
        if FakeBroker.drop_after_publishes is not None:
            if FakeBroker.drop_after_publishes == 0:
                FakeBroker.drop_after_publishes = None
                raise MqttError("The client is not currently connected.")
            FakeBroker.drop_after_publishes -= 1
        FakeBroker.published.append((topic, payload, qos, retain))


async def until(predicate: Any) -> None:
    async with asyncio.timeout(10.0):
        while not predicate():
            await asyncio.sleep(0.005)


class TestMirror:
    @pytest.fixture(autouse=True)
    def broker(self, monkeypatch: pytest.MonkeyPatch) -> None:
        FakeBroker.published = []
        FakeBroker.connections = []
        FakeBroker.fail_connections = 0
        FakeBroker.drop_after_publishes = None
        monkeypatch.setattr(mqtt_publisher, "Client", FakeBroker)
        monkeypatch.setattr(mqtt_publisher, "RECONNECT_DELAY_S", 0.001)

    async def test_every_snapshot_is_published_plant_first(self) -> None:
        runtime = PhysicsRuntime(PhysicsRuntimeConfig(start_paused=True))
        await runtime.start()
        publisher = MQTTPublisher(
            MQTTConfig(
                host="broker", client_id="physics-engine", username="u", password="p"
            )
        )
        mirror = asyncio.create_task(publisher.mirror_runtime(runtime))
        try:
            await until(lambda: len(FakeBroker.published) >= 5)
            await runtime.step(1)
            await until(lambda: len(FakeBroker.published) >= 9)
        finally:
            mirror.cancel()
            await runtime.stop()
        topics = [topic for topic, *_ in FakeBroker.published]
        assert topics[:9] == [
            TOPIC_AVAILABILITY,
            TOPIC_PLANT,
            TOPIC_BOILER,
            TOPIC_TURBINE,
            TOPIC_HEARTBEAT,
            TOPIC_PLANT,
            TOPIC_BOILER,
            TOPIC_TURBINE,
            TOPIC_HEARTBEAT,
        ]
        assert FakeBroker.published[0][1:] == ("online", 1, True)
        options = FakeBroker.connections[0]
        assert (options["username"], options["identifier"]) == ("u", "physics-engine")
        assert options["will"].topic == TOPIC_AVAILABILITY
        assert options["will"].retain is True
        status = pb.PlantStatusMsg.FromString(FakeBroker.published[5][1])
        assert status.simulation.step_count == 1

    async def test_the_mirror_reconnects_after_the_broker_fails(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        FakeBroker.fail_connections = 2
        runtime = PhysicsRuntime(PhysicsRuntimeConfig(start_paused=True))
        await runtime.start()
        publisher = MQTTPublisher(MQTTConfig())
        with caplog.at_level(logging.WARNING, logger="physics_engine.mqtt_publisher"):
            mirror = asyncio.create_task(publisher.mirror_runtime(runtime))
            try:
                await until(lambda: FakeBroker.published)
            finally:
                mirror.cancel()
                await runtime.stop()
        assert len(FakeBroker.connections) == 3
        # Two refused connections in one outage: warned about once, not twice.
        assert caplog.text.count("MQTT error") == 1

    async def test_a_connection_lost_while_publishing_is_reconnected_not_spun_on(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        FakeBroker.drop_after_publishes = 3
        runtime = PhysicsRuntime(PhysicsRuntimeConfig(start_paused=True))
        await runtime.start()
        publisher = MQTTPublisher(MQTTConfig())
        with caplog.at_level(logging.WARNING, logger="physics_engine.mqtt_publisher"):
            mirror = asyncio.create_task(publisher.mirror_runtime(runtime))
            try:
                await until(lambda: len(FakeBroker.connections) == 2)
                await until(lambda: len(FakeBroker.published) >= 4)
                await runtime.step(1)
                await until(lambda: len(FakeBroker.published) >= 8)
            finally:
                mirror.cancel()
                await runtime.stop()
        assert len(FakeBroker.connections) == 2
        assert caplog.text.count("MQTT error") == 1
        assert "not currently connected" in caplog.text
        topics = [topic for topic, *_ in FakeBroker.published]
        assert topics[3] == TOPIC_AVAILABILITY
        assert topics[4:8] == [
            TOPIC_PLANT,
            TOPIC_BOILER,
            TOPIC_TURBINE,
            TOPIC_HEARTBEAT,
        ]
        assert publisher.errors == 1

    async def test_the_mirror_ends_when_the_runtime_stops(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        runtime = PhysicsRuntime(PhysicsRuntimeConfig(start_paused=True))
        await runtime.start()
        publisher = MQTTPublisher(MQTTConfig())
        mirror = asyncio.create_task(publisher.mirror_runtime(runtime))
        await until(lambda: len(FakeBroker.published) >= 5)
        with caplog.at_level(logging.ERROR, logger="physics_engine.mqtt_publisher"):
            await runtime.stop()
            await asyncio.wait_for(mirror, timeout=5.0)
        assert "MQTT session stopped: physics runtime stopped" in caplog.text


class TestEntryPoint:
    def test_the_command_line_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("sys.argv", ["physics_engine"])
        args = entry.parse_args()
        assert (args.scenario, args.speed, args.step_s) == ("steady_state", 1.0, 1.0)
        assert (args.grpc_port, args.metrics_port, args.paused) == (50052, 9101, False)

    @pytest.mark.parametrize("disable_mqtt", [True, False])
    async def test_it_serves_the_runtime_and_mirrors_it(
        self, monkeypatch: pytest.MonkeyPatch, disable_mqtt: bool
    ) -> None:
        served: list[PhysicsRuntime] = []
        mirrored: list[bool] = []

        async def serve(runtime: PhysicsRuntime, *, port: int) -> None:
            served.append(runtime)
            await asyncio.sleep(0.01)

        async def mirror(self: MQTTPublisher, runtime: PhysicsRuntime) -> None:
            mirrored.append(True)
            await asyncio.Event().wait()

        monkeypatch.setattr(entry, "serve", serve)
        monkeypatch.setattr(MQTTPublisher, "mirror_runtime", mirror)
        monkeypatch.setattr(entry, "start_metrics_server", lambda port, host: None)
        args = argparse.Namespace(
            scenario="part_load",
            speed=2.0,
            step_s=1.0,
            paused=True,
            mqtt_host="broker",
            mqtt_port=1883,
            grpc_port=0,
            disable_mqtt=disable_mqtt,
            metrics_port=0,
            metrics_host="127.0.0.1",
        )
        await entry.main(args)
        (runtime,) = served
        status = runtime.simulation_status()
        assert (status.scenario, status.speed_factor) == (ScenarioName.PART_LOAD, 2.0)
        assert status.run_state.value == "paused"
        assert mirrored == ([] if disable_mqtt else [True])
