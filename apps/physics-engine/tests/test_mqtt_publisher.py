"""
Tests for MQTTPublisher (protobuf edition).

Strategy: mock aiomqtt.Client — no real broker needed.
We verify:
  - correct topic names and their order (sensors/plant first)
  - payload deserializes to correct protobuf message type
  - field values match the physics state
  - error counting on publish failure
  - heartbeat format
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import cogniboiler_pb2 as pb
import pytest
from physics_engine.constants import (
    MAX_STEAM_FLOW,
    PRESSURE_NOMINAL,
    TEMP_STEAM_NOMINAL,
)
from physics_engine.models import BoilerParameters, BoilerState
from physics_engine.mqtt_publisher import (
    TOPIC_BOILER,
    TOPIC_HEARTBEAT,
    TOPIC_PLANT,
    TOPIC_TURBINE,
    MQTTConfig,
    MQTTPublisher,
)
from physics_engine.plant import PlantSimulator, PlantSnapshot
from physics_engine.proto_mapping import boiler_state_to_proto, turbine_state_to_proto
from physics_engine.runtime import RunState, SimulationStatus
from physics_engine.turbine import TurbineModel, TurbineState

# ─── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def boiler_state() -> BoilerState:
    """Nominal boiler state from BoilerParameters."""
    return BoilerParameters().nominal_initial_state()


@pytest.fixture
def turbine_state() -> TurbineState:
    """Turbine state at the nominal inlet: 552.5 °C, 140 bar, 277.8 kg/s."""
    return TurbineModel().calculate(
        steam_temp_in=TEMP_STEAM_NOMINAL,
        steam_pressure_in=PRESSURE_NOMINAL,
        steam_flow=MAX_STEAM_FLOW,
    )


@pytest.fixture(scope="module")
def snapshot() -> PlantSnapshot:
    """The nominal plant after one step."""
    return PlantSimulator().step(1)


@pytest.fixture
def status(snapshot: PlantSnapshot) -> SimulationStatus:
    return SimulationStatus(
        run_state=RunState.RUNNING,
        speed_factor=1.0,
        simulation_time_s=snapshot.simulation_time_s,
        step_count=snapshot.step_count,
        scenario=snapshot.scenario,
        run_id=snapshot.run_id,
        step_s=snapshot.step_s,
    )


@pytest.fixture
def publisher() -> MQTTPublisher:
    """MQTTPublisher with default config (no real broker)."""
    return MQTTPublisher(MQTTConfig())


@pytest.fixture
def mock_client() -> MagicMock:
    """Async mock of aiomqtt.Client."""
    client = MagicMock()
    client.publish = AsyncMock()
    return client


# ─── boiler_state_to_proto tests ──────────────────────────────────────────────


class TestBoilerStateToProto:
    def test_returns_boiler_state_msg(self, boiler_state: BoilerState) -> None:
        msg = boiler_state_to_proto(boiler_state)
        assert isinstance(msg, pb.BoilerStateMsg)

    def test_pressure_field_matches(self, boiler_state: BoilerState) -> None:
        msg = boiler_state_to_proto(boiler_state)
        assert msg.pressure_pa == pytest.approx(boiler_state.pressure)

    def test_water_level_field_matches(self, boiler_state: BoilerState) -> None:
        msg = boiler_state_to_proto(boiler_state)
        assert msg.water_level_m == pytest.approx(boiler_state.water_level)

    def test_water_temp_field_matches(self, boiler_state: BoilerState) -> None:
        msg = boiler_state_to_proto(boiler_state)
        assert msg.water_temp_k == pytest.approx(boiler_state.water_temp)

    def test_quality_is_good(self, boiler_state: BoilerState) -> None:
        msg = boiler_state_to_proto(boiler_state)
        assert msg.quality == pb.SensorQuality.GOOD

    def test_timestamp_is_positive(self, boiler_state: BoilerState) -> None:
        msg = boiler_state_to_proto(boiler_state)
        assert msg.timestamp_ms > 0

    def test_serializes_and_roundtrips(self, boiler_state: BoilerState) -> None:
        """Serialize -> bytes -> deserialize -> same pressure."""
        msg = boiler_state_to_proto(boiler_state)
        raw = msg.SerializeToString()
        restored = pb.BoilerStateMsg()
        restored.ParseFromString(raw)
        assert restored.pressure_pa == pytest.approx(boiler_state.pressure)


# ─── turbine_state_to_proto tests ─────────────────────────────────────────────


class TestTurbineStateToProto:
    def test_returns_turbine_state_msg(self, turbine_state: TurbineState) -> None:
        msg = turbine_state_to_proto(turbine_state)
        assert isinstance(msg, pb.TurbineStateMsg)

    def test_electrical_power_field_matches(self, turbine_state: TurbineState) -> None:
        msg = turbine_state_to_proto(turbine_state)
        assert msg.electrical_power_w == pytest.approx(turbine_state.electrical_power)

    def test_shaft_power_field_matches(self, turbine_state: TurbineState) -> None:
        msg = turbine_state_to_proto(turbine_state)
        assert msg.shaft_power_w == pytest.approx(turbine_state.shaft_power)

    def test_steam_flow_field_matches(self, turbine_state: TurbineState) -> None:
        msg = turbine_state_to_proto(turbine_state)
        assert msg.steam_flow_kg_s == pytest.approx(turbine_state.steam_flow)

    def test_exhaust_pressure_field_matches(self, turbine_state: TurbineState) -> None:
        msg = turbine_state_to_proto(turbine_state)
        assert msg.exhaust_pressure_pa == pytest.approx(turbine_state.exhaust_pressure)

    def test_timestamp_is_positive(self, turbine_state: TurbineState) -> None:
        msg = turbine_state_to_proto(turbine_state)
        assert msg.timestamp_ms > 0

    def test_serializes_and_roundtrips(self, turbine_state: TurbineState) -> None:
        msg = turbine_state_to_proto(turbine_state)
        raw = msg.SerializeToString()
        restored = pb.TurbineStateMsg()
        restored.ParseFromString(raw)
        assert restored.electrical_power_w == pytest.approx(
            turbine_state.electrical_power
        )


# ─── MQTTPublisher tests ──────────────────────────────────────────────────────


class TestMQTTPublisher:
    async def test_a_snapshot_goes_out_plant_first_as_four_messages(
        self,
        publisher: MQTTPublisher,
        mock_client: MagicMock,
        snapshot: PlantSnapshot,
        status: SimulationStatus,
    ) -> None:
        await publisher.publish_snapshot(mock_client, snapshot, status)
        topics = [call.args[0] for call in mock_client.publish.call_args_list]
        assert topics == [TOPIC_PLANT, TOPIC_BOILER, TOPIC_TURBINE, TOPIC_HEARTBEAT]
        assert publisher.published == 4

    async def test_the_boiler_payload_carries_the_measured_pressure(
        self,
        publisher: MQTTPublisher,
        mock_client: MagicMock,
        snapshot: PlantSnapshot,
        status: SimulationStatus,
    ) -> None:
        await publisher.publish_snapshot(mock_client, snapshot, status)
        raw = mock_client.publish.call_args_list[1].args[1]
        msg = pb.BoilerStateMsg.FromString(raw)
        assert msg.pressure_pa == pytest.approx(snapshot.boiler.pressure)

    async def test_the_turbine_payload_carries_the_power(
        self,
        publisher: MQTTPublisher,
        mock_client: MagicMock,
        snapshot: PlantSnapshot,
        status: SimulationStatus,
    ) -> None:
        await publisher.publish_snapshot(mock_client, snapshot, status)
        raw = mock_client.publish.call_args_list[2].args[1]
        msg = pb.TurbineStateMsg.FromString(raw)
        assert msg.electrical_power_w == pytest.approx(
            snapshot.turbine.electrical_power
        )

    async def test_publish_heartbeat_payload_is_numeric_string(
        self,
        publisher: MQTTPublisher,
        mock_client: MagicMock,
    ) -> None:
        await publisher.publish_heartbeat(mock_client)
        topic, payload = mock_client.publish.call_args.args[:2]
        assert topic == TOPIC_HEARTBEAT
        assert int(payload.decode()) > 0

    async def test_a_failed_publish_is_counted_and_raised_so_the_session_reconnects(
        self,
        publisher: MQTTPublisher,
        snapshot: PlantSnapshot,
        status: SimulationStatus,
    ) -> None:
        from aiomqtt import MqttError

        error_client = MagicMock()
        error_client.publish = AsyncMock(
            side_effect=MqttError("The client is not currently connected.")
        )
        with pytest.raises(MqttError, match="not currently connected"):
            await publisher.publish_snapshot(error_client, snapshot, status)
        assert publisher.errors == 1
        assert publisher.published == 0

    def test_config_defaults(self) -> None:
        cfg = MQTTConfig()
        assert cfg.host == "localhost"
        assert cfg.port == 1883

    def test_initial_stats_zero(self, publisher: MQTTPublisher) -> None:
        assert publisher.published == 0
        assert publisher.errors == 0
