"""
Tests for MQTTPublisher (protobuf edition).

Strategy: mock asyncio_mqtt.Client — no real broker needed.
We verify:
  - correct topic names (sensors/boiler, sensors/turbine)
  - payload deserializes to correct protobuf message type
  - field values match the physics state
  - error counting on publish failure
  - heartbeat format
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import cogniboiler_pb2 as pb
import pytest
from physics_engine.models import BoilerParameters, BoilerState
from physics_engine.mqtt_publisher import (
    TOPIC_BOILER,
    TOPIC_HEARTBEAT,
    TOPIC_TURBINE,
    MQTTConfig,
    MQTTPublisher,
    boiler_state_to_proto,
    turbine_state_to_proto,
)
from physics_engine.turbine import TurbineModel, TurbineState

# ─── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def boiler_state() -> BoilerState:
    """Nominal boiler state from BoilerParameters."""
    return BoilerParameters().nominal_initial_state()


@pytest.fixture
def turbine_state() -> TurbineState:
    """Nominal turbine state from TurbineModel."""
    return TurbineModel().nominal_state()


@pytest.fixture
def publisher() -> MQTTPublisher:
    """MQTTPublisher with default config (no real broker)."""
    return MQTTPublisher(MQTTConfig())


@pytest.fixture
def mock_client() -> MagicMock:
    """Async mock of asyncio_mqtt.Client."""
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
        """Serialize → bytes → deserialize → same pressure."""
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
    @pytest.mark.asyncio
    async def test_publish_boiler_calls_publish_once(
        self,
        publisher: MQTTPublisher,
        mock_client: MagicMock,
        boiler_state: BoilerState,
    ) -> None:
        await publisher.publish_boiler(mock_client, boiler_state)
        assert mock_client.publish.call_count == 1

    @pytest.mark.asyncio
    async def test_publish_boiler_uses_correct_topic(
        self,
        publisher: MQTTPublisher,
        mock_client: MagicMock,
        boiler_state: BoilerState,
    ) -> None:
        await publisher.publish_boiler(mock_client, boiler_state)
        topic = mock_client.publish.call_args.args[0]
        assert topic == TOPIC_BOILER

    @pytest.mark.asyncio
    async def test_publish_boiler_payload_is_valid_protobuf(
        self,
        publisher: MQTTPublisher,
        mock_client: MagicMock,
        boiler_state: BoilerState,
    ) -> None:
        await publisher.publish_boiler(mock_client, boiler_state)
        raw = mock_client.publish.call_args.args[1]
        msg = pb.BoilerStateMsg()
        msg.ParseFromString(raw)
        assert msg.pressure_pa == pytest.approx(boiler_state.pressure)

    @pytest.mark.asyncio
    async def test_publish_turbine_calls_publish_once(
        self,
        publisher: MQTTPublisher,
        mock_client: MagicMock,
        turbine_state: TurbineState,
    ) -> None:
        await publisher.publish_turbine(mock_client, turbine_state)
        assert mock_client.publish.call_count == 1

    @pytest.mark.asyncio
    async def test_publish_turbine_uses_correct_topic(
        self,
        publisher: MQTTPublisher,
        mock_client: MagicMock,
        turbine_state: TurbineState,
    ) -> None:
        await publisher.publish_turbine(mock_client, turbine_state)
        topic = mock_client.publish.call_args.args[0]
        assert topic == TOPIC_TURBINE

    @pytest.mark.asyncio
    async def test_publish_turbine_payload_is_valid_protobuf(
        self,
        publisher: MQTTPublisher,
        mock_client: MagicMock,
        turbine_state: TurbineState,
    ) -> None:
        await publisher.publish_turbine(mock_client, turbine_state)
        raw = mock_client.publish.call_args.args[1]
        msg = pb.TurbineStateMsg()
        msg.ParseFromString(raw)
        assert msg.electrical_power_w == pytest.approx(turbine_state.electrical_power)

    @pytest.mark.asyncio
    async def test_publish_heartbeat_uses_correct_topic(
        self,
        publisher: MQTTPublisher,
        mock_client: MagicMock,
    ) -> None:
        await publisher.publish_heartbeat(mock_client)
        topic = mock_client.publish.call_args.args[0]
        assert topic == TOPIC_HEARTBEAT

    @pytest.mark.asyncio
    async def test_publish_heartbeat_payload_is_numeric_string(
        self,
        publisher: MQTTPublisher,
        mock_client: MagicMock,
    ) -> None:
        await publisher.publish_heartbeat(mock_client)
        payload = mock_client.publish.call_args.args[1]
        ts = int(payload.decode())
        assert ts > 0

    @pytest.mark.asyncio
    async def test_published_counter_increments(
        self,
        publisher: MQTTPublisher,
        mock_client: MagicMock,
        boiler_state: BoilerState,
    ) -> None:
        assert publisher.published == 0
        await publisher.publish_boiler(mock_client, boiler_state)
        assert publisher.published == 1

    @pytest.mark.asyncio
    async def test_error_counter_increments_on_mqtt_error(
        self,
        publisher: MQTTPublisher,
        boiler_state: BoilerState,
    ) -> None:
        from asyncio_mqtt import MqttError

        error_client = MagicMock()
        error_client.publish = AsyncMock(side_effect=MqttError("broker down"))
        await publisher.publish_boiler(error_client, boiler_state)
        assert publisher.errors == 1

    @pytest.mark.asyncio
    async def test_error_does_not_raise(
        self,
        publisher: MQTTPublisher,
        boiler_state: BoilerState,
    ) -> None:
        """Publish errors must be swallowed — never crash the publisher loop."""
        from asyncio_mqtt import MqttError

        error_client = MagicMock()
        error_client.publish = AsyncMock(side_effect=MqttError("timeout"))
        await publisher.publish_boiler(error_client, boiler_state)  # must not raise

    def test_config_defaults(self) -> None:
        cfg = MQTTConfig()
        assert cfg.host == "localhost"
        assert cfg.port == 1883
        assert cfg.interval_s == pytest.approx(0.1)

    def test_initial_stats_zero(self, publisher: MQTTPublisher) -> None:
        assert publisher.published == 0
        assert publisher.errors == 0
