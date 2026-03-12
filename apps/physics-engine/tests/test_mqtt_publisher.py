"""
Tests for MQTTPublisher.

Strategy: mock the asyncio_mqtt.Client so no real broker is needed.
We verify:
  - correct topic names
  - correct payload structure (valid JSON with required keys)
  - correct field values
  - error counting on publish failure
  - heartbeat format
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from physics_engine.models import BoilerParameters, BoilerState
from physics_engine.mqtt_publisher import (
    BOILER_TOPICS,
    HEARTBEAT_TOPIC,
    TOPIC_PREFIX,
    TURBINE_TOPICS,
    MQTTConfig,
    MQTTPublisher,
    build_payload,
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
    """
    Async mock of asyncio_mqtt.Client.
    .publish() is an AsyncMock so we can await it and inspect calls.
    """
    client = MagicMock()
    client.publish = AsyncMock()
    return client


# ─── build_payload tests ──────────────────────────────────────────────────────


class TestBuildPayload:
    def test_returns_bytes(self) -> None:
        result = build_payload(42.0, "Pa")
        assert isinstance(result, bytes)

    def test_valid_json(self) -> None:
        result = build_payload(100.0, "K")
        doc = json.loads(result)
        assert isinstance(doc, dict)

    def test_required_keys_present(self) -> None:
        doc = json.loads(build_payload(1.0, "m"))
        assert "value" in doc
        assert "unit" in doc
        assert "quality" in doc
        assert "ts" in doc

    def test_value_correct(self) -> None:
        doc = json.loads(build_payload(14_000_000.0, "Pa"))
        assert doc["value"] == pytest.approx(14_000_000.0)

    def test_unit_correct(self) -> None:
        doc = json.loads(build_payload(1.0, "kg/s"))
        assert doc["unit"] == "kg/s"

    def test_default_quality_is_good(self) -> None:
        doc = json.loads(build_payload(1.0, "K"))
        assert doc["quality"] == "good"

    def test_custom_quality(self) -> None:
        doc = json.loads(build_payload(1.0, "K", quality="uncertain"))
        assert doc["quality"] == "uncertain"

    def test_timestamp_is_positive_int(self) -> None:
        doc = json.loads(build_payload(1.0, "Pa"))
        assert isinstance(doc["ts"], int)
        assert doc["ts"] > 0


# ─── MQTTPublisher unit tests ─────────────────────────────────────────────────


class TestMQTTPublisher:
    @pytest.mark.asyncio
    async def test_publish_boiler_calls_correct_number_of_publishes(
        self,
        publisher: MQTTPublisher,
        mock_client: MagicMock,
        boiler_state: BoilerState,
    ) -> None:
        """One publish per boiler field."""
        await publisher.publish_boiler(mock_client, boiler_state)
        assert mock_client.publish.call_count == len(BOILER_TOPICS)

    @pytest.mark.asyncio
    async def test_publish_turbine_calls_correct_number_of_publishes(
        self,
        publisher: MQTTPublisher,
        mock_client: MagicMock,
        turbine_state: TurbineState,
    ) -> None:
        """One publish per turbine field."""
        await publisher.publish_turbine(mock_client, turbine_state)
        assert mock_client.publish.call_count == len(TURBINE_TOPICS)

    @pytest.mark.asyncio
    async def test_publish_boiler_topics_start_with_prefix(
        self,
        publisher: MQTTPublisher,
        mock_client: MagicMock,
        boiler_state: BoilerState,
    ) -> None:
        await publisher.publish_boiler(mock_client, boiler_state)
        for call in mock_client.publish.call_args_list:
            topic = call.args[0]
            assert topic.startswith(TOPIC_PREFIX)

    @pytest.mark.asyncio
    async def test_publish_boiler_pressure_topic_correct(
        self,
        publisher: MQTTPublisher,
        mock_client: MagicMock,
        boiler_state: BoilerState,
    ) -> None:
        await publisher.publish_boiler(mock_client, boiler_state)
        topics = [call.args[0] for call in mock_client.publish.call_args_list]
        assert f"{TOPIC_PREFIX}/boiler/pressure_pa" in topics

    @pytest.mark.asyncio
    async def test_publish_boiler_payload_is_valid_json(
        self,
        publisher: MQTTPublisher,
        mock_client: MagicMock,
        boiler_state: BoilerState,
    ) -> None:
        await publisher.publish_boiler(mock_client, boiler_state)
        for call in mock_client.publish.call_args_list:
            payload_bytes = call.args[1]
            doc = json.loads(payload_bytes)
            assert "value" in doc

    @pytest.mark.asyncio
    async def test_publish_boiler_pressure_value_matches_state(
        self,
        publisher: MQTTPublisher,
        mock_client: MagicMock,
        boiler_state: BoilerState,
    ) -> None:
        await publisher.publish_boiler(mock_client, boiler_state)
        pressure_topic = f"{TOPIC_PREFIX}/boiler/pressure_pa"
        for call in mock_client.publish.call_args_list:
            if call.args[0] == pressure_topic:
                doc = json.loads(call.args[1])
                assert doc["value"] == pytest.approx(boiler_state.pressure)
                return
        pytest.fail("Pressure topic not published")

    @pytest.mark.asyncio
    async def test_publish_turbine_electrical_power_topic_correct(
        self,
        publisher: MQTTPublisher,
        mock_client: MagicMock,
        turbine_state: TurbineState,
    ) -> None:
        await publisher.publish_turbine(mock_client, turbine_state)
        topics = [call.args[0] for call in mock_client.publish.call_args_list]
        assert f"{TOPIC_PREFIX}/turbine/electrical_power_w" in topics

    @pytest.mark.asyncio
    async def test_publish_turbine_payload_has_unit(
        self,
        publisher: MQTTPublisher,
        mock_client: MagicMock,
        turbine_state: TurbineState,
    ) -> None:
        await publisher.publish_turbine(mock_client, turbine_state)
        for call in mock_client.publish.call_args_list:
            doc = json.loads(call.args[1])
            assert "unit" in doc
            assert doc["unit"] != ""

    @pytest.mark.asyncio
    async def test_publish_heartbeat_uses_correct_topic(
        self,
        publisher: MQTTPublisher,
        mock_client: MagicMock,
    ) -> None:
        await publisher.publish_heartbeat(mock_client)
        topic = mock_client.publish.call_args.args[0]
        assert topic == HEARTBEAT_TOPIC

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
        assert publisher.published == len(BOILER_TOPICS)

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
        assert publisher.errors == len(BOILER_TOPICS)

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

        await publisher.publish_boiler(error_client, boiler_state)

    @pytest.mark.asyncio
    async def test_all_boiler_fields_published(
        self,
        publisher: MQTTPublisher,
        mock_client: MagicMock,
        boiler_state: BoilerState,
    ) -> None:
        """Every entry in BOILER_TOPICS must appear in publish calls."""
        await publisher.publish_boiler(mock_client, boiler_state)
        published_topics = {call.args[0] for call in mock_client.publish.call_args_list}
        for _, (sub_topic, _) in BOILER_TOPICS.items():
            assert f"{TOPIC_PREFIX}/{sub_topic}" in published_topics

    @pytest.mark.asyncio
    async def test_all_turbine_fields_published(
        self,
        publisher: MQTTPublisher,
        mock_client: MagicMock,
        turbine_state: TurbineState,
    ) -> None:
        """Every entry in TURBINE_TOPICS must appear in publish calls."""
        await publisher.publish_turbine(mock_client, turbine_state)
        published_topics = {call.args[0] for call in mock_client.publish.call_args_list}
        for _, (sub_topic, _) in TURBINE_TOPICS.items():
            assert f"{TOPIC_PREFIX}/{sub_topic}" in published_topics

    def test_config_defaults(self) -> None:
        cfg = MQTTConfig()
        assert cfg.host == "localhost"
        assert cfg.port == 1883
        assert cfg.interval_s == pytest.approx(0.1)

    def test_initial_stats_zero(self, publisher: MQTTPublisher) -> None:
        assert publisher.published == 0
        assert publisher.errors == 0
