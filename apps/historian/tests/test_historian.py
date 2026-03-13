"""
Tests for Historian writer and subscriber.

No real InfluxDB or MQTT broker needed.
We test:
  1. topic_to_measurement()  — topic routing logic
  2. build_point()           — Point construction and field/tag values
  3. HistorianSubscriber._handle_message() — full pipeline with mock writer
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest
from historian.subscriber import HistorianSubscriber
from historian.writer import (
    MEASUREMENT_BOILER,
    MEASUREMENT_TURBINE,
    TOPIC_TO_SENSOR_NAME,
    InfluxWriter,
    build_point,
    topic_to_measurement,
)
from influxdb_client import Point

# ─── Helpers ──────────────────────────────────────────────────────────────────


def make_payload(
    value: float,
    unit: str = "Pa",
    quality: str = "good",
    ts: int = 1_741_000_000_000,
) -> bytes:
    return json.dumps(
        {"value": value, "unit": unit, "quality": quality, "ts": ts}
    ).encode()


def parse_payload(raw: bytes) -> dict[str, Any]:
    return json.loads(raw)  # type: ignore[no-any-return]


def make_mock_writer() -> MagicMock:
    writer = MagicMock(spec=InfluxWriter)
    writer.write_point = MagicMock()
    return writer


# ─── topic_to_measurement tests ───────────────────────────────────────────────


class TestTopicToMeasurement:
    def test_boiler_pressure_maps_to_boiler_sensors(self) -> None:
        assert topic_to_measurement("sensors/boiler/pressure_pa") == MEASUREMENT_BOILER

    def test_boiler_water_level_maps_to_boiler_sensors(self) -> None:
        assert (
            topic_to_measurement("sensors/boiler/water_level_m") == MEASUREMENT_BOILER
        )

    def test_turbine_electrical_power_maps_to_turbine_sensors(self) -> None:
        result = topic_to_measurement("sensors/turbine/electrical_power_w")
        assert result == MEASUREMENT_TURBINE

    def test_heartbeat_returns_none(self) -> None:
        assert topic_to_measurement("sensors/system/timestamp_ms") is None

    def test_unknown_topic_returns_none(self) -> None:
        assert topic_to_measurement("sensors/unknown/xyz") is None

    def test_all_boiler_topics_map_correctly(self) -> None:
        boiler_topics = [t for t in TOPIC_TO_SENSOR_NAME if "/boiler/" in t]
        for topic in boiler_topics:
            assert topic_to_measurement(topic) == MEASUREMENT_BOILER

    def test_all_turbine_topics_map_correctly(self) -> None:
        turbine_topics = [t for t in TOPIC_TO_SENSOR_NAME if "/turbine/" in t]
        for topic in turbine_topics:
            assert topic_to_measurement(topic) == MEASUREMENT_TURBINE


# ─── build_point tests ────────────────────────────────────────────────────────


class TestBuildPoint:
    def test_returns_point_for_known_topic(self) -> None:
        payload = parse_payload(make_payload(14_000_000.0))
        result = build_point("sensors/boiler/pressure_pa", payload)
        assert isinstance(result, Point)

    def test_returns_none_for_heartbeat(self) -> None:
        payload = parse_payload(make_payload(1.0))
        result = build_point("sensors/system/timestamp_ms", payload)
        assert result is None

    def test_returns_none_for_unknown_topic(self) -> None:
        payload = parse_payload(make_payload(1.0))
        result = build_point("sensors/unknown/xyz", payload)
        assert result is None

    def test_returns_none_on_missing_value_key(self) -> None:
        payload: dict[str, Any] = {"unit": "Pa", "quality": "good", "ts": 1000}
        result = build_point("sensors/boiler/pressure_pa", payload)
        assert result is None

    def test_returns_none_on_missing_ts_key(self) -> None:
        payload: dict[str, Any] = {"value": 1.0, "unit": "Pa", "quality": "good"}
        result = build_point("sensors/boiler/pressure_pa", payload)
        assert result is None

    def test_point_measurement_is_boiler_sensors(self) -> None:
        payload = parse_payload(make_payload(14_000_000.0))
        point = build_point("sensors/boiler/pressure_pa", payload)
        assert point is not None
        line = point.to_line_protocol()
        assert line.startswith(MEASUREMENT_BOILER)

    def test_point_measurement_is_turbine_sensors(self) -> None:
        payload = parse_payload(make_payload(341_000_000.0, unit="W"))
        point = build_point("sensors/turbine/electrical_power_w", payload)
        assert point is not None
        line = point.to_line_protocol()
        assert line.startswith(MEASUREMENT_TURBINE)

    def test_point_contains_value_field(self) -> None:
        payload = parse_payload(make_payload(14_000_000.0))
        point = build_point("sensors/boiler/pressure_pa", payload)
        assert point is not None
        line = point.to_line_protocol()
        assert "value=14000000" in line

    def test_point_contains_unit_tag(self) -> None:
        payload = parse_payload(make_payload(1.0, unit="kg/s"))
        point = build_point("sensors/turbine/steam_flow_kg_s", payload)
        assert point is not None
        line = point.to_line_protocol()
        assert "unit=kg/s" in line

    def test_point_contains_quality_tag(self) -> None:
        payload = parse_payload(make_payload(1.0, quality="uncertain"))
        point = build_point("sensors/boiler/pressure_pa", payload)
        assert point is not None
        line = point.to_line_protocol()
        assert "quality=uncertain" in line

    def test_point_timestamp_converted_to_nanoseconds(self) -> None:
        ts_ms = 1_741_000_000_000
        payload = parse_payload(make_payload(1.0, ts=ts_ms))
        point = build_point("sensors/boiler/pressure_pa", payload)
        assert point is not None
        line = point.to_line_protocol()
        ts_ns = int(line.split()[-1])
        assert ts_ns == ts_ms * 1_000_000

    def test_point_sensor_tag_is_pressure(self) -> None:
        payload = parse_payload(make_payload(1.0))
        point = build_point("sensors/boiler/pressure_pa", payload)
        assert point is not None
        line = point.to_line_protocol()
        assert "sensor=Pressure" in line


# ─── HistorianSubscriber fixtures ────────────────────────────────────────────


@pytest.fixture  # type: ignore[misc]
def writer() -> MagicMock:
    return make_mock_writer()


@pytest.fixture  # type: ignore[misc]
def subscriber(writer: MagicMock) -> HistorianSubscriber:
    return HistorianSubscriber(writer=writer)


# ─── HistorianSubscriber tests ────────────────────────────────────────────────


class TestHistorianSubscriber:
    @pytest.mark.asyncio  # type: ignore[misc]
    async def test_known_topic_calls_write_point(
        self, subscriber: HistorianSubscriber, writer: MagicMock
    ) -> None:
        await subscriber._handle_message(
            "sensors/boiler/pressure_pa", make_payload(14_000_000.0)
        )
        writer.write_point.assert_called_once()

    @pytest.mark.asyncio  # type: ignore[misc]
    async def test_write_point_receives_point_instance(
        self, subscriber: HistorianSubscriber, writer: MagicMock
    ) -> None:
        await subscriber._handle_message(
            "sensors/boiler/pressure_pa", make_payload(14_000_000.0)
        )
        call_arg = writer.write_point.call_args.args[0]
        assert isinstance(call_arg, Point)

    @pytest.mark.asyncio  # type: ignore[misc]
    async def test_heartbeat_skipped(
        self, subscriber: HistorianSubscriber, writer: MagicMock
    ) -> None:
        await subscriber._handle_message(
            "sensors/system/timestamp_ms", b"1741000000000"
        )
        writer.write_point.assert_not_called()

    @pytest.mark.asyncio  # type: ignore[misc]
    async def test_malformed_json_skipped(
        self, subscriber: HistorianSubscriber, writer: MagicMock
    ) -> None:
        await subscriber._handle_message("sensors/boiler/pressure_pa", b"not-json")
        writer.write_point.assert_not_called()

    @pytest.mark.asyncio  # type: ignore[misc]
    async def test_unknown_topic_skipped(
        self, subscriber: HistorianSubscriber, writer: MagicMock
    ) -> None:
        await subscriber._handle_message("sensors/unknown/xyz", make_payload(1.0))
        writer.write_point.assert_not_called()

    @pytest.mark.asyncio  # type: ignore[misc]
    async def test_received_counter_increments(
        self, subscriber: HistorianSubscriber, writer: MagicMock
    ) -> None:
        await subscriber._handle_message(
            "sensors/boiler/pressure_pa", make_payload(1.0)
        )
        assert subscriber.stats["received"] == 1

    @pytest.mark.asyncio  # type: ignore[misc]
    async def test_stored_counter_increments_on_success(
        self, subscriber: HistorianSubscriber, writer: MagicMock
    ) -> None:
        await subscriber._handle_message(
            "sensors/boiler/pressure_pa", make_payload(1.0)
        )
        assert subscriber.stats["stored"] == 1

    @pytest.mark.asyncio  # type: ignore[misc]
    async def test_skipped_counter_increments_on_heartbeat(
        self, subscriber: HistorianSubscriber, writer: MagicMock
    ) -> None:
        await subscriber._handle_message("sensors/system/timestamp_ms", b"12345")
        assert subscriber.stats["skipped"] == 1

    @pytest.mark.asyncio  # type: ignore[misc]
    async def test_skipped_counter_increments_on_bad_json(
        self, subscriber: HistorianSubscriber, writer: MagicMock
    ) -> None:
        await subscriber._handle_message("sensors/boiler/pressure_pa", b"{{bad}}")
        assert subscriber.stats["skipped"] == 1

    @pytest.mark.asyncio  # type: ignore[misc]
    async def test_all_sensor_topics_stored(
        self, subscriber: HistorianSubscriber, writer: MagicMock
    ) -> None:
        """Every sensor topic must result in a write_point call."""
        for topic in TOPIC_TO_SENSOR_NAME:
            writer.write_point.reset_mock()
            await subscriber._handle_message(topic, make_payload(1.0))
            writer.write_point.assert_called_once()
