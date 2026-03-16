"""
Tests for Historian writer and subscriber (protobuf edition).

No real InfluxDB or MQTT broker needed.
We test:
  1. build_boiler_point()  — Point construction from BoilerStateMsg
  2. build_turbine_point() — Point construction from TurbineStateMsg
  3. HistorianSubscriber._handle_message() — full pipeline with mock writer
"""

from __future__ import annotations

from unittest.mock import MagicMock

import cogniboiler_pb2 as pb
import pytest
from historian.subscriber import (
    TOPIC_BOILER,
    TOPIC_HEARTBEAT,
    TOPIC_TURBINE,
    HistorianSubscriber,
)
from historian.writer import (
    MEASUREMENT_BOILER,
    MEASUREMENT_TURBINE,
    InfluxWriter,
    build_boiler_point,
    build_turbine_point,
)
from influxdb_client import Point

# ─── Helpers ──────────────────────────────────────────────────────────────────

TS_MS: int = 1_741_000_000_000
TS_NS: int = TS_MS * 1_000_000


def make_boiler_msg(
    pressure_pa: float = 14_000_000.0,
    water_level_m: float = 4.8,
    water_temp_k: float = 611.0,
    flue_gas_temp_k: float = 1200.0,
    internal_energy_j: float = 2.5e12,
    quality: int = pb.SensorQuality.GOOD,
    timestamp_ms: int = TS_MS,
) -> pb.BoilerStateMsg:
    return pb.BoilerStateMsg(
        pressure_pa=pressure_pa,
        water_level_m=water_level_m,
        water_temp_k=water_temp_k,
        flue_gas_temp_k=flue_gas_temp_k,
        internal_energy_j=internal_energy_j,
        quality=quality,
        timestamp_ms=timestamp_ms,
    )


def make_turbine_msg(
    electrical_power_w: float = 200_000_000.0,
    shaft_power_w: float = 205_000_000.0,
    enthalpy_in_j_kg: float = 3_400_000.0,
    enthalpy_out_j_kg: float = 2_200_000.0,
    exhaust_pressure_pa: float = 7_000.0,
    steam_flow_kg_s: float = 150.0,
    timestamp_ms: int = TS_MS,
) -> pb.TurbineStateMsg:
    return pb.TurbineStateMsg(
        electrical_power_w=electrical_power_w,
        shaft_power_w=shaft_power_w,
        enthalpy_in_j_kg=enthalpy_in_j_kg,
        enthalpy_out_j_kg=enthalpy_out_j_kg,
        exhaust_pressure_pa=exhaust_pressure_pa,
        steam_flow_kg_s=steam_flow_kg_s,
        timestamp_ms=timestamp_ms,
    )


def make_mock_writer() -> MagicMock:
    writer = MagicMock(spec=InfluxWriter)
    writer.write_point = MagicMock()
    return writer


# ─── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture  # type: ignore[misc]
def writer() -> MagicMock:
    return make_mock_writer()


@pytest.fixture  # type: ignore[misc]
def subscriber(writer: MagicMock) -> HistorianSubscriber:
    return HistorianSubscriber(writer=writer)


# ─── build_boiler_point tests ─────────────────────────────────────────────────


class TestBuildBoilerPoint:
    def test_returns_point_instance(self) -> None:
        point = build_boiler_point(make_boiler_msg())
        assert isinstance(point, Point)

    def test_measurement_is_boiler_sensors(self) -> None:
        line = build_boiler_point(make_boiler_msg()).to_line_protocol()
        assert line.startswith(MEASUREMENT_BOILER)

    def test_pressure_field_present(self) -> None:
        line = build_boiler_point(
            make_boiler_msg(pressure_pa=14_000_000.0)
        ).to_line_protocol()
        assert "pressure_pa=14000000" in line

    def test_water_level_field_present(self) -> None:
        line = build_boiler_point(make_boiler_msg(water_level_m=4.8)).to_line_protocol()
        assert "water_level_m=4.8" in line

    def test_water_temp_field_present(self) -> None:
        line = build_boiler_point(
            make_boiler_msg(water_temp_k=611.0)
        ).to_line_protocol()
        assert "water_temp_k=611" in line

    def test_flue_gas_temp_field_present(self) -> None:
        line = build_boiler_point(
            make_boiler_msg(flue_gas_temp_k=1200.0)
        ).to_line_protocol()
        assert "flue_gas_temp_k=1200" in line

    def test_internal_energy_field_present(self) -> None:
        line = build_boiler_point(
            make_boiler_msg(internal_energy_j=2.5e12)
        ).to_line_protocol()
        assert "internal_energy_j=" in line

    def test_quality_good_tag(self) -> None:
        line = build_boiler_point(
            make_boiler_msg(quality=pb.SensorQuality.GOOD)
        ).to_line_protocol()
        assert "quality=good" in line

    def test_quality_uncertain_tag(self) -> None:
        line = build_boiler_point(
            make_boiler_msg(quality=pb.SensorQuality.UNCERTAIN)
        ).to_line_protocol()
        assert "quality=uncertain" in line

    def test_quality_bad_tag(self) -> None:
        line = build_boiler_point(
            make_boiler_msg(quality=pb.SensorQuality.BAD)
        ).to_line_protocol()
        assert "quality=bad" in line

    def test_timestamp_converted_to_nanoseconds(self) -> None:
        line = build_boiler_point(
            make_boiler_msg(timestamp_ms=TS_MS)
        ).to_line_protocol()
        ts_ns = int(line.split()[-1])
        assert ts_ns == TS_NS

    def test_all_five_fields_present_in_one_point(self) -> None:
        line = build_boiler_point(make_boiler_msg()).to_line_protocol()
        for field in (
            "pressure_pa",
            "water_level_m",
            "water_temp_k",
            "flue_gas_temp_k",
            "internal_energy_j",
        ):
            assert field in line, f"Missing field: {field}"


# ─── build_turbine_point tests ────────────────────────────────────────────────


class TestBuildTurbinePoint:
    def test_returns_point_instance(self) -> None:
        point = build_turbine_point(make_turbine_msg())
        assert isinstance(point, Point)

    def test_measurement_is_turbine_sensors(self) -> None:
        line = build_turbine_point(make_turbine_msg()).to_line_protocol()
        assert line.startswith(MEASUREMENT_TURBINE)

    def test_electrical_power_field_present(self) -> None:
        line = build_turbine_point(
            make_turbine_msg(electrical_power_w=200_000_000.0)
        ).to_line_protocol()
        assert "electrical_power_w=" in line

    def test_shaft_power_field_present(self) -> None:
        line = build_turbine_point(make_turbine_msg()).to_line_protocol()
        assert "shaft_power_w=" in line

    def test_steam_flow_field_present(self) -> None:
        line = build_turbine_point(
            make_turbine_msg(steam_flow_kg_s=150.0)
        ).to_line_protocol()
        assert "steam_flow_kg_s=150" in line

    def test_exhaust_pressure_field_present(self) -> None:
        line = build_turbine_point(
            make_turbine_msg(exhaust_pressure_pa=7000.0)
        ).to_line_protocol()
        assert "exhaust_pressure_pa=7000" in line

    def test_enthalpy_in_field_present(self) -> None:
        line = build_turbine_point(make_turbine_msg()).to_line_protocol()
        assert "enthalpy_in_j_kg=" in line

    def test_enthalpy_out_field_present(self) -> None:
        line = build_turbine_point(make_turbine_msg()).to_line_protocol()
        assert "enthalpy_out_j_kg=" in line

    def test_timestamp_converted_to_nanoseconds(self) -> None:
        line = build_turbine_point(
            make_turbine_msg(timestamp_ms=TS_MS)
        ).to_line_protocol()
        ts_ns = int(line.split()[-1])
        assert ts_ns == TS_NS

    def test_all_six_fields_present_in_one_point(self) -> None:
        line = build_turbine_point(make_turbine_msg()).to_line_protocol()
        for field in (
            "electrical_power_w",
            "shaft_power_w",
            "enthalpy_in_j_kg",
            "enthalpy_out_j_kg",
            "exhaust_pressure_pa",
            "steam_flow_kg_s",
        ):
            assert field in line, f"Missing field: {field}"


# ─── HistorianSubscriber tests ────────────────────────────────────────────────


class TestHistorianSubscriber:
    @pytest.mark.asyncio  # type: ignore[misc]
    async def test_boiler_topic_calls_write_point(
        self, subscriber: HistorianSubscriber, writer: MagicMock
    ) -> None:
        payload = make_boiler_msg().SerializeToString()
        await subscriber._handle_message(TOPIC_BOILER, payload)
        writer.write_point.assert_called_once()

    @pytest.mark.asyncio  # type: ignore[misc]
    async def test_boiler_topic_writes_point_instance(
        self, subscriber: HistorianSubscriber, writer: MagicMock
    ) -> None:
        payload = make_boiler_msg().SerializeToString()
        await subscriber._handle_message(TOPIC_BOILER, payload)
        arg = writer.write_point.call_args.args[0]
        assert isinstance(arg, Point)

    @pytest.mark.asyncio  # type: ignore[misc]
    async def test_boiler_point_has_correct_measurement(
        self, subscriber: HistorianSubscriber, writer: MagicMock
    ) -> None:
        payload = make_boiler_msg().SerializeToString()
        await subscriber._handle_message(TOPIC_BOILER, payload)
        arg = writer.write_point.call_args.args[0]
        assert arg.to_line_protocol().startswith(MEASUREMENT_BOILER)

    @pytest.mark.asyncio  # type: ignore[misc]
    async def test_turbine_topic_calls_write_point(
        self, subscriber: HistorianSubscriber, writer: MagicMock
    ) -> None:
        payload = make_turbine_msg().SerializeToString()
        await subscriber._handle_message(TOPIC_TURBINE, payload)
        writer.write_point.assert_called_once()

    @pytest.mark.asyncio  # type: ignore[misc]
    async def test_turbine_point_has_correct_measurement(
        self, subscriber: HistorianSubscriber, writer: MagicMock
    ) -> None:
        payload = make_turbine_msg().SerializeToString()
        await subscriber._handle_message(TOPIC_TURBINE, payload)
        arg = writer.write_point.call_args.args[0]
        assert arg.to_line_protocol().startswith(MEASUREMENT_TURBINE)

    @pytest.mark.asyncio  # type: ignore[misc]
    async def test_heartbeat_skipped(
        self, subscriber: HistorianSubscriber, writer: MagicMock
    ) -> None:
        await subscriber._handle_message(TOPIC_HEARTBEAT, b"1741000000000")
        writer.write_point.assert_not_called()

    @pytest.mark.asyncio  # type: ignore[misc]
    async def test_unknown_topic_skipped(
        self, subscriber: HistorianSubscriber, writer: MagicMock
    ) -> None:
        await subscriber._handle_message("sensors/unknown/xyz", b"\x00\x01\x02")
        writer.write_point.assert_not_called()

    @pytest.mark.asyncio  # type: ignore[misc]
    async def test_malformed_protobuf_boiler_skipped(
        self, subscriber: HistorianSubscriber, writer: MagicMock
    ) -> None:
        await subscriber._handle_message(TOPIC_BOILER, b"not-protobuf-\xff\xfe")
        writer.write_point.assert_not_called()

    @pytest.mark.asyncio  # type: ignore[misc]
    async def test_malformed_protobuf_turbine_skipped(
        self, subscriber: HistorianSubscriber, writer: MagicMock
    ) -> None:
        await subscriber._handle_message(TOPIC_TURBINE, b"not-protobuf-\xff\xfe")
        writer.write_point.assert_not_called()

    @pytest.mark.asyncio  # type: ignore[misc]
    async def test_received_counter_increments_on_boiler(
        self, subscriber: HistorianSubscriber, writer: MagicMock
    ) -> None:
        payload = make_boiler_msg().SerializeToString()
        await subscriber._handle_message(TOPIC_BOILER, payload)
        assert subscriber.stats["received"] == 1

    @pytest.mark.asyncio  # type: ignore[misc]
    async def test_stored_counter_increments_on_boiler(
        self, subscriber: HistorianSubscriber, writer: MagicMock
    ) -> None:
        payload = make_boiler_msg().SerializeToString()
        await subscriber._handle_message(TOPIC_BOILER, payload)
        assert subscriber.stats["stored"] == 1

    @pytest.mark.asyncio  # type: ignore[misc]
    async def test_stored_counter_increments_on_turbine(
        self, subscriber: HistorianSubscriber, writer: MagicMock
    ) -> None:
        payload = make_turbine_msg().SerializeToString()
        await subscriber._handle_message(TOPIC_TURBINE, payload)
        assert subscriber.stats["stored"] == 1

    @pytest.mark.asyncio  # type: ignore[misc]
    async def test_skipped_counter_on_heartbeat(
        self, subscriber: HistorianSubscriber, writer: MagicMock
    ) -> None:
        await subscriber._handle_message(TOPIC_HEARTBEAT, b"12345")
        assert subscriber.stats["skipped"] == 1

    @pytest.mark.asyncio  # type: ignore[misc]
    async def test_skipped_counter_on_bad_protobuf(
        self, subscriber: HistorianSubscriber, writer: MagicMock
    ) -> None:
        await subscriber._handle_message(TOPIC_BOILER, b"\xff\xfe\xfd")
        assert subscriber.stats["skipped"] == 1

    @pytest.mark.asyncio  # type: ignore[misc]
    async def test_skipped_counter_on_unknown_topic(
        self, subscriber: HistorianSubscriber, writer: MagicMock
    ) -> None:
        await subscriber._handle_message("sensors/weird/topic", b"data")
        assert subscriber.stats["skipped"] == 1

    @pytest.mark.asyncio  # type: ignore[misc]
    async def test_both_topics_stored_independently(
        self, subscriber: HistorianSubscriber, writer: MagicMock
    ) -> None:
        await subscriber._handle_message(
            TOPIC_BOILER, make_boiler_msg().SerializeToString()
        )
        await subscriber._handle_message(
            TOPIC_TURBINE, make_turbine_msg().SerializeToString()
        )
        assert subscriber.stats["stored"] == 2
        assert writer.write_point.call_count == 2

    @pytest.mark.asyncio  # type: ignore[misc]
    async def test_boiler_roundtrip_pressure_value(
        self, subscriber: HistorianSubscriber, writer: MagicMock
    ) -> None:
        """Verify that the exact pressure value survives the protobuf roundtrip."""
        payload = make_boiler_msg(pressure_pa=15_500_000.0).SerializeToString()
        await subscriber._handle_message(TOPIC_BOILER, payload)
        arg = writer.write_point.call_args.args[0]
        line = arg.to_line_protocol()
        assert "pressure_pa=15500000" in line

    @pytest.mark.asyncio  # type: ignore[misc]
    async def test_turbine_roundtrip_steam_flow_value(
        self, subscriber: HistorianSubscriber, writer: MagicMock
    ) -> None:
        payload = make_turbine_msg(steam_flow_kg_s=175.5).SerializeToString()
        await subscriber._handle_message(TOPIC_TURBINE, payload)
        arg = writer.write_point.call_args.args[0]
        line = arg.to_line_protocol()
        assert "steam_flow_kg_s=175.5" in line
