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
    writer.write_points = MagicMock(side_effect=len)
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


def written(writer: MagicMock) -> list[str]:
    return [
        point.to_line_protocol()
        for call in writer.write_points.call_args_list
        for point in call.args[0]
    ]


class TestHistorianSubscriber:
    @pytest.mark.parametrize(
        ("topic", "payload", "measurement", "field"),
        [
            (
                TOPIC_BOILER,
                make_boiler_msg(pressure_pa=15_500_000.0).SerializeToString(),
                MEASUREMENT_BOILER,
                "pressure_pa=15500000",
            ),
            (
                TOPIC_TURBINE,
                make_turbine_msg(steam_flow_kg_s=175.5).SerializeToString(),
                MEASUREMENT_TURBINE,
                "steam_flow_kg_s=175.5",
            ),
        ],
        ids=["boiler", "turbine"],
    )
    async def test_a_message_becomes_one_stored_point(
        self,
        subscriber: HistorianSubscriber,
        writer: MagicMock,
        topic: str,
        payload: bytes,
        measurement: str,
        field: str,
    ) -> None:
        await subscriber._handle_message(topic, payload)
        writer.write_points.assert_called_once()
        assert isinstance(writer.write_points.call_args.args[0][0], Point)
        (line,) = written(writer)
        assert line.startswith(measurement)
        assert field in line
        assert subscriber.stats == {"received": 1, "stored": 1, "skipped": 0}

    @pytest.mark.parametrize(
        ("topic", "payload"),
        [
            (TOPIC_HEARTBEAT, b"1741000000000"),
            ("sensors/unknown/xyz", b"\x00\x01\x02"),
            (TOPIC_BOILER, b"not-protobuf-\xff\xfe"),
            (TOPIC_TURBINE, b"\xff\xfe\xfd"),
        ],
        ids=["heartbeat", "unknown-topic", "bad-boiler", "bad-turbine"],
    )
    async def test_what_cannot_be_recorded_is_skipped(
        self,
        subscriber: HistorianSubscriber,
        writer: MagicMock,
        topic: str,
        payload: bytes,
    ) -> None:
        await subscriber._handle_message(topic, payload)
        writer.write_points.assert_not_called()
        assert subscriber.stats == {"received": 1, "stored": 0, "skipped": 1}

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
        assert writer.write_points.call_count == 2

    async def test_points_the_database_refused_are_not_counted_as_stored(
        self, subscriber: HistorianSubscriber, writer: MagicMock
    ) -> None:
        # "stored" fed a healthy-looking stats line while InfluxDB refused everything.
        writer.write_points.side_effect = lambda points: 0
        await subscriber._handle_message(
            TOPIC_BOILER, make_boiler_msg().SerializeToString()
        )
        assert subscriber.stats == {"received": 1, "stored": 0, "skipped": 0}


class TestValuesInfluxDbCannotStore:
    """A diverging model publishes NaN; the line protocol has no spelling for it.

    Written as a field, it makes InfluxDB refuse the whole batch — with the stack's batch
    of fifty, one bad value would cost forty-nine good points on every flush. The field is
    left out instead, so the gap shows in that one series and nothing else is lost.
    """

    def test_a_not_a_number_field_is_left_out_of_the_point(self) -> None:
        line = build_boiler_point(
            make_boiler_msg(pressure_pa=float("nan"), water_level_m=4.8)
        ).to_line_protocol()
        assert "pressure_pa" not in line
        assert "water_level_m=4.8" in line

    def test_an_infinite_field_is_left_out_too(self) -> None:
        for value in (float("inf"), float("-inf")):
            line = build_boiler_point(
                make_boiler_msg(pressure_pa=value, water_level_m=4.8)
            ).to_line_protocol()
            assert "pressure_pa" not in line
            assert "water_level_m=4.8" in line

    def test_a_point_that_keeps_its_other_fields_is_still_written(self) -> None:
        line = build_turbine_point(
            make_turbine_msg(steam_flow_kg_s=float("nan"))
        ).to_line_protocol()
        assert line.startswith(MEASUREMENT_TURBINE)
        assert "steam_flow_kg_s" not in line
