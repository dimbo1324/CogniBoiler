"""
InfluxDB writer for CogniBoiler sensor telemetry.

Converts protobuf messages into InfluxDB Points and writes them
via the official influxdb-client-python library.

Data model (one Point per MQTT message, multiple fields):
    boiler_sensors measurement:
        tags:   quality (GOOD | UNCERTAIN | BAD)
        fields: pressure_pa, water_level_m, water_temp_k,
                flue_gas_temp_k, internal_energy_j
        time:   BoilerStateMsg.timestamp_ms -> nanoseconds

    turbine_sensors measurement:
        fields: electrical_power_w, shaft_power_w,
                enthalpy_in_j_kg, enthalpy_out_j_kg,
                exhaust_pressure_pa, steam_flow_kg_s
        time:   TurbineStateMsg.timestamp_ms -> nanoseconds

Writing one multi-field Point per message (vs one Point per field)
gives atomic writes and faster range queries.
"""

from __future__ import annotations

import logging
from typing import Protocol, cast

import cogniboiler_pb2 as pb
from influxdb_client.client.influxdb_client import InfluxDBClient as _InfluxDBClient
from influxdb_client.client.write.point import Point as _Point
from influxdb_client.client.write_api import SYNCHRONOUS
from influxdb_client.domain.write_precision import WritePrecision

logger = logging.getLogger(__name__)

# ─── Measurement names ────────────────────────────────────────────────────────

MEASUREMENT_BOILER: str = "boiler_sensors"
MEASUREMENT_TURBINE: str = "turbine_sensors"

# ─── Quality enum -> tag string ────────────────────────────────────────────────

_QUALITY_TAG: dict[int, str] = {
    pb.SensorQuality.GOOD: "good",
    pb.SensorQuality.UNCERTAIN: "uncertain",
    pb.SensorQuality.BAD: "bad",
}


class PointLike(Protocol):
    """Typed subset of the fluent Point API used in this module."""

    def tag(self, key: str, value: str) -> PointLike: ...

    def field(self, key: str, value: bool | int | float | str) -> PointLike: ...

    def time(self, time_value: int, write_precision: object) -> PointLike: ...


class WriteApiLike(Protocol):
    """Minimal synchronous write API surface used by InfluxWriter."""

    def write(self, *, bucket: str, record: PointLike | list[PointLike]) -> object: ...


class InfluxDBClientLike(Protocol):
    """Minimal client surface used by InfluxWriter."""

    def write_api(self, *, write_options: object) -> WriteApiLike: ...

    def close(self) -> None: ...


def _new_point(measurement: str) -> PointLike:
    """Create a Point while containing the untyped third-party constructor."""
    return cast(PointLike, _Point(measurement))  # type: ignore[no-untyped-call]


def _new_client(url: str, token: str, org: str) -> InfluxDBClientLike:
    """Create an InfluxDB client while containing the untyped constructor."""
    return cast(
        InfluxDBClientLike,
        _InfluxDBClient(url=url, token=token, org=org),
    )


# ─── Point builders ───────────────────────────────────────────────────────────


def build_boiler_point(msg: pb.BoilerStateMsg) -> PointLike:
    """
    Build an InfluxDB Point from a BoilerStateMsg protobuf message.

    All five sensor fields are written as separate fields on a single Point.
    The quality enum is stored as a tag for fast filtering.

    Args:
        msg: Parsed BoilerStateMsg from MQTT payload.

    Returns:
        Point ready for writing to InfluxDB.
    """
    ts_ns = msg.timestamp_ms * 1_000_000
    quality_tag = _QUALITY_TAG.get(msg.quality, "unknown")

    return (
        _new_point(MEASUREMENT_BOILER)
        .tag("quality", quality_tag)
        .field("pressure_pa", msg.pressure_pa)
        .field("water_level_m", msg.water_level_m)
        .field("water_temp_k", msg.water_temp_k)
        .field("flue_gas_temp_k", msg.flue_gas_temp_k)
        .field("internal_energy_j", msg.internal_energy_j)
        .time(ts_ns, WritePrecision.NS)
    )


def build_turbine_point(msg: pb.TurbineStateMsg) -> PointLike:
    """
    Build an InfluxDB Point from a TurbineStateMsg protobuf message.

    All six sensor fields are written as separate fields on a single Point.

    Args:
        msg: Parsed TurbineStateMsg from MQTT payload.

    Returns:
        Point ready for writing to InfluxDB.
    """
    ts_ns = msg.timestamp_ms * 1_000_000

    return (
        _new_point(MEASUREMENT_TURBINE)
        .field("electrical_power_w", msg.electrical_power_w)
        .field("shaft_power_w", msg.shaft_power_w)
        .field("enthalpy_in_j_kg", msg.enthalpy_in_j_kg)
        .field("enthalpy_out_j_kg", msg.enthalpy_out_j_kg)
        .field("exhaust_pressure_pa", msg.exhaust_pressure_pa)
        .field("steam_flow_kg_s", msg.steam_flow_kg_s)
        .field("steam_temp_in_k", msg.steam_temp_in_k)
        .time(ts_ns, WritePrecision.NS)
    )


# ─── Writer ───────────────────────────────────────────────────────────────────


class InfluxWriter:
    """
    Thin wrapper around InfluxDB write API.

    Usage:
        writer = InfluxWriter(url="http://localhost:8086",
                              token="...", org="cogniboiler",
                              bucket="sensors")
        writer.write_point(point)
        writer.close()

    In tests, inject a mock write_api instead of a real client.
    """

    def __init__(
        self,
        url: str,
        token: str,
        org: str,
        bucket: str,
    ) -> None:
        self._bucket = bucket
        self._org = org
        self._client: InfluxDBClientLike = _new_client(url=url, token=token, org=org)
        self._write_api = self._client.write_api(write_options=SYNCHRONOUS)
        self._written: int = 0
        self._errors: int = 0

    @property
    def written(self) -> int:
        return self._written

    @property
    def errors(self) -> int:
        return self._errors

    def write_point(self, point: PointLike) -> None:
        """Write a single Point to InfluxDB. Errors are counted, not raised."""
        try:
            self._write_api.write(bucket=self._bucket, record=point)
            self._written += 1
        except Exception as exc:
            self._errors += 1
            logger.warning("InfluxDB write error: %s", exc)

    def write_points(self, points: list[PointLike]) -> None:
        """Write a batch of points to InfluxDB in one call."""
        if not points:
            return
        try:
            self._write_api.write(bucket=self._bucket, record=points)
            self._written += len(points)
        except Exception as exc:
            self._errors += len(points)
            logger.warning("InfluxDB batch write error: %s", exc)

    def close(self) -> None:
        """Flush and close the InfluxDB client."""
        self._client.close()
