"""
InfluxDB writer for CogniBoiler sensor telemetry.

Converts protobuf messages into InfluxDB Points and writes them
via the official influxdb-client-python library.

Data model (one Point per MQTT message, multiple fields, SI units):
    boiler_sensors measurement:
        tags:   quality (good | uncertain | bad), scenario
        fields: every numeric field of BoilerStateMsg except timestamp_ms
        time:   BoilerStateMsg.timestamp_ms -> nanoseconds

    turbine_sensors measurement:
        tags:   scenario
        fields: every numeric field of TurbineStateMsg except timestamp_ms
        time:   TurbineStateMsg.timestamp_ms -> nanoseconds

The scenario tag is the scenario of the latest plant status; it is omitted until the
first plant status arrives. Plant status, KPIs and events are built in historian.points.

Writing one multi-field Point per message (vs one Point per field)
gives atomic writes and faster range queries.
"""

from __future__ import annotations

import logging
from typing import Protocol, cast

import cogniboiler_pb2 as pb
from google.protobuf.message import Message
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

_NOT_FIELDS: frozenset[str] = frozenset({"timestamp_ms", "quality"})


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


def new_point(measurement: str) -> PointLike:
    """Create a Point while containing the untyped third-party constructor."""
    return cast(PointLike, _Point(measurement))  # type: ignore[no-untyped-call]


def _new_client(url: str, token: str, org: str) -> InfluxDBClientLike:
    """Create an InfluxDB client while containing the untyped constructor."""
    return cast(
        InfluxDBClientLike,
        _InfluxDBClient(url=url, token=token, org=org),
    )


def timestamp_ns(timestamp_ms: int) -> int:
    return timestamp_ms * 1_000_000


def add_numeric_fields(
    point: PointLike,
    message: Message,
    *,
    prefix: str = "",
    skip: frozenset[str] = _NOT_FIELDS,
) -> PointLike:
    """Every scalar numeric field of a flat message as a float field."""
    for descriptor in message.DESCRIPTOR.fields:
        if descriptor.name in skip or descriptor.type == descriptor.TYPE_MESSAGE:
            continue
        value = getattr(message, descriptor.name)
        if isinstance(value, bool):
            point = point.field(f"{prefix}{descriptor.name}", 1.0 if value else 0.0)
        elif isinstance(value, int | float):
            point = point.field(f"{prefix}{descriptor.name}", float(value))
    return point


# ─── Point builders ───────────────────────────────────────────────────────────


def build_boiler_point(
    msg: pb.BoilerStateMsg, scenario: str | None = None
) -> PointLike:
    """
    Build an InfluxDB Point from a BoilerStateMsg protobuf message.

    Every sensor, flow and heat-duty field is written on a single Point.
    The quality enum is stored as a tag for fast filtering.
    """
    point = new_point(MEASUREMENT_BOILER).tag(
        "quality", _QUALITY_TAG.get(msg.quality, "unknown")
    )
    if scenario:
        point = point.tag("scenario", scenario)
    point = add_numeric_fields(point, msg)
    return point.time(timestamp_ns(msg.timestamp_ms), WritePrecision.NS)


def build_turbine_point(
    msg: pb.TurbineStateMsg, scenario: str | None = None
) -> PointLike:
    """Build an InfluxDB Point from a TurbineStateMsg protobuf message."""
    point = new_point(MEASUREMENT_TURBINE)
    if scenario:
        point = point.tag("scenario", scenario)
    point = add_numeric_fields(point, msg)
    return point.time(timestamp_ns(msg.timestamp_ms), WritePrecision.NS)


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
