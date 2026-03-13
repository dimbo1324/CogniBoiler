"""
InfluxDB writer for CogniBoiler sensor telemetry.

Converts protobuf messages into InfluxDB Points and writes them
via the official influxdb-client-python library.

Data model (one Point per MQTT message, multiple fields):
    boiler_sensors measurement:
        tags:   quality (GOOD | UNCERTAIN | BAD)
        fields: pressure_pa, water_level_m, water_temp_k,
                flue_gas_temp_k, internal_energy_j
        time:   BoilerStateMsg.timestamp_ms → nanoseconds

    turbine_sensors measurement:
        fields: electrical_power_w, shaft_power_w,
                enthalpy_in_j_kg, enthalpy_out_j_kg,
                exhaust_pressure_pa, steam_flow_kg_s
        time:   TurbineStateMsg.timestamp_ms → nanoseconds

Writing one multi-field Point per message (vs one Point per field)
gives atomic writes and faster range queries.
"""

from __future__ import annotations

import logging

import cogniboiler_pb2 as pb
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS

logger = logging.getLogger(__name__)

# ─── Measurement names ────────────────────────────────────────────────────────

MEASUREMENT_BOILER: str = "boiler_sensors"
MEASUREMENT_TURBINE: str = "turbine_sensors"

# ─── Quality enum → tag string ────────────────────────────────────────────────

_QUALITY_TAG: dict[int, str] = {
    pb.SensorQuality.GOOD: "good",
    pb.SensorQuality.UNCERTAIN: "uncertain",
    pb.SensorQuality.BAD: "bad",
}


# ─── Point builders ───────────────────────────────────────────────────────────


def build_boiler_point(msg: pb.BoilerStateMsg) -> Point:
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
        Point(MEASUREMENT_BOILER)
        .tag("quality", quality_tag)
        .field("pressure_pa", msg.pressure_pa)
        .field("water_level_m", msg.water_level_m)
        .field("water_temp_k", msg.water_temp_k)
        .field("flue_gas_temp_k", msg.flue_gas_temp_k)
        .field("internal_energy_j", msg.internal_energy_j)
        .time(ts_ns, WritePrecision.NS)
    )


def build_turbine_point(msg: pb.TurbineStateMsg) -> Point:
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
        Point(MEASUREMENT_TURBINE)
        .field("electrical_power_w", msg.electrical_power_w)
        .field("shaft_power_w", msg.shaft_power_w)
        .field("enthalpy_in_j_kg", msg.enthalpy_in_j_kg)
        .field("enthalpy_out_j_kg", msg.enthalpy_out_j_kg)
        .field("exhaust_pressure_pa", msg.exhaust_pressure_pa)
        .field("steam_flow_kg_s", msg.steam_flow_kg_s)
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
        self._client = InfluxDBClient(url=url, token=token, org=org)
        self._write_api = self._client.write_api(write_options=SYNCHRONOUS)
        self._written: int = 0
        self._errors: int = 0

    @property
    def written(self) -> int:
        return self._written

    @property
    def errors(self) -> int:
        return self._errors

    def write_point(self, point: Point) -> None:
        """Write a single Point to InfluxDB. Errors are counted, not raised."""
        try:
            self._write_api.write(bucket=self._bucket, record=point)
            self._written += 1
        except Exception as exc:
            self._errors += 1
            logger.warning("InfluxDB write error: %s", exc)

    def close(self) -> None:
        """Flush and close the InfluxDB client."""
        self._client.close()
