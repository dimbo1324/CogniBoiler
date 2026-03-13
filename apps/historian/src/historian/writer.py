"""
InfluxDB writer for CogniBoiler sensor telemetry.

Converts MQTT JSON payloads into InfluxDB Points and writes them
via the official influxdb-client-python library.

Data model:
    Measurement:  "boiler_sensors"  or  "turbine_sensors"
    Fields:       value (float)
    Tags:         sensor   (browse_name, e.g. "Pressure")
                  unit     (engineering unit, e.g. "Pa")
                  quality  ("good" | "uncertain" | "bad")
    Timestamp:    taken from payload["ts"] (milliseconds → nanoseconds)

One Point per MQTT message — no batching delay, InfluxDB client
handles internal write batching transparently.
"""

from __future__ import annotations

import logging
from typing import Any

from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS

logger = logging.getLogger(__name__)

# ─── Measurement names ────────────────────────────────────────────────────────

MEASUREMENT_BOILER: str = "boiler_sensors"
MEASUREMENT_TURBINE: str = "turbine_sensors"

# MQTT sub-topic prefix → measurement name
TOPIC_PREFIX_TO_MEASUREMENT: dict[str, str] = {
    "boiler": MEASUREMENT_BOILER,
    "turbine": MEASUREMENT_TURBINE,
}

# MQTT sub-topic → sensor browse name (for the "sensor" tag)
TOPIC_TO_SENSOR_NAME: dict[str, str] = {
    "sensors/boiler/pressure_pa": "Pressure",
    "sensors/boiler/water_level_m": "WaterLevel",
    "sensors/boiler/water_temp_k": "WaterTemp",
    "sensors/boiler/flue_gas_temp_k": "FlueGasTemp",
    "sensors/boiler/internal_energy_j": "InternalEnergy",
    "sensors/turbine/electrical_power_w": "ElectricalPower",
    "sensors/turbine/shaft_power_w": "ShaftPower",
    "sensors/turbine/steam_flow_kg_s": "SteamFlow",
    "sensors/turbine/exhaust_pressure_pa": "ExhaustPressure",
}


# ─── Point builder ────────────────────────────────────────────────────────────


def topic_to_measurement(topic: str) -> str | None:
    """
    Derive InfluxDB measurement name from MQTT topic.

    "sensors/boiler/pressure_pa"  → "boiler_sensors"
    "sensors/turbine/shaft_power" → "turbine_sensors"
    "sensors/system/timestamp_ms" → None  (skip)

    Returns None for topics that should not be stored.
    """
    parts = topic.split("/")
    if len(parts) < 2:
        return None
    sub = parts[1]  # "boiler" | "turbine" | "system"
    return TOPIC_PREFIX_TO_MEASUREMENT.get(sub)


def build_point(topic: str, payload: dict[str, Any]) -> Point | None:
    """
    Build an InfluxDB Point from a parsed MQTT payload dict.

    Args:
        topic:   Full MQTT topic string.
        payload: Parsed JSON dict with keys: value, unit, quality, ts.

    Returns:
        Point ready for writing, or None if topic should be skipped.
    """
    measurement = topic_to_measurement(topic)
    if measurement is None:
        return None

    sensor_name = TOPIC_TO_SENSOR_NAME.get(topic, topic.split("/")[-1])

    try:
        value = float(payload["value"])
        unit = str(payload.get("unit", ""))
        quality = str(payload.get("quality", "good"))
        ts_ms = int(payload["ts"])
    except (KeyError, ValueError, TypeError) as exc:
        logger.warning("Cannot build Point from payload on %s: %s", topic, exc)
        return None

    # InfluxDB client expects nanoseconds for WritePrecision.NS
    ts_ns = ts_ms * 1_000_000

    point = (
        Point(measurement)
        .tag("sensor", sensor_name)
        .tag("unit", unit)
        .tag("quality", quality)
        .field("value", value)
        .time(ts_ns, WritePrecision.NS)
    )
    return point


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
