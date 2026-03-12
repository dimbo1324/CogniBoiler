"""
MQTT publisher for Physics Engine telemetry.

Publishes boiler and turbine state to Mosquitto broker
at a configurable rate. Each physical quantity gets its
own topic so subscribers can filter precisely.

Topic tree (Poток 1):
    sensors/boiler/pressure_pa
    sensors/boiler/water_level_m
    sensors/boiler/water_temp_k
    sensors/boiler/flue_gas_temp_k
    sensors/boiler/internal_energy_j
    sensors/turbine/electrical_power_w
    sensors/turbine/shaft_power_w
    sensors/turbine/steam_flow_kg_s
    sensors/turbine/exhaust_pressure_pa
    sensors/system/timestamp_ms          ← sync heartbeat

Payload format: plain UTF-8 JSON
    {"value": 14000000.0, "unit": "Pa", "quality": "good", "ts": 1741000000123}

All publishes use QoS 0 (fire-and-forget) — sensor telemetry
can tolerate occasional loss; throughput matters more.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from asyncio_mqtt import Client, MqttError

from physics_engine.models import BoilerState
from physics_engine.turbine import TurbineState

logger = logging.getLogger(__name__)

# ─── Topic constants ──────────────────────────────────────────────────────────

TOPIC_PREFIX = "sensors"

BOILER_TOPICS: dict[str, tuple[str, str]] = {
    # field_name: (sub_topic, unit)
    "pressure": ("boiler/pressure_pa", "Pa"),
    "water_level": ("boiler/water_level_m", "m"),
    "water_temp": ("boiler/water_temp_k", "K"),
    "flue_gas_temp": ("boiler/flue_gas_temp_k", "K"),
    "internal_energy": ("boiler/internal_energy_j", "J"),
}

TURBINE_TOPICS: dict[str, tuple[str, str]] = {
    "electrical_power": ("turbine/electrical_power_w", "W"),
    "shaft_power": ("turbine/shaft_power_w", "W"),
    "steam_flow_kg_s": ("turbine/steam_flow_kg_s", "kg/s"),
    "exhaust_pressure": ("turbine/exhaust_pressure_pa", "Pa"),
}

HEARTBEAT_TOPIC = f"{TOPIC_PREFIX}/system/timestamp_ms"


# ─── Payload builder ──────────────────────────────────────────────────────────


def build_payload(value: float, unit: str, quality: str = "good") -> bytes:
    """
    Serialize a single sensor reading to JSON bytes.

    Args:
        value:   Engineering value.
        unit:    Engineering unit string (e.g. "Pa", "K", "W").
        quality: OPC UA quality string ("good" | "uncertain" | "bad").

    Returns:
        UTF-8 encoded JSON bytes.
    """
    doc: dict[str, Any] = {
        "value": value,
        "unit": unit,
        "quality": quality,
        "ts": int(time.time() * 1000),
    }
    return json.dumps(doc).encode()


# ─── Publisher config ─────────────────────────────────────────────────────────


@dataclass
class MQTTConfig:
    """MQTT broker connection parameters."""

    host: str = "localhost"
    port: int = 1883
    keepalive: int = 60  # seconds
    client_id: str = "physics-engine"
    interval_s: float = 0.1  # publish every 100 ms = 10 Hz


# ─── Publisher ────────────────────────────────────────────────────────────────


class MQTTPublisher:
    """
    Async MQTT publisher for boiler + turbine telemetry.

    Usage (production):
        config = MQTTConfig(host="mosquitto", port=1883)
        pub = MQTTPublisher(config)
        await pub.run(state_generator)   # blocks forever

    Usage (one-shot, for testing):
        async with pub.connected() as client:
            await pub.publish_boiler(client, boiler_state)
            await pub.publish_turbine(client, turbine_state)
    """

    def __init__(self, config: MQTTConfig | None = None) -> None:
        self.config = config or MQTTConfig()
        self._published: int = 0
        self._errors: int = 0

    # ─── Stats ───────────────────────────────────────────────────────────────

    @property
    def published(self) -> int:
        """Total messages successfully published."""
        return self._published

    @property
    def errors(self) -> int:
        """Total publish errors."""
        return self._errors

    # ─── Publish helpers ─────────────────────────────────────────────────────

    async def publish_boiler(
        self,
        client: Client,
        state: BoilerState,
        quality: str = "good",
    ) -> None:
        """
        Publish all boiler state fields to their individual topics.

        Each field → one MQTT message, QoS 0.
        """
        field_values = {
            "pressure": state.pressure,
            "water_level": state.water_level,
            "water_temp": state.water_temp,
            "flue_gas_temp": state.flue_gas_temp,
            "internal_energy": state.internal_energy,
        }
        for field, (sub_topic, unit) in BOILER_TOPICS.items():
            topic = f"{TOPIC_PREFIX}/{sub_topic}"
            payload = build_payload(field_values[field], unit, quality)
            try:
                await client.publish(topic, payload, qos=0)
                self._published += 1
            except MqttError as exc:
                self._errors += 1
                logger.warning("Publish failed [%s]: %s", topic, exc)

    async def publish_turbine(
        self,
        client: Client,
        state: TurbineState,
        quality: str = "good",
    ) -> None:
        """
        Publish all turbine state fields to their individual topics.
        """
        field_values = {
            "electrical_power": state.electrical_power,
            "shaft_power": state.shaft_power,
            "steam_flow_kg_s": state.steam_flow,
            "exhaust_pressure": state.exhaust_pressure,
        }
        for field, (sub_topic, unit) in TURBINE_TOPICS.items():
            topic = f"{TOPIC_PREFIX}/{sub_topic}"
            payload = build_payload(field_values[field], unit, quality)
            try:
                await client.publish(topic, payload, qos=0)
                self._published += 1
            except MqttError as exc:
                self._errors += 1
                logger.warning("Publish failed [%s]: %s", topic, exc)

    async def publish_heartbeat(self, client: Client) -> None:
        """Publish system sync heartbeat with current timestamp."""
        payload = str(int(time.time() * 1000)).encode()
        try:
            await client.publish(HEARTBEAT_TOPIC, payload, qos=0)
            self._published += 1
        except MqttError as exc:
            self._errors += 1
            logger.warning("Heartbeat publish failed: %s", exc)

    # ─── Context manager ─────────────────────────────────────────────────────

    def connected(self) -> Client:
        """
        Return an asyncio_mqtt Client context manager.

        Usage:
            async with publisher.connected() as client:
                await publisher.publish_boiler(client, state)
        """
        return Client(
            hostname=self.config.host,
            port=self.config.port,
            keepalive=self.config.keepalive,
            client_id=self.config.client_id,
        )

    # ─── Continuous publish loop ──────────────────────────────────────────────

    async def run(
        self,
        boiler_state_fn: Callable[[], BoilerState],
        turbine_state_fn: Callable[[], TurbineState],
    ) -> None:
        """
        Continuously publish telemetry at config.interval_s rate.

        Args:
            boiler_state_fn:  Callable[[], BoilerState]
            turbine_state_fn: Callable[[], TurbineState]

        Reconnects automatically on broker disconnect.
        """
        while True:
            try:
                async with self.connected() as client:
                    logger.info(
                        "MQTT connected to %s:%d",
                        self.config.host,
                        self.config.port,
                    )
                    while True:
                        boiler_state = boiler_state_fn()
                        turbine_state = turbine_state_fn()

                        await self.publish_boiler(client, boiler_state)
                        await self.publish_turbine(client, turbine_state)
                        await self.publish_heartbeat(client)

                        await asyncio.sleep(self.config.interval_s)

            except MqttError as exc:
                self._errors += 1
                logger.warning("MQTT disconnected: %s — retrying in 5s", exc)
                await asyncio.sleep(5.0)
