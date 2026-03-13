"""
MQTT publisher for Physics Engine telemetry.

Publishes boiler and turbine state to Mosquitto broker
at a configurable rate using Protocol Buffers serialization.

Topic tree (Flow 1):
    sensors/boiler          ← serialized BoilerStateMsg
    sensors/turbine         ← serialized TurbineStateMsg
    sensors/system/heartbeat ← UTF-8 epoch-ms string

Protocol Buffers (not JSON) are used for:
  - ~3× smaller payload vs equivalent JSON
  - Strict schema — no silent field renames
  - Native gRPC/OPC UA compatibility

All publishes use QoS 0 (fire-and-forget) — sensor telemetry
can tolerate occasional loss; throughput matters more.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass

import cogniboiler_pb2 as pb  # shared/generated — added to sys.path by conftest
from asyncio_mqtt import Client, MqttError

from physics_engine.models import BoilerState
from physics_engine.turbine import TurbineState

logger = logging.getLogger(__name__)

# ─── Topic constants ──────────────────────────────────────────────────────────

TOPIC_BOILER: str = "sensors/boiler"
TOPIC_TURBINE: str = "sensors/turbine"
TOPIC_HEARTBEAT: str = "sensors/system/heartbeat"


# ─── Protobuf serializers ─────────────────────────────────────────────────────


def boiler_state_to_proto(state: BoilerState) -> pb.BoilerStateMsg:
    """
    Convert a BoilerState dataclass to a BoilerStateMsg protobuf message.

    Args:
        state: Current boiler physics state.

    Returns:
        Populated BoilerStateMsg ready for serialization.
    """
    return pb.BoilerStateMsg(
        pressure_pa=state.pressure,
        water_level_m=state.water_level,
        water_temp_k=state.water_temp,
        flue_gas_temp_k=state.flue_gas_temp,
        internal_energy_j=state.internal_energy,
        timestamp_ms=int(time.time() * 1000),
        quality=pb.SensorQuality.GOOD,
    )


def turbine_state_to_proto(state: TurbineState) -> pb.TurbineStateMsg:
    """
    Convert a TurbineState dataclass to a TurbineStateMsg protobuf message.

    Args:
        state: Current turbine physics state.

    Returns:
        Populated TurbineStateMsg ready for serialization.
    """
    return pb.TurbineStateMsg(
        electrical_power_w=state.electrical_power,
        shaft_power_w=state.shaft_power,
        enthalpy_in_j_kg=state.enthalpy_in,
        enthalpy_out_j_kg=state.enthalpy_out_actual,
        exhaust_pressure_pa=state.exhaust_pressure,
        steam_flow_kg_s=state.steam_flow,
        timestamp_ms=int(time.time() * 1000),
    )


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

    Publishes protobuf-serialized messages to two topics:
        sensors/boiler   ← BoilerStateMsg
        sensors/turbine  ← TurbineStateMsg

    Usage (production):
        config = MQTTConfig(host="mosquitto", port=1883)
        pub = MQTTPublisher(config)
        await pub.run(boiler_state_fn, turbine_state_fn)   # blocks forever

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
    ) -> None:
        """
        Serialize BoilerState to BoilerStateMsg and publish to sensors/boiler.

        One MQTT message per call, QoS 0.
        """
        msg = boiler_state_to_proto(state)
        payload = msg.SerializeToString()
        try:
            await client.publish(TOPIC_BOILER, payload, qos=0)
            self._published += 1
        except MqttError as exc:
            self._errors += 1
            logger.warning("Publish failed [%s]: %s", TOPIC_BOILER, exc)

    async def publish_turbine(
        self,
        client: Client,
        state: TurbineState,
    ) -> None:
        """
        Serialize TurbineState to TurbineStateMsg and publish to sensors/turbine.

        One MQTT message per call, QoS 0.
        """
        msg = turbine_state_to_proto(state)
        payload = msg.SerializeToString()
        try:
            await client.publish(TOPIC_TURBINE, payload, qos=0)
            self._published += 1
        except MqttError as exc:
            self._errors += 1
            logger.warning("Publish failed [%s]: %s", TOPIC_TURBINE, exc)

    async def publish_heartbeat(self, client: Client) -> None:
        """Publish system sync heartbeat with current timestamp."""
        payload = str(int(time.time() * 1000)).encode()
        try:
            await client.publish(TOPIC_HEARTBEAT, payload, qos=0)
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
