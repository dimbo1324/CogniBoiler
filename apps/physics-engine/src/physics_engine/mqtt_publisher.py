"""
MQTT publisher for Physics Engine telemetry.

Publishes the live plant to the Mosquitto broker using Protocol Buffers.

Topic tree:
    sensors/boiler           ← BoilerStateMsg (measured values)
    sensors/turbine          ← TurbineStateMsg (measured values)
    sensors/plant            ← PlantStatusMsg: emissions, condenser, wear, faults,
                               instrument qualities, simulation status
    sensors/system/heartbeat ← UTF-8 epoch-ms string
    status/physics-engine    ← retained "online" / "offline" (MQTT will)

Protocol Buffers (not JSON) are used for:
  - ~3× smaller payload vs equivalent JSON
  - Strict schema — no silent field renames
  - Native gRPC/OPC UA compatibility

Telemetry publishes use QoS 0 (fire-and-forget) — sensor telemetry
can tolerate occasional loss; throughput matters more.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from aiomqtt import Client, Will
from aiomqtt import MqttError as AioMqttError
from cogniboiler_observability import MQTT_PUBLISH_ERRORS, MQTT_PUBLISHED
from cogniboiler_runtime import MqttSession

from physics_engine.models import BoilerState
from physics_engine.plant import PlantSnapshot
from physics_engine.proto_mapping import (
    boiler_state_to_proto,
    boiler_to_proto,
    plant_status_to_proto,
    turbine_state_to_proto,
    turbine_to_proto,
)
from physics_engine.runtime import (
    PhysicsRuntime,
    RuntimeUnavailableError,
    SimulationStatus,
)
from physics_engine.turbine import TurbineState

logger = logging.getLogger(__name__)


MQTT_ERRORS: tuple[type[BaseException], ...] = (AioMqttError,)
RECONNECT_DELAY_S: float = 5.0

# ─── Topic constants ──────────────────────────────────────────────────────────

TOPIC_BOILER: str = "sensors/boiler"
TOPIC_TURBINE: str = "sensors/turbine"
TOPIC_PLANT: str = "sensors/plant"
TOPIC_HEARTBEAT: str = "sensors/system/heartbeat"
TOPIC_AVAILABILITY: str = "status/physics-engine"


# ─── Publisher config ─────────────────────────────────────────────────────────


@dataclass
class MQTTConfig:
    """MQTT broker connection parameters."""

    host: str = "localhost"
    port: int = 1883
    keepalive: int = 60  # seconds
    client_id: str = "physics-engine"
    interval_s: float = 0.1  # publish every 100 ms = 10 Hz
    username: str | None = None
    password: str | None = None


# ─── Publisher ────────────────────────────────────────────────────────────────


class MQTTPublisher:
    """
    Async MQTT publisher for plant telemetry.

    Usage (production):
        pub = MQTTPublisher(MQTTConfig(host="mosquitto", port=1883))
        await pub.mirror_runtime(runtime)   # blocks, reconnects on its own

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

    async def _publish(
        self,
        client: Client,
        topic: str,
        payload: bytes | str,
        *,
        qos: int = 0,
        retain: bool = False,
    ) -> None:
        try:
            await client.publish(topic, payload, qos=qos, retain=retain)
        except MQTT_ERRORS:
            # aiomqtt does not reconnect by itself: a failed publish almost always means
            # the connection is gone, so the session ends and mirror_runtime reconnects.
            self._errors += 1
            MQTT_PUBLISH_ERRORS.labels(topic).inc()
            raise
        self._published += 1
        MQTT_PUBLISHED.labels(topic).inc()

    async def publish_boiler(
        self,
        client: Client,
        state: BoilerState,
    ) -> None:
        """
        Serialize BoilerState to BoilerStateMsg and publish to sensors/boiler.

        One MQTT message per call, QoS 0.
        """
        payload = boiler_state_to_proto(state).SerializeToString()
        await self._publish(client, TOPIC_BOILER, payload)

    async def publish_turbine(
        self,
        client: Client,
        state: TurbineState,
    ) -> None:
        """
        Serialize TurbineState to TurbineStateMsg and publish to sensors/turbine.

        One MQTT message per call, QoS 0.
        """
        payload = turbine_state_to_proto(state).SerializeToString()
        await self._publish(client, TOPIC_TURBINE, payload)

    async def publish_heartbeat(self, client: Client) -> None:
        """Publish system sync heartbeat with current timestamp."""
        payload = str(int(time.time() * 1000)).encode()
        await self._publish(client, TOPIC_HEARTBEAT, payload)

    async def publish_availability(self, client: Client, status: str) -> None:
        """Publish retained availability state for broker-side liveness tracking."""
        await self._publish(client, TOPIC_AVAILABILITY, status, qos=1, retain=True)

    async def publish_snapshot(
        self,
        client: Client,
        snapshot: PlantSnapshot,
        status: SimulationStatus,
    ) -> None:
        """
        Publish one plant snapshot: plant status, boiler, turbine and heartbeat.

        Plant status goes first: it carries the scenario and fault labels, so a recorder
        can label the boiler and turbine values of the same step.
        """
        await self._publish(
            client,
            TOPIC_PLANT,
            plant_status_to_proto(snapshot, status).SerializeToString(),
        )
        await self._publish(
            client, TOPIC_BOILER, boiler_to_proto(snapshot).SerializeToString()
        )
        await self._publish(
            client, TOPIC_TURBINE, turbine_to_proto(snapshot).SerializeToString()
        )
        await self.publish_heartbeat(client)

    # ─── Context manager ─────────────────────────────────────────────────────

    def connected(self) -> Client:
        """
        Return an aiomqtt Client context manager.

        Usage:
            async with publisher.connected() as client:
                await publisher.publish_boiler(client, state)
        """
        return Client(
            hostname=self.config.host,
            port=self.config.port,
            keepalive=self.config.keepalive,
            identifier=self.config.client_id,  # aiomqtt uses 'identifier'
            username=self.config.username,
            password=self.config.password,
            will=Will(TOPIC_AVAILABILITY, payload="offline", qos=1, retain=True),
        )

    # ─── Mirror of the live runtime ───────────────────────────────────────────

    async def mirror_runtime(self, runtime: PhysicsRuntime) -> None:
        """Publish every runtime snapshot, reconnecting for as long as the runtime lives.

        A runtime that has stopped is not a broker failure: there is nothing left to
        mirror, so the session ends instead of retrying forever.
        """
        session: MqttSession[Client] = MqttSession(
            self.connected,
            name="Physics mirror",
            reconnect_delay_s=RECONNECT_DELAY_S,
            logger=logger,
        )
        last_sequence = -1

        async def mirror(client: Client) -> None:
            nonlocal last_sequence
            await self.publish_availability(client, "online")
            while True:
                last_sequence, snapshot = await runtime.wait_for_update(last_sequence)
                await self.publish_snapshot(
                    client, snapshot, runtime.simulation_status()
                )

        await session.run(mirror, fatal=(RuntimeUnavailableError,))
