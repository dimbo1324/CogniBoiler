"""
Upstreams of the WebSocket channels.

- telemetry: the PhysicsService state stream;
- plc: PLCService status polled at a fixed interval, and PLC events from `plc/events`;
- alarms: alarm changes from `alarms/changes`, published by alert-manager.

Each source reconnects on its own after a delay and logs once per outage, not once per
retry. A message that cannot be turned into a frame (a NaN from a diverging physics step,
an enum value this gateway does not know yet) is skipped with one warning per run of bad
messages; anything else a source did not expect is an error with its traceback, after
which the source starts over. Only cancellation ends a source.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable
from typing import Any
from uuid import uuid4

import grpc
from aiomqtt import Client
from cogniboiler_runtime import MqttSession, OutageLog, subscribe_all
from pydantic import BaseModel

from api_gateway.clients import PhysicsGatewayClient, PLCGatewayClient
from api_gateway.plant_state import plant_state
from api_gateway.plc_state import plc_status_from_proto
from api_gateway.realtime.hub import Channel, RealtimeHub

logger = logging.getLogger(__name__)

RECONNECT_DELAY_S = 3.0
TOPIC_PLC_EVENTS = "plc/events"
TOPIC_ALARM_CHANGES = "alarms/changes"
# Both topics are published at QoS 1 and matter to the operator's screen.
EVENT_QOS = 1


class _Frames:
    """Publishes one kind of frame; a message that cannot become one is skipped."""

    def __init__(
        self, hub: RealtimeHub, channel: Channel, kind: str, source: str
    ) -> None:
        self._hub = hub
        self._channel = channel
        self._kind = kind
        self._rejects = OutageLog(logger, f"Realtime source {source}: valid frames")

    def publish[M](self, convert: Callable[[M], BaseModel], message: M) -> None:
        try:
            data = convert(message).model_dump(mode="json")
            self._hub.publish(self._channel, self._kind, data)
        except (ValueError, TypeError) as exc:
            # pydantic's ValidationError is a ValueError, and so is a NaN the strict
            # JSON encoder refuses.
            self._rejects.failed(exc)
            return
        self._rejects.recovered()


class _Defects:
    """A failure nobody planned for: an error with traceback once, until a success."""

    def __init__(self, source: str) -> None:
        self._source = source
        self._failing = False

    def failed(self) -> None:
        if self._failing:
            logger.debug("Realtime source %s failed again", self._source, exc_info=True)
            return
        self._failing = True
        logger.error(
            "Realtime source %s failed; restarting it in %.0f s",
            self._source,
            RECONNECT_DELAY_S,
            exc_info=True,
        )

    def cleared(self) -> None:
        self._failing = False


def report_source_end(task: asyncio.Task[None]) -> None:
    """Done callback of a source task: a source that stops for any reason but
    cancellation is a defect, and must not stop silently."""
    if task.cancelled():
        return
    error = task.exception()
    if error is None:
        logger.error(
            "Realtime source %s returned; its channel is dead", task.get_name()
        )
    else:
        logger.error(
            "Realtime source %s died; its channel is dead",
            task.get_name(),
            exc_info=error,
        )


async def run_telemetry(hub: RealtimeHub, physics: PhysicsGatewayClient) -> None:
    outage = OutageLog(logger, "Realtime source telemetry")
    defects = _Defects("telemetry")
    frames = _Frames(hub, Channel.TELEMETRY, "state", "telemetry")
    while True:
        try:
            async for message in physics.stream_system_state(interval_s=0.0):
                outage.recovered()
                defects.cleared()
                frames.publish(plant_state, message)
        except grpc.RpcError as exc:
            outage.failed(exc)
        except Exception:
            defects.failed()
        await asyncio.sleep(RECONNECT_DELAY_S)


async def run_plc_status(
    hub: RealtimeHub, plc: PLCGatewayClient, interval_s: float
) -> None:
    outage = OutageLog(logger, "Realtime source plc status")
    defects = _Defects("plc status")
    frames = _Frames(hub, Channel.PLC, "status", "plc status")
    while True:
        try:
            status = await plc.get_control_status()
            outage.recovered()
            defects.cleared()
            frames.publish(plc_status_from_proto, status)
        except grpc.RpcError as exc:
            outage.failed(exc)
            await asyncio.sleep(RECONNECT_DELAY_S)
            continue
        except Exception:
            defects.failed()
            await asyncio.sleep(RECONNECT_DELAY_S)
            continue
        await asyncio.sleep(interval_s)


def _json_object(payload: bytes | bytearray | Any) -> dict[str, Any] | None:
    if not isinstance(payload, bytes | bytearray):
        return None
    try:
        value = json.loads(payload)
    except UnicodeDecodeError, json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


async def run_mqtt_events(
    hub: RealtimeHub,
    host: str,
    port: int,
    username: str | None = None,
    password: str | None = None,
) -> None:
    """Forward the two JSON topics the console listens to, across broker outages.

    The gateway is a reader here: a payload that is not a JSON object is dropped with a
    debug line, never passed on to the browser as-is.
    """
    routes: dict[str, tuple[Channel, str]] = {
        TOPIC_PLC_EVENTS: (Channel.PLC, "event"),
        TOPIC_ALARM_CHANGES: (Channel.ALARMS, "change"),
    }

    def open_client() -> Client:
        return Client(
            hostname=host,
            port=port,
            identifier=f"api-gateway-{uuid4().hex[:12]}",
            username=username,
            password=password,
        )

    async def consume(client: Client) -> None:
        await subscribe_all(client, [(topic, EVENT_QOS) for topic in routes])
        async for message in client.messages:
            route = routes.get(str(message.topic))
            payload = _json_object(message.payload)
            if route is None or payload is None:
                logger.debug("Ignored MQTT message on %s", message.topic)
                continue
            hub.publish(route[0], route[1], payload)

    session: MqttSession[Client] = MqttSession(
        open_client,
        name="Realtime MQTT events",
        reconnect_delay_s=RECONNECT_DELAY_S,
        logger=logger,
    )
    await session.run(consume)
