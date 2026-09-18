"""
Upstreams of the WebSocket channels.

- telemetry: the PhysicsService state stream;
- plc: PLCService status polled at a fixed interval, and PLC events from `plc/events`;
- alarms: alarm changes from `alarms/changes`, published by alert-manager.

Each source reconnects on its own after a delay and logs once per outage, not once per
retry.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any
from uuid import uuid4

import grpc
from aiomqtt import Client, MqttError

from api_gateway.clients import PhysicsGatewayClient, PLCGatewayClient
from api_gateway.plant_state import plant_state
from api_gateway.plc_state import plc_status_from_proto
from api_gateway.realtime.hub import Channel, RealtimeHub

logger = logging.getLogger(__name__)

RECONNECT_DELAY_S = 3.0
TOPIC_PLC_EVENTS = "plc/events"
TOPIC_ALARM_CHANGES = "alarms/changes"


class _OutageLog:
    """Logs the first failure of an outage and the recovery, nothing in between."""

    def __init__(self, source: str) -> None:
        self._source = source
        self._down = False

    def failed(self, exc: BaseException) -> None:
        if not self._down:
            logger.warning("Realtime source %s unavailable: %s", self._source, exc)
            self._down = True

    def recovered(self) -> None:
        if self._down:
            logger.info("Realtime source %s recovered", self._source)
            self._down = False


async def run_telemetry(hub: RealtimeHub, physics: PhysicsGatewayClient) -> None:
    outage = _OutageLog("telemetry")
    while True:
        try:
            async for message in physics.stream_system_state(interval_s=0.0):
                outage.recovered()
                hub.publish(
                    Channel.TELEMETRY,
                    "state",
                    plant_state(message).model_dump(mode="json"),
                )
        except grpc.RpcError as exc:
            outage.failed(exc)
        await asyncio.sleep(RECONNECT_DELAY_S)


async def run_plc_status(
    hub: RealtimeHub, plc: PLCGatewayClient, interval_s: float
) -> None:
    outage = _OutageLog("plc status")
    while True:
        try:
            status = await plc.get_control_status()
        except grpc.RpcError as exc:
            outage.failed(exc)
            await asyncio.sleep(RECONNECT_DELAY_S)
            continue
        outage.recovered()
        hub.publish(
            Channel.PLC, "status", plc_status_from_proto(status).model_dump(mode="json")
        )
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
    outage = _OutageLog("mqtt events")
    routes: dict[str, tuple[Channel, str]] = {
        TOPIC_PLC_EVENTS: (Channel.PLC, "event"),
        TOPIC_ALARM_CHANGES: (Channel.ALARMS, "change"),
    }
    while True:
        try:
            async with Client(
                hostname=host,
                port=port,
                identifier=f"api-gateway-{uuid4().hex[:12]}",
                username=username,
                password=password,
            ) as client:
                for topic in routes:
                    await client.subscribe(topic, qos=1)
                outage.recovered()
                async for message in client.messages:
                    route = routes.get(str(message.topic))
                    payload = _json_object(message.payload)
                    if route is None or payload is None:
                        logger.debug("Ignored MQTT message on %s", message.topic)
                        continue
                    hub.publish(route[0], route[1], payload)
        except MqttError as exc:
            outage.failed(exc)
        await asyncio.sleep(RECONNECT_DELAY_S)
