"""
What the PLC tells the rest of the platform over MQTT.

    alerts/warning, alerts/critical   JSON alarm condition, state "active" or "cleared"
    alerts/snapshot                   JSON list of every active condition key, sent on
                                      connect and periodically, so a consumer that missed
                                      a "cleared" message can reconcile
    plc/events                        JSON PLC event: mode change, trip, reset, targets
    status/plc-controller             retained "online" / "offline" (MQTT will)

One persistent connection carries everything. Messages queue while the broker is away and
go out in order once it is back; the queue is bounded, so a long outage costs the oldest
messages, never unbounded memory.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from aiomqtt import Client, MqttError, Will
from cogniboiler_observability import MQTT_PUBLISHED

from plc_controller.alarms import (
    SOURCE_SERVICE,
    AlarmCondition,
    AlarmTransition,
    Severity,
)

logger = logging.getLogger(__name__)

TOPIC_ALERT_WARNING: str = "alerts/warning"
TOPIC_ALERT_CRITICAL: str = "alerts/critical"
TOPIC_ALERT_SNAPSHOT: str = "alerts/snapshot"
TOPIC_PLC_EVENTS: str = "plc/events"
TOPIC_AVAILABILITY: str = "status/plc-controller"

QUEUE_LIMIT: int = 1000
RECONNECT_DELAY_S: float = 5.0
SNAPSHOT_INTERVAL_S: float = 10.0
CLOSE_DRAIN_TIMEOUT_S: float = 1.0


def now_ms() -> int:
    """Current UTC epoch milliseconds."""
    return int(time.time() * 1000)


class PlcEventKind(StrEnum):
    MODE_CHANGED = "mode_changed"
    INTERLOCK_TRIPPED = "interlock_tripped"
    MANUAL_TRIP = "manual_trip"
    ESTOP_RESET = "estop_reset"
    ESTOP_RESET_REFUSED = "estop_reset_refused"
    LOAD_DEMAND_CHANGED = "load_demand_changed"
    SETPOINTS_CHANGED = "setpoints_changed"
    RUN_CHANGED = "run_changed"


@dataclass(frozen=True)
class PlcEvent:
    """Something the PLC did or refused, with who asked for it."""

    kind: PlcEventKind
    operator_id: str
    detail: Mapping[str, Any] = field(default_factory=dict)
    timestamp_ms: int = field(default_factory=now_ms)


@dataclass(frozen=True)
class _Message:
    topic: str
    payload: bytes
    qos: int = 1
    retain: bool = False


def alarm_topic(transition: AlarmTransition) -> str:
    if transition.condition.rule.severity is Severity.CRITICAL:
        return TOPIC_ALERT_CRITICAL
    return TOPIC_ALERT_WARNING


def alarm_payload(transition: AlarmTransition) -> bytes:
    condition = transition.condition
    rule = condition.rule
    return json.dumps(
        {
            "alarm_id": f"{rule.key}:{transition.timestamp_ms}",
            "key": rule.key,
            "state": "active" if transition.active else "cleared",
            "source_service": SOURCE_SERVICE,
            "severity": rule.severity.value,
            "parameter": rule.parameter,
            "direction": rule.direction.value,
            "unit": rule.unit,
            "value": condition.value,
            "threshold": rule.threshold,
            "action": rule.action,
            "message": condition.message,
            "raised_at_ms": condition.since_ms,
            "timestamp_ms": transition.timestamp_ms,
        }
    ).encode("utf-8")


def snapshot_payload(active: Sequence[AlarmCondition], timestamp_ms: int) -> bytes:
    return json.dumps(
        {
            "source_service": SOURCE_SERVICE,
            "active_keys": [condition.key for condition in active],
            "timestamp_ms": timestamp_ms,
        }
    ).encode("utf-8")


def event_payload(event: PlcEvent) -> bytes:
    return json.dumps(
        {
            "event_id": f"{event.kind.value}:{event.timestamp_ms}",
            "kind": event.kind.value,
            "source_service": SOURCE_SERVICE,
            "operator_id": event.operator_id,
            "detail": dict(event.detail),
            "timestamp_ms": event.timestamp_ms,
        }
    ).encode("utf-8")


class PlcPublisher:
    """Queue-backed MQTT publisher; disabled, every publish is a no-op."""

    def __init__(
        self,
        host: str,
        port: int,
        *,
        enabled: bool = True,
        client_id: str = SOURCE_SERVICE,
        active_conditions: Callable[[], Sequence[AlarmCondition]] = tuple,
    ) -> None:
        self._host = host
        self._port = port
        self._enabled = enabled
        self._client_id = client_id
        self._active_conditions = active_conditions
        self._queue: deque[_Message] = deque()
        self._wakeup = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._failing = False
        self._dropped = 0

    @property
    def dropped(self) -> int:
        """Messages lost to a full queue."""
        return self._dropped

    # ─── Publishing API (non-blocking) ───────────────────────────────────────

    def publish_alarm(self, transition: AlarmTransition) -> None:
        self._enqueue(_Message(alarm_topic(transition), alarm_payload(transition)))

    def publish_event(self, event: PlcEvent) -> None:
        self._enqueue(_Message(TOPIC_PLC_EVENTS, event_payload(event)))

    def _enqueue(self, message: _Message) -> None:
        if not self._enabled:
            return
        if len(self._queue) >= QUEUE_LIMIT:
            self._queue.popleft()
            self._dropped += 1
            if self._dropped == 1 or self._dropped % 100 == 0:
                logger.error(
                    "PLC publish queue full: %d messages dropped so far", self._dropped
                )
        self._queue.append(message)
        self._wakeup.set()

    # ─── Lifecycle ───────────────────────────────────────────────────────────

    def start(self) -> None:
        if not self._enabled or (self._task is not None and not self._task.done()):
            return
        self._task = asyncio.create_task(self._run(), name="plc-mqtt-publisher")

    async def aclose(self) -> None:
        """Announce "offline", give the queue a moment to drain, then stop."""
        task = self._task
        if task is None:
            return
        self._enqueue(_Message(TOPIC_AVAILABILITY, b"offline", retain=True))
        deadline = time.monotonic() + CLOSE_DRAIN_TIMEOUT_S
        while self._queue and not self._failing and time.monotonic() < deadline:
            await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        self._task = None

    async def _run(self) -> None:
        while True:
            try:
                async with Client(
                    hostname=self._host,
                    port=self._port,
                    identifier=self._client_id,
                    will=Will(
                        TOPIC_AVAILABILITY, payload="offline", qos=1, retain=True
                    ),
                ) as client:
                    await client.publish(
                        TOPIC_AVAILABILITY, "online", qos=1, retain=True
                    )
                    if self._failing:
                        logger.info("PLC publisher reconnected to MQTT")
                    self._failing = False
                    await self._drain(client)
            except MqttError as exc:
                if not self._failing:
                    logger.warning(
                        "PLC publisher lost MQTT %s:%d: %s — retrying every %.0fs",
                        self._host,
                        self._port,
                        exc,
                        RECONNECT_DELAY_S,
                    )
                self._failing = True
                await asyncio.sleep(RECONNECT_DELAY_S)

    async def _drain(self, client: Client) -> None:
        next_snapshot = 0.0
        while True:
            while self._queue:
                message = self._queue[0]
                await client.publish(
                    message.topic,
                    message.payload,
                    qos=message.qos,
                    retain=message.retain,
                )
                MQTT_PUBLISHED.labels(message.topic).inc()
                self._queue.popleft()
            if time.monotonic() >= next_snapshot:
                await client.publish(
                    TOPIC_ALERT_SNAPSHOT,
                    snapshot_payload(self._active_conditions(), now_ms()),
                    qos=1,
                )
                MQTT_PUBLISHED.labels(TOPIC_ALERT_SNAPSHOT).inc()
                next_snapshot = time.monotonic() + SNAPSHOT_INTERVAL_S
            self._wakeup.clear()
            if self._queue:
                continue
            try:
                await asyncio.wait_for(
                    self._wakeup.wait(),
                    timeout=max(next_snapshot - time.monotonic(), 0.05),
                )
            except TimeoutError:
                pass
