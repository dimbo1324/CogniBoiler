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

import json
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from aiomqtt import Client, Will
from cogniboiler_observability import MQTT_PUBLISH_ERRORS, MQTT_PUBLISHED
from cogniboiler_runtime import (
    DEFAULT_QUEUE_LIMIT,
    DEFAULT_RECONNECT_DELAY_S,
    MqttSession,
    QueuedMessage,
    QueuedMqttPublisher,
    now_ms,
)
from cogniboiler_runtime.topics import (
    TOPIC_ALERT_CRITICAL,
    TOPIC_ALERT_SNAPSHOT,
    TOPIC_ALERT_WARNING,
    TOPIC_PLC_EVENTS,
    TOPIC_STATUS_PLC_CONTROLLER,
)

from plc_controller.alarms import (
    SOURCE_SERVICE,
    AlarmCondition,
    AlarmTransition,
    Severity,
)

logger = logging.getLogger(__name__)

DEFAULT_MQTT_HOST: str = "localhost"
DEFAULT_MQTT_PORT: int = 1883

QUEUE_LIMIT: int = DEFAULT_QUEUE_LIMIT
RECONNECT_DELAY_S: float = DEFAULT_RECONNECT_DELAY_S
SNAPSHOT_INTERVAL_S: float = 10.0
CLOSE_DRAIN_TIMEOUT_S: float = 1.0


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
        username: str | None = None,
        password: str | None = None,
    ) -> None:
        self._host = host
        self._port = port
        self._username = username
        self._password = password
        self._enabled = enabled
        self._client_id = client_id
        self._active_conditions = active_conditions
        self._session: MqttSession[Client] = MqttSession(
            self._open_client,
            name="PLC publisher",
            reconnect_delay_s=RECONNECT_DELAY_S,
            logger=logger,
        )
        self._queue: QueuedMqttPublisher[Client] = QueuedMqttPublisher(
            self._session,
            name="PLC publisher",
            limit=QUEUE_LIMIT,
            availability_topic=TOPIC_STATUS_PLC_CONTROLLER,
            periodic=self._publish_snapshot,
            periodic_interval_s=SNAPSHOT_INTERVAL_S,
            on_published=_count_published,
            on_publish_error=_count_publish_error,
            logger=logger,
        )

    @property
    def dropped(self) -> int:
        """Messages lost to a full queue."""
        return self._queue.dropped

    @property
    def connected(self) -> bool:
        """True while the broker connection is up."""
        return self._session.connected

    @property
    def failures(self) -> int:
        """Broker connections that failed or dropped."""
        return self._session.failures

    # ─── Publishing API (non-blocking) ───────────────────────────────────────

    def publish_alarm(self, transition: AlarmTransition) -> None:
        self._enqueue(QueuedMessage(alarm_topic(transition), alarm_payload(transition)))

    def publish_event(self, event: PlcEvent) -> None:
        self._enqueue(QueuedMessage(TOPIC_PLC_EVENTS, event_payload(event)))

    def _enqueue(self, message: QueuedMessage) -> None:
        if self._enabled:
            self._queue.enqueue(message)

    # ─── Lifecycle ───────────────────────────────────────────────────────────

    def start(self) -> None:
        if self._enabled:
            self._queue.start()

    async def aclose(self) -> None:
        """Announce "offline", give the queue a moment to drain, then stop."""
        await self._queue.aclose(CLOSE_DRAIN_TIMEOUT_S)

    def _open_client(self) -> Client:
        return Client(
            hostname=self._host,
            port=self._port,
            identifier=self._client_id,
            username=self._username,
            password=self._password,
            will=Will(
                TOPIC_STATUS_PLC_CONTROLLER, payload="offline", qos=1, retain=True
            ),
        )

    async def _publish_snapshot(self, client: Client) -> None:
        """The active condition keys, on connect and every SNAPSHOT_INTERVAL_S."""
        payload = snapshot_payload(self._active_conditions(), now_ms())
        try:
            await client.publish(TOPIC_ALERT_SNAPSHOT, payload, qos=1)
        except Exception:
            _count_publish_error(TOPIC_ALERT_SNAPSHOT)
            raise
        _count_published(TOPIC_ALERT_SNAPSHOT)


def _count_published(topic: str) -> None:
    MQTT_PUBLISHED.labels(topic).inc()


def _count_publish_error(topic: str) -> None:
    MQTT_PUBLISH_ERRORS.labels(topic).inc()
