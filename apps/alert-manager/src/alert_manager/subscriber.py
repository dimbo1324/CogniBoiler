"""
MQTT subscriber that feeds alarm conditions and snapshots to the alarm processor.

The broker session is persistent (a fixed client id, clean_session off), so condition
messages published with QoS 1 while the alert manager restarts are delivered when it is
back. The broker acknowledges a QoS 1 message as soon as the client has it, so a message
given up here is lost for good: a database outage is waited out with the message kept and
the intake blocked, which also keeps the order. Only an error the database will repeat for
the same message gives it up.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Protocol

from aiomqtt import Client
from cogniboiler_observability import MQTT_RECEIVED
from cogniboiler_runtime import (
    DEFAULT_RECONNECT_DELAY_S,
    MqttSession,
    subscribe_all,
    unsubscribe_all,
)
from cogniboiler_runtime.topics import (
    FILTER_ALERTS,
    TOPIC_ALERT_CRITICAL,
    TOPIC_ALERT_SNAPSHOT,
    TOPIC_ALERT_WARNING,
)
from sqlalchemy.exc import (
    DBAPIError,
    InterfaceError,
    OperationalError,
    SQLAlchemyError,
)
from sqlalchemy.exc import TimeoutError as PoolTimeoutError

from alert_manager.metrics import MESSAGES_FAILED, MESSAGES_REJECTED
from alert_manager.payloads import (
    ConditionReport,
    PayloadError,
    SnapshotReport,
    parse_condition,
    parse_snapshot,
)

logger = logging.getLogger(__name__)

RECONNECT_DELAY_S: float = DEFAULT_RECONNECT_DELAY_S
STORE_RETRY_DELAY_S: float = 1.0
STORE_RETRY_MAX_DELAY_S: float = 30.0
INCOMING_QUEUE_LIMIT: int = 10_000
SUBSCRIPTIONS: tuple[tuple[str, int], ...] = (
    (TOPIC_ALERT_WARNING, 1),
    (TOPIC_ALERT_CRITICAL, 1),
    (TOPIC_ALERT_SNAPSHOT, 1),
)
# What this persistent session subscribed to before it named its topics.
RETIRED_FILTERS: tuple[str, ...] = (FILTER_ALERTS,)
STALL_LIMIT_S: float = 60.0


def store_retry_delay_s(attempt: int) -> float:
    """The wait after a failed attempt: doubling from the first delay, capped."""
    exponent = min(max(attempt - 1, 0), 32)
    return min(STORE_RETRY_DELAY_S * 2.0**exponent, STORE_RETRY_MAX_DELAY_S)


def is_transient(exc: BaseException) -> bool:
    """A failure of the database connection rather than of this message.

    asyncpg raises connection refusals and command timeouts as they are, not wrapped
    in a SQLAlchemy error, hence OSError (TimeoutError is one).
    """
    if isinstance(exc, OperationalError | InterfaceError | PoolTimeoutError | OSError):
        return True
    return isinstance(exc, DBAPIError) and exc.connection_invalidated


class MessageHandler(Protocol):
    async def handle_condition(self, report: ConditionReport) -> None: ...

    async def handle_snapshot(self, report: SnapshotReport) -> None: ...


class AlertSubscriber:
    """Consume alarm messages from MQTT and hand them to the processor."""

    def __init__(
        self,
        mqtt_host: str = "localhost",
        mqtt_port: int = 1883,
        handler: MessageHandler | None = None,
        *,
        client_id: str = "alert-manager",
        username: str | None = None,
        password: str | None = None,
    ) -> None:
        self._host = mqtt_host
        self._port = mqtt_port
        self._username = username
        self._password = password
        self._handler = handler
        self._client_id = client_id
        self._busy_since: float | None = None
        self._session: MqttSession[Client] = MqttSession(
            self._open_client,
            name="AlertManager",
            reconnect_delay_s=RECONNECT_DELAY_S,
            logger=logger,
        )

    @property
    def connected(self) -> bool:
        """True while subscribed to the broker; the liveness file follows it."""
        return self._session.connected

    @property
    def stalled(self) -> bool:
        """True while one message has been in processing for STALL_LIMIT_S or more."""
        busy_since = self._busy_since
        return busy_since is not None and time.monotonic() - busy_since >= STALL_LIMIT_S

    @property
    def healthy(self) -> bool:
        """Connected, and not stuck behind a database that stopped answering."""
        return self.connected and not self.stalled

    async def _handle_message(self, topic: str, raw_payload: bytes) -> None:
        MQTT_RECEIVED.labels(topic).inc()
        handler = self._handler
        if handler is None:
            return
        self._busy_since = time.monotonic()
        try:
            await self._process(handler, topic, raw_payload)
        finally:
            self._busy_since = None

    async def _process(
        self, handler: MessageHandler, topic: str, raw_payload: bytes
    ) -> None:
        try:
            if topic == TOPIC_ALERT_SNAPSHOT:
                snapshot = parse_snapshot(raw_payload)
                await self._with_retries(
                    topic, lambda: handler.handle_snapshot(snapshot)
                )
            else:
                report = parse_condition(topic, raw_payload)
                await self._with_retries(
                    topic, lambda: handler.handle_condition(report)
                )
        except PayloadError as exc:
            MESSAGES_REJECTED.labels(exc.reason).inc()
            logger.warning("Alarm message on %s rejected: %s", topic, exc)
            return
        except SQLAlchemyError as exc:
            MESSAGES_FAILED.inc()
            logger.warning(
                "Alarm message on %s could not be stored and is dropped: %s", topic, exc
            )
            return
        except Exception:
            MESSAGES_FAILED.inc()
            logger.exception("Alarm message on %s could not be processed", topic)
            return

    @staticmethod
    async def _with_retries(
        topic: str, operation: Callable[[], Awaitable[None]]
    ) -> None:
        attempt = 0
        while True:
            attempt += 1
            try:
                await operation()
            except Exception as exc:
                if not is_transient(exc):
                    raise
                delay = store_retry_delay_s(attempt)
                log = logger.warning if attempt == 1 else logger.debug
                log(
                    "Alarm intake paused, database unavailable (%s: %s); "
                    "the message from %s is kept, retrying in %.0f s",
                    type(exc).__name__,
                    exc,
                    topic,
                    delay,
                )
                await asyncio.sleep(delay)
                continue
            if attempt > 1:
                logger.info(
                    "Alarm message from %s stored again after %d attempts",
                    topic,
                    attempt,
                )
            return

    def _open_client(self) -> Client:
        return Client(
            hostname=self._host,
            port=self._port,
            identifier=self._client_id,
            username=self._username,
            password=self._password,
            clean_session=False,
            max_queued_incoming_messages=INCOMING_QUEUE_LIMIT,
        )

    async def _consume(self, client: Client) -> None:
        """One session: subscribe, then hand every payload to the processor.

        The session is persistent and the subscription is QoS 1, so conditions published
        while the alert manager was away are delivered once it is back.
        """
        await unsubscribe_all(client, RETIRED_FILTERS)
        await subscribe_all(client, SUBSCRIPTIONS)
        logger.info(
            "AlertManager subscribed to %s on %s:%d",
            ", ".join(topic for topic, _ in SUBSCRIPTIONS),
            self._host,
            self._port,
        )
        async for message in client.messages:
            payload = message.payload
            if not isinstance(payload, bytes | bytearray):
                MQTT_RECEIVED.labels(str(message.topic)).inc()
                MESSAGES_REJECTED.labels("not_bytes").inc()
                logger.warning("Alarm message on %s is not bytes", message.topic)
                continue
            await self._handle_message(str(message.topic), bytes(payload))

    async def run(self) -> None:
        """Consume alarm messages for as long as the service lives."""
        await self._session.run(self._consume)
