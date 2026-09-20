"""One broker session per service, and one way to lose it and get it back.

Every service that speaks MQTT had the same outer loop written out by hand: open a
session, work with the live client, and when the broker goes away say so, wait, and try
again — forever, without spinning and without a wall of identical warnings. Five copies
of that loop had already drifted apart (one of them, in physics-engine, swallowed the
failure and published into a dead link until 2026-09-19). This is the loop, written once.

The client itself is still built by the caller: the address, the credentials, the will and
whether the session is persistent are the service's decisions, and building it there also
lets a service's own tests hand in a fake broker.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Sequence
from contextlib import AbstractAsyncContextManager
from typing import Protocol

DEFAULT_RECONNECT_DELAY_S: float = 5.0

_LOGGER = logging.getLogger(__name__)


class SupportsSubscribe(Protocol):
    """The part of an MQTT client `subscribe_all` needs."""

    async def subscribe(self, topic: str, qos: int) -> object: ...


async def subscribe_all(
    client: SupportsSubscribe, subscriptions: Sequence[tuple[str, int]]
) -> None:
    """Subscribe to every (topic, qos) of a service's contract, in order."""
    for topic, qos in subscriptions:
        await client.subscribe(topic, qos=qos)


class MqttSession[ClientT]:
    """A broker session that outlives any one connection.

    `run` opens a client, hands it to the work, and when anything fails logs it once,
    waits `reconnect_delay_s` and opens the next one. Only cancellation ends it:
    `asyncio.CancelledError` is not an `Exception`, so it passes straight through.

    A failure that repeats is logged once as a warning and then at debug level, so a
    broker that stays down for an hour costs one line, not seven hundred; the line that
    says the session is back only appears if one was lost.
    """

    def __init__(
        self,
        open_client: Callable[[], AbstractAsyncContextManager[ClientT]],
        *,
        name: str,
        reconnect_delay_s: float = DEFAULT_RECONNECT_DELAY_S,
        logger: logging.Logger | None = None,
    ) -> None:
        if reconnect_delay_s < 0.0:
            raise ValueError("reconnect_delay_s must be >= 0")
        self._open_client = open_client
        self._name = name
        self._reconnect_delay_s = reconnect_delay_s
        self._logger = logger or _LOGGER
        self._connected = False
        self._failing = False
        self._failures = 0

    @property
    def connected(self) -> bool:
        """True between a successful connect and the end of that session."""
        return self._connected

    @property
    def failures(self) -> int:
        """How many sessions ended in a failure since this object was made."""
        return self._failures

    @property
    def reconnect_delay_s(self) -> float:
        return self._reconnect_delay_s

    async def run(
        self,
        work: Callable[[ClientT], Awaitable[None]],
        *,
        on_failure: Callable[[], Awaitable[None]] | None = None,
        fatal: tuple[type[BaseException], ...] = (),
    ) -> None:
        """Keep a session open and `work` running inside it until cancelled.

        `on_failure` runs after a failed session and before the wait — where a service
        flushes what it had buffered for the connection that just died.

        `fatal` names the failures that are not the broker's: when what the session
        exists to feed is gone, reconnecting to the broker forever would only hide it,
        so the loop says so at error level and returns.
        """
        while True:
            try:
                async with self._open_client() as client:
                    self._announce_connected()
                    self._connected = True
                    try:
                        await work(client)
                    finally:
                        self._connected = False
            except fatal as exc:
                self._logger.error("%s: MQTT session stopped: %s", self._name, exc)
                return
            except Exception as exc:
                self._failures += 1
                self._announce_failure(exc)
                if on_failure is not None:
                    await on_failure()
            else:
                # The work returned without an error: the broker closed the session
                # politely, or the message stream simply ended. Waiting before the next
                # attempt is what keeps this from becoming a busy loop.
                self._logger.info("%s: MQTT session ended; reconnecting", self._name)
            await asyncio.sleep(self._reconnect_delay_s)

    def _announce_connected(self) -> None:
        if self._failing:
            self._logger.info("%s: MQTT connection is back", self._name)
        else:
            self._logger.info("%s: connected to MQTT", self._name)
        self._failing = False

    def _announce_failure(self, exc: Exception) -> None:
        message = "%s: MQTT error: %s — retrying every %.0f s"
        if self._failing:
            self._logger.debug(message, self._name, exc, self._reconnect_delay_s)
        else:
            self._logger.warning(message, self._name, exc, self._reconnect_delay_s)
        self._failing = True
