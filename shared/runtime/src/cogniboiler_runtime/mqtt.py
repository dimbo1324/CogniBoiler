"""One broker session per service, and one way to lose it and get it back.

Every service that speaks MQTT had the same outer loop written out by hand: open a
session, work with the live client, and when the broker goes away say so, wait, and try
again — forever, without spinning and without a wall of identical warnings. Seven copies
of that loop had drifted apart: two warned on every retry instead of once per outage, one
kept no `connected` flag for its healthcheck, and one (physics-engine, until 2026-09-19)
swallowed the failure and published into a dead link. This is the loop, written once.

The client itself is still built by the caller: the address, the credentials, the will and
whether the session is persistent are the service's decisions, and building it there also
lets a service's own tests hand in a fake broker.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections.abc import Awaitable, Callable, Sequence
from contextlib import AbstractAsyncContextManager
from typing import Protocol

DEFAULT_RECONNECT_DELAY_S: float = 5.0
RECONNECT_JITTER: float = 0.2


def _library_errors() -> tuple[type[BaseException], ...]:
    # The package must import without the MQTT library, but every service that runs a
    # session has it, and its MqttError always means the link, never the work.
    try:
        from aiomqtt import MqttError
    except ImportError:
        return ()
    return (MqttError,)


DEFAULT_BROKER_ERRORS: tuple[type[BaseException], ...] = (
    OSError,
    *_library_errors(),
)

_LOGGER = logging.getLogger(__name__)


def reconnect_jitter() -> float:
    """A factor within ±20 % of 1, so sessions that lost one broker do not retry in step."""
    return random.uniform(1.0 - RECONNECT_JITTER, 1.0 + RECONNECT_JITTER)


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

    `run` opens a client, hands it to the work, and when anything fails logs it, waits
    about `reconnect_delay_s` and opens the next one. Only cancellation ends it:
    `asyncio.CancelledError` is not an `Exception`, so it passes straight through.

    A failure of the link (`broker_errors`) is an outage: its first failure is a warning
    and every repeat is debug, until a session has stayed up for `reconnect_delay_s` or
    ended cleanly. A broker that accepts the connection and drops it at once is still the
    same outage, not a new one per cycle.

    Any other exception is a defect in the work, not the broker's fault: it is logged at
    error level with its traceback, once per exception type, and the session is opened
    again so the plant keeps running.
    """

    def __init__(
        self,
        open_client: Callable[[], AbstractAsyncContextManager[ClientT]],
        *,
        name: str,
        reconnect_delay_s: float = DEFAULT_RECONNECT_DELAY_S,
        logger: logging.Logger | None = None,
        broker_errors: tuple[type[BaseException], ...] = DEFAULT_BROKER_ERRORS,
        jitter: Callable[[], float] = reconnect_jitter,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if reconnect_delay_s < 0.0:
            raise ValueError("reconnect_delay_s must be >= 0")
        self._open_client = open_client
        self._name = name
        self._reconnect_delay_s = reconnect_delay_s
        self._logger = logger or _LOGGER
        self._broker_errors = broker_errors
        self._jitter = jitter
        self._clock = clock
        self._connected = False
        self._connected_at: float | None = None
        self._failing = False
        self._quiet_connect = False
        self._reported_defects: set[type[BaseException]] = set()
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
        flushes what it had buffered for the connection that just died. A hook that
        raises is logged and the loop goes on.

        `fatal` names the failures that are not the broker's: when what the session
        exists to feed is gone, reconnecting to the broker forever would only hide it,
        so the loop says so at error level and returns.
        """
        while True:
            self._connected_at = None
            try:
                async with self._open_client() as client:
                    self._announce_connected()
                    self._connected_at = self._clock()
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
                if self._stayed_up():
                    self._settle()
                if isinstance(exc, self._broker_errors):
                    self._announce_outage(exc)
                else:
                    self._announce_defect(exc)
                if on_failure is not None:
                    await self._run_hook(on_failure)
            else:
                # The work returned without an error: the broker closed the session
                # politely, or the message stream simply ended. Waiting before the next
                # attempt is what keeps this from becoming a busy loop.
                self._settle()
                self._logger.info("%s: MQTT session ended; reconnecting", self._name)
            await asyncio.sleep(self._reconnect_delay_s * self._jitter())

    def _stayed_up(self) -> bool:
        if self._connected_at is None:
            return False
        return self._clock() - self._connected_at >= self._reconnect_delay_s

    def _settle(self) -> None:
        self._failing = False
        self._quiet_connect = False

    def _announce_connected(self) -> None:
        if self._quiet_connect:
            self._logger.debug("%s: connected to MQTT again", self._name)
        elif self._failing:
            self._logger.info("%s: MQTT connection is back", self._name)
        else:
            self._logger.info("%s: connected to MQTT", self._name)
        self._quiet_connect = True

    def _announce_outage(self, exc: Exception) -> None:
        message = "%s: MQTT error: %s — retrying every %.0f s"
        if self._failing:
            self._logger.debug(message, self._name, exc, self._reconnect_delay_s)
        else:
            self._logger.warning(message, self._name, exc, self._reconnect_delay_s)
            self._failing = True
            self._quiet_connect = False

    def _announce_defect(self, exc: Exception) -> None:
        kind = type(exc)
        if kind in self._reported_defects:
            self._logger.debug("%s: session work failed again: %r", self._name, exc)
            return
        self._reported_defects.add(kind)
        self._logger.error(
            "%s: session work failed; reconnecting in %.0f s",
            self._name,
            self._reconnect_delay_s,
            exc_info=exc,
        )

    async def _run_hook(self, hook: Callable[[], Awaitable[None]]) -> None:
        try:
            await hook()
        except Exception:
            self._logger.exception("%s: on_failure hook failed", self._name)
