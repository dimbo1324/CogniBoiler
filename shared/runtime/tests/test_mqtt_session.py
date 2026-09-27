"""The session loop: what it survives, what it says, and what it never does.

The awkward cases matter more than the happy one here — this loop is the only thing
standing between a broker restart and a service that publishes into a dead link.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from types import TracebackType

import pytest
from cogniboiler_runtime.mqtt import (
    DEFAULT_RECONNECT_DELAY_S,
    MqttSession,
    reconnect_jitter,
    subscribe_all,
)

type Work = Callable[[FakeClient], Awaitable[None]]


class FakeClient:
    """A broker that can refuse the connection, or drop it while the work runs."""

    def __init__(self) -> None:
        self.subscriptions: list[tuple[str, int]] = []
        self.closed = False

    async def subscribe(self, topic: str, qos: int = 0) -> None:
        self.subscriptions.append((topic, qos))


def opener(
    *outcomes: Exception | None, clients: list[FakeClient] | None = None
) -> tuple[object, list[FakeClient]]:
    """A client factory that fails on connect exactly where `outcomes` says None is a
    successful connection, an exception is a broker that refuses it."""
    made: list[FakeClient] = clients if clients is not None else []
    attempts = iter(outcomes)

    @asynccontextmanager
    async def open_client() -> AsyncIterator[FakeClient]:
        outcome = next(attempts, None)
        if outcome is not None:
            raise outcome
        client = FakeClient()
        made.append(client)
        try:
            yield client
        finally:
            client.closed = True

    return open_client, made


async def run_until(
    session: MqttSession[FakeClient], work: Work, done: asyncio.Event
) -> None:
    """Run a session until `done`, then cancel it — the only way one ever ends."""
    task = asyncio.create_task(session.run(work))
    try:
        async with asyncio.timeout(5.0):
            await done.wait()
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


class TestReconnecting:
    async def test_a_failure_inside_the_work_opens_the_next_session(self) -> None:
        open_client, made = opener(None, None, None)
        session: MqttSession[FakeClient] = MqttSession(
            open_client, name="test", reconnect_delay_s=0.0
        )
        attempts = 0

        async def work(client: FakeClient) -> None:
            nonlocal attempts
            attempts += 1
            if attempts >= 3:
                raise asyncio.CancelledError
            raise OSError("connection lost")

        with pytest.raises(asyncio.CancelledError):
            await session.run(work)
        assert attempts == 3
        assert session.failures == 2
        assert [client.closed for client in made] == [True, True, True]
        assert session.connected is False

    async def test_a_broker_that_refuses_the_connection_is_retried(self) -> None:
        open_client, made = opener(OSError("refused"), OSError("refused"), None)
        session: MqttSession[FakeClient] = MqttSession(
            open_client, name="test", reconnect_delay_s=0.0
        )
        reached = asyncio.Event()

        async def work(client: FakeClient) -> None:
            reached.set()
            await asyncio.sleep(3600)

        task = asyncio.create_task(session.run(work))
        async with asyncio.timeout(5.0):
            await reached.wait()
        assert session.failures == 2
        assert session.connected is True
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert len(made) == 1

    async def test_cancellation_ends_the_loop_instead_of_being_retried(self) -> None:
        open_client, _ = opener(None)
        session: MqttSession[FakeClient] = MqttSession(
            open_client, name="test", reconnect_delay_s=0.0
        )

        async def work(client: FakeClient) -> None:
            raise asyncio.CancelledError

        with pytest.raises(asyncio.CancelledError):
            await session.run(work)
        assert session.failures == 0

    async def test_work_that_returns_waits_before_the_next_session(self) -> None:
        """A politely closed session must not become a busy loop."""
        open_client, made = opener()
        session: MqttSession[FakeClient] = MqttSession(
            open_client, name="test", reconnect_delay_s=3600.0
        )
        done = asyncio.Event()

        async def work(client: FakeClient) -> None:
            done.set()

        await run_until(session, work, done)
        # One session opened, and the next one is an hour away rather than immediate.
        assert len(made) == 1
        assert session.failures == 0

    async def test_the_failure_hook_runs_before_the_wait(self) -> None:
        open_client, _ = opener(None, None)
        session: MqttSession[FakeClient] = MqttSession(
            open_client, name="test", reconnect_delay_s=0.0
        )
        order: list[str] = []
        sessions = 0

        async def work(client: FakeClient) -> None:
            nonlocal sessions
            sessions += 1
            if sessions > 2:
                raise asyncio.CancelledError
            order.append("work")
            raise OSError("connection lost")

        async def on_failure() -> None:
            order.append("flush")

        with pytest.raises(asyncio.CancelledError):
            await session.run(work, on_failure=on_failure)
        assert order == ["work", "flush", "work", "flush"]


class TestAFailureThatIsNotTheBrokers:
    async def test_a_fatal_failure_ends_the_loop_instead_of_reconnecting(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        class RuntimeGoneError(RuntimeError):
            pass

        open_client, made = opener()
        session: MqttSession[FakeClient] = MqttSession(
            open_client, name="physics", reconnect_delay_s=0.0
        )

        async def work(client: FakeClient) -> None:
            raise RuntimeGoneError("the runtime stopped")

        with caplog.at_level(logging.ERROR, logger="cogniboiler_runtime.mqtt"):
            await session.run(work, fatal=(RuntimeGoneError,))
        assert len(made) == 1
        assert session.failures == 0
        assert "the runtime stopped" in caplog.text

    async def test_everything_else_is_still_the_brokers_fault(self) -> None:
        class RuntimeGoneError(RuntimeError):
            pass

        open_client, _ = opener()
        session: MqttSession[FakeClient] = MqttSession(
            open_client, name="physics", reconnect_delay_s=0.0
        )
        attempts = 0

        async def work(client: FakeClient) -> None:
            nonlocal attempts
            attempts += 1
            if attempts >= 2:
                raise asyncio.CancelledError
            raise OSError("connection lost")

        with pytest.raises(asyncio.CancelledError):
            await session.run(work, fatal=(RuntimeGoneError,))
        assert attempts == 2
        assert session.failures == 1


class TestLogging:
    async def test_a_broker_that_stays_down_costs_one_warning_not_a_wall(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        logger = logging.getLogger("test.session")
        open_client, _ = opener(*(OSError("down") for _ in range(20)))
        session: MqttSession[FakeClient] = MqttSession(
            open_client, name="historian", reconnect_delay_s=0.0, logger=logger
        )

        async def work(client: FakeClient) -> None:
            raise asyncio.CancelledError

        with caplog.at_level(logging.DEBUG, logger="test.session"):
            with pytest.raises(asyncio.CancelledError):
                await session.run(work)

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        debugs = [r for r in caplog.records if r.levelno == logging.DEBUG]
        assert len(warnings) == 1
        assert "historian" in warnings[0].getMessage()
        assert len(debugs) == 19
        infos = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
        assert any("connection is back" in message for message in infos)


class TestConstruction:
    def test_a_negative_delay_is_refused(self) -> None:
        open_client, _ = opener()
        with pytest.raises(ValueError, match="reconnect_delay_s"):
            MqttSession(open_client, name="test", reconnect_delay_s=-1.0)  # type: ignore[var-annotated]

    def test_the_default_delay_is_the_shared_one(self) -> None:
        open_client, _ = opener()
        session: MqttSession[FakeClient] = MqttSession(open_client, name="test")
        assert session.reconnect_delay_s == DEFAULT_RECONNECT_DELAY_S
        assert session.connected is False
        assert session.failures == 0


class TestSubscribeAll:
    async def test_every_topic_of_a_contract_is_subscribed_in_order(self) -> None:
        client = FakeClient()
        await subscribe_all(client, (("sensors/#", 0), ("alarms/changes", 1)))
        assert client.subscriptions == [("sensors/#", 0), ("alarms/changes", 1)]

    async def test_a_refused_subscription_is_not_swallowed(self) -> None:
        class Refusing(FakeClient):
            async def subscribe(self, topic: str, qos: int = 0) -> None:
                raise PermissionError(f"not allowed on {topic}")

        with pytest.raises(PermissionError):
            await subscribe_all(Refusing(), (("alerts/#", 1),))


class TestWithARealAsyncContextManager:
    """aiomqtt's Client is an object with __aenter__/__aexit__, not a generator."""

    async def test_an_object_context_manager_works_the_same(self) -> None:
        entered: list[str] = []

        class ClientObject:
            async def __aenter__(self) -> FakeClient:
                entered.append("in")
                return FakeClient()

            async def __aexit__(
                self,
                exc_type: type[BaseException] | None,
                exc: BaseException | None,
                tb: TracebackType | None,
            ) -> bool:
                entered.append("out")
                return False

        session: MqttSession[FakeClient] = MqttSession(
            ClientObject, name="test", reconnect_delay_s=3600.0
        )
        done = asyncio.Event()

        async def work(client: FakeClient) -> None:
            done.set()

        await run_until(session, work, done)
        assert entered == ["in", "out"]


def no_wait() -> float:
    """A jitter of zero: the loop still goes through its wait, but it lasts no time."""
    return 0.0


class ManualClock:
    """Monotonic time that moves only when a test says so."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def failing_work(*errors: BaseException) -> Work:
    """Work that raises each of `errors` in turn, then ends the test by cancelling."""
    pending = iter(errors)

    async def work(client: FakeClient) -> None:
        raise next(pending, asyncio.CancelledError())

    return work


def records(
    caplog: pytest.LogCaptureFixture, level: int, text: str = ""
) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.levelno == level and text in r.getMessage()]


class TestAnOutageThatReconnectsAndFailsAgain:
    async def test_a_failure_right_after_connect_warns_once(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # The broker accepts the connection and drops it at once, three times over: one
        # outage, not three, because no session lived long enough to count as healthy.
        open_client, made = opener(None, None, None, None)
        session: MqttSession[FakeClient] = MqttSession(
            open_client,
            name="historian",
            reconnect_delay_s=5.0,
            jitter=no_wait,
            clock=ManualClock(),
        )
        work = failing_work(*(OSError("connection lost") for _ in range(3)))
        with caplog.at_level(logging.DEBUG, logger="cogniboiler_runtime.mqtt"):
            with pytest.raises(asyncio.CancelledError):
                await session.run(work)

        assert len(made) == 4
        assert len(records(caplog, logging.WARNING)) == 1
        assert len(records(caplog, logging.INFO, "connection is back")) == 1
        assert len(records(caplog, logging.INFO, "connected to MQTT")) == 1
        assert session.failures == 3

    async def test_a_failure_after_a_healthy_session_is_a_new_outage(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        clock = ManualClock()
        open_client, _ = opener(None, None, None)
        session: MqttSession[FakeClient] = MqttSession(
            open_client,
            name="historian",
            reconnect_delay_s=5.0,
            jitter=no_wait,
            clock=clock,
        )
        sessions = 0

        async def work(client: FakeClient) -> None:
            nonlocal sessions
            sessions += 1
            if sessions > 2:
                raise asyncio.CancelledError
            clock.now += 60.0
            raise OSError("connection lost")

        with caplog.at_level(logging.DEBUG, logger="cogniboiler_runtime.mqtt"):
            with pytest.raises(asyncio.CancelledError):
                await session.run(work)

        assert len(records(caplog, logging.WARNING, "MQTT error")) == 2


class TestAFailureThatIsABug:
    async def test_a_non_broker_error_is_logged_with_its_traceback(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        open_client, made = opener(None, None, None, None)
        session: MqttSession[FakeClient] = MqttSession(
            open_client, name="gateway", reconnect_delay_s=0.0
        )
        work = failing_work(*(TypeError("a handler bug") for _ in range(3)))
        with caplog.at_level(logging.DEBUG, logger="cogniboiler_runtime.mqtt"):
            with pytest.raises(asyncio.CancelledError):
                await session.run(work)

        [error] = records(caplog, logging.ERROR)
        assert error.exc_info is not None
        assert error.exc_info[0] is TypeError
        assert "gateway" in error.getMessage()
        assert records(caplog, logging.WARNING) == []
        # The plant keeps running: the session is opened again after each failure.
        assert len(made) == 4
        assert session.failures == 3

    async def test_each_distinct_bug_gets_one_traceback(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        open_client, _ = opener(None, None, None, None, None)
        session: MqttSession[FakeClient] = MqttSession(
            open_client, name="gateway", reconnect_delay_s=0.0
        )
        work = failing_work(
            TypeError("one"), KeyError("two"), TypeError("three"), KeyError("four")
        )
        with caplog.at_level(logging.DEBUG, logger="cogniboiler_runtime.mqtt"):
            with pytest.raises(asyncio.CancelledError):
                await session.run(work)

        errors = records(caplog, logging.ERROR)
        assert [r.exc_info[0] for r in errors if r.exc_info] == [TypeError, KeyError]

    async def test_a_repeating_bug_does_not_log_a_connect_line_every_cycle(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # A retained message the handler cannot process comes back on every connect.
        open_client, _ = opener(*(None for _ in range(6)))
        session: MqttSession[FakeClient] = MqttSession(
            open_client,
            name="gateway",
            reconnect_delay_s=5.0,
            jitter=no_wait,
            clock=ManualClock(),
        )
        work = failing_work(*(ValueError("NaN in payload") for _ in range(5)))
        with caplog.at_level(logging.INFO, logger="cogniboiler_runtime.mqtt"):
            with pytest.raises(asyncio.CancelledError):
                await session.run(work)

        assert len(records(caplog, logging.INFO)) == 1
        assert len(records(caplog, logging.ERROR)) == 1

    async def test_the_mqtt_librarys_own_error_is_a_broker_failure_by_default(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        from aiomqtt import MqttError

        open_client, _ = opener(None, None)
        session: MqttSession[FakeClient] = MqttSession(
            open_client, name="plc", reconnect_delay_s=0.0
        )
        with caplog.at_level(logging.DEBUG, logger="cogniboiler_runtime.mqtt"):
            with pytest.raises(asyncio.CancelledError):
                await session.run(failing_work(MqttError("broker gone")))

        assert len(records(caplog, logging.WARNING, "MQTT error")) == 1
        assert records(caplog, logging.ERROR) == []

    async def test_the_caller_names_its_broker_errors(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        class LinkError(Exception):
            pass

        open_client, _ = opener(None, None, None)
        session: MqttSession[FakeClient] = MqttSession(
            open_client,
            name="plc",
            reconnect_delay_s=0.0,
            broker_errors=(LinkError,),
        )
        work = failing_work(LinkError("link down"), OSError("not a broker error here"))
        with caplog.at_level(logging.DEBUG, logger="cogniboiler_runtime.mqtt"):
            with pytest.raises(asyncio.CancelledError):
                await session.run(work)

        assert len(records(caplog, logging.WARNING, "link down")) == 1
        [error] = records(caplog, logging.ERROR)
        assert error.exc_info is not None
        assert error.exc_info[0] is OSError


class TestAFailingHook:
    async def test_a_failing_hook_does_not_end_the_session(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        open_client, made = opener(None, None, None)
        session: MqttSession[FakeClient] = MqttSession(
            open_client, name="historian", reconnect_delay_s=0.0
        )
        calls = 0

        async def on_failure() -> None:
            nonlocal calls
            calls += 1
            raise RuntimeError("hook failed")

        work = failing_work(OSError("lost"), OSError("lost"))
        with caplog.at_level(logging.ERROR, logger="cogniboiler_runtime.mqtt"):
            with pytest.raises(asyncio.CancelledError):
                await session.run(work, on_failure=on_failure)

        assert calls == 2
        assert len(made) == 3
        hook_errors = records(caplog, logging.ERROR, "on_failure")
        assert hook_errors
        assert hook_errors[0].exc_info is not None
        assert hook_errors[0].exc_info[0] is RuntimeError


class TestJitter:
    async def test_the_wait_is_the_delay_scaled_by_the_jitter(self) -> None:
        # An hour of delay scaled by zero: the second session opens at once, which it
        # could not if the delay were slept as configured.
        open_client, made = opener(OSError("refused"), None)
        factors: list[float] = []

        def jitter() -> float:
            factors.append(0.0)
            return 0.0

        session: MqttSession[FakeClient] = MqttSession(
            open_client, name="test", reconnect_delay_s=3600.0, jitter=jitter
        )
        done = asyncio.Event()

        async def work(client: FakeClient) -> None:
            done.set()
            await asyncio.sleep(3600)

        await run_until(session, work, done)
        assert len(made) == 1
        assert factors == [0.0]

    def test_the_default_jitter_spreads_retries_within_twenty_percent(self) -> None:
        samples = [reconnect_jitter() for _ in range(500)]
        assert all(0.8 <= sample <= 1.2 for sample in samples)
        assert len(set(samples)) > 1
