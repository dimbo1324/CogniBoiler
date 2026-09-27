"""Running a service until it is told to stop, so its cleanup actually runs.

In a container the service is PID 1, and Python installs no SIGTERM handler: without one,
`docker compose stop` waits ten seconds and kills it, and no `finally` ever runs.
"""

from __future__ import annotations

import asyncio
import os
import signal
import sys
from collections.abc import Callable
from typing import Any

import pytest
from cogniboiler_runtime import service
from cogniboiler_runtime.service import (
    STOP_SIGNALS,
    run_service,
    run_until_signalled,
)


class Service:
    """A main coroutine that runs until cancelled and records its cleanup."""

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.cleaned_up = False

    async def main(self) -> int:
        self.started.set()
        try:
            await asyncio.Event().wait()
        finally:
            self.cleaned_up = True
        return 0


def capture_loop_handlers(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[int, Callable[[], None]]:
    """Record what the helper registers on the loop instead of touching the process."""
    loop = asyncio.get_running_loop()
    handlers: dict[int, Callable[[], None]] = {}

    def add(sig: int, callback: Callable[..., None], *args: Any) -> None:
        handlers[sig] = lambda: callback(*args)

    def remove(sig: int) -> bool:
        return handlers.pop(sig, None) is not None

    monkeypatch.setattr(loop, "add_signal_handler", add)
    monkeypatch.setattr(loop, "remove_signal_handler", remove)
    return handlers


class TestRunUntilSignalled:
    async def test_a_main_that_finishes_returns_its_result(self) -> None:
        async def main() -> int:
            return 7

        assert await run_until_signalled(main) == 7

    async def test_a_signal_cancels_main_runs_its_cleanup_and_returns_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        handlers = capture_loop_handlers(monkeypatch)
        svc = Service()
        runner = asyncio.create_task(run_until_signalled(svc.main))
        await svc.started.wait()
        assert set(handlers) == {int(sig) for sig in STOP_SIGNALS}

        handlers[int(signal.SIGTERM)]()

        assert await runner is None
        assert svc.cleaned_up is True
        # The handlers are the helper's only while it runs.
        assert handlers == {}

    async def test_without_loop_signal_support_it_falls_back_to_signal_handlers(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Windows event loops have no add_signal_handler.
        loop = asyncio.get_running_loop()

        def unsupported(*_: object) -> None:
            raise NotImplementedError

        monkeypatch.setattr(loop, "add_signal_handler", unsupported)
        installed: dict[int, Any] = {}

        def fake_signal(sig: int, handler: Any) -> Any:
            previous = installed.get(sig, signal.SIG_DFL)
            installed[sig] = handler
            return previous

        monkeypatch.setattr(service.signal, "signal", fake_signal)
        svc = Service()
        runner = asyncio.create_task(run_until_signalled(svc.main))
        await svc.started.wait()

        installed[int(signal.SIGTERM)](int(signal.SIGTERM), None)

        assert await runner is None
        assert svc.cleaned_up is True
        assert all(handler == signal.SIG_DFL for handler in installed.values())

    async def test_where_no_handler_can_be_installed_main_still_runs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Not the main thread: neither the loop nor the signal module will help.
        loop = asyncio.get_running_loop()

        def refuse_loop(*_: object) -> None:
            raise RuntimeError("not the main thread")

        def refuse_signal(*_: object) -> None:
            raise ValueError("signal only works in main thread")

        monkeypatch.setattr(loop, "add_signal_handler", refuse_loop)
        monkeypatch.setattr(service.signal, "signal", refuse_signal)

        async def main() -> str:
            return "ran"

        assert await run_until_signalled(main) == "ran"

    async def test_cancelling_the_helper_itself_is_not_swallowed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        capture_loop_handlers(monkeypatch)
        svc = Service()
        runner = asyncio.create_task(run_until_signalled(svc.main))
        await svc.started.wait()
        runner.cancel()
        with pytest.raises(asyncio.CancelledError):
            await runner
        assert svc.cleaned_up is True

    async def test_an_error_in_main_propagates(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        handlers = capture_loop_handlers(monkeypatch)

        async def main() -> None:
            raise RuntimeError("could not start")

        with pytest.raises(RuntimeError, match="could not start"):
            await run_until_signalled(main)
        assert handlers == {}

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX signal delivery")
    async def test_a_real_sigterm_stops_the_service_cleanly(self) -> None:
        svc = Service()
        runner = asyncio.create_task(run_until_signalled(svc.main))
        await svc.started.wait()
        os.kill(os.getpid(), signal.SIGTERM)
        async with asyncio.timeout(5.0):
            assert await runner is None
        assert svc.cleaned_up is True


class TestRunService:
    def test_the_exit_code_is_mains_or_zero(self) -> None:
        async def three() -> int:
            return 3

        async def nothing() -> None:
            return None

        assert run_service(three) == 3
        assert run_service(nothing) == 0

    def test_windows_gets_the_selector_loop_that_aiomqtt_needs(self) -> None:
        seen: list[type[asyncio.AbstractEventLoop]] = []

        async def main() -> None:
            seen.append(type(asyncio.get_running_loop()))

        run_service(main)
        if sys.platform == "win32":
            assert issubclass(seen[0], asyncio.SelectorEventLoop)
        else:
            assert seen
