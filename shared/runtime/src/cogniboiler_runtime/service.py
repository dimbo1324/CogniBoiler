"""Run a service's main coroutine until SIGTERM or SIGINT, then let its cleanup run.

In a container the service is PID 1, and Linux delivers no signal with a default
disposition to PID 1; Python installs no SIGTERM handler of its own. So `docker compose
stop` used to wait its ten seconds and kill the process, and none of the `finally` blocks
that stop gRPC servers, flush buffers or announce "offline" ever ran.

Here a stop signal cancels the main task, the task's cleanup runs, and the process exits
normally. Windows event loops have no `add_signal_handler`, so there the handler goes
through the `signal` module instead; where neither is possible (not the main thread),
main simply runs without one.
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
from collections.abc import Awaitable, Callable, Sequence
from types import FrameType

logger = logging.getLogger(__name__)

STOP_SIGNALS: tuple[signal.Signals, ...] = (signal.SIGTERM, signal.SIGINT)
EXIT_INTERRUPTED = 130


def _install(
    loop: asyncio.AbstractEventLoop, sig: signal.Signals, stop: Callable[[int], None]
) -> Callable[[], None]:
    """Route `sig` to `stop`; return what undoes it."""
    try:
        loop.add_signal_handler(sig, stop, int(sig))
    except NotImplementedError, RuntimeError:
        pass
    else:
        return lambda: _remove_loop_handler(loop, sig)

    def handler(signum: int, frame: FrameType | None) -> None:
        loop.call_soon_threadsafe(stop, signum)

    try:
        previous = signal.signal(sig, handler)
    except ValueError:
        logger.debug("No handler for %s outside the main thread", sig.name)
        return lambda: None
    restore = signal.SIG_DFL if previous is None else previous

    def undo() -> None:
        signal.signal(sig, restore)

    return undo


def _remove_loop_handler(loop: asyncio.AbstractEventLoop, sig: signal.Signals) -> None:
    loop.remove_signal_handler(sig)


async def run_until_signalled[T](
    main: Callable[[], Awaitable[T]],
    *,
    signals: Sequence[signal.Signals] = STOP_SIGNALS,
) -> T | None:
    """Await `main()`; on a stop signal cancel it, await its cleanup and return None.

    An exception from `main` propagates, and so does a cancellation of this coroutine
    itself: only the signal counts as a clean stop.
    """
    loop = asyncio.get_running_loop()
    task = asyncio.ensure_future(main())
    received: list[int] = []

    def stop(signum: int) -> None:
        if not task.done():
            received.append(signum)
            task.cancel()

    undo = [_install(loop, sig, stop) for sig in signals]
    try:
        return await task
    except asyncio.CancelledError:
        current = asyncio.current_task()
        if not received or (current is not None and current.cancelling()):
            raise
        logger.info("Stopping on %s", signal.Signals(received[0]).name)
        return None
    finally:
        for restore in undo:
            restore()


def run_service(main: Callable[[], Awaitable[int | None]]) -> int:
    """Run `main` in a new event loop until it returns or a stop signal arrives.

    Returns the exit code: what `main` returned, 0 for None or a clean stop. On Windows
    the loop is a SelectorEventLoop, because aiomqtt needs `add_reader()`.
    """
    loop_factory = asyncio.SelectorEventLoop if sys.platform == "win32" else None
    try:
        code = asyncio.run(run_until_signalled(main), loop_factory=loop_factory)
    except KeyboardInterrupt:
        return EXIT_INTERRUPTED
    return 0 if code is None else code
