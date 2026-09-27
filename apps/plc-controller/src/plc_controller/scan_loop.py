"""The PLC's clock: one scan for every plant state PhysicsService publishes.

The loop tells two kinds of failure apart. A failure of the plant link — a gRPC error, a
refused connection, the stream ending — is a warning, logged once per outage, and the
stream is opened again after a pause. A fault of the PLC itself is an error with its
traceback, logged once per exception type; a fault in one scan skips that scan and the
healthy stream carries on.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import Awaitable, Callable

import cogniboiler_pb2 as pb2

from plc_controller.client import PhysicsClient
from plc_controller.forwarding import LINK_ERRORS, link_error
from plc_controller.metrics import SCAN_SECONDS

logger = logging.getLogger(__name__)

type Scan = Callable[[pb2.SystemStateMsg], Awaitable[None]]


class ScanLoop:
    """Subscribes to the plant's states and runs `scan` once for each of them."""

    def __init__(
        self,
        physics: PhysicsClient,
        scan: Scan,
        *,
        retry_delay_s: float,
        on_new_stream: Callable[[], None],
    ) -> None:
        self._physics = physics
        self._scan = scan
        self._retry_delay_s = retry_delay_s
        self._on_new_stream = on_new_stream
        self._task: asyncio.Task[None] | None = None
        self._failing = False
        self._faults_logged: set[type[Exception]] = set()
        self.error = ""
        self.link_up = False
        self.scans = 0
        self.scan_failures = 0
        self.stream_failures = 0
        self.last_scanned_step = -1

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run(), name="plc-scan-loop")

    async def close(self) -> None:
        task = self._task
        if task is None:
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        self._task = None

    async def _run(self) -> None:
        while True:
            # A new stream may reach a restarted plant, whose valves are not ours.
            self._on_new_stream()
            try:
                stream = self._physics.stream_system_state()
                async with contextlib.aclosing(stream) as states:
                    async for state in states:
                        self.link_up = True
                        await self._scan_one(state)
                raise ConnectionError("physics state stream ended")
            except LINK_ERRORS as exc:
                self._stream_failed(link_error(exc))
            except Exception as exc:
                self._stream_failed(type(exc).__name__)
                self._log_fault_once("PLC scan stream broke", exc)
            await asyncio.sleep(self._retry_delay_s)

    async def _scan_one(self, state: pb2.SystemStateMsg) -> None:
        started = time.perf_counter()
        try:
            await self._scan(state)
        except Exception as exc:
            self.scan_failures += 1
            self.error = f"scan failed: {type(exc).__name__}"
            self._log_fault_once(
                f"PLC scan failed on plant step {state.simulation.step_count}", exc
            )
            return
        SCAN_SECONDS.observe(time.perf_counter() - started)
        self.scans += 1
        self.last_scanned_step = state.simulation.step_count
        if self._failing:
            logger.info("PLC scan stream restored")
        self._failing = False
        self.error = ""

    def _stream_failed(self, reason: str) -> None:
        self.link_up = False
        self.stream_failures += 1
        self.error = reason
        if not self._failing:
            logger.warning(
                "PLC scan stream failed: %s — retrying every %.2fs",
                reason,
                self._retry_delay_s,
            )
        self._failing = True

    def _log_fault_once(self, message: str, exc: Exception) -> None:
        """A fault of the PLC itself: an error with its traceback, once per kind."""
        kind = type(exc)
        if kind in self._faults_logged:
            logger.debug("%s: %s (logged before)", message, kind.__name__)
            return
        self._faults_logged.add(kind)
        logger.error("%s", message, exc_info=exc)
