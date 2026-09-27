"""The historian's own counters: logged and stored in InfluxDB every interval."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping
from typing import Protocol

from historian.points import build_stats_point
from historian.writer import PointLike

logger = logging.getLogger(__name__)

STATS_INTERVAL_S: float = 30.0


class CountsMessages(Protocol):
    @property
    def stats(self) -> Mapping[str, int]: ...


class StoresPoints(Protocol):
    @property
    def errors(self) -> int: ...

    def write_point(self, point: PointLike) -> int: ...


async def report_stats(
    subscriber: CountsMessages, writer: StoresPoints, interval_s: float
) -> None:
    """Every interval, log the counters and write them as a historian_stats point."""
    while True:
        await asyncio.sleep(interval_s)
        stats = subscriber.stats
        logger.info(
            "Stats: received=%d stored=%d skipped=%d writer_errors=%d",
            stats["received"],
            stats["stored"],
            stats["skipped"],
            writer.errors,
        )
        point = build_stats_point(stats, writer.errors)
        await asyncio.to_thread(writer.write_point, point)
