"""A liveness file the container healthcheck reads.

The historian and the alert manager expose no port, so their health is a file whose
modification time the service refreshes only while it is connected to the MQTT broker.
The healthcheck runs `python -m cogniboiler_runtime.liveness <path>` and gets exit code 0
while the file is fresh. Both services had a byte-identical copy of this until 2026-09-20.

The file says "this service was alive and connected a moment ago" and nothing else. Every
way of being wrong about that — a missing file, a directory left by a mount, a stat that
is refused — counts as dead, because a healthcheck that passes while a service is deaf is
worse than no healthcheck at all.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import stat
import sys
import time
from collections.abc import Callable
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_INTERVAL_S = 5.0
DEFAULT_MAX_AGE_S = 30.0


class LivenessFile:
    def __init__(self, path: Path, interval_s: float = DEFAULT_INTERVAL_S) -> None:
        if interval_s <= 0.0:
            raise ValueError("interval_s must be > 0")
        self._path = path
        self._interval_s = interval_s
        self._write_failed = False

    def beat(self, healthy: bool) -> None:
        """Stamp the file while the service is healthy; leave it to age out when not."""
        if not healthy:
            return
        try:
            self._path.write_text(str(int(time.time())), encoding="ascii")
        except OSError as exc:
            # Keep beating: the file simply stays stale, the healthcheck fails and the
            # container is restarted — but say once why, or the restart looks unexplained.
            if not self._write_failed:
                logger.warning(
                    "Liveness file %s cannot be written: %s", self._path, exc
                )
            self._write_failed = True
            return
        if self._write_failed:
            logger.info("Liveness file %s can be written again", self._path)
        self._write_failed = False

    async def run(self, is_healthy: Callable[[], bool]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        while True:
            self.beat(is_healthy())
            await asyncio.sleep(self._interval_s)


def is_fresh(path: Path, max_age_s: float) -> bool:
    try:
        status = path.stat()
    except OSError:
        return False
    if not stat.S_ISREG(status.st_mode):
        return False
    return time.time() - status.st_mtime <= max_age_s


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m cogniboiler_runtime.liveness",
        description="Exit 0 when the liveness file was refreshed recently.",
    )
    parser.add_argument("path", type=Path)
    parser.add_argument("--max-age-s", type=float, default=DEFAULT_MAX_AGE_S)
    args = parser.parse_args(argv)
    return 0 if is_fresh(args.path, args.max_age_s) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
