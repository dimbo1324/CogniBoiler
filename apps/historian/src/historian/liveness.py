"""A liveness file the container healthcheck reads.

The historian exposes no port, so its health is a file whose modification time the
service refreshes only while it is connected to the MQTT broker. The healthcheck runs
`python -m historian.liveness <path>` and gets exit code 0 while the file is fresh.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from collections.abc import Callable
from pathlib import Path

DEFAULT_INTERVAL_S = 5.0
DEFAULT_MAX_AGE_S = 30.0


class LivenessFile:
    def __init__(self, path: Path, interval_s: float = DEFAULT_INTERVAL_S) -> None:
        self._path = path
        self._interval_s = interval_s

    def beat(self, healthy: bool) -> None:
        if healthy:
            self._path.write_text(str(int(time.time())), encoding="ascii")

    async def run(self, is_healthy: Callable[[], bool]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        while True:
            self.beat(is_healthy())
            await asyncio.sleep(self._interval_s)


def is_fresh(path: Path, max_age_s: float) -> bool:
    try:
        modified = path.stat().st_mtime
    except FileNotFoundError:
        return False
    return time.time() - modified <= max_age_s


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m historian.liveness",
        description="Exit 0 when the liveness file was refreshed recently.",
    )
    parser.add_argument("path", type=Path)
    parser.add_argument("--max-age-s", type=float, default=DEFAULT_MAX_AGE_S)
    args = parser.parse_args(argv)
    return 0 if is_fresh(args.path, args.max_age_s) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
