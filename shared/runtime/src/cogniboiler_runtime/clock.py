"""The moment, in the unit every contract of this platform uses.

UTC epoch milliseconds is the time of `timestamp_ms` in the protobuf messages, the MQTT
payloads, the REST schemas and the database rows. Seven modules had written the same
`int(time.time() * 1000)` by hand; this is that line, with the unit named once.

Never truncate what this returns to a day or a minute for storage: precision that was
never captured cannot be recovered, and on a project where several things happen in one
second, the seconds are the answer to "when".
"""

from __future__ import annotations

import time

MILLISECONDS_PER_SECOND = 1_000
MILLISECONDS_PER_DAY = 86_400_000
NANOSECONDS_PER_MILLISECOND = 1_000_000


def now_ms() -> int:
    """The current moment as UTC epoch milliseconds."""
    return int(time.time() * MILLISECONDS_PER_SECOND)
