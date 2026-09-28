"""Upper bounds of request integers, set by what the upstreams can hold.

A value past one of these used to fail while the gRPC message or the Flux query was
built — a 500 with an ERROR traceback, or a 503 blaming the upstream — instead of a 422.
"""

from __future__ import annotations

# proto int32 fields (offset, limit) and the alarm tables' Integer primary key.
MAX_INT32 = 2**31 - 1
# 2100-01-01T00:00:00Z: far past any recorded telemetry, well inside Flux's time range.
MAX_EPOCH_MS = 4_102_444_800_000
