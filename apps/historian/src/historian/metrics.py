"""Prometheus metrics of the historian: what it received, wrote and failed to write."""

from __future__ import annotations

from prometheus_client import Counter, Histogram

POINTS_WRITTEN = Counter(
    "historian_points_written_total", "Points written to InfluxDB."
)
POINTS_FAILED = Counter(
    "historian_points_failed_total", "Points InfluxDB did not accept."
)
WRITE_SECONDS = Histogram(
    "historian_write_seconds",
    "Time of one write call to InfluxDB.",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)
MESSAGES_SKIPPED = Counter(
    "historian_messages_skipped_total",
    "Messages that produced no point: heartbeats, unknown topics, bad payloads.",
)
FIELDS_DROPPED = Counter(
    "historian_fields_dropped_total",
    "Numeric fields left out of a point because they were NaN or infinite.",
    ["field"],
)
