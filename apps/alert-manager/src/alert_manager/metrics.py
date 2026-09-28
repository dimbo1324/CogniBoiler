"""Prometheus metrics of the alarm lifecycle."""

from __future__ import annotations

from prometheus_client import Counter

TRANSITIONS = Counter(
    "alarm_transitions_total",
    "Alarm state changes.",
    ["severity", "to_state"],
)
MESSAGES_FAILED = Counter(
    "alarm_messages_failed_total",
    "Condition messages the database refused for good (not an outage).",
)
MESSAGES_REJECTED = Counter(
    "alarm_messages_rejected_total",
    "Alarm messages refused as malformed, by reason.",
    ["reason"],
)
SNAPSHOT_UNMATCHED_KEYS = Counter(
    "alarm_snapshot_unmatched_keys_total",
    "Keys a source snapshot lists as active with no active alarm: a lost activation.",
)
CHANGES_DROPPED = Counter(
    "alarm_changes_dropped_total",
    "Alarm changes lost unpublished to a full publish queue during a broker outage.",
)
