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
    "Condition messages that could not be stored after retries.",
)
