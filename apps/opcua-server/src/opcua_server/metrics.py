"""Prometheus metrics of the OPC UA server."""

from __future__ import annotations

from prometheus_client import Counter

METHOD_CALLS = Counter(
    "opcua_method_calls_total",
    "OPC UA method calls: accepted or refused by the PLC or alarms, or failed before.",
    ["method", "outcome"],
)
