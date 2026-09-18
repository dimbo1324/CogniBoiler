"""
Prometheus metrics of the PLC.

The scan duration is measured where the scan runs; everything the service already counts
(commands, trips, warnings, scans, mode) is read from it when Prometheus scrapes, so the
control code keeps a single set of counters.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

from prometheus_client import REGISTRY, Histogram
from prometheus_client.core import CounterMetricFamily, GaugeMetricFamily, Metric
from prometheus_client.registry import Collector

if TYPE_CHECKING:
    from plc_controller.service import PLCService

SCAN_SECONDS = Histogram(
    "plc_scan_seconds",
    "Time of one scan: measurements, interlocks, alarms, control and the command sent.",
    buckets=(0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5),
)

_MODES = ("auto", "manual", "estop")


class PlcCollector(Collector):
    def __init__(self, service: PLCService) -> None:
        self._service = service

    def collect(self) -> Iterator[Metric]:
        stats = self._service.stats
        for name, key, documentation in (
            ("plc_scans", "scans", "Scans completed."),
            (
                "plc_commands_received",
                "commands_received",
                "Operator commands received.",
            ),
            ("plc_commands_rejected", "commands_rejected", "Commands the PLC refused."),
            (
                "plc_commands_forwarded",
                "commands_forwarded",
                "Commands sent to the plant.",
            ),
            ("plc_warnings", "warnings", "Interlock warnings raised."),
            ("plc_trips", "trips", "Interlock trips."),
        ):
            counter = CounterMetricFamily(name, documentation)
            counter.add_metric([], float(stats[key]))
            yield counter

        mode = GaugeMetricFamily(
            "plc_mode", "1 for the PLC's current mode.", labels=["mode"]
        )
        current = self._service.mode.value
        for value in _MODES:
            mode.add_metric([value], 1.0 if value == current else 0.0)
        yield mode

        conditions = GaugeMetricFamily(
            "plc_alarm_conditions_active", "Alarm conditions standing now."
        )
        conditions.add_metric([], float(self._service.active_condition_count))
        yield conditions


def observe_service(service: PLCService) -> None:
    """Expose the service's counters; call once per process."""
    REGISTRY.register(PlcCollector(service))
