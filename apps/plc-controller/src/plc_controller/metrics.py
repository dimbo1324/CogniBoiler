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
            ("plc_warnings", "warnings", "Scans that raised an interlock warning."),
            ("plc_trips", "trips", "Interlock trips."),
            ("plc_scan_failures", "scan_failures", "Scans a fault of the PLC broke."),
            (
                "plc_plant_stream_failures",
                "stream_failures",
                "Times the plant's state stream failed or ended.",
            ),
            (
                "plc_command_forward_failures",
                "forward_failures",
                "Commands the plant did not acknowledge: link failures.",
            ),
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

        dropped, connected, failures = self._service.publisher_health
        lost = CounterMetricFamily(
            "plc_publish_dropped", "Alarm and event messages lost to a full queue."
        )
        lost.add_metric([], float(dropped))
        yield lost
        broker_failures = CounterMetricFamily(
            "plc_mqtt_connection_failures",
            "Broker connections that failed or dropped.",
        )
        broker_failures.add_metric([], float(failures))
        yield broker_failures
        broker = GaugeMetricFamily(
            "plc_mqtt_connected", "1 while the broker connection is up."
        )
        broker.add_metric([], 1.0 if connected else 0.0)
        yield broker

        link = GaugeMetricFamily(
            "plc_plant_link_up", "1 while the plant's state stream is delivering."
        )
        link.add_metric([], 1.0 if self._service.plant_link_up else 0.0)
        yield link

        conditions = GaugeMetricFamily(
            "plc_alarm_conditions_active", "Alarm conditions standing now."
        )
        conditions.add_metric([], float(self._service.active_condition_count))
        yield conditions


def observe_service(service: PLCService) -> None:
    """Expose the service's counters; call once per process."""
    REGISTRY.register(PlcCollector(service))
