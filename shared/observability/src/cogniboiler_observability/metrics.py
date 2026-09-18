"""
The Prometheus endpoint of a service without its own HTTP server.

`start_metrics_server` serves `/metrics` from a background thread. On the host it listens on
the loopback interface by default; the Compose services bind every interface so that
Prometheus, on the same network, can scrape them.
"""

from __future__ import annotations

import logging

from prometheus_client import Counter, start_http_server

logger = logging.getLogger(__name__)

# Every service speaks MQTT; one definition per process, whichever services share it
# (the PLC's tests run the physics engine in-process).
MQTT_PUBLISHED = Counter(
    "mqtt_messages_published_total", "MQTT messages published.", ["topic"]
)
MQTT_PUBLISH_ERRORS = Counter(
    "mqtt_publish_errors_total", "MQTT publishes that failed.", ["topic"]
)
MQTT_RECEIVED = Counter(
    "mqtt_messages_received_total", "MQTT messages received.", ["topic"]
)


def start_metrics_server(port: int, host: str = "127.0.0.1") -> None:
    """Serve /metrics on host:port; a port of 0 or less disables it."""
    if port <= 0:
        logger.info("Metrics endpoint disabled")
        return
    start_http_server(port, addr=host)
    logger.info("Metrics on http://%s:%d/metrics", host, port)
