"""The exposure rules of docker-compose.yml, which `stack` starts.

Every published port binds 127.0.0.1, the gRPC ports are never published (that would open
a path to the valves around the PLC), and every long-running service has a healthcheck
(`stack up --wait` relies on it). One edit like ``- "8080:8080"`` would expose the console
on every interface with the gate still green, so the file is read here line by line —
stdlib only, no YAML library.

Run with:  python -m unittest discover -s scripts -t .
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

from scripts._toolkit.config import repo_root

LOOPBACK_PORT = re.compile(r'^- "127\.0\.0\.1:\d+:\d+"$')
GRPC_PORTS = ("50051", "50052", "50053")
ONE_SHOT_SERVICES = frozenset({"migrate"})
SERVICE = re.compile(r"^  ([a-z][a-z0-9-]*):\s*$")
TOP_LEVEL = re.compile(r"^[A-Za-z]")


def services(text: str) -> dict[str, list[str]]:
    """The lines of each service block under the top-level ``services:`` key."""
    blocks: dict[str, list[str]] = {}
    inside = False
    current: str | None = None
    for line in text.splitlines():
        if TOP_LEVEL.match(line):
            inside = line.rstrip() == "services:"
            current = None
            continue
        if not inside:
            continue
        match = SERVICE.match(line)
        if match:
            current = match.group(1)
            blocks[current] = []
        elif current is not None:
            blocks[current].append(line)
    return blocks


def published_ports(block: list[str]) -> list[str]:
    """The list items under a service's ``ports:`` key, stripped."""
    ports: list[str] = []
    collecting = False
    for line in block:
        stripped = line.strip()
        if line.startswith("    ") and not line.startswith("     "):
            collecting = stripped == "ports:"
            continue
        if collecting and stripped.startswith("-"):
            ports.append(stripped)
    return ports


class ComposeExposureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        text = (repo_root() / "docker-compose.yml").read_text(encoding="utf-8")
        cls.services = services(text)

    def test_the_services_were_found(self) -> None:
        # The parser must see the file, or every rule below passes vacuously.
        for name in ("postgresql", "api-gateway", "plc-controller", "web"):
            self.assertIn(name, self.services)
        self.assertTrue(any(published_ports(b) for b in self.services.values()))

    def test_every_published_port_binds_loopback_only(self) -> None:
        for name, block in self.services.items():
            for port in published_ports(block):
                with self.subTest(service=name, port=port):
                    self.assertRegex(port, LOOPBACK_PORT)

    def test_no_grpc_port_is_published(self) -> None:
        for name, block in self.services.items():
            for port in published_ports(block):
                with self.subTest(service=name, port=port):
                    self.assertFalse(any(grpc in port for grpc in GRPC_PORTS))

    def test_every_long_running_service_has_a_healthcheck(self) -> None:
        for name, block in self.services.items():
            if name in ONE_SHOT_SERVICES:
                continue
            with self.subTest(service=name):
                self.assertTrue(
                    any(line.strip() == "healthcheck:" for line in block),
                    f"{name} has no healthcheck",
                )


class ParserTest(unittest.TestCase):
    SAMPLE = """\
x-defaults:
  logging:
    driver: json-file
services:
  web:
    image: web
    ports:
      - "8080:8080"
      - "127.0.0.1:8443:8443"
    environment:
      - "NOT_A_PORT=1"
    healthcheck:
      test: ["CMD", "true"]
  migrate:
    image: migrate
volumes:
  data:
"""

    def test_a_port_on_every_interface_is_seen(self) -> None:
        found = services(self.SAMPLE)
        self.assertEqual(set(found), {"web", "migrate"})
        ports = published_ports(found["web"])
        self.assertEqual(ports, ['- "8080:8080"', '- "127.0.0.1:8443:8443"'])
        self.assertNotRegex(ports[0], LOOPBACK_PORT)
        self.assertRegex(ports[1], LOOPBACK_PORT)

    def test_keys_outside_services_are_not_services(self) -> None:
        self.assertNotIn("logging", services(self.SAMPLE))
        self.assertNotIn("data", services(self.SAMPLE))


if __name__ == "__main__":
    unittest.main()
