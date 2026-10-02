"""Check the Compose exposure rules that review alone used to guard.

Reads the effective model from ``docker compose config --no-interpolate`` (no .env value is
ever rendered) and fails when a published port is not bound to 127.0.0.1, a gRPC port
(50051-50053) is published, a long-running service has no healthcheck, a secret variable
falls back to a non-empty default, or a container that is not a caller shares a network
with physics-engine or plc-controller.

    python infrastructure/checks/compose_exposure.py            # runs docker compose
    python infrastructure/checks/compose_exposure.py model.json  # a saved config
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
GRPC_PORTS = {50051, 50052, 50053}
ONE_SHOT = {"migrate"}
SECRET_NAME = re.compile(r"PASSWORD|TOKEN|SECRET|KEY")
DEFAULTED = re.compile(r"\$\{(?P<name>[A-Z0-9_]+):?-(?P<default>[^}]*)\}")

# Who may share a network with each gRPC server: its callers, plus the broker and
# Prometheus, which reach the same containers for MQTT and /metrics.
ALLOWED_PEERS = {
    "physics-engine": {"plc-controller", "api-gateway", "mosquitto", "prometheus"},
    "plc-controller": {
        "physics-engine",
        "api-gateway",
        "opcua-server",
        "mosquitto",
        "prometheus",
    },
}


def load_model(argv: list[str]) -> dict[str, Any]:
    if argv:
        return dict(json.loads(Path(argv[0]).read_text(encoding="utf-8")))
    command = [
        "docker",
        "compose",
        "--file",
        str(ROOT / "docker-compose.yml"),
        "--profile",
        "full",
        "config",
        "--no-interpolate",
        "--format",
        "json",
    ]
    completed = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True, check=True
    )
    return dict(json.loads(completed.stdout))


def port_problems(name: str, service: dict[str, Any]) -> list[str]:
    problems = []
    for port in service.get("ports", []):
        if port.get("host_ip") != "127.0.0.1":
            problems.append(
                f"{name}: port {port.get('target')} is not bound to 127.0.0.1"
            )
        published = {int(port["target"])}
        if port.get("published"):
            published.add(int(str(port["published"]).split("-")[0]))
        if published & GRPC_PORTS:
            problems.append(f"{name}: gRPC port {port.get('target')} is published")
    return problems


def healthcheck_problems(name: str, service: dict[str, Any]) -> list[str]:
    if name in ONE_SHOT:
        return []
    check = service.get("healthcheck") or {}
    if check.get("disable") or not check.get("test"):
        return [f"{name}: a long-running service without a healthcheck"]
    return []


def secret_default_problems(name: str, service: dict[str, Any]) -> list[str]:
    problems = []
    for key, value in (service.get("environment") or {}).items():
        for match in DEFAULTED.finditer(str(value or "")):
            secret = SECRET_NAME.search(match["name"]) or SECRET_NAME.search(key)
            if secret and match["default"]:
                problems.append(f"{name}: {key} falls back to a default secret")
    return problems


def network_problems(services: dict[str, Any]) -> list[str]:
    networks = {
        name: set(service.get("networks") or {"default": None})
        for name, service in services.items()
    }
    problems = []
    for server, allowed in ALLOWED_PEERS.items():
        if server not in networks:
            problems.append(f"{server}: missing from the Compose model")
            continue
        peers = {
            name
            for name, joined in networks.items()
            if name != server and joined & networks[server]
        }
        for intruder in sorted(peers - allowed):
            problems.append(f"{intruder}: shares a network with {server}'s gRPC port")
    return problems


def main(argv: list[str]) -> int:
    services: dict[str, Any] = load_model(argv)["services"]
    problems: list[str] = []
    for name, service in sorted(services.items()):
        problems += port_problems(name, service)
        problems += healthcheck_problems(name, service)
        problems += secret_default_problems(name, service)
    problems += network_problems(services)
    for problem in problems:
        print(f"compose exposure: {problem}", file=sys.stderr)
    if not problems:
        print(f"compose exposure: {len(services)} services follow the rules")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
