"""Check a running stack end to end through the API gateway.

Every check runs even after one fails, and each names the data path it proves. The only
write is a setpoint update with nominal values, which the PLC accepts idempotently.
Credentials are read from .env, the same file the stack was started with.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from scripts._toolkit.config import load_config, repo_root
from scripts._toolkit.console import fail, heading, ok, summary
from scripts._toolkit.envfile import EnvFileError, parse, values

SCRIPT_DIR = Path(__file__).resolve().parent


class GatewayUnreachableError(RuntimeError):
    """The gateway did not answer at all."""


@dataclass(frozen=True)
class Reply:
    status: int
    body: Any


@dataclass
class Report:
    rows: list[tuple[str, str]] = field(default_factory=list)
    failed: bool = False

    def record(self, name: str, passed: bool, detail: str) -> None:
        if passed:
            ok(f"{name} — {detail}")
            self.rows.append((name, "ok"))
            return
        fail(f"{name} — {detail}")
        self.rows.append((name, f"FAILED ({detail})"))
        self.failed = True


def dig(body: Any, *keys: str) -> Any:
    for key in keys:
        if not isinstance(body, dict):
            return None
        body = body.get(key)
    return body


def call(
    base_url: str,
    method: str,
    path: str,
    *,
    token: str | None = None,
    payload: dict[str, Any] | None = None,
    timeout: float = 10.0,
) -> Reply:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(f"{base_url}{path}", data=data, method=method)
    request.add_header("Accept", "application/json")
    if data is not None:
        request.add_header("Content-Type", "application/json")
    if token is not None:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
            return Reply(response.status, json.loads(raw) if raw else None)
    except urllib.error.HTTPError as error:
        return Reply(error.code, None)
    except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
        raise GatewayUnreachableError(f"{method} {path}: {error}") from error


def wait_for_history(base_url: str, token: str, wait_s: float) -> tuple[bool, str]:
    deadline = time.monotonic() + wait_s
    path = "/api/v1/history?measurement=boiler_sensors&limit=5"
    while True:
        reply = call(base_url, "GET", path, token=token)
        points = dig(reply.body, "points")
        if reply.status == 200 and isinstance(points, list) and points:
            return True, f"{len(points)} recent points"
        if time.monotonic() >= deadline:
            return False, f"HTTP {reply.status}, no points after {wait_s:.0f} s"
        time.sleep(3)


def login_all(
    base_url: str, users: dict[str, dict[str, str]], env: dict[str, str], report: Report
) -> dict[str, str]:
    tokens: dict[str, str] = {}
    for role, user in users.items():
        password = env.get(user["password_env"], "")
        if not password:
            report.record(f"login as {role}", False, f"{user['password_env']} is empty")
            continue
        reply = call(
            base_url,
            "POST",
            "/auth/login",
            payload={"username": user["username"], "password": password},
        )
        token = dig(reply.body, "access_token")
        report.record(
            f"login as {role}",
            reply.status == 200 and isinstance(token, str),
            f"HTTP {reply.status}",
        )
        if isinstance(token, str):
            tokens[role] = token
    return tokens


def run_checks(
    base_url: str, config: dict[str, Any], env: dict[str, str], wait_s: float
) -> Report:
    report = Report()
    setpoints = config["nominal_setpoints"]

    health = call(base_url, "GET", "/health")
    report.record(
        "gateway health",
        health.status == 200 and dig(health.body, "status") == "running",
        f"HTTP {health.status}",
    )

    tokens = login_all(base_url, config["users"], env, report)

    anonymous = call(base_url, "GET", "/api/v1/status")
    report.record(
        "status refuses an anonymous caller",
        anonymous.status == 401,
        f"HTTP {anonymous.status}",
    )

    operator = tokens.get("operator")
    if operator:
        status = call(base_url, "GET", "/api/v1/status", token=operator)
        pressure = dig(status.body, "boiler", "pressure_pa")
        report.record(
            "live state through physics-engine gRPC",
            status.status == 200 and isinstance(pressure, int | float) and pressure > 0,
            f"HTTP {status.status}, drum pressure {pressure} Pa",
        )
        denied = call(
            base_url,
            "POST",
            "/api/v1/commands/setpoint",
            token=operator,
            payload=setpoints,
        )
        report.record(
            "setpoints refuse the operator role",
            denied.status == 403,
            f"HTTP {denied.status}",
        )
        alarms = call(base_url, "GET", "/api/v1/alarms?limit=10", token=operator)
        report.record(
            "alarm events from PostgreSQL",
            alarms.status == 200 and isinstance(alarms.body, list),
            f"HTTP {alarms.status}",
        )
        passed, detail = wait_for_history(base_url, operator, wait_s)
        report.record("telemetry history from InfluxDB", passed, detail)

    engineer = tokens.get("engineer")
    if engineer:
        accepted = call(
            base_url,
            "POST",
            "/api/v1/commands/setpoint",
            token=engineer,
            payload=setpoints,
        )
        report.record(
            "setpoints accepted by plc-controller gRPC",
            accepted.status == 200 and dig(accepted.body, "accepted") is True,
            f"HTTP {accepted.status}, reason {dig(accepted.body, 'reason')!r}",
        )

    admin = tokens.get("admin")
    if admin:
        audit = call(base_url, "GET", "/api/v1/audit?limit=20", token=admin)
        entries = len(audit.body) if isinstance(audit.body, list) else 0
        report.record(
            "audit log records requests",
            audit.status == 200 and entries > 0,
            f"HTTP {audit.status}, {entries} entries",
        )
    return report


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="smoke",
        description="Check a running stack end to end through the API gateway.",
    )
    parser.add_argument("--base-url", help="gateway URL (default from config)")
    parser.add_argument(
        "--history-wait-s",
        type=float,
        help="how long to wait for the first telemetry in InfluxDB",
    )
    args = parser.parse_args(argv)

    root = repo_root()
    config = load_config(SCRIPT_DIR, "smoke.json")
    base_url = str(args.base_url or config["base_url"]).rstrip("/")
    wait_s = float(args.history_wait_s or config["history_wait_s"])

    env_path = root / str(config["env_file"])
    if not env_path.exists():
        fail(f"{config['env_file']} is missing — the stack cannot have been started")
        return 1
    try:
        env = values(parse(env_path.read_text(encoding="utf-8")))
    except EnvFileError as error:
        fail(f"{config['env_file']}: {error}")
        return 1

    heading(f"smoke — {base_url}")
    try:
        report = run_checks(base_url, config, env, wait_s)
    except GatewayUnreachableError as error:
        report = Report()
        report.record("gateway reachable", False, str(error))

    summary("smoke", report.rows)
    return 1 if report.failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
