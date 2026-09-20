"""Play the five-minute demo of VISION §7 through the gateway, then read the service logs.

Every step uses the same REST API an operator uses, so the console can stay open beside it
and show the story as it happens: the load change, the feedwater pump failure, the alarms,
the trip, the acknowledgement, the reset and the return to AUTO. The simulation runs at
`--speed` (10 by default), so a scenario of many minutes takes about one.

Whatever happens, the nominal scenario and real time are put back before the script
returns. It ends by reading `logs/*.log`: one `error` line written while the demo ran fails
the run, because a demo without a single error in the logs is a 1.0 criterion (VISION §10)
and a trip is logged as a warning, not an error (owner decision 2026-09-19).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from scripts._toolkit.config import load_config, repo_root
from scripts._toolkit.console import fail, heading, info, ok, summary
from scripts._toolkit.envfile import EnvFileError, parse, values

SCRIPT_DIR = Path(__file__).resolve().parent

MW = 1_000_000.0


class GatewayUnreachableError(RuntimeError):
    """The gateway did not answer at all."""


@dataclass(frozen=True)
class Reply:
    status: int
    body: Any

    @property
    def accepted(self) -> bool:
        return self.status == 200 and bool(dig(self.body, "accepted"))

    @property
    def refusal(self) -> str:
        reason = dig(self.body, "reason") or dig(self.body, "detail")
        return f"HTTP {self.status}" + (f": {reason}" if reason else "")


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
        raw = error.read()
        try:
            return Reply(error.code, json.loads(raw) if raw else None)
        except json.JSONDecodeError:
            return Reply(error.code, None)
    except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
        raise GatewayUnreachableError(f"{method} {path}: {error}") from error


def utc_now() -> str:
    """The instant this returns, to the second, as the log files write it."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S")


def clock(seconds: float) -> str:
    """Elapsed wall time as a demo script reads it: 2:20."""
    return f"{int(seconds) // 60}:{int(seconds) % 60:02d}"


def audit_line(entry: dict[str, Any]) -> str:
    """One audit row: when, exactly, who, what and how it ended."""
    at_ms = int(entry.get("timestamp_ms", 0))
    at = datetime.fromtimestamp(at_ms / 1000, UTC).isoformat(timespec="seconds")
    outcome = entry.get("outcome") or entry.get("response_status")
    return (
        f"{at}  {entry.get('username', '—'):9} "
        f"{entry.get('method', '')} {entry.get('endpoint', '')} → {outcome}"
    )


def error_lines(log_dir: Path, since_utc: str) -> list[tuple[str, dict[str, Any]]]:
    """Every `error` or `critical` line written to a service log since `since_utc`."""
    found: list[tuple[str, dict[str, Any]]] = []
    for path in sorted(log_dir.glob("*.log")):
        for raw in path.read_text("utf-8", errors="replace").splitlines():
            try:
                record = json.loads(raw)
            except ValueError:
                continue
            if not isinstance(record, dict):
                continue
            if str(record.get("timestamp", "")) < since_utc:
                continue
            if str(record.get("level", "")) in {"error", "critical"}:
                found.append((path.stem, record))
    return found


@dataclass
class Demo:
    """The narrator: it times the story, prints it, and remembers what failed."""

    base_url: str
    started: float = field(default_factory=time.monotonic)
    rows: list[tuple[str, str]] = field(default_factory=list)
    failed: bool = False

    def say(self, text: str) -> None:
        info(f"{clock(time.monotonic() - self.started):>5}  {text}")

    def step(self, name: str, passed: bool, detail: str) -> bool:
        stamp = clock(time.monotonic() - self.started)
        if passed:
            ok(f"{stamp:>5}  {name} — {detail}")
            self.rows.append((name, "ok"))
            return True
        fail(f"{stamp:>5}  {name} — {detail}")
        self.rows.append((name, f"FAILED ({detail})"))
        self.failed = True
        return False

    def wait_for(
        self,
        name: str,
        probe: Callable[[], Any],
        ready: Callable[[Any], bool],
        describe: Callable[[Any], str],
        timeout_s: float,
    ) -> bool:
        deadline = time.monotonic() + timeout_s
        while True:
            latest = probe()
            if ready(latest):
                return self.step(name, True, describe(latest))
            if time.monotonic() >= deadline:
                return self.step(
                    name, False, f"not within {timeout_s:.0f} s: {describe(latest)}"
                )
            time.sleep(1.0)


def sign_in(base_url: str, username: str, password: str) -> str:
    reply = call(
        base_url,
        "POST",
        "/auth/login",
        payload={"username": username, "password": password},
    )
    token = dig(reply.body, "access_token")
    if not isinstance(token, str):
        raise RuntimeError(f"{username} could not sign in: {reply.refusal}")
    return token


def plant(base_url: str, token: str) -> Any:
    return call(base_url, "GET", "/api/v1/status", token=token).body


def plc(base_url: str, token: str) -> Any:
    return call(base_url, "GET", "/api/v1/plc/status", token=token).body


def active_alarms(base_url: str, token: str) -> list[dict[str, Any]]:
    reply = call(base_url, "GET", "/api/v1/alarms?active_only=true", token=token)
    return reply.body if isinstance(reply.body, list) else []


def power_mw(state: Any) -> float:
    value = dig(state, "turbine", "electrical_power_w")
    return float(value) / MW if isinstance(value, (int, float)) else float("nan")


def level_m(state: Any) -> float:
    value = dig(state, "boiler", "water_level_m")
    return float(value) if isinstance(value, (int, float)) else float("nan")


def severities(alarms: list[dict[str, Any]]) -> set[str]:
    return {str(alarm.get("severity", "")) for alarm in alarms}


def nominal_again(
    base_url: str, engineer: str, scenario: str, nominal_mw: float
) -> tuple[bool, str]:
    """Clear the faults, reload the scenario and put the load demand back to nominal.

    A demo must be repeatable: the next run finds the unit where this one started, with the
    PLC's load demand at nominal and no E-Stop still latched.
    """
    faults = call(base_url, "DELETE", "/api/v1/simulation/faults", token=engineer)
    loaded = call(
        base_url,
        "POST",
        "/api/v1/simulation/scenario",
        token=engineer,
        payload={"name": scenario},
    )
    released = True
    if dig(plc(base_url, engineer), "emergency_stop_active"):
        released = call(
            base_url,
            "POST",
            "/api/v1/commands/reset",
            token=engineer,
            payload={"operator_id": "demo"},
        ).accepted
    demand = call(
        base_url,
        "POST",
        "/api/v1/commands/load",
        token=engineer,
        payload={"load_w": nominal_mw * MW},
    )
    done = faults.accepted and loaded.accepted and released and demand.accepted
    return done, f"{scenario} at {nominal_mw:.0f} MW, no fault, no latched trip"


def restore(
    base_url: str, engineer: str, scenario: str, nominal_mw: float, demo: Demo
) -> None:
    """Put the unit back the way the demo found it, even after a failed step."""
    done, detail = nominal_again(base_url, engineer, scenario, nominal_mw)
    speed = call(
        base_url,
        "POST",
        "/api/v1/simulation/speed",
        token=engineer,
        payload={"speed_factor": 1.0},
    )
    demo.step(
        "the unit is back at nominal in real time",
        done and speed.accepted,
        f"{detail}, real time",
    )


def play(
    base_url: str, config: dict[str, Any], env: dict[str, str], speed: float
) -> Demo:
    demo = Demo(base_url)
    users = config["users"]
    tokens = {
        role: sign_in(base_url, user["username"], env.get(user["password_env"], ""))
        for role, user in users.items()
    }
    operator, engineer, admin = tokens["operator"], tokens["engineer"], tokens["admin"]
    scenario = str(config["scenario"])
    limits = config["timeouts"]
    demo.say(f"the operator, the engineer and the admin are signed in at {base_url}")

    nominal = float(config["nominal_load_mw"])
    ready, detail = nominal_again(base_url, engineer, scenario, nominal)
    call(base_url, "POST", "/api/v1/simulation/resume", token=engineer)
    call(base_url, "POST", "/api/v1/alarms/ack-all", token=operator, payload={})
    fast = call(
        base_url,
        "POST",
        "/api/v1/simulation/speed",
        token=engineer,
        payload={"speed_factor": speed},
    )
    if not demo.step(
        "the engineer sets the scene",
        ready and fast.accepted,
        f"{detail}, running at {speed:g}×",
    ):
        return demo

    demo.wait_for(
        "the unit stands at nominal in AUTO",
        lambda: (plant(base_url, operator), plc(base_url, operator)),
        lambda pair: (
            abs(power_mw(pair[0]) - nominal) < 10.0 and dig(pair[1], "mode") == "auto"
        ),
        lambda pair: f"{power_mw(pair[0]):.1f} MW, PLC {dig(pair[1], 'mode')}",
        float(limits["nominal_s"]),
    )

    target = float(config["target_load_mw"])
    demanded = call(
        base_url,
        "POST",
        "/api/v1/commands/load",
        token=operator,
        payload={"load_w": target * MW},
    )
    demo.step(
        f"the operator asks for {target:.0f} MW",
        demanded.accepted,
        demanded.refusal if not demanded.accepted else "accepted by the PLC",
    )
    demo.wait_for(
        "the regulators bring the unit up",
        lambda: plant(base_url, operator),
        lambda state: power_mw(state) >= target - 5.0,
        lambda state: f"{power_mw(state):.1f} MW",
        float(limits["load_s"]),
    )

    fault = config["fault"]
    injected = call(
        base_url,
        "POST",
        "/api/v1/simulation/faults",
        token=engineer,
        payload=fault,
        timeout=20.0,
    )
    demo.step(
        f"the engineer injects the {str(fault['kind']).replace('_', ' ')}",
        injected.accepted,
        injected.refusal if not injected.accepted else "the fault is in",
    )

    demo.wait_for(
        "the drum level falls and a warning is raised",
        lambda: active_alarms(base_url, operator),
        lambda alarms: bool(severities(alarms)),
        lambda alarms: ", ".join(sorted(severities(alarms))) or "no alarm yet",
        float(limits["warning_s"]),
    )
    demo.wait_for(
        "the interlock trips the unit and the PLC latches E-Stop",
        lambda: (plc(base_url, operator), active_alarms(base_url, operator)),
        lambda pair: bool(dig(pair[0], "emergency_stop_active")),
        lambda pair: (
            f"PLC {dig(pair[0], 'mode')}, "
            f"alarms: {', '.join(sorted(severities(pair[1]))) or 'none'}"
        ),
        float(limits["trip_s"]),
    )

    acknowledged = call(
        base_url, "POST", "/api/v1/alarms/ack-all", token=operator, payload={}
    )
    count = len(dig(acknowledged.body, "alarms") or [])
    demo.step(
        "the operator acknowledges the alarms",
        acknowledged.accepted,
        f"{count} alarm(s) acknowledged"
        if acknowledged.accepted
        else acknowledged.refusal,
    )

    cleared = call(base_url, "DELETE", "/api/v1/simulation/faults", token=engineer)
    demo.step(
        "the engineer repairs the pump",
        cleared.accepted,
        cleared.refusal if not cleared.accepted else "no fault is active",
    )
    demo.wait_for(
        "the level recovers and the trip may be reset",
        lambda: (plc(base_url, operator), plant(base_url, operator)),
        lambda pair: bool(dig(pair[0], "reset_permitted")),
        lambda pair: (
            f"drum level {level_m(pair[1]):.2f} m, "
            f"blockers: {', '.join(dig(pair[0], 'reset_blockers') or []) or 'none'}"
        ),
        float(limits["recovery_s"]),
    )

    reset = call(
        base_url,
        "POST",
        "/api/v1/commands/reset",
        token=engineer,
        payload={"operator_id": "demo"},
    )
    demo.step(
        "the engineer resets the E-Stop",
        reset.accepted,
        reset.refusal if not reset.accepted else "the latch is open",
    )
    back_on_load = float(config["restart_load_mw"])
    demo.wait_for(
        f"the unit comes back on load in AUTO, above {back_on_load:.0f} MW",
        lambda: (plc(base_url, operator), plant(base_url, operator)),
        lambda pair: (
            dig(pair[0], "mode") == "auto" and power_mw(pair[1]) >= back_on_load
        ),
        lambda pair: f"PLC {dig(pair[0], 'mode')}, {power_mw(pair[1]):.1f} MW",
        float(limits["restart_s"]),
    )

    audit = call(base_url, "GET", "/api/v1/audit?limit=12", token=admin)
    entries = dig(audit.body, "items") or []
    demo.step(
        "the admin reads who did what, to the second",
        audit.status == 200 and bool(entries),
        f"{len(entries)} audit entries",
    )
    for entry in list(reversed(entries))[-8:]:
        demo.say(f"    {audit_line(entry)}")
    return demo


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="demo",
        description="Play the VISION §7 demo through the gateway and read the logs.",
    )
    parser.add_argument("--base-url", help="the console's origin (nginx)")
    parser.add_argument(
        "--speed", type=float, help="simulation speed factor during the demo"
    )
    parser.add_argument(
        "--log-dir", help="where the services write their JSON logs (default: logs)"
    )
    parser.add_argument(
        "--keep-logs-unread",
        action="store_true",
        help="skip the final scan of the service logs",
    )
    return parser


def main(argv: list[str]) -> int:
    args = _parser().parse_args(argv)
    root = repo_root()
    config = load_config(SCRIPT_DIR, "demo.json")
    base_url = str(args.base_url or config["base_url"])
    speed = float(args.speed or config["speed_factor"])
    log_dir = root / str(args.log_dir or config["log_dir"])

    env_path = root / str(config["env_file"])
    if not env_path.exists():
        fail(f"{config['env_file']} is missing — the stack cannot have been started")
        return 1
    try:
        env = values(parse(env_path.read_text(encoding="utf-8")))
    except EnvFileError as error:
        fail(f"{config['env_file']}: {error}")
        return 1

    heading(f"demo — {base_url}")
    since = utc_now()
    demo = Demo(base_url)
    try:
        demo = play(base_url, config, env, speed)
    except GatewayUnreachableError as error:
        demo.step("the gateway answers", False, str(error))
    except RuntimeError as error:
        demo.step("the demo runs", False, str(error))
    finally:
        try:
            engineer = sign_in(
                base_url,
                config["users"]["engineer"]["username"],
                env.get(config["users"]["engineer"]["password_env"], ""),
            )
            restore(
                base_url,
                engineer,
                str(config["scenario"]),
                float(config["nominal_load_mw"]),
                demo,
            )
        except (GatewayUnreachableError, RuntimeError) as error:
            demo.step("the unit is back at nominal in real time", False, str(error))

    if not args.keep_logs_unread:
        errors = error_lines(log_dir, since)
        demo.step(
            "no service logged an error during the demo",
            not errors,
            f"{log_dir.name}/: {len(errors)} error line(s)"
            if errors
            else f"{log_dir.name}/ is clean",
        )
        for service, record in errors[:5]:
            info(f"    {service}: {str(record.get('event', ''))[:160]}")

    summary("demo", demo.rows)
    return 1 if demo.failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
