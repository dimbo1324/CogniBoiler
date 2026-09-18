<!-- tier: extended -->

# Command Reference: Services, Stack, Contracts and Platform Notes

> **Essence.** Lookup material — running each service from the host, the Compose stack with its ports and credentials flow, the MQTT topics, protobuf regeneration, the frontend commands, and the Windows notes (selector event loop, a `.venv` held by VS Code, `AppData` redirection under the Claude desktop app). The always-apply policies live in `11-commands.md`.

## The local stack

```powershell
python dev_tools_scripts_runner.py dev-secrets        # once: .env with generated secrets
python dev_tools_scripts_runner.py stack up           # build and start everything, wait for health
python dev_tools_scripts_runner.py stack up --infra-only
python dev_tools_scripts_runner.py stack status
python dev_tools_scripts_runner.py stack logs api-gateway
python dev_tools_scripts_runner.py smoke              # end-to-end check through the gateway
python dev_tools_scripts_runner.py stack down         # --volumes also wipes the databases
```

The gateway seeds demo users `admin`, `engineer`, `operator` and `viewer` with the
`DEMO_*_PASSWORD` values from `.env`. A one-shot `migrate` service applies Alembic before
the gateway starts. InfluxDB and PostgreSQL read credentials only on first start: after
regenerating secrets, run `stack down --volumes`.

| Component | Host port | Notes |
|---|---|---|
| Mosquitto (MQTT / WebSocket) | 1883 / 9001 | anonymous access, development only |
| InfluxDB 2 | 8086 | org and raw bucket (7 days) from `.env`; `sensors_1m` one-minute aggregates (90 days), set up by the historian |
| PostgreSQL 16 | 5432 | users, roles, sessions, append-only audit log, scenario runs; alarm lifecycle tables owned by alert-manager |
| Grafana | 3000 | provisioned InfluxDB datasource (uid `influxdb`) and dashboards Process, Efficiency and emissions, Alarms, Platform |
| physics-engine gRPC | — | `PhysicsService` on :50052 inside the Compose network only: publishing it would open a path to the valves around the PLC |
| plc-controller gRPC | — | `PLCService` on :50051 inside the Compose network only |
| alert-manager gRPC | — | `AlarmService` on :50053 inside the Compose network only |
| opcua-server | 4840 | `opc.tcp://localhost:4840/cogniboiler`; anonymous read, methods need a gateway user (username and password) |
| api-gateway | 8000 | REST under `/api/v1`, `/auth`, `/health`, `/ready`, OpenAPI at `/docs`, WebSocket `/ws` (token in the first frame) |
| web console | 5173 (dev) | Vite dev server on 127.0.0.1 proxies `/api`, `/auth`, `/health` and `/ws` to the gateway |

## Running a service from the host

Start the infrastructure with `stack up --infra-only`, then, each in its own terminal:

```powershell
uv run --package physics-engine python -m physics_engine --speed 1   # --scenario hot_start, --paused
uv run --package plc-controller python -m plc_controller --physics-target localhost:50052
uv run --package historian python -m historian
uv run --package alert-manager python -m alert_manager               # AlarmService on :50053
uv run --package opcua-server python -m opcua_server                 # --gateway-url http://localhost:8000
uv run --package api-gateway uvicorn api_gateway.main:app --reload --port 8000
```

Host-run services read `.env` from the repository root for secrets. alert-manager refuses to
start until the migrations have created its tables (`stack up --infra-only` runs `migrate`
only with the full stack; from the host, apply `alembic upgrade head` first).

Simulation control for engineers is REST: `/api/v1/simulation/pause`, `/resume`, `/speed`,
`/step`, `/scenario`, `/faults` (the gateway calls the `PhysicsService` RPCs and records
scenario loads and fault changes in `scenario_runs`). Errors are Problem Details with a
stable `code`.

## MQTT topics

| Topic | Payload | Publisher → subscribers |
|---|---|---|
| `sensors/plant` | protobuf `PlantStatusMsg`, first in each step | physics-engine → historian, opcua-server |
| `sensors/boiler` | protobuf `BoilerStateMsg` (measured values) | physics-engine → historian, opcua-server |
| `sensors/turbine` | protobuf `TurbineStateMsg` (measured values) | physics-engine → historian, opcua-server |
| `sensors/system/heartbeat` | text timestamp | physics-engine → (skipped) |
| `alerts/warning`, `alerts/critical` | JSON alarm condition, `state` active/cleared | plc-controller → alert-manager |
| `alerts/snapshot` | JSON active condition keys, every 10 s | plc-controller → alert-manager |
| `plc/events` | JSON PLC event | plc-controller → api-gateway (`/ws` plc), historian |
| `alarms/changes` | JSON alarm state change | alert-manager → api-gateway (`/ws` alarms), historian, opcua-server |
| `status/physics-engine`, `status/plc-controller` | retained `online`/`offline` | the service itself (MQTT will) → historian |

## Contracts and code generation

- Edit `shared/proto/cogniboiler.proto`, then `generate-proto`; commit both together.
- Change a gateway route or schema, then `generate-openapi`: it rewrites
  `shared/openapi/api-gateway.json` and the console types `apps/web/src/api/schema.gen.ts`;
  commit them with the change. The console never declares a gateway shape by hand.
- Schema change: add an Alembic revision under `apps/api-gateway/migrations/versions/`,
  then `uv run --package api-gateway alembic -c apps/api-gateway/alembic.ini upgrade head`.

## Frontend

```powershell
pnpm --dir apps/web install
pnpm --dir apps/web dev          # http://localhost:5173
pnpm --dir apps/web run lint ; pnpm --dir apps/web run typecheck ; pnpm --dir apps/web run test
python dev_tools_scripts_runner.py console-e2e   # Playwright against the running stack
```

`console-e2e` installs Playwright's Chromium on first use (`--no-install` skips it) and starts
the Vite dev server when no `--url` is given. Its demo check reloads the nominal scenario, runs
the simulation at 10×, trips and resets the unit, then restores real time. The checks sign in with the demo passwords from
`.env`; they never print them. The access token stays in memory and the refresh token in the
gateway's httpOnly cookie, so the console needs the gateway on the same origin (the Vite proxy,
or nginx in the stack).

## Platform notes

- **Windows event loop.** aiomqtt needs `add_reader()`; entry points pass
  `loop_factory=asyncio.SelectorEventLoop` on Windows. A new entry point must do the same.
- **A `.venv` held by VS Code.** Python extensions (Black Formatter, Pylance) run the
  project interpreter and restart it the moment it is killed, so `uv sync` cannot
  recreate `.venv` while VS Code is open. Close VS Code for the rebuild (found 2026-09-14).
- **Claude desktop app on Windows.** Commands an agent runs from the desktop app write to
  `%APPDATA%` and `%LOCALAPPDATA%` inside the app's package container, invisible outside
  it. Per-user tooling therefore goes outside AppData (`UV_PYTHON_INSTALL_DIR`,
  `UV_CACHE_DIR` in the user environment; `PLAYWRIGHT_BROWSERS_PATH` for Playwright's
  browsers). Such a session reaches only IPv4 loopback, which is why every dev server and
  published port binds 127.0.0.1. Docker Desktop started from such a session
  runs inside the container and its backend crashes on its AppData sockets: the owner
  starts Docker Desktop (found 2026-09-16).
- **Git Bash path conversion.** `git show origin/branch:path` gets mangled into a Windows
  path; prefix the command with `MSYS_NO_PATHCONV=1`.
- **Line endings.** The repository normalizes to LF (pre-commit `mixed-line-ending`).
