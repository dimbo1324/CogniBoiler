# Architecture: what exists today

> This document describes what is **actually in the code**, not what is planned. It is
> updated whenever the shape of the system changes: a new service, endpoint group, topic,
> table, screen, or operational job. The plan lives in the internal roadmap.

**Last revised:** 2026-09-14 · **Version:** 0.1.0 · **Python:** 3.14

## The shape of the system

```text
                      api-gateway (FastAPI, :8000)
                  JWT RS256 · RBAC · audit · REST · WebSocket
                     │ gRPC            │ gRPC           │ SQL        │ Flux
                     ▼                 ▼                ▼            ▼
             plc-controller ──gRPC──▶ physics-engine   PostgreSQL   InfluxDB
               (:50051)                 (:50052)          ▲            ▲
                  │ alerts/*             │ sensors/*       │            │
                  ▼                      ▼                 │            │
               ┌──────────── Mosquitto (:1883, :9001) ─────┼────────────┤
               ▼                         ▼                 │            │
         alert-manager ──────────────────┼─────────────────┘            │
                                          ├──────────▶ historian ────────┘
                                          └──────────▶ opcua-server (:4840)
                                                    Grafana (:3000) reads InfluxDB
```

`physics-engine` owns process state. `plc-controller` polls it, runs control and safety,
and sends validated commands back. Telemetry fans out over MQTT to the historian and the
OPC UA server. Alarms travel from the PLC over MQTT to the alert manager. The gateway is
the only edge for people.

## Services

| Service | Package | What it does today |
|---|---|---|
| physics-engine | `apps/physics-engine` | Boiler model (ODEs, IAPWS water/steam tables), turbine, combustion, heat exchanger; standalone condenser, emissions and equipment-health models not yet wired into the runtime; live runtime stepping in a worker thread; `PhysicsService` gRPC; MQTT telemetry publisher; offline scenario runner |
| plc-controller | `apps/plc-controller` | `PLCService` gRPC: command validation, setpoints, AUTO/MANUAL/ESTOP, cascade PID scan loop, safety interlocks with E-Stop latch, alarm and availability publishing to MQTT |
| api-gateway | `apps/api-gateway` | FastAPI app, JWT RS256 access/refresh tokens, refresh blacklist, role hierarchy, request audit middleware, gRPC clients to PLC and physics, InfluxDB history queries, Alembic migrations |
| historian | `apps/historian` | Subscribes to `sensors/#`, batches protobuf telemetry into InfluxDB |
| alert-manager | `apps/alert-manager` | Subscribes to `alerts/#`, stores deduplicated alarm events in PostgreSQL |
| opcua-server | `apps/opcua-server` | OPC UA address space (namespace 2) for boiler and turbine values, fed from MQTT; read-only |
| ai-predictor | `apps/ai-predictor` | Deferred placeholder; **not** a workspace member |

## Contracts

### gRPC (`shared/proto/cogniboiler.proto`, stubs committed in `shared/generated/`)

| Service | RPCs |
|---|---|
| `PhysicsService` | `Health`, `GetSystemState`, `StreamSystemState`, `ApplyControlCommand` |
| `PLCService` | `Health`, `SendCommand`, `GetSetpoints`, `UpdateSetpoints`, `GetControlStatus`, `ResetEmergencyStop`, `StreamCommands` |

### MQTT

| Topic | Payload | Publisher → subscribers |
|---|---|---|
| `sensors/boiler` | protobuf `BoilerStateMsg` | physics-engine → historian, opcua-server |
| `sensors/turbine` | protobuf `TurbineStateMsg` | physics-engine → historian, opcua-server |
| `sensors/system/heartbeat` | text timestamp | physics-engine → historian, opcua-server |
| `alerts/warning`, `alerts/critical` | JSON alarm event (`alarm_id`, `source_service`, `severity`, `parameter`, `value`, `threshold`, `action`, `message`, `timestamp_ms`) | plc-controller → alert-manager |
| `status/physics-engine`, `status/plc-controller` | retained `online` / `offline` | the service itself (MQTT will) |

Reserved for the deferred AI stage, not implemented: `insights/*`.

### REST and WebSocket (api-gateway)

| Route | Minimum role |
|---|---|
| `GET /health` | public |
| `POST /auth/login`, `POST /auth/refresh`, `POST /auth/logout` | public (credentials / token) |
| `GET /api/v1/status` | viewer |
| `GET /api/v1/history?measurement=&start_ms=&end_ms=&limit=` | viewer |
| `GET /api/v1/alarms?active_only=&limit=` | viewer |
| `POST /api/v1/commands/valve` | operator |
| `POST /api/v1/commands/setpoint` | engineer |
| `POST /api/v1/commands/reset` | engineer |
| `GET /api/v1/audit?limit=` | admin |
| `WS /ws/realtime?token=` | any valid access token |

OpenAPI is served at `/docs`.

### Storage

- **PostgreSQL** (Alembic revision `0001_initial_schema`): `users`, `roles`, `user_roles`,
  `token_blacklist`, `audit_log`. `alarm_events` is created by alert-manager with
  `create_all`, outside the migration chain.
- **InfluxDB** bucket from `.env`: measurements `boiler_sensors` and `turbine_sensors`,
  tag `quality`, fields in SI units named as in the protobuf messages.

## Runtime and tooling

- `docker-compose.yml` + `Dockerfile` run infrastructure and all services from one
  Python 3.14 image (`uv sync --frozen --no-dev`, non-root, `python -m` entry points).
  Every secret is interpolated from `.env`; host ports bind to `127.0.0.1`. A one-shot
  `migrate` service applies Alembic before the gateway and alert-manager start; every
  long-running service except historian and alert-manager has a healthcheck. Grafana is
  provisioned with an InfluxDB datasource and the `core-overview` dashboard.
- The gateway seeds demo users `admin`, `engineer`, `operator`, `viewer` from
  `DEMO_*_PASSWORD` when `AUTO_INIT_DB` is set; no credential is hardcoded.
- `smoke` checks a running stack through the gateway: health, logins, role refusals,
  live state (physics gRPC), a setpoint accepted by the PLC, alarms (PostgreSQL),
  history (InfluxDB) and the audit log. CI runs it against a freshly built stack.
- `apps/web` is the operator console skeleton: React 19 + TypeScript (6.0) + Vite 8, a
  gateway health screen, the network client module and the unit-conversion module.
- `python dev_tools_scripts_runner.py` is the developer-tools orchestrator: `quality-gate`,
  `format-code`, `sync-agents`, `stack`, `dev-secrets`, `generate-proto`, `doctor`,
  `install-hooks`, `clean-caches`, `selftest`.
- The quality gate runs ruff, strict mypy, every service test suite, the protobuf and
  `AGENTS.md` sync checks, the scripts' own tests, and the frontend checks when
  `apps/web/node_modules` exists.
- `ml/preprocessing/generate_dataset.py` exports scenario runs for the deferred AI stage.

## Known gaps

Recorded with their planned fix in the internal roadmap:

- the nominal operating point is not an equilibrium and AUTO does not hold it; the
  corresponding PLC test is `xfail(strict=True)`;
- PID and safety code live in `physics_engine` and are imported by the PLC;
- the gateway imports `alert_manager.models`, and `alarm_events` is outside Alembic;
- the audit middleware records every request, swallows write failures at debug level, and
  the table is not protected against `UPDATE`/`DELETE`;
- CORS allows any origin with credentials;
- MQTT is anonymous; OPC UA is read-only;
- the web console is only a skeleton, and logs are plain text;
- historian and alert-manager have no healthcheck, and historian receives its InfluxDB
  token as a command-line argument.
