# Architecture: what exists today

> This document describes what is **actually in the code**, not what is planned. It is
> updated whenever the shape of the system changes: a new service, endpoint group, topic,
> table, screen, or operational job. The plan lives in the internal roadmap.

**Last revised:** 2026-09-15 · **Version:** 0.1.0 · **Python:** 3.14

## The shape of the system

```text
                          api-gateway (FastAPI, :8000)
                   JWT RS256 · RBAC · audit · REST · WebSocket
          │ gRPC               │ gRPC              │ gRPC           │ SQL      │ Flux
          ▼                    ▼                   ▼                ▼          ▼
   plc-controller ──gRPC──▶ physics-engine    alert-manager ──▶ PostgreSQL  InfluxDB
     (:50051)                  (:50052)          (:50053)                      ▲
        │ alerts/*, plc/events    │ sensors/*        ▲  │ alarms/changes       │
        ▼                         ▼                  │  ▼                      │
   ┌────────────────────── Mosquitto (:1883, :9001) ─┴──────────────────────────┤
                                  │                                            │
                                  ├──────────▶ historian ──────────────────────┘
                                  └──────────▶ opcua-server (:4840)
                                               Grafana (:3000) reads InfluxDB
```

`physics-engine` owns process state and publishes what the instruments report.
`plc-controller` scans every published plant state, runs protection and coordinated control,
and sends validated valve commands back. It also publishes alarm conditions and its events.
`alert-manager` turns conditions into alarms with a lifecycle and serves them over
`AlarmService`. Telemetry fans out over MQTT to the historian and the OPC UA server. The
gateway is the only edge for people.

## Services

| Service | Package | What it does today |
|---|---|---|
| physics-engine | `apps/physics-engine` | Energy-conserving lumped model of a 300 MW gas-fired drum unit: furnace, superheater, evaporator bank, economizer, drum mass and enthalpy balances, attemperator spray, choked turbine admission, drum safety valves, turbine, condenser with lag, emissions, equipment wear. IAPWS-IF97 saturation tables for speed. Steady operating points solved for any load (the plant holds them open-loop). Deterministic `PlantSimulator` (RK4, 1 s steps) behind an async runtime with pause, step, speed and scenario load. Labelled faults: burner fouling, steam leak, feedwater pump failure, stuck valve, sensor drift and failure; instruments report GOOD/UNCERTAIN/BAD. `PhysicsService` gRPC; MQTT telemetry |
| plc-controller | `apps/plc-controller` | `PLCService` gRPC. One scan per published plant state. Coordinated control: turbine master on power with a pressure guard, boiler master on pressure through fuel with load feedforward and a steam-flow firing limit, three-element drum level, spray steam temperature, ramped setpoints, bumpless takeover. AUTO / MANUAL / ESTOP. Interlocks armed by operating state, a cause-dependent trip response, an E-Stop latch that resets only once its cause is gone. Alarm conditions with deadband and PLC events over one persistent MQTT connection |
| api-gateway | `apps/api-gateway` | FastAPI app, JWT RS256 access/refresh tokens, refresh blacklist, role hierarchy, request audit middleware, gRPC clients to PLC, physics and alert-manager, InfluxDB history queries, the Alembic migration chain |
| historian | `apps/historian` | Subscribes to `sensors/#`, batches boiler and turbine protobuf telemetry into InfluxDB |
| alert-manager | `apps/alert-manager` | Alarm lifecycle (ACTIVE_UNACK → ACTIVE_ACK → CLEARED, CLEARED_UNACK), one open alarm per condition, chatter hold on clears, reconciliation from source snapshots, transition history, `AlarmService` gRPC, changes on `alarms/changes` |
| opcua-server | `apps/opcua-server` | OPC UA address space (namespace 2) for boiler and turbine values, fed from MQTT; read-only |
| ai-predictor | `apps/ai-predictor` | Deferred placeholder; **not** a workspace member |

## Contracts

### gRPC (`shared/proto/cogniboiler.proto`, stubs committed in `shared/generated/`)

| Service | RPCs |
|---|---|
| `PhysicsService` (:50052) | `Health`, `GetSystemState`, `StreamSystemState`, `ApplyControlCommand`, `GetSimulationStatus`, `PauseSimulation`, `ResumeSimulation`, `SetSimulationSpeed`, `StepSimulation`, `ListScenarios`, `LoadScenario`, `InjectFault`, `ClearFault` |
| `PLCService` (:50051) | `Health`, `SendCommand`, `GetSetpoints`, `UpdateSetpoints`, `GetControlStatus`, `ResetEmergencyStop`, `StreamCommands`, `SetLoadDemand`, `SetControlMode` |
| `AlarmService` (:50053) | `Health`, `ListAlarms`, `GetAlarm`, `AcknowledgeAlarm`, `AcknowledgeAll` |

`SystemStateMsg` carries measured boiler and turbine values, flows and heat duties, valve
commands and positions (spray included), emissions, condenser, equipment health, active
faults with labels, per-instrument quality and the simulation status (scenario, run id,
time). Scenarios: `steady_state` (250 MW), `part_load` (180 MW), `full_load` (300 MW),
`hot_start`, `cold_start`, `feedwater_pump_drill`.

### MQTT

| Topic | Payload | Publisher → subscribers |
|---|---|---|
| `sensors/boiler` | protobuf `BoilerStateMsg` (measured) | physics-engine → historian, opcua-server |
| `sensors/turbine` | protobuf `TurbineStateMsg` (measured) | physics-engine → historian, opcua-server |
| `sensors/plant` | protobuf `PlantStatusMsg`: emissions, condenser, health, faults, instrument qualities, simulation status | physics-engine → (none yet) |
| `sensors/system/heartbeat` | text timestamp | physics-engine → historian, opcua-server |
| `alerts/warning`, `alerts/critical` | JSON condition: `key`, `state` (`active`/`cleared`), `source_service`, `severity`, `parameter`, `direction`, `unit`, `value`, `threshold`, `action`, `message`, `raised_at_ms`, `timestamp_ms`, `alarm_id` | plc-controller → alert-manager |
| `alerts/snapshot` | JSON `source_service`, `active_keys`, `timestamp_ms` (on connect and every 10 s) | plc-controller → alert-manager |
| `plc/events` | JSON PLC event: mode change, trip, manual trip, reset, refused reset, load demand, setpoints, run change | plc-controller → (none yet) |
| `alarms/changes` | JSON alarm after a state change with its transition | alert-manager → (none yet) |
| `status/physics-engine`, `status/plc-controller` | retained `online` / `offline` | the service itself (MQTT will) |

Reserved for the deferred AI stage, not implemented: `insights/*`.

### REST and WebSocket (api-gateway)

| Route | Minimum role |
|---|---|
| `GET /health` | public |
| `POST /auth/login`, `POST /auth/refresh`, `POST /auth/logout` | public (credentials / token) |
| `GET /api/v1/status` | viewer |
| `GET /api/v1/plc/status` | viewer |
| `GET /api/v1/history?measurement=&start_ms=&end_ms=&limit=` | viewer |
| `GET /api/v1/alarms?active_only=&limit=` | viewer |
| `GET /api/v1/alarms/history?severity=&parameter=&from_ms=&to_ms=&limit=&offset=` | viewer |
| `GET /api/v1/alarms/{alarm_id}` | viewer |
| `POST /api/v1/alarms/{alarm_id}/ack`, `POST /api/v1/alarms/ack-all` | operator |
| `POST /api/v1/commands/valve` (spray optional), `POST /api/v1/commands/load`, `POST /api/v1/commands/mode` | operator |
| `POST /api/v1/commands/setpoint` | engineer |
| `POST /api/v1/commands/reset` | engineer |
| `GET /api/v1/audit?limit=` | admin |
| `WS /ws/realtime?token=` | any valid access token |

OpenAPI is served at `/docs`. Simulation control is available over gRPC only.

### Storage

- **PostgreSQL**, one Alembic chain in `apps/api-gateway/migrations`: revision
  `0001_initial_schema` — `users`, `roles`, `user_roles`, `token_blacklist`, `audit_log`;
  revision `0002_alarm_lifecycle` — `alarm_events` (one row per alarm, a partial unique
  index keeps one open alarm per condition key) and `alarm_transitions`. The alarm tables
  belong to alert-manager, which reads and writes them but never creates schema.
- **InfluxDB** bucket from `.env`: measurements `boiler_sensors` and `turbine_sensors`,
  tag `quality`, fields in SI units named as in the protobuf messages.

## Runtime and tooling

- `docker-compose.yml` + `Dockerfile` run infrastructure and all services from one
  Python 3.14 image (`uv sync --frozen --no-dev`, non-root, `python -m` entry points).
  Every secret is interpolated from `.env`; host ports bind to `127.0.0.1`. A one-shot
  `migrate` service applies Alembic before the gateway and alert-manager start; every
  long-running service has a healthcheck — historian and alert-manager, which expose no
  HTTP port, refresh a liveness file only while subscribed to the broker. Grafana is
  provisioned with an InfluxDB datasource and the `core-overview` dashboard.
- The gateway seeds demo users `admin`, `engineer`, `operator`, `viewer` from
  `DEMO_*_PASSWORD` when `AUTO_INIT_DB` is set; no credential is hardcoded.
- `smoke` checks a running stack through the gateway: health, logins, role refusals,
  live state (physics gRPC), a setpoint accepted by the PLC, alarms (AlarmService),
  history (InfluxDB) and the audit log. CI runs it against a freshly built stack.
- `apps/web` is the operator console skeleton: React 19 + TypeScript (6.0) + Vite 8, a
  gateway health screen, the network client module and the unit-conversion module.
- `python dev_tools_scripts_runner.py` is the developer-tools orchestrator: `quality-gate`,
  `format-code`, `sync-agents`, `stack`, `dev-secrets`, `smoke`, `generate-proto`, `doctor`,
  `install-hooks`, `clean-caches`, `selftest`.
- The quality gate runs ruff, strict mypy, every service test suite, the protobuf and
  `AGENTS.md` sync checks, the scripts' own tests, and the frontend checks when
  `apps/web/node_modules` exists.
- `ml/preprocessing/generate_dataset.py` runs labelled closed-loop episodes — the
  deterministic plant scanned by the real PLC logic — for the deferred AI stage.

## Known gaps

Recorded with their planned fix in the internal roadmap:

- the plant, PLC and alarm logic added in S2–S4 has no dedicated tests yet (owner decision
  for that work); the PLC integration tests still pace the plant by wall clock instead of
  stepping it deterministically;
- simulation control, alarm and PLC events are not yet exposed over REST or WebSocket;
- the audit middleware records every request, swallows write failures at debug level, and
  the table is not protected against `UPDATE`/`DELETE`;
- CORS allows any origin with credentials;
- MQTT is anonymous; OPC UA is read-only;
- the web console is only a skeleton, and logs are plain text.
