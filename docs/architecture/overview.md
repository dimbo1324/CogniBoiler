# Architecture: what exists today

> This document describes what is **actually in the code**, not what is planned. It is
> updated whenever the shape of the system changes: a new service, endpoint group, topic,
> table, screen, or operational job. The plan lives in the internal roadmap.

**Last revised:** 2026-09-18 · **Version:** 0.1.0 · **Python:** 3.14

## The shape of the system

```text
   web console in a browser (REST, WebSocket)          OPC UA clients (opc.tcp :4840)
                  │                                              │ methods as the signed-in user
                  ▼                                              ▼
        api-gateway (FastAPI, :8000) ◀──────── REST ─────── opcua-server
   sessions · RBAC · audit · REST · /ws                       │ reads PLC and alarms (gRPC)
     │ gRPC        │ gRPC          │ gRPC       │ SQL   │ Flux
     ▼             ▼               ▼            ▼       ▼
 plc-controller ─gRPC─▶ physics-engine   alert-manager ─▶ PostgreSQL   InfluxDB ◀── Grafana (:3000)
   (:50051)             (:50052)          (:50053)                        ▲
     │ alerts/*, plc/events │ sensors/*      ▲  │ alarms/changes          │
     ▼                      ▼                │  ▼                         │
 ┌──────────────────────── Mosquitto (:1883, :9001) ───────────────────────┤
     │                      │                                             │
     └──▶ api-gateway (plc/events, alarms/changes → /ws)                  │
                            ├──────▶ historian ───────────────────────────┘
                            └──────▶ opcua-server
```

`physics-engine` owns process state and publishes what the instruments report, plus the
unit's performance computed from the true heat balance. `plc-controller` scans every plant
state, runs protection and coordinated control, and sends validated valve commands back; it
publishes alarm conditions and its events. `alert-manager` turns conditions into alarms with
a lifecycle. The historian records telemetry, KPIs, labels and events into InfluxDB. The
gateway is the edge for people; the OPC UA server is the edge for industrial clients and
performs their writes through the gateway, as the signed-in user.

## Services

| Service | Package | What it does today |
|---|---|---|
| physics-engine | `apps/physics-engine` | Energy-conserving lumped model of a 300 MW gas-fired drum unit: furnace, superheater, evaporator bank, economizer, drum mass and enthalpy balances, attemperator spray, choked turbine admission throttled at part load (Stodola) with a part-load isentropic efficiency, drum safety valves, turbine, condenser with lag, emissions from the fuel as fired, equipment wear. IAPWS-IF97 saturation tables. Steady operating points for any load. Deterministic `PlantSimulator` (RK4, 1 s steps) behind an async runtime with pause, step, speed and scenario load. Labelled faults; instruments report GOOD/UNCERTAIN/BAD. Unit performance (efficiencies, heat rates, CO2 intensity). `PhysicsService` gRPC; MQTT telemetry |
| plc-controller | `apps/plc-controller` | `PLCService` gRPC. One scan per published plant state. Coordinated control: turbine master on power with a pressure guard, boiler master on pressure through fuel with load feedforward and a firing limit, three-element drum level, spray steam temperature, ramped setpoints, bumpless takeover. AUTO / MANUAL / ESTOP. Interlocks armed by operating state, a cause-dependent trip response, an E-Stop latch that resets only once its cause is gone. Alarm conditions with deadband and PLC events over one persistent MQTT connection; scan counters |
| api-gateway | `apps/api-gateway` | FastAPI edge. Sessions as refresh-token families with rotation and reuse detection; every request resolves the token to an active user, an open session and the current role. Sign-in throttling, httpOnly refresh cookie, user administration, append-only audit with filters and pages, Problem Details errors, readiness, WebSocket channels, plant snapshot, simulation control, history at automatic resolution, KPIs. gRPC clients to PLC, physics and alarms; MQTT subscriber for the live channels; the Alembic migration chain |
| historian | `apps/historian` | Records `sensors/*`, `alarms/changes`, `plc/events` and `status/+` into InfluxDB, labels values with the scenario, writes simulation events, its own counters, and applies the storage policy (7-day raw bucket, 90-day one-minute aggregates, downsampling task) |
| alert-manager | `apps/alert-manager` | Alarm lifecycle (ACTIVE_UNACK → ACTIVE_ACK → CLEARED, CLEARED_UNACK), one open alarm per condition, chatter hold on clears, reconciliation from source snapshots, transition history, `AlarmService` gRPC, changes on `alarms/changes` |
| opcua-server | `apps/opcua-server` | OPC UA (asyncua 2) address space of 84 read-only variables in ten folders with engineering units and instrument quality as status codes; PLC and alarm folders from `PLCService` and `AlarmService`; methods for load, mode, E-Stop reset, valves and acknowledgement, performed through the gateway as the signed-in user |
| web | `apps/web` | Operator console (React, TypeScript, Vite): sign-in, live SVG mimic, trends with history and KPIs, alarms with acknowledgement and an audible annunciator; control (load, mode, trip, manual valves, setpoints, E-Stop reset) behind confirmations; the engineer panel (simulation, scenarios, faults, run log); audit log, user administration and platform health; light and dark themes. Screens follow the role; the gateway enforces it. Talks only to the gateway — REST and `/ws` on the same origin |
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
faults with labels, per-instrument quality, the simulation status and `PerformanceMsg`: fuel
heat input, heat to the cycle, boiler and net efficiency, turbine and plant heat rates
[J/J], CO2 per joule of electricity and the true electrical output. Scenarios:
`steady_state` (250 MW), `part_load` (180 MW), `full_load` (300 MW), `hot_start`,
`cold_start`, `feedwater_pump_drill`.

### MQTT

Every client signs in with its own account (`physics-engine`, `plc-controller`,
`alert-manager`, `historian`, `api-gateway`, `opcua-server`, and `monitor` for the broker's
healthcheck); anonymous clients are refused. The ACL in
`infrastructure/docker/mosquitto/acl` lets each account publish only its own topics below
and read only what it consumes; a denied publish is dropped. There is no WebSocket
listener: the browser never speaks MQTT.

| Topic | Payload | Publisher → subscribers |
|---|---|---|
| `sensors/plant` | protobuf `PlantStatusMsg`: emissions, condenser, health, faults, instrument qualities, simulation status, valves, performance; published first in each step | physics-engine → historian, opcua-server |
| `sensors/boiler` | protobuf `BoilerStateMsg` (measured) | physics-engine → historian, opcua-server |
| `sensors/turbine` | protobuf `TurbineStateMsg` (measured) | physics-engine → historian, opcua-server |
| `sensors/system/heartbeat` | text timestamp | physics-engine → (skipped by subscribers) |
| `alerts/warning`, `alerts/critical` | JSON condition: `key`, `state` (`active`/`cleared`), `source_service`, `severity`, `parameter`, `direction`, `unit`, `value`, `threshold`, `action`, `message`, `raised_at_ms`, `timestamp_ms`, `alarm_id` | plc-controller → alert-manager |
| `alerts/snapshot` | JSON `source_service`, `active_keys`, `timestamp_ms` (on connect and every 10 s) | plc-controller → alert-manager |
| `plc/events` | JSON PLC event: `event_id`, `kind`, `source_service`, `operator_id`, `detail`, `timestamp_ms` | plc-controller → api-gateway (WebSocket `plc`), historian |
| `alarms/changes` | JSON `alarm`, `transition`, `timestamp_ms` | alert-manager → api-gateway (WebSocket `alarms`), historian, opcua-server |
| `status/physics-engine`, `status/plc-controller` | retained `online` / `offline` | the service itself (MQTT will) → historian |

Reserved for the deferred AI stage, not implemented: `insights/*`.

### REST (api-gateway)

People and tools reach the gateway through the `web` service (nginx) on
`http://localhost:8080`, or `https://localhost:8443` with the self-signed certificate from
`.env`; the gateway itself has no host port. nginx serves the console, proxies `/api`,
`/auth`, `/health`, `/ready`, `/ws` and the OpenAPI pages, keeps `/metrics` inside the
network, and adds a strict Content-Security-Policy, `X-Frame-Options`, `nosniff` and
`Referrer-Policy` (HSTS on HTTPS). The gateway takes the client address for the audit log
from nginx's `X-Forwarded-For`.

The OpenAPI schema is committed as `shared/openapi/api-gateway.json`; the console's
TypeScript types are generated from it (`apps/web/src/api/schema.gen.ts`, script
`generate-openapi`) and the gate fails when either falls behind the gateway's routes.

Every error is `application/problem+json` (RFC 9457) with `type`, `title`, `status`,
`detail`, `instance` and a stable `code` (for example `auth.invalid_credentials`,
`auth.forbidden`, `upstream.unavailable`, `request.invalid` with `errors`).

| Route | Minimum role |
|---|---|
| `GET /health` (liveness), `GET /ready` (database and upstreams; 503 without a database) | public |
| `POST /auth/login`, `POST /auth/refresh`, `POST /auth/logout` | public (credentials / token / cookie) |
| `GET /auth/me`, `POST /auth/password` | any signed-in user |
| `GET /api/v1/status`, `GET /api/v1/plant`, `GET /api/v1/plc/status`, `GET /api/v1/platform` | viewer |
| `GET /api/v1/simulation`, `GET /api/v1/simulation/scenarios`, `GET /api/v1/simulation/runs` | viewer |
| `GET /api/v1/history?measurement=&start_ms=&end_ms=&limit=&fields=` | viewer |
| `GET /api/v1/kpi?start_ms=&end_ms=` | viewer |
| `GET /api/v1/alarms`, `GET /api/v1/alarms/history`, `GET /api/v1/alarms/{alarm_id}` | viewer |
| `POST /api/v1/alarms/{alarm_id}/ack`, `POST /api/v1/alarms/ack-all` | operator |
| `POST /api/v1/commands/valve`, `/load`, `/mode` | operator |
| `POST /api/v1/commands/setpoint`, `/reset` | engineer |
| `POST /api/v1/simulation/pause`, `/resume`, `/speed`, `/step`, `/scenario`, `/faults`; `DELETE /api/v1/simulation/faults[/{fault_id}]` | engineer |
| `GET /api/v1/audit?user_id=&username=&method=&endpoint=&status=&min_status=&from_ms=&to_ms=&limit=&offset=` | admin |
| `GET, POST /api/v1/users`; `GET, PATCH /api/v1/users/{id}`; `POST /api/v1/users/{id}/password`, `/revoke-sessions` | admin |

Sessions: `/auth/login` returns an access token (15 min) and a refresh token that is also set
as an httpOnly, `SameSite=Strict` cookie on `/auth`. A refresh exchanges the token for a
successor with the session's original expiry (7 days after sign-in); presenting an exchanged
token again after 5 s closes the session. Sign-out, a password or role change and blocking an
account close sessions immediately. Five failed sign-ins per account name (twenty per client)
within 15 minutes answer 429, alike for existing and unknown accounts.

History picks a standard aggregation window (1 s … 1 day) so a range fits the point limit,
reads raw data for recent ranges of up to a day and one-minute means otherwise, and spans at
most 90 days. KPIs are ratios of means over the range (energy-weighted).

### WebSocket `/ws` (api-gateway)

JSON frames. The client authenticates with `{"type": "auth", "access_token": …}` within 5 s,
then `subscribe` / `unsubscribe` to channels, may renew its token in-band and `ping`. Channels:
`telemetry` (plant snapshots, at most the requested rate, capped at 10 Hz), `plc` (status
every second and PLC events), `alarms` (alarm changes). New subscribers first receive the
latest telemetry and PLC status. The server closes with 4401 when the token expires without
renewal, the session is closed or the account blocked (checked every 30 s), 4400 on a bad
frame and 1013 when a client cannot keep up with events.

### OPC UA (`opc.tcp://localhost:4840/cogniboiler`, namespace `urn:cogniboiler:simulation`)

Folders under `Objects/CogniBoiler` (NodeIds ns=2): Boiler (21xx), Turbine (22xx), Valves
(23xx), Emissions (240x), Condenser (245x), Performance (250x), Health (255x), Simulation
(26xx), PLC (27xx), Alarms (28xx). Variables are read-only with `EngineeringUnits`; status
codes carry instrument quality (`UncertainSensorNotAccurate`, `BadSensorFailure`) and stale
upstreams (`UncertainLastUsableValue`). Methods (29xx): `PLC/SetLoadDemand`,
`PLC/SetControlMode`, `PLC/ResetEmergencyStop`, `PLC/ApplyValveCommand`,
`Alarms/AcknowledgeAlarm`, `Alarms/AcknowledgeAllAlarms`, each returning `Accepted` and
`Reason`. Anonymous sessions browse and read; methods need a username session, signed in at
the gateway, and run with that user's role and audit trail.

Security: an application certificate (`urn:cogniboiler:opcua-server`, from `.env`, or a
temporary self-signed one) and two endpoints — `None`, where a username's password must
travel encrypted with the server's key (the token policy names Basic256Sha256; a password
sent in clear is refused with `BadIdentityTokenRejected`), and `Basic256Sha256 /
SignAndEncrypt`. Client certificates are accepted without a trust list: users are
authenticated by their password at the gateway.

### Logs and metrics

Every service writes one JSON object per line to standard output: `timestamp` (UTC),
`level`, `service`, `logger`, `event` and `correlation_id`, plus `exception` when there is
one. The gateway adopts a caller's `X-Correlation-ID` of a safe shape or starts a new id,
returns it in the response header, and passes it in gRPC metadata (`x-correlation-id`) to
the PLC, physics and alarm services, which log under it for the duration of the call; OPC
UA method calls start their own id and hand it to the gateway. The shared package
`shared/observability` (`cogniboiler_observability`) holds the log setup, the correlation
scope, the gRPC interceptors and the MQTT counters.

| Service | Metrics endpoint | Its own metrics |
|---|---|---|
| api-gateway | `:8000/metrics` | `http_requests_total` and `http_request_seconds` by method and route template, `gateway_websocket_clients`, `gateway_telemetry_age_seconds` |
| physics-engine | `:9100/metrics` | `physics_steps_total`, `physics_step_seconds`, simulation time, speed, running, run id, active faults |
| plc-controller | `:9100/metrics` | `plc_scan_seconds`, scans, commands received, refused and forwarded, warnings, trips, `plc_mode`, standing alarm conditions |
| historian | `:9100/metrics` | points written and failed, `historian_write_seconds`, skipped messages |
| alert-manager | `:9100/metrics` | `alarm_transitions_total` by severity and target state, failed messages |
| opcua-server | `:9100/metrics` | `opcua_method_calls_total` by method and outcome |

Every gRPC server counts and times its calls (`grpc_server_handled_total`,
`grpc_server_handling_seconds`); every MQTT client counts messages by topic
(`mqtt_messages_published_total`, `mqtt_messages_received_total`). Prometheus scrapes them
all every 10 s in the Compose profiles `observability` and `full` (the default of
`stack up`).

### Storage

- **PostgreSQL**, one Alembic chain in `apps/api-gateway/migrations`:
  `0001_initial_schema` — `users`, `roles`, `user_roles`, `audit_log` (and the former
  `token_blacklist`); `0002_alarm_lifecycle` — `alarm_events` (a partial unique index keeps
  one open alarm per condition key) and `alarm_transitions`, owned by alert-manager;
  `0003_sessions_and_append_only_audit` — `refresh_tokens` (replacing `token_blacklist`),
  `users.last_login_at_ms`, `audit_log.username`, `role`, `outcome`, triggers that refuse
  `UPDATE`, `DELETE` and `TRUNCATE` on `audit_log`, and `scenario_runs` (who loaded a
  scenario or injected or cleared a fault); `0004_application_roles` — roles
  `cogniboiler_gateway` (read and write its tables, only `SELECT` and `INSERT` on
  `audit_log`) and `cogniboiler_alarms` (the alarm tables). Neither owns a table, so neither
  can alter one, disable a trigger or truncate. Only the `migrate` job connects as the owner;
  it also gives the roles their passwords (`python -m api_gateway.db_roles`).
- **InfluxDB**: raw bucket from `.env` (7 days) with `boiler_sensors` (tags `quality`,
  `scenario`), `turbine_sensors` (tag `scenario`), `plant_status` (tag `scenario`; emissions,
  condenser, health, valves, performance, simulation, fault labels), `simulation_events`,
  `alarm_changes`, `plc_events`, `service_availability`, `historian_stats`; bucket
  `sensors_1m` (90 days) with one-minute `mean`/`min`/`max` (tag `agg`) of every float field
  of the first three, filled by the task `cogniboiler-downsample-1m`.

## Runtime and tooling

- `Dockerfile` has one target per Python service: uv builds each environment from
  `uv.lock` alone with the workspace packages as wheels, and the runtime is Python 3.14
  with that environment, the protobuf stubs and (for the gateway) the migrations — no uv,
  no sources, no pip, Debian security updates applied, user 10001. `apps/web/Dockerfile`
  builds the console and serves it from an unprivileged nginx 1.30. Images are named
  `${COGNIBOILER_REGISTRY:-cogniboiler}/<service>:${COGNIBOILER_TAG:-dev}`: built locally
  by default, or pulled from GitHub Container Registry with `stack up --no-build` when the
  two variables name a release.
- `docker-compose.yml` has four profiles: `infra` (broker, PostgreSQL, InfluxDB — for
  running the services from the host), `core` (infra, every service and the console),
  `observability` (Prometheus, Grafana, InfluxDB) and `full` (everything; the default of
  `stack up`). Every secret is interpolated from `.env`; host ports bind to `127.0.0.1`: the console
  and API (8080, 8443), MQTT (1883), OPC UA (4840), and for developers PostgreSQL,
  InfluxDB, Grafana and Prometheus. A one-shot
  `migrate` service applies Alembic before the gateway and alert-manager start; the gateway
  only seeds roles and demo users. Every long-running service has a healthcheck.
- Grafana is provisioned with InfluxDB and Prometheus datasources of fixed uids and four
  dashboards: Process, Efficiency and emissions, Alarms, and Platform — service health,
  request rates and latency, physics step and PLC scan times, gRPC, MQTT, historian writes
  and alarm transitions from Prometheus, then availability and data flow from InfluxDB —
  with scenario, fault, PLC and critical-alarm annotations.
- Prometheus (profiles `observability` and `full`, port 9090 on loopback) scrapes every
  service.
- The gateway seeds demo users `admin`, `engineer`, `operator`, `viewer` from
  `DEMO_*_PASSWORD` when `AUTO_INIT_DB` is set; no credential is hardcoded.
- `smoke` checks a running stack through the gateway: health and readiness, logins, role
  refusals, live state, a setpoint accepted by the PLC, alarms, history, KPIs and the audit
  log of sign-ins.
- CI (`.github/workflows/ci.yml`): `gate` runs the quality gate; `audit` audits the locked
  dependencies (`audit-deps`: pip-audit and pnpm audit); `stack` builds every image, starts
  the whole stack with throwaway secrets, runs `smoke` and the Playwright console checks
  through nginx and scans all seven images with Trivy, failing on a HIGH or CRITICAL
  finding that has a fix. Reports are kept as artifacts. On `main` and on tags `v*`, after
  all three, `publish` pushes every image to `ghcr.io/<owner>/cogniboiler/<service>`
  (tags `main`, `sha-…`, and for a release `vX.Y.Z`, `vX.Y` and `latest`); a tag `v*`
  then gets a GitHub release with generated notes, `docker-compose.yml` and
  `.env.example`.
- `apps/web` is the operator console: React 19 + TypeScript (6.0) + Vite 8, React Router,
  TanStack Query and uPlot. One HTTP module keeps the access token in memory and refreshes
  it through the httpOnly cookie; one WebSocket client authenticates in the first frame and
  renews in-band; one module converts SI units for display. The Vite dev server proxies
  `/api`, `/auth`, `/health` and `/ws` to the gateway, so the browser sees one origin.
  Vitest covers the modules and screens; Playwright (`apps/web/e2e`, script `console-e2e`)
  checks the console against a running stack with the demo users from `.env`, including the
  five-minute demo played by an operator, an engineer and an admin at once.
- `python dev_tools_scripts_runner.py` is the developer-tools orchestrator: `quality-gate`,
  `format-code`, `audit-deps`, `sync-agents`, `stack`, `dev-secrets`, `smoke`, `console-e2e`,
  `generate-proto`, `generate-openapi`, `doctor`, `install-hooks`, `clean-caches`,
  `selftest`.
- The quality gate runs ruff, strict mypy, every service test suite, the protobuf,
  OpenAPI and `AGENTS.md` sync checks, the scripts' own tests, and the frontend checks when
  `apps/web/node_modules` exists.
- `ml/preprocessing/generate_dataset.py` runs labelled closed-loop episodes — the
  deterministic plant scanned by the real PLC logic — for the deferred AI stage.

## Known gaps

Recorded with their planned fix in the internal roadmap:

- the logic added in S2–S5, S8 and S10 has no dedicated tests yet (owner decision for that
  work); of the PLC integration tests, only the AUTO hold test runs in lockstep with the
  plant, the others still pace it by wall clock;
- sign-in throttling state lives in the single gateway process;
- OPC UA accepts any client certificate (no trust list).
