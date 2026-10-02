# Architecture: what exists today

> This document describes what is **actually in the code**, not what is planned. It is
> updated whenever the shape of the system changes: a new service, endpoint group, topic,
> table, screen, or operational job. The plan lives in the internal roadmap.

**Last revised:** 2026-10-01 · **Version:** 0.1.0 · **Python:** 3.14

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
 ┌───────────────────────────── Mosquitto (:1883) ─────────────────────────┤
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
| `InsightService` | reserved for the deferred AI stage; no service, no port, nothing depends on it |

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
listener: the browser never speaks MQTT. Every topic name is declared once, in
`cogniboiler_runtime.topics`, and a test reads the ACL against that module.

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

Reserved for the deferred AI stage, not implemented: the MQTT topics `insights/*`
(anomalies, recommendations, equipment health), the gRPC `InsightService` and the REST
route group `/api/v1/insights`. The console keeps a hidden slot for them: the
`Recommendations` panel of the overview renders nothing unless a build sets
`VITE_INSIGHTS=true` (`apps/web/src/insights`). No service, screen or test needs any of
them to run.

### REST (api-gateway)

People and tools reach the gateway through the `web` service (nginx) on
`http://localhost:8080`, or `https://localhost:8443` with the self-signed certificate from
`.env`; the gateway itself has no host port. nginx serves the console, proxies `/api`,
`/auth`, `/health`, `/ready`, `/ws` and the OpenAPI pages, keeps `/metrics` inside the
network, and adds the same headers to every location: a strict Content-Security-Policy
without `unsafe-inline`, `X-Frame-Options`, `nosniff`, `Referrer-Policy`, and HSTS on HTTPS
for any host but `localhost`, `127.0.0.1` and `[::1]`. `/docs` and `/redoc` load Swagger UI
5.33.0 and ReDoc 2.5.4 from the CDN at pinned versions with Subresource Integrity, so those
two pages need the internet; nothing else does. The gateway itself answers `Cache-Control:
no-store` under `/auth`, `/api/v1/users` and `/api/v1/audit` and `nosniff` everywhere, and
takes the client address for the audit log from nginx's `X-Forwarded-For` (the OPC UA server
forwards its own client's address the same way).

The OpenAPI schema is committed as `shared/openapi/api-gateway.json`; the console's
TypeScript types are generated from it (`apps/web/src/api/schema.gen.ts`, script
`generate-openapi`) and the gate fails when either falls behind the gateway's routes.

Every error is `application/problem+json` (RFC 9457) with `type`, `title`, `status`,
`detail`, `instance` and a stable `code` (for example `auth.invalid_credentials`,
`auth.forbidden`, `upstream.unavailable`, `request.invalid` with `errors`). An upstream gRPC
failure answers 503 `upstream.unavailable`, a deadline 504 `upstream.timeout`, and a request
the upstream refuses as invalid 422 `request.invalid`; each is counted in
`gateway_upstream_failures_total{service,code}`.

| Route | Minimum role |
|---|---|
| `GET /health` (liveness), `GET /ready` (database and upstreams, cached for 1.5 s; 503 without a database) | public |
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
account close sessions immediately. Access tokens are RS256 and carry `iss`, `aud` and `sid`;
the gateway refuses to start without a matching RSA key pair of at least 2048 bits. Five
failed sign-ins per account name (twenty per client, an IPv6 client counted by its /64)
within 15 minutes answer 429, alike for existing and unknown accounts; an attempt is charged
before its password is checked, so a burst of parallel guesses cannot pass the limit, and
at most four Argon2 checks run at once. The audit row of a request that carries a password
(sign-in, password change, user creation, password reset) keeps no body digest.

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
or binary frame, and 1013 when a client cannot keep up with events or `WS_MAX_CONNECTIONS`
(200) sockets are already open; frames are at most 64 KiB. A full send queue drops the
incoming telemetry frame, never a queued event. Every non-normal close is logged and
counted.

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
authenticated by their password at the gateway. Only RSA-OAEP password encryption is
accepted; session tokens are random; a re-activated session must present the certificate of
the channel that created it; requests are capped near 4 MiB, sessions and connections at 50,
subscriptions at 100. Each OPC UA session signs in at the gateway once and signs out once.

### Logs and metrics

Every service writes one JSON object per line to standard output: `timestamp` (UTC),
`level`, `service`, `logger`, `event` and `correlation_id`, plus `exception` when there is
one. The gateway adopts a caller's `X-Correlation-ID` of a safe shape or starts a new id,
returns it in the response header, and passes it in gRPC metadata (`x-correlation-id`) to
the PLC, physics and alarm services, which log under it for the duration of the call; OPC
UA method calls start their own id and hand it to the gateway. The shared package
`shared/observability` (`cogniboiler_observability`) holds the log setup — which masks any
field whose name contains password, secret, token, authorization, cookie or api key, and escapes control
characters in console mode — the correlation scope, the gRPC interceptors with
`observed_channel` and `serve_until_cancelled`, and the MQTT counters.

A second shared package, `shared/runtime` (`cogniboiler_runtime`), holds what every
service does the same way and no service owns: the MQTT session loop — connect, work,
warn once per broker outage, log a defect in the work at `error` with its traceback, wait
with jitter, reconnect, forever, and never spin — the queued publisher
(`QueuedMqttPublisher`: bounded, drops the oldest and counts it, sends a retained `offline`
before it stops), `run_service` (every asyncio entry point: the selector loop on Windows,
and SIGTERM cancels the service so its cleanup runs in a container), `OutageLog`, the JSON
payload decoder (`decode_json_object`: no NaN or Infinity, bounded size and depth, never
raises), the topic names, the Flux string literal, the clock, and the liveness file behind
the healthchecks of the two services that expose no port. A service still builds
its own client, so the address, the credentials, the will and a persistent session stay
its own decision; physics-engine names `RuntimeUnavailableError` fatal, so a stopped
runtime ends its mirror instead of reconnecting to a broker with nothing to say.

With `LOG_DIR` set, a service also writes the same lines, always as JSON, to
`<LOG_DIR>/<service>.log`, rotated by size (`LOG_FILE_MAX_BYTES`, 10 MiB by default; the
`LOG_FILE_BACKUPS` newest older files, 5 by default). In the Compose stack every Python
service writes into the repository's `logs/` directory through a bind mount
(`db-roles.log` from the one-shot `migrate`, then one file per service), and `stack up`
creates that directory; Docker keeps at most 3 × 10 MB of every container's own output. A
directory that cannot be written leaves the service on standard output with a warning — it
never stops a service.

| Service | Metrics endpoint | Its own metrics |
|---|---|---|
| api-gateway | `:8000/metrics` | `http_requests_total` and `http_request_seconds` by method (unknown methods as `other`) and route template, `gateway_websocket_clients`, `gateway_telemetry_age_seconds`, `gateway_login_failures_total{reason}`, `gateway_refresh_rejections_total{code}`, `gateway_audit_write_failures_total`, `gateway_upstream_failures_total`, `gateway_websocket_closes_total`, `gateway_websocket_dropped_frames_total` |
| physics-engine | `:9100/metrics` | `physics_steps_total`, `physics_step_seconds`, simulation time, speed, run id, active faults, `physics_simulation_running` (1 only while healthy and not paused), `physics_runtime_failures_total`, `physics_commands_refused_total{rpc}`, `physics_property_fallbacks_total{function}` |
| plc-controller | `:9100/metrics` | `plc_scan_seconds`, scans, commands received, refused and forwarded, warnings, trips, `plc_mode`, standing alarm conditions, `plc_plant_link_up`, plant stream, forward and scan failures, `plc_publish_dropped_total`, `plc_mqtt_connected` |
| historian | `:9100/metrics` | points written and failed, `historian_write_seconds`, skipped messages and dropped fields |
| alert-manager | `:9100/metrics` | `alarm_transitions_total` by severity and target state, messages failed and rejected by reason, `alarm_snapshot_unmatched_keys_total`, `alarm_changes_dropped_total` |
| opcua-server | `:9100/metrics` | `opcua_method_calls_total` by method and outcome, skipped bridge messages |

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
  it also gives the roles their passwords (`python -m api_gateway.db_roles`);
  `0005_narrow_application_grants` — the gateway role may not update or delete
  `scenario_runs` nor delete users and roles, the alarm role may not update or delete
  `alarm_transitions` nor delete `alarm_events`; `0006_case_insensitive_usernames` — a unique
  index on `lower(username)`; `0007_audit_log_keeps_its_authors` — deleting a user who has
  audit rows is refused. `smoke` proves on the running database that the gateway role cannot
  update, delete or truncate `audit_log` and that the triggers refuse the owner too.
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
  InfluxDB, Grafana and Prometheus. Seven networks keep each container next to what it
  talks to: `plant` (physics-engine with plc-controller, the gateway, the broker and
  Prometheus), `control` (plc-controller with the gateway and opcua-server), `services`
  (the broker, the MQTT services, opcua-server, the gateway and Prometheus), `postgres`,
  `influx`, `edge` (nginx and the gateway) and `dashboards` (Grafana and Prometheus). Every
  container runs with `no-new-privileges` and memory and process limits; the services, the
  console, Grafana and Prometheus also drop every capability and run on a read-only root
  with a `tmpfs` for `/tmp`. Base and third-party images are pinned by digest. A host run of
  physics-engine, plc-controller or the gateway listens on `127.0.0.1` unless told otherwise
  (`--grpc-host`, `--host`); Compose and the images pass `0.0.0.0` inside the private
  networks. The physics healthcheck requires the runtime status `running`. A one-shot
  `migrate` service applies Alembic before the gateway and alert-manager start; the gateway
  only seeds roles and demo users. Every long-running service has a healthcheck. Every
  service reconnects by itself when the broker, PostgreSQL or InfluxDB restarts: after
  `docker compose restart` of the whole stack, or of any one of them, every container is
  healthy and the gateway ready again within about 15 s.
- Grafana 13 is provisioned with read-only InfluxDB and Prometheus datasources of fixed
  uids, makes no calls to grafana.com, has alert rules for a failed audit write and a reused
  refresh token, and four dashboards: Process, Efficiency and emissions, Alarms, and Platform — service health,
  request rates and latency, physics step and PLC scan times, gRPC, MQTT, historian writes
  and alarm transitions from Prometheus, then availability and data flow from InfluxDB —
  with scenario, fault, PLC and critical-alarm annotations.
- Prometheus (profiles `observability` and `full`, port 9090 on loopback) scrapes every
  service.
- The gateway seeds demo users `admin`, `engineer`, `operator`, `viewer` from
  `DEMO_*_PASSWORD` when `AUTO_INIT_DB` is set; no credential is hardcoded.
- `backup` writes `backups/<UTC stamp>/` with a `pg_dump` of PostgreSQL and an
  `influx backup` of InfluxDB, both taken inside their own container so no credential
  reaches a command line; `restore` puts a folder back after asking, stopping the four
  services that hold connections while it does. The PostgreSQL restore runs in a single
  transaction, stops at the first failing step and restarts the services; a folder without
  its manifest is refused unless `--force` is given. The InfluxDB restore is `--full`, so a
  backup belongs to the `.env` it was taken with. `.env` and backups are written readable
  by their owner only.
- `demo` plays the five-minute scenario of VISION §7 against a running stack through the
  gateway — nominal scenario, 300 MW, a feedwater pump failure, warning and critical
  alarms, the trip, acknowledgement, repair, E-Stop reset, back on load, then the audit
  log — at ten times speed, leaves the unit as it found it, and fails when any service
  wrote an `error` line while it ran.
- `smoke` checks a running stack through the gateway: health and readiness, logins, role
  refusals, live state, a setpoint accepted by the PLC, alarms, telemetry recorded in the
  last 30 s, KPIs, the audit log of sign-ins, and that `audit_log` is append-only on the
  database itself (`--skip-database` leaves that last check out).
- CI (`.github/workflows/ci.yml`): `gate` runs the quality gate; `audit` audits the locked
  dependencies (`audit-deps`: pip-audit and pnpm audit); `stack` builds every image, starts
  the whole stack with throwaway secrets, runs `smoke` and the Playwright console checks
  through nginx, checks that every service wrote its log file, and scans all seven images
  with Trivy, failing on a HIGH or CRITICAL finding that has a fix, and checks the
  Compose exposure rules (`infrastructure/checks/compose_exposure.py`). Reports and the log
  files are kept as artifacts. Third-party actions are pinned by commit SHA. On `main` and
  on tags `v*`, after all three, `publish` pushes every image by digest, scans exactly that
  digest, and only then tags it, to `ghcr.io/<owner>/cogniboiler/<service>`
  (tags `main`, `sha-…`, and for a release `vX.Y.Z`, `vX.Y` and `latest`); a tag `v*`
  then gets a GitHub release with generated notes, `docker-compose.yml` and
  `.env.example`.
- `apps/web` is the operator console: React 19 + TypeScript (6.0) + Vite 8, React Router,
  TanStack Query, uPlot and Lucide icons. One HTTP module keeps the access token in memory
  and refreshes it through the httpOnly cookie; one WebSocket client authenticates in the
  first frame and renews in-band; one module converts SI units for display, and one names
  the SI unit a contract field is called after. Every screen is built from the same few
  primitives — a titled panel, a state badge, an error, empty or framed line, an icon —
  and the icon vocabulary is chosen once in `components/ui/icons.ts`, so one meaning is
  one glyph everywhere. An icon is always `aria-hidden` beside a real label. The console
  is dark by default (a control room is dim and a white mimic is glare); the operator can
  choose light or follow the system, and only the resolved theme reaches the page as
  `data-theme`, so the stylesheet declares each palette once. The Vite dev server proxies
  `/api`, `/auth`, `/health` and `/ws` to the gateway, so the browser sees one origin.
  Vitest covers the modules and screens; Playwright (`apps/web/e2e`, script `console-e2e`)
  checks the console against a running stack with the demo users from `.env`, including the
  five-minute demo played by an operator, an engineer and an admin at once, every screen on
  a tablet (768×1024, 1024×768) and a projector (1280×720), and how a session ends when an
  administrator closes it, when the browser loses it and on sign-out.
- `python dev_tools_scripts_runner.py` is the developer-tools orchestrator: `quality-gate`,
  `format-code`, `audit-deps`, `sync-agents`, `stack`, `dev-secrets`, `smoke`, `console-e2e`,
  `generate-proto`, `generate-openapi`, `doctor`, `install-hooks`, `clean-caches`,
  `demo`, `backup`, `restore`, `readme-media`, `selftest`. `readme-media` takes the README's
  screenshots and GIF from a running stack with a Playwright spec of its own
  (`apps/web/readme/`, which neither `console-e2e` nor CI runs) and writes them, each under
  the 500 KiB large-file limit, into `docs/images/`.
- The quality gate runs ruff, strict mypy (the developer scripts included), a check that
  every test module name is unique, every service test suite with a floor of 95 % line
  coverage, the protobuf, OpenAPI and `AGENTS.md` sync checks, the scripts' own tests, and
  the frontend checks when `apps/web/node_modules` exists. The service suites need no broker, database or network:
  gRPC servers run in process on port 0, MQTT and HTTP peers are fakes, storage is SQLite
  or a parsed line protocol, and the PLC tests step the physics runtime in lockstep instead
  of waiting on the wall clock.
- `ml/preprocessing/generate_dataset.py` runs labelled closed-loop episodes — the
  deterministic plant scanned by the real PLC logic — for the deferred AI stage.

## Known gaps

Recorded with their planned fix in the internal roadmap:

- an InfluxDB restart leaves a gap in the recorded history: the historian counts and drops
  the batches it cannot write (`historian_points_failed_total`) instead of buffering them;
- sign-in throttling state lives in the single gateway process;
- OPC UA accepts any client certificate (no trust list);
- the internal gRPC services authenticate no caller: only the Compose networks keep other
  containers away from `PhysicsService` and `PLCService` (the broker and Prometheus share
  the plant network);
- the gateway, Grafana and the historian share one all-access InfluxDB token;
- the E-Stop latch lives in the PLC's memory: a PLC restarted while tripped comes back in
  AUTO without an operator reset, and says so in a warning at start-up;
- the superheater model has no metal heat capacity: at low steam flow its outlet
  temperature jumps (below saturation on a sudden flow, to furnace gas temperature behind
  shut valves), and a restart after an E-Stop reset at ten times real speed can trip the
  unit a second time on high steam temperature. Restart at three times speed or slower —
  `demo` does — and reset a latched trip before reloading a scenario.
