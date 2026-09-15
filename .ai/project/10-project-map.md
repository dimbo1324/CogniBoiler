# Project: CogniBoiler

A digital twin of a 300 MW gas-fired steam unit — boiler, turbine and a virtual PLC —
built as Python microservices speaking industrial protocols (MQTT, gRPC, OPC UA), with a
time-series historian, an authenticated API and a web operator console. It is a
portfolio project: it must run and demo end to end on one machine with Docker Compose.

**AI is deferred.** The anomaly, efficiency and maintenance models planned for
`apps/ai-predictor` and `ml/` are out of scope until the owner reopens that stage
(decision 2026-09-14). The platform must be complete and demonstrable without them.

## Repository map

uv workspace on Python 3.14; every service is a package with `src/` and `tests/`:

- `apps/physics-engine` — thermodynamic boiler/turbine model, the live runtime, the
  `PhysicsService` gRPC API, MQTT telemetry. The single owner of process state.
- `apps/plc-controller` — virtual PLC: command validation, setpoints, cascade PID in
  AUTO, safety interlocks and the E-Stop latch, `PLCService` gRPC API, alarm publishing.
- `apps/api-gateway` — FastAPI edge: JWT RS256, RBAC, audit log, REST and WebSocket,
  gRPC clients to the PLC and physics services, PostgreSQL via SQLAlchemy async, Alembic
  migrations in `apps/api-gateway/migrations`.
- `apps/historian` — MQTT telemetry subscriber writing to InfluxDB.
- `apps/alert-manager` — MQTT alarm subscriber persisting alarm events to PostgreSQL.
- `apps/opcua-server` — OPC UA (IEC 62541) projection of live state.
- `apps/web` — the operator console (React + TypeScript + Vite, pnpm).
- `apps/ai-predictor` — deferred placeholder, **not** a workspace member.

Shared and supporting areas:

- `shared/proto/cogniboiler.proto` — the gRPC and telemetry contract; `shared/generated/`
  holds the committed stubs; `shared/models/` holds cross-service Pydantic models.
- `docker-compose.yml`, `Dockerfile`, `.env.example` — the local stack.
- `infrastructure/` — Mosquitto and Grafana provisioning; Kubernetes/Helm later.
- `ml/` — deferred AI material.
- `dev_tools_scripts_runner.py` + `scripts/` — the developer-tools orchestrator.
  `scripts/runner` is the orchestrator itself, `scripts/_toolkit` the only shared code,
  one directory per script with its own `config/*.json`. Scripts never import each other.
- `.ai/`, `.claude/`, `.codex/` — assistant rules and workspaces.

## Internal vs external documents

A document serves exactly one audience.

**Internal — everything in `docs/__arch__/`, written in Russian.** For whoever builds
the project: `VISION.txt` (the product vision without AI), `ROADMAP.md` (stages and what
is done), `open-questions.md` (owner decisions), `archive/` (superseded plans). Nothing
a user reads links to them.

**External — written in English.** For whoever picks the project up: `README.md` (the
hub; every external document is reachable from it), `docs/architecture/overview.md`,
`docs/architecture/invariants.md`, `docs/architecture/service-boundaries.md`.

## Language policy

- **English**: `.ai/`, `.claude/`, `.codex/`, `CLAUDE.md`, `AGENTS.md`,
  `task-checklist.md`, code, comments, commit messages, test names, UI copy, and every
  external document.
- **Russian**: the internal documents in `docs/__arch__/`, and reports to the owner.
- Do not mix languages inside one file.

## Documentation policy

- `docs/__arch__/VISION.txt` changes only when the product intent changes, by owner
  agreement. `docs/__arch__/ROADMAP.md` is the plan and progress record.
- The no-new-docs exception (`03-scope-and-code-style.md`) covers `docs/architecture/`,
  `README.md` and `docs/__arch__/ROADMAP.md`.

## Product guardrails

- `physics-engine` is the only writer of process state; every other service observes.
- Every actuator command reaches the plant through `plc-controller`; nothing bypasses
  validation, the interlocks, or an active E-Stop.
- The whole platform starts with `stack up` on one machine and demos without internet.
- AI stays optional: no service may need `ai-predictor` to start or to pass its tests.
- **Stage order in `docs/__arch__/ROADMAP.md` is binding.** Skipping ahead needs an owner
  decision recorded in `docs/__arch__/open-questions.md`.
