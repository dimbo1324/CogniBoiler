# Domain Rules and House Style (CogniBoiler)

These sharpen the universal rules for this codebase. Stricter wins.

## Python code

- Python 3.14 with full type hints; `mypy --strict` over `apps/` and `shared/` is a gate
  section. A `# type: ignore` carries its error code.
- A module past roughly 700 lines is split by meaning. `__main__.py` entry points hold
  argument parsing and wiring only.
- A service never imports another service's internals. The two existing exceptions
  (`plc-controller` → `physics_engine`, `api-gateway` → `alert_manager.models`) are
  recorded debt in `docs/architecture/service-boundaries.md`; add no new ones — talk over
  gRPC, MQTT or the database contract instead.
- Configuration comes from environment variables (pydantic-settings, or argparse defaults
  for local runs), never from constants edited per machine.

## Units, time and contracts

- SI units in every contract, message and column: Pa, K, kg/s, W, m, s. Convert to bar,
  °C or MW only at a presentation edge, and name the unit in the field (`pressure_pa`).
- Timestamps are UTC epoch milliseconds (`timestamp_ms`) in contracts and storage.
- `shared/proto/cogniboiler.proto` is a contract: add fields, never renumber or reuse a
  field number; regenerate the stubs with `generate-proto` in the same commit.
- MQTT topics and payloads (`sensors/*` protobuf, `alerts/*` JSON, `status/*` retained
  availability) are a contract listed in `docs/architecture/overview.md`. Changing one
  updates every publisher and subscriber in the same task.
- Database schema changes go through Alembic migrations only.

## Control and safety

- The PLC is the only path from an operator to an actuator. The API gateway calls
  `PLCService`; it never calls `PhysicsService.ApplyControlCommand`.
- Interlocks and the E-Stop latch cannot be disabled by configuration, an API parameter,
  or a flag. Resetting a trip is an explicit, role-checked, audited action.
- Valve commands are validated to [0, 1] at the PLC and again in the physics runtime.
- A change to plant parameters, PID tuning or interlock thresholds states its physical
  reasoning in the commit body and is covered by a test of the behavior it changes.

## Async services

- No blocking work on the event loop: physics steps run in `asyncio.to_thread`, and
  blocking client libraries are wrapped the same way.
- Network clients reconnect with a delay and log once per failure; they never spin.
- Entry points pass `loop_factory=asyncio.SelectorEventLoop` on Windows, because aiomqtt
  needs `add_reader()`, which the proactor loop lacks.

## Security

- Keys, passwords and tokens come from `.env` or the environment; `dev-secrets` generates
  local ones. `.env` and `certs/*.pem` are never committed.
- Every mutating API route requires a role (`viewer` < `operator` < `engineer` < `admin`)
  and writes an audit entry. Default users are seeded only for local development.
- Passwords are Argon2id hashes; a failed login never reveals whether the user exists.

## Tests

- Unit tests need no broker, database or internet: use fakes, SQLite in memory, or
  in-process gRPC servers on port 0.
- A test whose outcome depends on machine speed is a defect to fix, not a test to loosen.
- Never delete, skip or weaken a test to make the gate green.

## Frontend (`apps/web`)

- React + TypeScript in strict mode + Vite. The browser talks only to the API gateway
  (REST and WebSocket) — never to MQTT, gRPC or a database.
- Unit conversion and formatting live in one presentation module; business rules stay in
  the backend. The console is minimal and functional unless a task is about appearance.

## Assistant workspaces

- `.claude/agents|skills` and `.codex/agents|skills` are name-for-name mirrors.
- `.claude/settings.json` allowlists routine read and verification commands and denies
  destructive git and Docker volume operations. Extend the allowlist rather than routing
  around it; never remove a deny entry without explicit owner approval.
