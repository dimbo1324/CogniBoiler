# CogniBoiler Service Boundaries

Which service owns which responsibility. New logic goes to the owning service and is never
duplicated across services. Invariants I1–I3 in `invariants.md` rest on this table.

## Ownership

| Service | Owns | Must not own |
| --- | --- | --- |
| `physics-engine` | Process state, the integration step, the live runtime, MQTT telemetry, `PhysicsService` | Control policy, operator auth, alarm routing |
| `plc-controller` | Control intent: setpoints, AUTO/MANUAL/ESTOP, PID loops, safety interlocks, E-Stop latch, validated command forwarding, alarm publishing | Canonical process state, HTTP/auth, telemetry storage |
| `alert-manager` | Alarm persistence, deduplication and (planned) lifecycle: acknowledge, return to normal | Process state, authentication, control |
| `historian` | Time-series ingestion into InfluxDB | Control decisions, alarm policy |
| `opcua-server` | Projection of live state to OPC UA clients | Control ownership (writes, when added, go through the PLC), persistence |
| `api-gateway` | The edge for people: authentication, RBAC, audit, REST, WebSocket, orchestration of calls | The integration step, PLC internals, alarm state |
| `web` (planned, `apps/web`) | Presentation: screens, unit conversion for display | Business rules, authorization decisions |

## Command and state flow

1. `api-gateway` authenticates and authorizes the caller, audits the request, and forwards a
   control request to `plc-controller`.
2. `plc-controller` validates it, applies the interlocks, and forwards the accepted command
   to `physics-engine`.
3. `physics-engine` applies it on the next step and exposes the resulting state through
   `PhysicsService` and MQTT.
4. `historian` and `opcua-server` consume telemetry as observers only.
5. `plc-controller` publishes alarm events; `alert-manager` stores them. Nobody but the
   physics engine mutates the simulator.

## Recorded boundary debt

These cross the table above today. They are debt with a planned fix, not a pattern to copy —
add no new cross-service imports.

| Debt | Where | Planned fix |
| --- | --- | --- |
| PID and safety code live in `physics_engine.controller`, `physics_engine.pid` and `physics_engine.safety`, imported by the PLC | `apps/plc-controller/src/plc_controller/service.py` | move into `plc_controller` (roadmap S3) |
| The gateway imports `alert_manager.models` and reads `alarm_events` directly | `apps/api-gateway/src/api_gateway/routers/alarms.py`, `db_init.py` | an `AlarmService` gRPC API owned by alert-manager (roadmap S4) |
| `alarm_events` is created with `create_all` outside the Alembic chain | `apps/alert-manager/src/alert_manager/db.py` | one migration chain applied by a `migrate` job (roadmap S1/S4) |
