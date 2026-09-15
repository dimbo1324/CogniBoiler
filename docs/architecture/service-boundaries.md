# CogniBoiler Service Boundaries

Which service owns which responsibility. New logic goes to the owning service and is never
duplicated across services. Invariants I1–I3 in `invariants.md` rest on this table.

## Ownership

| Service | Owns | Must not own |
| --- | --- | --- |
| `physics-engine` | Process state, the integration step, the live runtime and simulation control (pause, speed, scenarios), faults and instrument behaviour, MQTT telemetry, `PhysicsService` | Control policy, operator auth, alarm routing |
| `plc-controller` | Control intent: load demand, setpoints, AUTO/MANUAL/ESTOP, coordinated control loops, safety interlocks, E-Stop latch and reset permission, validated command forwarding, alarm condition detection, PLC events | Canonical process state, HTTP/auth, alarm lifecycle, telemetry storage |
| `alert-manager` | Alarm lifecycle: activation, acknowledgement, return to normal, history; the `alarm_events` and `alarm_transitions` data; `AlarmService` | Process state, authentication, control, deciding what is an alarm condition |
| `historian` | Time-series ingestion into InfluxDB | Control decisions, alarm policy |
| `opcua-server` | Projection of live state to OPC UA clients | Control ownership (writes, when added, go through the PLC), persistence |
| `api-gateway` | The edge for people: authentication, RBAC, audit, REST, WebSocket, orchestration of calls; the Alembic migration chain | The integration step, PLC internals, alarm state |
| `web` (planned, `apps/web`) | Presentation: screens, unit conversion for display | Business rules, authorization decisions |

## Command and state flow

1. `api-gateway` authenticates and authorizes the caller, audits the request, and forwards a
   control request to `plc-controller`.
2. `plc-controller` validates it, applies the interlocks, and forwards the accepted command
   to `physics-engine`.
3. `physics-engine` applies it on the next step and publishes the resulting state through
   `PhysicsService` and MQTT.
4. `plc-controller` scans every published state; `historian` and `opcua-server` consume
   telemetry as observers only.
5. `plc-controller` publishes alarm conditions; `alert-manager` owns the alarms they become,
   and the gateway reads and acknowledges them only through `AlarmService`. Nobody but the
   physics engine mutates the simulator.

## Recorded boundary notes

- The migration chain for every PostgreSQL table, alarm tables included, lives in
  `apps/api-gateway/migrations` and is applied by the `migrate` job; alert-manager owns the
  alarm data but not the chain (decision in the internal decision log, Q2).
- `plc-controller` imports `physics_engine` only in its tests, as an in-process plant; the
  dependency is a development group, not a runtime dependency.
