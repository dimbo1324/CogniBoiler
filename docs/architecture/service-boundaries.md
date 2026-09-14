# CogniBoiler Service Boundaries

This document is the current source of truth for service ownership while Phase 3-5 logic is being split into stable microservice boundaries.

## Target split

| Service | Owns | Must not own |
| --- | --- | --- |
| `physics-engine` | Physical process state, integration step, live simulator runtime, MQTT telemetry publishing, `PhysicsService` gRPC state API | PID policy, operator auth, alarm routing |
| `plc-controller` | PID loops, safety policy, validated operator commands, setpoints, command forwarding to physics | Canonical process state, HTTP/auth, telemetry storage |
| `historian` | Time-series ingestion from telemetry and persistence to InfluxDB | Control decisions, alarm policy |
| `opcua-server` | Projection of current process state to industrial clients | Control ownership, persistence |
| `api-gateway` | Edge API, auth, RBAC, audit, client-facing orchestration | Physics integration step, PLC internals |
| `alert-manager` | Alarm rules, deduplication, severity, notification fan-out | Physics state ownership, authentication |

## Command and state flow

1. `api-gateway` authenticates the caller and forwards a control request to `plc-controller`.
2. `plc-controller` validates the command, applies PID/safety policy, and forwards the accepted command to `physics-engine`.
3. `physics-engine` updates the live simulator state and exposes the resulting process snapshot through `PhysicsService`.
4. `historian` and `opcua-server` consume state as downstream observers only.
5. `alert-manager` evaluates alarms from state/telemetry; it never mutates the simulator.

## What stays where now

- `physics-engine` remains the owner of:
  - live boiler/turbine state
  - physics integration and MQTT publishing
  - the new `PhysicsService`
- `plc-controller` remains the owner of:
  - command validation
  - setpoint storage
  - forwarding accepted commands to `PhysicsService`
- `historian`, `opcua-server`, `api-gateway`, and `alert-manager` keep their existing responsibilities unchanged.

## What moves next

- Move PID ownership from `apps/physics-engine/src/physics_engine/controller.py` into `plc-controller`.
- Move protection ownership from `apps/physics-engine/src/physics_engine/safety.py` into `plc-controller`.
- Replace `api-gateway` status/command stubs with real gRPC clients to `plc-controller` and `physics-engine`.
- Add alarm evaluation and storage flow in `alert-manager`.

## Non-negotiable rules

- `physics-engine` is the single source of truth for live process state.
- `plc-controller` is the single source of truth for control intent and setpoints.
- Downstream services consume state; they do not rewrite it.
- New logic must be added to the owning service, not duplicated across services.
