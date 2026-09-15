# Task: S2–S4 business logic — physics, PLC, alarms

Owner request of 2026-09-15: implement the business logic of roadmap stages S2, S3 and S4
"without tests and the rest", carefully and at senior level. Work on branch
`feat/s2-s4-physics-plc-alarms`, merged into `main` locally after a green gate; no push until
the owner asks.

Marks: `[ ]` open, `+` done, `-` not done or partially done (with a note).

## Preparation

+ Orientation: roadmap S2–S4, vision §5.1–5.3, every affected service, its contracts and tests
+ Owner instruction recorded: no new tests for S2–S4 (decision log, debt Д11)

## S2 — physics

+ Energy-conserving 300 MW plant recalibrated from a rated heat balance; IAPWS-IF97 saturation tables
+ Steady operating points for any load; the 250 MW point holds 8 h open loop with zero drift
+ Deterministic PlantSimulator; runtime with pause, step on request, speed, scenario load and faults
+ Condenser, emissions and equipment wear in the live loop; telemetry of measured values and instrument quality
+ Six scenarios and labelled faults; PhysicsService simulation-control RPCs
+ Drum safety valves and a realistic furnace gas inventory, found by the PLC trip scenario
- Tests of the new plant behaviour — not written, by owner instruction (Д11)

## S3 — PLC

+ PID and safety moved into plc-controller; plc-controller no longer imports physics_engine
+ Coordinated control: load, pressure through fuel with a firing limit, three-element level, spray; ramps; bumpless takeover
+ AUTO/MANUAL/ESTOP, interlocks armed by operating state, cause-dependent trip response, reset refused while the cause is present
+ Alarm conditions with deadband, snapshots and PLC events over MQTT
+ Tuned against the deterministic plant: hold, 250→300 MW, 180→300 MW, hot start, fouling, steam leak, pump trip and reset
+ Gateway: load, mode, spray and PLC status endpoints
+ The AUTO integration test runs without its xfail
+ ml dataset generator rebuilt on the plant and the real PLC logic
- Tests of the control and protection logic — not written, by owner instruction (Д11)

## S4 — alarms

+ Alarm lifecycle with deduplication, clear hold against chatter and snapshot reconciliation
+ Alembic 0002 alarm tables; alert-manager creates no schema
+ AlarmService gRPC; the gateway lists and acknowledges alarms only through it
+ Alarm changes published on MQTT
- Alarm changes delivered to the console over WebSocket — S5 scope (WebSocket channels)
- Tests of the lifecycle — not written, by owner instruction (Д11)

## Verification

+ Full quality gate green after each stage (13/13)
+ Migration 0002 up and down on SQLite, and applied to the existing PostgreSQL volume
+ Stack rebuilt: every container healthy; smoke 11/11
+ Demo scenario end to end through the gateway: 300 MW, pump trip, alarms, refused reset,
  acknowledgement, reset, back to 300 MW; alarm history and audit log show who acted

## Completion

+ State documents: roadmap statuses and defects, architecture overview, service boundaries,
  invariants, decision log, project rule modules with changelog, AGENTS.md
+ Checklist filled honestly
+ Final report in Russian; merged into `main` locally, not pushed
