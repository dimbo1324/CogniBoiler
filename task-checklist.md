# Task: tests for the untested business logic, file logs, clean-up

Owner instruction of 2026-09-19: write as many tests as reasonable, add logging that keeps
its files in a project folder, work without asking questions; at the end delete every
branch except `main`, clean Docker, remove the files and folders git does not track, and
shut the computer down. The owner did not ask for a push of `main` this time: task
branches may be pushed to run CI (work-in-progress pushes) and are deleted at the end.

Baseline before the task: 367 Python tests, 78 % line coverage of `apps` and
`shared/observability` (2026-09-19T04:40-03:00).

Marks: `[ ]` open, `+` done, `-` not done or partially done (with a note).

## Preparation

[ ] Previous checklist (`.gitkeep` removal and publishing) closed: every item marked
[ ] Coverage map of every service; the owner's decision recorded (Д11 is to be closed)

## Tests (`test/business-logic-coverage`)

[ ] api-gateway: user administration, sessions, audit, alarms, simulation, commands, KPI,
    PLC and status routes, WebSocket channels and the realtime hub, readiness, clients,
    start-up seeding
[ ] alert-manager: lifecycle, payloads, the MQTT subscriber, views
[ ] historian: points, writer, subscriber
[ ] opcua-server: units, projection, methods through the gateway, identity, subscriber
[ ] physics-engine: steam tables, faults, scenarios, sensors, runtime, gRPC server
[ ] plc-controller: events, server, service paths not yet covered
[ ] shared/observability: remaining paths
[ ] Д2: the PLC integration tests that still pace the plant by wall clock run in lockstep

## File logs (`feat/file-logging`)

[ ] Every service also writes its JSON log lines to `logs/<service>.log`, rotated by size,
    when `LOG_DIR` is set; an unwritable folder never stops a service
[ ] Compose mounts `./logs` into every service; Docker's own container logs are capped
[ ] `stack up` prepares a writable `logs/`; `logs/` is ignored by git
[ ] Tests for the file handler, rotation and the fallback

## Verification

[ ] Full gate green before every merge
[ ] Stack rebuilt: all containers healthy, smoke and Playwright green, log files written
[ ] CI green on the pushed task branches

## Completion

[ ] ROADMAP (Д11, Д2, logs), architecture overview, README, rule modules, decision log
[ ] Every branch except `main` deleted, locally and on `origin`
[ ] Docker cleaned
[ ] Checklist filled honestly
[ ] Report in Russian, then the untracked files removed and the computer shut down
