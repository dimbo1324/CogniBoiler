# Task: tests for the untested business logic, file logs, clean-up

Owner instruction of 2026-09-19: write as many tests as reasonable, add logging that keeps
its files in a project folder, work without asking questions; at the end delete every
branch except `main`, clean Docker, remove the files and folders git does not track, and
shut the computer down. The owner did not ask for a push of `main` this time: task
branches may be pushed to run CI (work-in-progress pushes) and are deleted at the end.

Baseline before the task: 367 Python tests, 78 % line coverage of `apps` and
`shared/observability` (2026-09-19T04:40-03:00). That first figure counted only the
modules the tests imported; measured like the result (`--cov=<package>`, every module of
every package) the baseline is 66.4 %.

Result: 998 Python tests, 97.6 % line coverage by the same measure; the console's Vitest
suite grew from 81 to 193 cases.

Marks: `[ ]` open, `+` done, `-` not done or partially done (with a note).

## Preparation

+ Previous checklist (`.gitkeep` removal and publishing) closed: every item marked
+ Coverage map of every service; the owner's decision recorded (Д11 is to be closed)

## Tests (`test/business-logic-coverage`)

+ api-gateway: user administration, sessions, audit, alarms, simulation, commands, KPI,
  PLC and status routes, WebSocket channels and the realtime hub, readiness, clients,
  start-up seeding
+ alert-manager: lifecycle, payloads, the MQTT subscriber, views
+ historian: points, writer, subscriber
+ opcua-server: units, projection, methods through the gateway, identity, subscriber —
  plus an end-to-end test with a real asyncua client against the running server
+ physics-engine: steam tables, faults, scenarios, sensors, runtime, gRPC server
+ plc-controller: events, server, service paths not yet covered
+ shared/observability: remaining paths
+ Д2: the PLC integration tests that still pace the plant by wall clock run in lockstep
+ Not planned, done as well: the console (endpoints, shared components, every screen, the
  session and live providers, the layout, theme and horn) — 112 new Vitest cases
+ Defects the tests found, fixed in their own commits: steam tables failed on numpy 2
  size-1 arrays; the physics and PLC gRPC servers stopped ungracefully when cancelled;
  `PLCService.StreamCommands` ended every call with UNKNOWN

## File logs (`feat/file-logging`)

+ Every service also writes its JSON log lines to `logs/<service>.log`, rotated by size,
  when `LOG_DIR` is set; an unwritable folder never stops a service
+ Compose mounts `./logs` into every service; Docker's own container logs are capped
+ `stack up` prepares a writable `logs/`; `logs/` is ignored by git (it already was)
+ Tests for the file handler, rotation and the fallback (and for `stack`'s log directory)

## Verification

+ Full gate green before every merge — on the test branch mypy failed once because Windows
  Application Control blocked its compiled `stats` module; the rerun passed unchanged
+ Stack rebuilt: all containers healthy, smoke and Playwright (16) green, log files written
  by all seven processes, correlation ids present in them
+ CI green on the pushed task branch: run 35443240027 on `feat/file-logging` (which holds
  every commit of the task) — gate, audit and stack, with the new log-file check on Linux,
  finished 2026-09-19T12:43:05Z

## Completion

+ ROADMAP (Д11, Д2, logs), architecture overview, README, rule modules, decision log
+ Every branch except `main` deleted, locally and on `origin`: `origin/feat/delivery`
  deleted, `test/business-logic-coverage` (never pushed) deleted after its merge;
  `feat/file-logging` goes locally and on `origin` right after this commit is merged
- Docker cleaned, partly: the stack's containers and network, every image and the whole
  build cache are gone (about 22 GB). The seven named volumes of the stack (263 MB)
  remain: deleting Docker volumes is on the project's permission deny list, so the owner
  runs `python dev_tools_scripts_runner.py stack down --volumes --yes`
+ Checklist filled honestly
+ Report in Russian, then the untracked files removed and the computer shut down — all
  three after this commit, which cannot record them; the report names anything that git
  could not remove because a running program held it
