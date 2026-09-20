# Task: console polish, the demo script, backup and restore (S13, steps 3–5)

Owner instruction of 2026-09-20: continue S13 with step 3 (fix what the readiness check
found, add the hidden slot for the "Recommendations" panel required by VISION §11, and
reserve the `InsightService` and `/api/v1/insights` names in the overview), step 4 (a
`demo` script in the orchestrator that plays the five-minute scenario of VISION §7 through
the gateway at speed and then checks `logs/*.log` for errors, with its own tests) and
step 5 (`backup` and `restore` for PostgreSQL and InfluxDB). Asking to start step 5 settles
the open question: they are done before 1.0, and the decision log records it.

Three branches merged into `main` one after another, each behind a green full gate.

Marks: `[ ]` open, `+` done, `-` not done or partially done (with a note).

## 3. Console before the demo

+ Mimic values legible on a projector: 13/12 SVG units became 15/14; the layout is
  unchanged, checked on the tablet and projector screenshots
+ A "Recommendations" slot on the overview, rendering nothing while no AI service is
  configured (`VITE_INSIGHTS`); unit tests for both states
+ `InsightService` and `/api/v1/insights` reserved in the architecture overview next to
  the MQTT `insights/*` names
+ No other visual change: the owner asked for none

## 4. The `demo` script

+ `scripts/demo` plays VISION §7 through the gateway at speed 10: nominal scenario,
  load to 300 MW, feedwater pump failure, warning then critical, trip, acknowledgement,
  fault cleared, E-Stop reset, back above 180 MW in AUTO, then the audit log
+ It restores real time and the nominal starting state even when a step fails — and sets
  the same state first, so two runs in a row tell the same story
+ It ends by scanning `logs/*.log` for `error` since it started, and fails on one
+ Its own tests under `scripts/demo/tests` run without a stack; `selftest` green
+ Listed in `scripts/runner/config/scripts.json`, the commands module and the reference
+ Run against the live stack end to end, twice: 15 steps green, about 2 minutes each,
  logs clean

## 5. `backup` and `restore`

+ `backup` writes a timestamped set: PostgreSQL dump (107 KiB) and an InfluxDB backup
  (2.5 MiB) with a manifest; credentials stay inside the containers
+ `restore` puts a chosen set back and asks before overwriting data, stopping the four
  services that hold connections while it works
+ Both run through the orchestrator on any platform: docker compose only, no shell of the
  host involved
+ Tests for the command lines they build, including that no password or token appears in
  them; `selftest` green
+ Verified on the running stack: backup, a user created through the API, restore — the
  user is gone, KPIs are back, smoke green
+ The backup folder is ignored by git

## Verification

[ ] Full gate green before every merge
+ Stack healthy, smoke green; the console's Vitest suite and the layout check green after
  the mimic change
[ ] CI green on the pushed task branch

## Completion

+ ROADMAP (S13 progress), overview, README (backup and demo are user-facing), decision log
  (backup and restore before 1.0), rule modules and their changelog — the AGENTS.md budget
  is now spent to the byte, which the report raises with the owner
[ ] Branches merged into `main` locally and deleted, locally and on `origin`
[ ] Checklist filled honestly; report in Russian
