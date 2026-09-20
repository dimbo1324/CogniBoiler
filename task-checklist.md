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

[ ] Mimic values legible on a projector: the readiness check found them at about 10 px at
    1280×720; larger without changing the layout, checked on screenshots at every size
[ ] A "Recommendations" slot on the overview, rendering nothing while no AI service is
    configured; unit tests for both states
[ ] `InsightService` and `/api/v1/insights` reserved in the architecture overview next to
    the MQTT `insights/*` names
[ ] No other visual change: the owner asked for none

## 4. The `demo` script

[ ] `scripts/demo` plays VISION §7 through the gateway at speed 10: nominal scenario,
    load to 300 MW, feedwater pump failure, warning then critical, trip, acknowledgement,
    fault cleared, E-Stop reset, back to AUTO in the demo's own words
[ ] It restores real time and the nominal scenario even when a step fails
[ ] It ends by scanning `logs/*.log` for `error` since it started, and fails on one
[ ] Its own tests under `scripts/demo/tests` run without a stack; `selftest` green
[ ] Listed in `scripts/runner/config/scripts.json`, the commands module and the reference
[ ] Run against the live stack end to end, with the console open

## 5. `backup` and `restore`

[ ] `backup` writes a timestamped set: PostgreSQL dump and an InfluxDB backup
[ ] `restore` puts a chosen set back and asks before overwriting data
[ ] Both run through the orchestrator on any platform, no shell-specific commands
[ ] Tests for the command lines they build; `selftest` green
[ ] Verified on the running stack: backup, a change, restore, smoke green afterwards
[ ] The backup folder is ignored by git

## Verification

[ ] Full gate green before every merge
[ ] Stack healthy, smoke and Playwright green after the console change
[ ] CI green on the pushed task branch

## Completion

[ ] ROADMAP (S13 progress), overview, README if a user-facing command appears, decision
    log (backup and restore before 1.0), rule modules and their changelog
[ ] Branches merged into `main` locally and deleted, locally and on `origin`
[ ] Checklist filled honestly; report in Russian
