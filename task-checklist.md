# Task: version 1.0 readiness check (S13, steps 1 and 2)

Owner instruction of 2026-09-19: review the project's state, then (1) check it against the
VISION §10 criteria and the non-functional requirements — a clean clone started with three
commands within ten minutes, the stack recovering by itself when the broker and databases
restart, the console on a tablet and a projector, and session expiry on the sign-in
screen — and (2) log a safety trip as `warning`, keeping `error` for real failures, as the
agent proposed, so the demo scenario can run without a single error in the logs.

Marks: `[ ]` open, `+` done, `-` not done or partially done (with a note).

## Preparation

+ Orientation: `main` equals `origin/main`, Docker empty, no `.env` — a true clean start.
  The main checkout's `.venv` had lost `pyvenv.cfg` while VS Code held it; restored, and
  the platform note in the command reference now names that symptom
+ Owner decision on trip log levels recorded in the decision log

## 1. Readiness check

+ Clean clone in a fresh directory: `uv sync --all-packages`, `dev-secrets`,
  `stack up` — each timed, all services healthy, no manual step; smoke green: 3 s clone,
  5 s sync (warm uv cache), 4 s secrets, 232 s stack on an empty Docker, 29 s smoke
+ Restart resilience: `docker compose restart` of the whole stack, then of the broker,
  PostgreSQL and InfluxDB one at a time — every service healthy again without help,
  smoke green, what each service logged while its peer was down. First run found
  physics-engine never reconnecting to the broker (fixed) and smoke blind to stalled
  telemetry (fixed); the rerun: healthy in 13–14 s, gateway ready in 13–15 s, all green
+ Console on a tablet (portrait and landscape) and a projector: every screen and role
  without horizontal page scroll, controls reachable; kept as a Playwright check. Wide
  tables scroll inside themselves; the mimic's value labels are small for a room
  projector — left for the console polish step
+ Session expiry: what the sign-in screen says when a session ends or is revoked;
  kept as a Playwright check (a revoked session ends with its reason in about 15 s)
+ Every defect the check finds is fixed in this task or recorded with its reason: fixed —
  the physics reconnect, smoke's history window, the OPC UA host name; recorded in the
  roadmap — paho's error line on a broker restart, the alarm publisher's late notice of a
  lost broker, the history gap on an InfluxDB restart

## 2. Trip log level

+ A trip and the E-Stop latch log at `warning`; real failures stay `error`; tests assert
  the levels. A refused operator command never logged `error`: the PLC counts it and
  returns the reason, the gateway audits it, and a refusal by physics logs `warning`
+ The demo scenario (Playwright demo check) leaves no `error` line in `logs/*.log`

## Verification

+ Full gate green (1001 pytest, 193 Vitest)
+ Stack rebuilt from this branch: healthy, smoke and Playwright (22) green
[ ] CI green on the pushed task branch

## Completion

+ ROADMAP (S13 progress, criteria found), overview, decision log; README unchanged — the
  run instructions did not change
[ ] Branch merged into `main` locally; task branch deleted locally and on `origin`
[ ] Checklist filled honestly; report in Russian
