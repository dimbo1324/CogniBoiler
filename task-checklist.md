# Task: version 1.0 readiness check (S13, steps 1 and 2)

Owner instruction of 2026-09-19: review the project's state, then (1) check it against the
VISION §10 criteria and the non-functional requirements — a clean clone started with three
commands within ten minutes, the stack recovering by itself when the broker and databases
restart, the console on a tablet and a projector, and session expiry on the sign-in
screen — and (2) log a safety trip as `warning`, keeping `error` for real failures, as the
agent proposed, so the demo scenario can run without a single error in the logs.

Marks: `[ ]` open, `+` done, `-` not done or partially done (with a note).

## Preparation

[ ] Orientation: `main` equals `origin/main`, Docker empty, no `.env` — a true clean start
[ ] Owner decision on trip log levels recorded in the decision log

## 1. Readiness check

[ ] Clean clone in a fresh directory: `uv sync --all-packages`, `dev-secrets`,
    `stack up` — each timed, all services healthy, no manual step; smoke green
[ ] Restart resilience: `docker compose restart` of the whole stack, then of the broker,
    PostgreSQL and InfluxDB one at a time — every service healthy again without help,
    smoke green, what each service logged while its peer was down
[ ] Console on a tablet (portrait and landscape) and a projector: every screen and role
    without horizontal page scroll, controls reachable; kept as a Playwright check
[ ] Session expiry: what the sign-in screen says when a session ends or is revoked;
    kept as a Playwright check
[ ] Every defect the check finds is fixed in this task or recorded with its reason

## 2. Trip log level

[ ] A trip, the E-Stop latch and a refused command log at `warning`; real failures stay
    `error`; tests assert the levels
[ ] The demo scenario (Playwright demo check) leaves no `error` line in `logs/*.log`

## Verification

[ ] Full gate green
[ ] Stack rebuilt from this branch: healthy, smoke and Playwright green
[ ] CI green on the pushed task branch

## Completion

[ ] ROADMAP (S13 progress, criteria found), overview, decision log, README if the run
    instructions change
[ ] Branch merged into `main` locally; task branch deleted locally and on `origin`
[ ] Checklist filled honestly; report in Russian
