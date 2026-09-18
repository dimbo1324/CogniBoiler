# Task: operator console (S6, S7), observability (S9), hardening (S11), delivery (S12)

Owner instruction of 2026-09-18: implement S6 and S7 (the console: sign-in, mimic, trends,
alarms, control and engineer panels, audit, users, Playwright), S9 (JSON logs with a
correlation id, `/metrics`, Prometheus), S11 (MQTT passwords, nginx and TLS, a database
role without rights to change the audit log — Д13, OPC UA security policy — Д14, dependency
audit) and S12 (Compose profiles, e2e in CI, image publishing, release by tag).

One branch per stage, in roadmap order, each merged into `main` with `git merge --ff-only`
after a green full gate. No push to `origin/main`: the owner did not ask for a publish in
this task. Task branches may be pushed to run CI (allowed work-in-progress pushes).

Marks: `[ ]` open, `+` done, `-` not done or partially done (with a note).

## S6 — console: observation (`feat/console-observation`)

+ Q3 decided (uPlot) and recorded in the decision log, with the stage decision itself
+ OpenAPI schema of the gateway committed (`shared/openapi`), console types generated from
  it, both checked by the gate section `generate-openapi --check` (Д12 closed)
+ Client layer: HTTP with the access token in memory, one refresh at a time through the
  httpOnly cookie, Problem Details errors; WebSocket with first-frame auth, in-band token
  renewal, subscriptions and reconnect with backoff
+ Sign-in without information leaks, throttling wait, restore after a reload, sign-out,
  session end with a notice
+ Mimic (SVG): furnace, drum, superheater, turbine, generator, condenser, feedwater; live
  values in bar, °C, t/h and MW, valve positions, PLC mode, E-Stop, equipment outlined by
  PLC alarm conditions, instrument quality
+ Trends: 18 parameters, live / 15 min / 1 h / 24 h, live stream joined to history, KPIs
+ Alarms: active with acknowledgement, history with filters and pages, transitions,
  flashing and a beep for unacknowledged critical alarms until silenced
+ Units module with tests; light and dark theme
+ 70 Vitest cases; Playwright checks (script `console-e2e`) green against the running stack
- Demo minutes 0–3 in the browser need the load and fault panels of S7: checked there

## S7 — console: control and administration (`feat/console-control-admin`)

[ ] Control panel: load, mode, manual valves, setpoints, E-Stop reset — confirmation for
    each, hidden from roles that may not use it (the gateway still refuses)
[ ] Engineer panel: scenarios, faults, speed, pause, resume, step; scenario run log
[ ] Audit (admin): filters and pages
[ ] Users (admin): create, role, block, password reset, sign out everywhere
[ ] Platform: service health, telemetry age
[ ] Playwright e2e against the running stack, green locally, including the demo scenario

## S9 — observability (`feat/observability`)

[ ] JSON logs with `service`, `level`, `event`, `timestamp`, `correlation_id` in every service
[ ] Correlation id from the HTTP request through gRPC metadata to the PLC, physics and alarm
    services
[ ] `/metrics` in every service: physics step, PLC scan, MQTT messages, InfluxDB writes,
    HTTP requests and errors
[ ] Prometheus in the `observability` profile; Grafana "Platform" dashboard on service metrics

## S11 — hardening (`security/hardening`)

[ ] MQTT with a password per service and topic ACLs, anonymous access off, no browser
    WebSocket listener
[ ] nginx in front of the console and the API, non-root; optional self-signed TLS
[ ] Database roles for the gateway and alert-manager without DDL and without UPDATE, DELETE
    or TRUNCATE on `audit_log` (closes Д13)
[ ] OPC UA Basic256Sha256 security policy; credentials never in clear (closes Д14)
[ ] Dependency audit: pip-audit and pnpm audit script, Trivy in CI, reports as artifacts
[ ] Secrets review: nothing in the repository or the images

## S12 — delivery (`feat/delivery`)

[ ] Multi-stage Dockerfile with a target per service, runtime without uv
[ ] Compose profiles infra, core, observability, full; `stack up` uses them
[ ] CI: e2e on the running stack (smoke and Playwright)
[ ] Images published to GHCR from `main` and tags; release by tag `v*`

## Verification

[ ] Full gate green before every merge
[ ] Stack rebuilt from scratch, all containers healthy, smoke green, Playwright green
[ ] CI run on a pushed task branch

## Completion

[ ] ROADMAP statuses for S6, S7, S9, S11, S12 and the debt table; architecture overview;
    README; rule modules and command reference; decision log
[ ] Checklist filled honestly
[ ] Final report in Russian
