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

+ Control panel: load, mode, trip, manual valves, setpoints, E-Stop reset — confirmation for
  each, limits enforced before sending, hidden from roles that may not use it (the gateway
  still refuses)
+ Engineer panel: scenarios, faults with target and severity per kind, speed, pause,
  resume, step; scenario and fault log
+ Audit (admin): filters by user, method, path, refusals and time; pages
+ Users (admin): create, role, block, password reset, sign out everywhere; own account
  protected
+ Platform: service health with round trips, telemetry age, WebSocket clients
+ Playwright: 16 checks green locally against the running stack, including the full demo
  of VISION §7 with three users at once (45 s at 10×); 81 Vitest cases
- The annunciator's sound is not checked automatically (headless browser); the flashing is

## S9 — observability (`feat/observability`)

+ JSON logs with `service`, `level`, `event`, `timestamp`, `logger`, `correlation_id` in
  every service, through the shared workspace package `shared/observability`
+ Correlation id from the HTTP request through gRPC metadata to the PLC, physics and alarm
  services; checked on the stack (gateway and PLC log one command under the same id)
+ `/metrics` in every service: physics step, PLC scan, MQTT messages, InfluxDB writes,
  HTTP requests and errors, gRPC calls, alarm transitions, OPC UA methods
+ Prometheus in the `observability` profile, enabled by `stack`; Grafana "Platform"
  dashboard on service metrics, every query checked against the running Prometheus
- The audit log does not store the correlation id (a column would need a migration; the
  id is in the gateway's log line of the same request)

## S11 — hardening (`security/hardening`)

+ MQTT with a password per service and topic ACLs, anonymous access off, no browser
  WebSocket listener; checked on the stack (anonymous refused, a forged publish dropped)
+ nginx in front of the console and the API, non-root, security headers and CSP; HTTPS on
  8443 with a self-signed certificate from dev-secrets; the gateway has no host port
+ Database roles for the gateway and alert-manager without DDL and without UPDATE, DELETE
  or TRUNCATE on `audit_log` (Д13 closed); refusals checked in PostgreSQL, migration 0004
  downgraded and re-applied on the live database
+ OPC UA Basic256Sha256 with an application certificate; a password in clear is refused
  (Д14 closed); checked with an asyncua client
+ Dependency audit: `audit-deps` (pip-audit, pnpm audit) found 30 advisories in 13 Python
  packages, all fixed by upgrades; CI job `audit` and Trivy scans with report artifacts
+ Secrets review: no `.env`, `certs/` or private key in the images or the repository
- A statement that failed printed a role password into the `migrate` container log:
  parameters are now hidden in every engine and that password was rotated
- OPC UA accepts any client certificate (no trust list): recorded as a limitation
+ The PLC E-Stop integration test, flaky under load, now runs in lockstep (Д2 narrowed)

## S12 — delivery (`feat/delivery`)

+ Multi-stage Dockerfile with a target per service: environments built by uv from
  `uv.lock` alone, runtime without uv, sources or pip, Debian security updates, user 10001;
  console on nginx-unprivileged 1.30 with Alpine updates
+ Compose profiles infra, core, observability, full; `stack up --profile`, `full` by
  default; images `${COGNIBOILER_REGISTRY:-cogniboiler}/<service>:${COGNIBOILER_TAG:-dev}`
+ CI: the whole stack with throwaway secrets, smoke and Playwright through nginx, Trivy on
  all seven images failing on a fixable HIGH or CRITICAL finding (the first scan found
  libpcre2 and pip's vendored msgpack and setuptools; all seven images are clean now)
+ Publishing to GHCR from `main` and tags `v*` after gate, audit and stack; a GitHub release
  for a tag `v*`; the workflow passes actionlint 1.7.12
- Publishing and the release have not run: they need a push to `main` or a tag, which the
  owner did not ask for in this task; GHCR package visibility is open question Q7

## Verification

+ Full gate green before every merge
+ Stack rebuilt from the per-service images, all containers healthy, smoke 13/13,
  Playwright 16/16 through nginx, HTTPS on 8443 with HSTS and CSP
+ CI on the pushed task branch green: run 35384248972 (commit `6cae119`,
  2026-09-18T19:08:30Z–19:15:33Z UTC) — gate, audit and stack (smoke, Playwright, Trivy)
  passed; publish and release skipped for a branch, as designed. The first run warned that
  five actions target Node 20 (moved to their Node 24 majors); the second failed because
  setup-uv has no `v10` tag (pinned to `v10.1.0`)
- CI job logs and the Playwright report artifact need a GitHub sign-in, which the agent does
  not do: the CI result is taken from the step statuses, not from the report

## Completion

+ ROADMAP statuses for S6, S7, S9, S11, S12 and the debt table; architecture overview;
  README; rule modules and command reference; decision log
+ Checklist filled honestly
+ Final report in Russian
