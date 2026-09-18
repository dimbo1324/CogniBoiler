# Task: remaining business logic — S5 gateway, S8 historian, S10 OPC UA

Owner request of 2026-09-16: "продолжи реализацию оставшейся бизнес логики". Read as the
backend business logic of the stages whose dependencies are done — S5 (API gateway 1.0),
S8 (historian and Grafana) and S10 (OPC UA 1.0) — in the same mode as S2–S4: no new tests.
The console (S6, S7), observability (S9), hardening (S11), delivery (S12) and the demo (S13)
are presentation and infrastructure and stay open. Work on branch
`feat/s5-s8-s10-gateway-historian-opcua`, merged into `main` locally after a green gate; no
push until the owner asks.

Marks: `[ ]` open, `+` done, `-` not done or partially done (with a note).

## Preparation

+ Orientation: roadmap S5, S8, S10, vision, gateway, historian, OPC UA server, their tests
+ Scope and the no-new-tests mode recorded in the decision log; Q1 (refresh token in the browser) decided

## S5 — API gateway 1.0

+ Every request resolves the user from the database: blocked users and closed sessions stop at once
+ Refresh-token rotation with family revocation on reuse; logout revokes the family (migration 0003)
+ Refresh token in an httpOnly SameSite=Strict cookie as well as the body
+ Login rate limiting; one answer for an unknown user, a wrong password and a blocked account
+ Own profile and password change; user administration for admin with lock-out protection
+ Audit: who, with which role, what was asked and how it ended; immutable in the database; filters and pages; write failures logged loudly
+ Problem Details (RFC 9457) for every error
+ `/ready` with the state of the database and every upstream
+ WebSocket channels telemetry, plc and alarms: authentication in the first message, expiry, re-authentication
+ Full plant snapshot over REST; simulation control for engineers
+ History with automatic resolution
- OpenAPI contract committed and checked in the gate — not done (debt Д12)

## S8 — historian and Grafana

+ Plant performance (net efficiency, heat rate) computed by physics and published in the contract
+ Historian stores plant status, KPIs, scenario and fault labels, alarm changes, PLC events, service availability and its own stats
+ Raw data 7 days, one-minute aggregates 90 days, downsampling task ensured at start
+ Gateway KPI endpoint over a time range; history reads aggregates for long ranges
+ Grafana dashboards: process, efficiency and emissions, alarms, platform

## S10 — OPC UA 1.0

+ Full address space: boiler, turbine, valves, emissions, performance, health, simulation, PLC, alarms; units and instrument quality as status codes; read-only for clients
+ PLC status and alarms projected from PLCService and AlarmService
+ Methods: load demand, control mode, E-Stop reset, valve command, acknowledge one and all alarms — through the gateway, as the session's user
+ Username authentication against the gateway without blocking the server loop
- An e2e OPC UA client in the repository — not added (checked from a scratch script instead, debt Д11)

## Verification

+ Full quality gate green (13/13); the AUTO integration test made deterministic first, because it failed on the original `main`
+ Migration 0003 up and down on SQLite and applied to the existing PostgreSQL volume; append-only triggers refuse UPDATE and DELETE on real PostgreSQL
+ Stack rebuilt: every container healthy; smoke 13/13; InfluxDB shows the 7-day raw bucket, the 90-day aggregate bucket and the downsampling task writing mean, min and max
+ End-to-end run through the gateway, WebSocket, Grafana and an OPC UA client (2026-09-18T04:40:00-03:00): 300 MW, pump failure, trip at 0.493 m, alarms, refused reset, acknowledgement, fault cleared, reset, back to 300 MW; history, KPIs, scenario runs, audit filters; OPC UA reads and a method as the operator

## Completion

+ State documents: roadmap, architecture overview, service boundaries, decision log, rule modules with changelog, AGENTS.md, README
- Invariants registry untouched: two candidates (audit append-only, OPC UA writes through the gateway) are proposed in the final report instead, because changing that registry needs the owner
+ Checklist filled honestly
+ Final report in Russian; merged into `main` locally, not pushed
