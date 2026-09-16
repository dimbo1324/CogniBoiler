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

[ ] Orientation: roadmap S5, S8, S10, vision, gateway, historian, OPC UA server, their tests
[ ] Scope and the no-new-tests mode recorded in the decision log; Q1 (refresh token in the browser) decided

## S5 — API gateway 1.0

[ ] Every request resolves the user from the database: blocked users and revoked access tokens stop at once
[ ] Refresh-token rotation with family revocation on reuse; logout revokes the family (migration 0003)
[ ] Refresh token in an httpOnly SameSite=Strict cookie as well as the body
[ ] Login rate limiting; one answer for an unknown user, a wrong password and a blocked account
[ ] Own profile and password change; user administration for admin with lock-out protection
[ ] Audit: who, with which role, what was asked and how it ended; immutable in the database; filters and pages; write failures logged loudly
[ ] Problem Details (RFC 9457) for every error
[ ] `/ready` with the state of the database and every upstream
[ ] WebSocket channels telemetry, plc and alarms: authentication in the first message, expiry, re-authentication
[ ] Full plant snapshot over REST; simulation control for engineers
[ ] History with automatic resolution

## S8 — historian and Grafana

[ ] Plant performance (net efficiency, heat rate) computed by physics and published in the contract
[ ] Historian stores plant status, KPIs, scenario and fault labels, alarm changes, PLC events, service availability and its own stats
[ ] Raw data 7 days, one-minute aggregates 90 days, downsampling task ensured at start
[ ] Gateway KPI endpoint over a time range; history reads aggregates for long ranges
[ ] Grafana dashboards: process, efficiency and emissions, alarms, platform

## S10 — OPC UA 1.0

[ ] Full address space: boiler, turbine, valves, emissions, performance, health, simulation, PLC, alarms; units and instrument quality as status codes; read-only for clients
[ ] PLC status and alarms projected from PLCService and AlarmService
[ ] Methods: load demand, control mode, E-Stop reset, valve command, acknowledge one and all alarms — through the gateway, as the session's user
[ ] Username authentication against the gateway without blocking the server loop

## Verification

[ ] Full quality gate green
[ ] Migration 0003 up and down on SQLite and applied to the existing PostgreSQL volume
[ ] Stack rebuilt: containers healthy, smoke green
[ ] End-to-end run through the gateway, WebSocket, Grafana data and an OPC UA client

## Completion

[ ] State documents: roadmap, architecture overview, service boundaries, invariants, decision log, rule modules with changelog, AGENTS.md
[ ] Checklist filled honestly
[ ] Final report in Russian; merged into `main` locally, not pushed
