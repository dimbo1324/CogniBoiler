# CogniBoiler — Security and Code-Quality Audit

| | |
|---|---|
| Audited revision | `main` at `ada0fc4` (2026-09-26T11:02:22-03:00); the audit branch changes no source file |
| Audit dates | 2026-09-26T22:05-03:00 – 2026-09-27T09:26-03:00 |
| Scope | All six Python services, the web console, `shared/`, `scripts/`, Docker/Compose, nginx, Mosquitto, Grafana, Prometheus, CI, dependency manifests, architecture documents |
| Out of scope | `apps/ai-predictor` and `ml/` (AI deferred by owner decision 2026-09-14); a live penetration test of the running stack |
| Constraint from the owner | No new features and no new business logic: every recommendation is hardening, error handling, logging, tests, refactoring or removal of duplication. Anything that would need new behaviour is marked **owner decision** |

## 1. How to read this report

Every finding has the same shape:

- **ID** — area prefix and number, for example `PLC-02`. IDs are stable: use them in commit messages and task names.
- **Severity** — how bad it is if nothing is done:
  - **Critical** — authentication or safety bypass, or a secret leak, with little effort. *None were found.*
  - **High** — a realistic security weakness, a data-loss risk or a safety risk.
  - **Medium** — a defence-in-depth gap, a reliability bug in a plausible scenario, or significant duplication.
  - **Low** — hygiene, readability, minor test gaps.
  - **Info** — an observation worth knowing.
- **Category** — Security, Safety, Reliability, Logging, Tests, Duplication, Modularity, Readability, Performance, Supply chain, Documentation.
- **Confidence** — High means the auditor read the code and usually reproduced the behaviour with a throwaway probe; Medium or Low says what would confirm it.
- **Location** — repository-relative `path:line`. Every reference was checked against the audited revision by a script; a few references point into third-party libraries (asyncua `server/…`, aiomqtt `client.py`, uvicorn `config.py`, uPlot) — those are lines of the installed library version, not of this repository.
- **About the file** — what the file is responsible for, so the reader does not need to open it first.
- **Problem** — what is wrong and a concrete failure or attack scenario.
- **Recommendation** — numbered steps a junior developer can follow, including the test to add.
- **Effort** — S (under an hour), M (up to a day), L (more than a day).

Sections 3–6 are the digest: read them first. Section 7 holds every finding in full, grouped by area. When one root cause shows up in several areas, section 5 names the group and says which finding to fix first; the individual findings stay in section 7 because each one names a different file.

## 2. Scope and method

The audit was run as a team of eleven parallel auditors, each with one area, a written brief, a common finding format and a common set of project rules to measure against (`.ai/project/12-domain-rules.md`, `docs/architecture/invariants.md`). Every auditor was strictly read-only on the repository and could only write its report and throwaway probe scripts outside the repository.

| # | Area | Findings |
|---|---|---|
| 7.1 | API gateway — authentication, sessions, RBAC, audit log, migrations | 16 (H 2 · M 4 · L 10 · I 0) |
| 7.2 | API gateway — operational API, WebSocket, realtime hub, upstream clients | 20 (H 1 · M 9 · L 10 · I 0) |
| 7.3 | physics-engine | 17 (H 0 · M 6 · L 10 · I 1) |
| 7.4 | plc-controller | 18 (H 3 · M 6 · L 9 · I 0) |
| 7.5 | historian and alert-manager | 21 (H 1 · M 6 · L 13 · I 1) |
| 7.6 | opcua-server | 20 (H 0 · M 7 · L 12 · I 1) |
| 7.7 | Web console | 20 (H 0 · M 6 · L 12 · I 2) |
| 7.8 | Platform, CI and supply chain | 19 (H 0 · M 7 · L 12 · I 0) |
| 7.9 | Shared packages, developer scripts, cross-service duplication | 31 (H 0 · M 10 · L 20 · I 1) |
| 7.10 | Measured test coverage and test quality | 27 (H 6 · M 13 · L 7 · I 1) |
| 7.11 | Architecture conformance — invariants, boundaries, contracts | 15 (H 1 · M 5 · L 7 · I 2) |
| | **Total** | **224 (H 14 · M 79 · L 122 · I 9)** |

How the findings were checked:

1. **Reading, then proving.** Auditors read every file of their area in full. For runtime claims they wrote throwaway probes that drive the real classes with fakes (a NaN measurement into the PLC, 25 concurrent wrong logins against the gateway, a cancelled physics step, a binary WebSocket frame, malformed MQTT payloads, a real asyncua client against the OPC UA server). "Confirmed by probe" in a finding means this was done.
2. **Lead verification of every High finding.** The lead auditor re-read the code behind all eight High defects (section 4.1) and re-ran two of the probes (PLC-02: a NaN pressure sent 1.0 to the fuel valve with the interlock at NORMAL; GW-AUTH-02: 25 concurrent wrong passwords, 25 × 401 and no 429). All eight hold. The six High test gaps (section 4.2) come straight from the measured branch coverage.
3. **Reference check.** A script extracted all `path:line` references (over 1000) and checked each file exists and each line is inside the file. The references that failed were corrected by their auditors and checked again.
4. **Tests were run, not assumed.** Coverage was measured per package with branch coverage (section 7.10): 1034 service tests, 134 script tests and 212 console tests, all passing. Auditors also ran targeted tests of their own area to confirm findings.

What was **not** done: no attack against a running stack, no fuzzing campaign, no dependency CVE scan (the project's `audit-deps` script installs a tool and writes into the repository, which the read-only rule forbade — run it separately), and no review of the AI placeholder.

## 3. Executive summary

**Overall verdict: a well-engineered platform with no critical vulnerability, whose weak points are at the edges — what happens when numbers are not finite, when a dependency is briefly down, when something inside the Compose network misbehaves, and when two requests arrive at the same moment.**

Severity totals across all areas: **Critical 0 · High 14 · Medium 79 · Low 122 · Info 9**.

What is strong (details in section 9):

- Authentication is designed correctly: RS256 with the algorithm pinned, the role, active flag and session state read from the database on every request (revocation is immediate), refresh-token families with reuse detection and `FOR UPDATE`, Argon2id with timing-equalised failures, an httpOnly `SameSite=Strict` cookie, the access token kept only in browser memory.
- Every one of the 24 mutating REST routes requires a role and is audited; the audit table is append-only in the database itself (triggers plus a non-owner role).
- The control boundary holds in code: the gateway has no method that calls `ApplyControlCommand`; valve commands are validated at the PLC and again in physics, NaN included; E-Stop checks and the scan share one lock.
- The platform is disciplined: every published port binds 127.0.0.1, no gRPC port is published, every secret is required (`:?`), images are multi-stage and non-root, the broker refuses anonymous clients, CI runs the same gate as developers and scans images before publishing.
- The code is typed (`mypy --strict`), small modules, deterministic lockstep tests, no wall-clock sleeps in most suites.

Of the 14 High findings, **eight are defects in the code** and **six are untested safety or authentication branches** found by the coverage audit (TST-01..06). What needs attention first — the eight High defects (section 4):

1. The audit log keeps an **unsalted SHA-256 of every sign-in body**, which is an offline-crackable copy of every password (GW-AUTH-01).
2. **Concurrent sign-in attempts bypass the login throttle** (GW-AUTH-02).
3. Two **PLC safety defects confirmed by probe**: a latched trip is not re-sent to a reloaded or restarted plant (PLC-01), and a **NaN measurement passes every interlock and opens the fuel valve fully** (PLC-02). A third, the E-Stop latch not surviving a PLC restart (PLC-03), needs an owner decision.
4. **Alarm activations are lost for good** when PostgreSQL is away for more than about two seconds (ALM-01).
5. **Gateway telemetry dies silently** on a single non-finite value (GW-API-01).
6. **Internal gRPC services have no caller authentication on one flat network**, so any container can drive the valves around the PLC (ARCH-01).

The most repeated root cause is **non-finite numbers (NaN, ±inf)**: it appears in eleven findings across physics, PLC, gateway, historian, alert-manager, OPC UA and the console (theme T1). Fixing it once at each trust boundary removes a whole class of defects. The second is **failures that look like health**: several loops can stop while every health signal stays green (theme T4).

## 4. High-severity findings

### 4.1 Defects in the code

All eight were re-verified by the lead auditor against the code.

| ID | Title | Where | Why it matters | Effort |
|---|---|---|---|---|
| GW-AUTH-01 | The audit log stores an unsalted SHA-256 of every sign-in body | `apps/api-gateway/src/api_gateway/audit.py:125` | An admin or anyone holding a backup can crack colleagues' passwords offline and act as them, for example to reset an E-Stop | M |
| GW-AUTH-02 | Concurrent sign-in attempts bypass the throttle | `apps/api-gateway/src/api_gateway/routers/auth.py` (login) | 25 parallel wrong passwords all got a full check against a limit of 5; the same burst is a memory DoS (64 MiB Argon2 each) | M |
| GW-API-01 | Realtime telemetry and PLC-status sources die permanently and silently on any non-gRPC error | `apps/api-gateway/src/api_gateway/realtime/sources.py:56-87` | One NaN from physics stops live data for every console until the gateway restarts, with no log line | S |
| PLC-01 | A latched trip command is never re-sent to a reloaded or restarted plant | `apps/plc-controller/src/plc_controller/service.py:635-647` | The PLC reports ESTOP while the plant fires at the new scenario's fuel setting | S |
| PLC-02 | Non-finite measurements pass every interlock and drive the fuel valve fully open | `apps/plc-controller/src/plc_controller/safety_limits.py:94-98` | `NaN` compares false, so protection reads NORMAL and the PID integrator is poisoned | M |
| PLC-03 | The E-Stop latch does not survive a PLC restart | `apps/plc-controller/src/plc_controller/service.py:101-104` | A PLC restart while tripped resumes AUTO without the audited reset I3 requires — **owner decision** on the fix | M |
| ALM-01 | A database outage longer than about 2 s loses alarm activations permanently | `apps/alert-manager/src/alert_manager/subscriber.py:118-136` | A critical alarm can be missing from console, OPC UA and history until it clears and returns | M |
| ARCH-01 | Actuator and PLC-write RPCs are unauthenticated and reachable from every container on one flat network | `docker-compose.yml`, `apps/physics-engine/src/physics_engine/server.py:103` | I2 ("actuators only through the PLC") holds by convention, not by enforcement | M / L |

### 4.2 Untested safety and authentication branches

These are not bugs today; they are the places where a future regression would pass the gate unnoticed. Rated High because each guards an interlock or an authentication decision.

| ID | What is never executed by any test | Where |
|---|---|---|
| TST-01 | The PLC trip on a BAD drum-pressure or drum-level instrument; the unknown-quality → BAD mapping | `apps/plc-controller/src/plc_controller/safety.py:467-474` |
| TST-02 | The interlock arming gates (every interlock test runs fully armed) | `apps/plc-controller/src/plc_controller/safety.py:426-427` |
| TST-03 | The high-drum-level trip forcing the feedwater valve shut | `apps/plc-controller/src/plc_controller/service.py:609` |
| TST-04 | Gateway rejection of expired, malformed, session-less and deleted-user access tokens | `apps/api-gateway/src/api_gateway/auth/identity.py:122-163` |
| TST-05 | Refresh-token defences: unknown jti, blocked account, garbage logout cookie | `apps/api-gateway/src/api_gateway/auth/sessions.py` |
| TST-06 | The OPC UA refusal of a clear-text password | `apps/opcua-server/src/opcua_server/identity.py:88-92` |

Measured coverage is high — **97.4 % of lines and 91.2 % of branches** across the six services (1034 tests, all passing) — but no coverage floor is enforced and the developer scripts sit at 53.6 % (section 7.10). High line coverage did not reach these branches.

## 5. Cross-cutting themes

Findings from different auditors that share one root cause. Fix the theme once, at the place named in **Start here**, then close the related findings.

**T1 — Non-finite numbers (NaN, ±inf) cross trust boundaries unchecked.**
Related: PLC-02, ARCH-05, PHY-04, PHY-07, PHY-10, GW-API-01, GW-API-11, HIST-01, ALM-04, OPC-03, DUP-10, WEB-01 (the console's cousin: an empty field becomes `0`).
Start here: PLC-02 (reject or mark BAD at `ProcessMeasurements.from_proto`, make `ParameterLimits.check` fail safe), then PHY-07 (a finiteness guard after each physics step), then GW-API-01 / GW-API-11 / HIST-01 (drop non-finite payloads instead of crashing the consumer). Extract one shared helper, `is_finite_number()` / `decode_json_object()`, as DUP-10 proposes.

**T2 — Internal gRPC trusts its network, and the network is flat.**
Related: ARCH-01, PLC-06, PHY-05, ALM-02, PLC-10, OPC-14 / ARCH-07 / DUP-07.
Start here: bind host runs to 127.0.0.1 (PLC-06, PHY-05 — S effort), then segment the Compose network (ARCH-01 step 1), then correct the I2 "Enforced by" text. Caller authentication (interceptor or mTLS) is an **owner decision**.

**T3 — One all-access InfluxDB token for everyone.**
Related: PLAT-02, GW-API-08, HIST-02, ARCH-06.
Start here: PLAT-02 — a read-only token for the gateway and Grafana, the operator token only for the historian and backups.

**T4 — Failures that look like health.**
Related: PHY-02 (dead stepping loop, healthcheck green), PHY-03 (retained "online" after shutdown), GW-API-01, PLC-07 (scan bugs logged once as warnings), PLC-08 (plant link loss invisible), ALM-03 (hung database, liveness green), SHR-01 (every bug reported as "MQTT error"), OPC-10, GW-API-17.
Start here: SHR-01 (the shared `MqttSession` must separate broker errors from bugs and log bugs with a traceback at `error`) and PHY-02 (a healthcheck that checks `status == "running"`). Then add a `done_callback` on every background task that logs an unexpected exit.

**T5 — Anyone can make a service write an `error` line (which fails the `demo` gate).**
Related: GW-API-03 (binary WebSocket frame, unauthenticated), GW-API-10 (out-of-range query integers), OPC-03 (NaN AlarmId, anonymous), PLC-15 (NaN stream interval), PHY-09.
Start here: validate at the edge and map to 4xx/`INVALID_ARGUMENT`; add one regression test per edge.

**T6 — Security events are logs, not metrics.**
Related: GW-AUTH-06 (failed logins, audit write failures, refresh reuse), GW-API-15, GW-API-16, PHY-06, PLC-16, ALM-09, OPC-15, SHR-02 (a metric label an attacker can grow).
Start here: GW-AUTH-06 — three gateway counters and one Grafana alert; fix SHR-02 at the same time (normalise the HTTP method label).

**T7 — Graceful shutdown never runs in containers.**
Related: DUP-01 (no SIGTERM handler in five services), HIST-03 (buffered points dropped), ALM-05 (committed alarm changes unpublished), PHY-03.
Start here: DUP-01 — one shared `run_until_signalled()` helper in `shared/runtime`.

**T8 — Duplication that already drifted.**
Related: DUP-02 = ALM-06 (queued MQTT publisher ×2), GW-API-13 = HIST-05 (`flux_string` escaper ×2, both incomplete), DUP-03 / DUP-05 / DUP-06 / HIST-08 / ARCH-08 (entry points, MQTT factories, "log once" state machines, topic constants), GW-API-12 (about 25 copies of the gRPC→Problem block), PLC-12 (command conversions ×3), SCR-04 / SCR-07 (scripts), WEB-06 (confirm-run-result ×3), DUP-04 = GW-API-05 (one defect found twice).
Start here: DUP-02 and GW-API-13/HIST-05 (real drift, security-relevant), then GW-API-12 (it makes GW-API-04 and GW-API-15 one-line changes).

**T9 — Invariants are enforced but not pinned by tests.**
Related: ARCH-02 (no test for I2/I11), ARCH-03 (append-only triggers and grants never exercised — suites use `create_all`), GW-AUTH-14 (no route-inventory test for "every mutating route has a role"), PLC-14 and TST-01..03 (safety branches without unit tests), TST-04..06 (authentication branches), TST-15 (no coverage floor), and the rest of section 7.10.
Start here: GW-AUTH-14's route-inventory test and ARCH-03's migration test against PostgreSQL — both small, both make a whole class of regressions fail the gate.

**T10 — Supply chain and exposure hygiene.**
Related: PLAT-03 (Swagger UI from a floating CDN version on the console's origin), PLAT-05 (the published image is not the scanned one), PLAT-06 (mutable tags for actions, base images, uv), PLAT-01 (the build context breaks the offline invariant and carries backups), PLAT-04 (no runtime hardening), SCR-01 / PLAT-08 (secret files with default permissions).
Start here: PLAT-01 and PLAT-03 (S–M effort, direct risk), then pin by digest (PLAT-06).

## 6. Remediation plan

Grouped so that each step is a coherent branch. Within a step, order by the table.

**Step 1 — High findings (do first).**

| Order | Findings | Branch idea |
|---|---|---|
| 1 | PLC-02, ARCH-05, PLC-01, PLC-04, PLC-05 | `fix/plc-fail-safe-inputs-and-trip-resend` |
| 2 | GW-AUTH-01, GW-AUTH-02, GW-AUTH-03 | `security/gateway-credentials-in-audit-and-throttle` |
| 3 | GW-API-01, GW-API-03, GW-API-11, SHR-01 | `fix/realtime-sources-never-die-silently` |
| 4 | ALM-01, ALM-03, ALM-04, HIST-01 | `fix/alarm-intake-survives-db-outage` |
| 5 | ARCH-01 (network segmentation and doc correction), PLC-06, PHY-05 | `security/segment-compose-network` |
| 6 | PLC-03 | **owner decision first** (safe hold, persisted latch, or plant-reported source) |

**Step 2 — Medium security and reliability.** Theme T3 (InfluxDB tokens), T4 (health), T5 (edge validation), T7 (SIGTERM), PHY-01 (cancel race), GW-API-02 (hub eviction), GW-AUTH-04 (re-seed undoes demotion), GW-AUTH-05 (RSA key parsed per token), OPC-01..07, WEB-01..04, PLAT-01..07, SCR-01..03.

**Step 3 — Tests that pin invariants (theme T9): TST-01..06 first, then ARCH-02, ARCH-03, GW-AUTH-14, TST-15 (a coverage floor in the gate), then the remaining TST findings.** Each is small and turns a convention into a failing test.

**Step 4 — Duplication and modularity (theme T8), dead code, readability.** PLC-11/PLC-13 (split `service.py` before it passes the 700-line limit), GW-API-19 (split `clients.py`), PHY-11/PHY-12, WEB-14, the DUP series.

**Step 5 — Low-severity hygiene** in any order, ideally alongside related work in the same file.

**Decisions only the owner can make** (each is marked "owner decision" in its finding):

- PLC-03 — how the E-Stop latch survives a PLC restart.
- ARCH-01 / PLC-06 — authentication for internal gRPC callers (interceptor secret or mTLS).
- GW-AUTH-01 step 4 — rotating demo passwords after the fix; old backups contain crackable digests.
- GW-AUTH-12, ALM-08 — narrowing DB grants on `scenario_runs` and alarm history (touches invariant I10).
- GW-API-07 — whether `/ready` stays public.
- PHY-02 step 4 — whether a dead stepping loop should end the process.
- PHY-15, ARCH-10, SHR-07 — contract changes (SI units, enum zero values).
- PLC-17 — whether a BAD non-trip instrument should trip.
- Any change to `docs/architecture/invariants.md` (ARCH-02, ARCH-09, GW-AUTH-12).

## 7. Findings by area

### 7.1 API gateway — authentication, sessions, RBAC, audit log, migrations

**Area summary.** The auth core is well designed. Access tokens are RS256 with the algorithm list pinned when decoding. The role, the active flag and whether the session is open are read from the database on every request, so revocation takes effect at once. Refresh tokens rotate inside families, keep an absolute expiry, detect reuse, and serialize concurrent refreshes with `FOR UPDATE`. Every one of the 24 mutating routes requires a role (except the three `/auth` bootstrap routes, which cannot) and is audited by the middleware. The weakest points are around the edges of that core:
- the audit table stores an unsalted SHA-256 of every sign-in body, which is an offline-crackable password verifier;
- the sign-in throttle can be bypassed with concurrent requests;
- the provisioning script can print a DB role password inside an exception;
- demo-user seeding silently undoes an admin's demotion on every restart;
- every token issue re-parses the RSA private key on the event loop (about 55 ms per token).

<details><summary>Scope reviewed by the area auditor</summary>

`apps/api-gateway/src/api_gateway/auth/` (identity.py, jwt_handler.py, password.py, rbac.py, sessions.py, throttle.py), `routers/auth.py`, `routers/users.py`, `routers/audit.py`, `accounts.py`, `audit.py`, `models/user.py`, `dependencies.py`, `config.py`, `main.py`, `__main__.py`, `db_init.py`, `db_roles.py`, `problems.py`, `observability.py`, `schemas/auth.py`, `schemas/users.py`, `schemas/ops.py` (audit response), `migrations/env.py`, `migrations/versions/0001`–`0004`, `alembic.ini`. For context I also read the token handling in `routers/websocket.py`, `docker-compose.yml` (migrate and api-gateway), `infrastructure/nginx/*`, `scripts/dev_secrets`, `apps/web/src/api/endpoints.ts`, `apps/opcua-server/src/opcua_server/gateway.py`. Tests read: `conftest.py`, `gateway_fakes.py`, `test_auth_flow.py`, `test_login_throttle.py`, `test_audit_log.py`, `test_user_administration.py`, `test_db_roles.py`, `test_gateway_startup.py`, `test_demo_users.py`, `test_api_gateway.py`. I ran 167 tests in those files: all pass (55 s). I also ran four throwaway probes from the scratchpad, never the repo: a concurrent-login burst, a re-seed after a demotion, the text of a SQLAlchemy error, and the cost of signing a JWT.

</details>

#### GW-AUTH-01 — The audit log stores an unsalted SHA-256 of every sign-in body, so admins and backups can crack passwords offline
- **Severity:** High
- **Category:** Security
- **Confidence:** High
- **Location:** `apps/api-gateway/src/api_gateway/audit.py:125` and `:168` (digest of the whole body); `apps/api-gateway/src/api_gateway/schemas/ops.py:132` (`request_body_hash` returned by `GET /api/v1/audit`); `apps/api-gateway/tests/test_audit_log.py:75-87` (the test asserts exactly this value); console body format `apps/web/src/api/endpoints.ts:42` + `apps/web/src/api/http.ts:97` (`JSON.stringify({username, password})`).
- **About the file:** `audit.py` is the ASGI middleware that writes one append-only `audit_log` row for each mutating or refused request.
- **Problem:** For `/auth/login` the body is `{"username":"<name>","password":"<pw>"}`. The username is also stored in `detail` (`username=<name>`), and the console's JSON serialization is deterministic. So `request_body_hash` equals `sha256(known_prefix + password + known_suffix)`, an unsalted fast hash. A GPU tries billions of candidates per second, so any human-chosen password (the policy allows 12 characters with 5 distinct ones) falls quickly, and the Argon2id protection is bypassed completely. The same applies to:
  - `/auth/password` (current and new password in one body);
  - `POST /api/v1/users` (initial password);
  - `POST /api/v1/users/{id}/password` (admin reset).

  Who can read these hashes:
  - every admin, through the API;
  - anyone holding a `backup` (`backups/<stamp>/postgres.sql`);
  - the error log, when an audit write fails (`body_sha256=` in `apps/api-gateway/src/api_gateway/audit.py:95-108`).

  Because the table is append-only, existing rows can never be purged. The impact: an admin (or a stolen backup) can recover a colleague's plaintext password and then sign in *as* that engineer to reset an E-Stop. That silently breaks the "safety action is attributable to a person" guarantee, and exposes passwords the users reuse elsewhere.
- **Recommendation:**
  1. In `audit.py`, add a constant `CREDENTIAL_PATHS` holding `/auth/login`, `/auth/password` and `/api/v1/users`, plus a regex for `^/api/v1/users/\d+/password$`. For these paths write `request_body_hash=None`, or better a keyed digest (step 2).
  2. For all other paths, consider switching to a keyed digest, `hmac.new(key, body, sha256)`, with the key read from a new setting `AUDIT_HASH_KEY` that `dev-secrets` generates. It still proves the body was not altered, but it is not crackable without the key. Update the `request_body_hash` column comment and `docs/architecture/invariants.md` I10 wording to say "keyed digest".
  3. Rewrite `test_the_body_is_kept_only_as_a_digest` so it asserts that a login row's `request_body_hash` is **not** `sha256(body)`, and add a case for `/auth/password` and for user creation.
  4. Existing rows keep crackable digests forever (append-only). Owner decision: rotate the demo passwords (`dev-secrets` + `stack down --volumes`) after the fix, and state in the README that backups taken before it hold password-derived data.
  5. Related, same place: `detail=username=<typed name>` is stored for *failed* sign-ins too (`apps/api-gateway/src/api_gateway/routers/auth.py:137`). Users sometimes type their password into the name field, and that plaintext then stays in the table permanently. Owner decision whether to store only names that match an existing account.
- **Effort:** M

#### GW-AUTH-02 — Concurrent sign-in attempts bypass the throttle: all of them pass the check before any failure is counted
- **Severity:** High
- **Category:** Security
- **Confidence:** High (confirmed by probe)
- **Location:** `apps/api-gateway/src/api_gateway/routers/auth.py:139-156` (check at 139, `await verify_password_async` at 150, `record_failure` at 154); same pattern in `apps/api-gateway/src/api_gateway/routers/auth.py:278-293` (password change); `apps/api-gateway/src/api_gateway/accounts.py:44-45`.
- **About the file:** `routers/auth.py` holds the sign-in, refresh, sign-out and password-change endpoints; `throttle.py` is the in-memory sliding-window failure counter.
- **Problem:** `retry_after_s()` only *reads* the counters, and the failure is recorded after an `await` on Argon2 in a worker thread. A burst of N parallel requests for one account therefore all see zero failures and all get a full password check. The probe sent 25 concurrent wrong passwords for `admin1` against the defaults (5 per account, 20 per client, 15-minute window): **25 × 401 and no 429**. Only the next sequential attempt got 429. The attacker gets as many guesses per window as they can send in one burst, not 5.

  Each attempt also allocates Argon2's 64 MiB (`m=65536`) in the default thread pool (up to min(32, CPU+4) threads). The same burst is therefore also a memory and CPU denial of service against the gateway, which also serves telemetry. (This is a separate problem from Д16, the per-worker counters, which is accepted.)
- **Recommendation:**
  1. In `throttle.py`, make the check and the charge one step. Add `LoginThrottle.begin_attempt(username, client) -> int`: it returns the wait if the key is locked, and otherwise appends a provisional failure at once. `record_success()` then clears the account key (it already does), and a failure keeps the provisional entry. Replace the `retry_after_s` + `record_failure` pair in `login()` and `change_password()` with it.
  2. Add a module-level `asyncio.Semaphore` (for example 4, configurable as `login_max_concurrent_hashes`) in `accounts.py` around `verify_password_async` and `hash_password_async`, so concurrent Argon2 work, and its memory, is bounded.
  3. Test in `test_auth_flow.py`: `asyncio.gather` 10 wrong logins with `tight_throttle(app, failures=2)` and assert that at most 2 answers are 401 and the rest are 429. Keep the existing sequential tests as the "allowed" side.
- **Effort:** M

#### GW-AUTH-03 — `db_roles` can print a role's plaintext password in an exception, and its test would not notice
- **Severity:** Medium
- **Category:** Security
- **Confidence:** High (SQLAlchemy message format confirmed by probe)
- **Location:** `apps/api-gateway/src/api_gateway/db_roles.py:41-49` (the password becomes part of the SQL text sent by `exec_driver_sql`), `:64` (no handling around `asyncio.run`); weak test `apps/api-gateway/tests/test_db_roles.py:138-143`.
- **About the file:** `db_roles.py` is run by the `migrate` job right after `alembic upgrade head`. It gives `cogniboiler_gateway` and `cogniboiler_alarms` a login and their passwords from the environment.
- **Problem:** `hide_parameters=True` hides only bound parameters. The executed statement is the literal `ALTER ROLE "x" WITH LOGIN PASSWORD '<secret>'`, and any `DBAPIError` renders it as `[SQL: ALTER ROLE ... PASSWORD 'S3cretProbe']` (verified). `main()` does not catch anything, so the traceback with the password goes to the `migrate` container's stderr (`docker logs`, `stack logs migrate`). Likely triggers are a missing role (0004 not applied), a lost connection, or a permission error. PostgreSQL itself also logs the failing statement text under its default `log_min_error_statement = error`. The test "a failure never echoes the parameters" only checks that `hide_parameters is True`, so it passes while the leak exists.
- **Recommendation:**
  1. In `provision()`, wrap each `exec_driver_sql` in `try/except SQLAlchemyError as exc:`. Log `logger.error("Could not give role %s a login: %s", role, type(exc).__name__)` without `str(exc)`, then `raise RuntimeError(f"provisioning {role} failed") from None`, so the chained message is dropped.
  2. Better, so the plaintext never reaches the server: compute a SCRAM-SHA-256 verifier in Python (`hashlib.pbkdf2_hmac` plus HMAC, about 15 lines, stdlib only) and send `PASSWORD 'SCRAM-SHA-256$4096:...'`. PostgreSQL accepts pre-hashed verifiers.
  3. Replace the weak test with one where the fake engine raises `DBAPIError.instance(statement, None, Exception(), Exception)` on execute. Assert that the password string appears neither in the raised exception chain (`str(exc)`, `exc.__cause__`, `exc.__context__`) nor in `caplog.text`.
- **Effort:** S

#### GW-AUTH-04 — Demo-user seeding gives back a role an admin removed, on every gateway restart, with no audit entry
- **Severity:** Medium
- **Category:** Security
- **Confidence:** High (confirmed by probe)
- **Location:** `apps/api-gateway/src/api_gateway/db_init.py:93-106`; `docker-compose.yml` sets `AUTO_INIT_DB: "true"`; the behaviour is codified in `apps/api-gateway/tests/test_gateway_startup.py:112-125`.
- **About the file:** `db_init.py` seeds the four roles and the demo users `admin`, `engineer`, `operator` and `viewer` at gateway start-up.
- **Problem:** `_ensure_default_user` adds the demo role assignment whenever that exact `(user, role)` row is missing, even for an existing user. An admin demoting `engineer` to `viewer` (for example because its password leaked) replaces the row. At the next start the seeder re-adds `engineer`, so `effective_role()` picks the higher role and the account is an engineer again. Probe: "after demotion by admin: viewer" became "after restart (re-seed): engineer". The same holds for a demoted `admin`. Nothing writes an audit row for the re-grant, and the account's sessions were already closed, so the user simply signs in again with the old privilege. Blocking (`is_active=False`) is not undone, which makes the behaviour inconsistent as well.
- **Recommendation:**
  1. In `_ensure_default_user`, add the role only when the user was created in this run, or when the user has **no** `user_roles` row at all. The second option keeps the "repair a half-seeded database" intent. Use `select(UserRole.id).where(UserRole.user_id == user.id).limit(1)`.
  2. Change `test_a_missing_role_assignment_is_restored` into two tests: "a user with no role gets its demo role back" (allowed) and "a demoted demo user keeps the role an admin gave it" (the demotion survives re-seeding).
  3. When the seeder does add a role, log it at `warning` with the username, so the change is visible.
- **Effort:** S

#### GW-AUTH-05 — Every token issue re-parses the RSA private key from PEM on the event loop (about 55 ms per token, two per sign-in or refresh)
- **Severity:** Medium
- **Category:** Performance
- **Confidence:** High (measured: 54.7 ms with a PEM string vs 0.93 ms with a loaded key, RSA-2048, PyJWT 2.14.0)
- **Location:** `apps/api-gateway/src/api_gateway/auth/jwt_handler.py:107` (`jwt.encode(payload, _private_key(), ...)` with a `str`), `:195-200` (decode re-parses the public key, cheap); called synchronously from `auth/sessions.py` `open_session`/`rotate_session`.
- **About the file:** `jwt_handler.py` signs and verifies access and refresh tokens.
- **Problem:** PyJWT loads a PEM string on every call, and `cryptography` validates the RSA key each time it is loaded. Each sign-in, refresh or password change therefore blocks the event loop for about 110 ms, stalling WebSocket telemetry fan-out and every other request. This violates the project rule "no blocking work on the event loop". Many console tabs refreshing together make it worse.
- **Recommendation:**
  1. Add `@functools.cache def _signing_key(pem: str) -> RSAPrivateKey` (using `serialization.load_pem_private_key`) and `_verifying_key(pem: str)`, and pass the loaded objects to `jwt.encode`/`jwt.decode`. Keying the cache by the PEM string keeps the test fixture that swaps keys working.
  2. Load and validate both keys once in `lifespan` (see GW-AUTH-09), so a bad key fails at start-up, not at the first sign-in.
  3. Test: check that `_signing_key` is called with the same PEM across two `issue_access_token` calls and parses once (`cache_info().misses == 1`). Do not add a timing assertion: that would depend on machine speed.
- **Effort:** S

#### GW-AUTH-06 — The audit write happens after the response: a failure only reaches the error log, and no security event has a metric
- **Severity:** Medium
- **Category:** Logging
- **Confidence:** High
- **Location:** `apps/api-gateway/src/api_gateway/audit.py:87-109` (`write_audit_entry`), `:149-175` (written in `finally`, after the response has been sent); `apps/api-gateway/src/api_gateway/auth/sessions.py:132-146` (reuse detection: log only); `apps/api-gateway/src/api_gateway/routers/auth.py:153-156` (failed sign-in: no log, no metric).
- **About the file:** see GW-AUTH-01; `sessions.py` handles refresh-token families.
- **Problem:** When the DB insert fails, the action (for example an E-Stop reset through gRPC) has already happened and the client got 200. The only trace is one `logger.error` line, as documented. That is acceptable as a design, but nothing makes it *alertable*:
  - there is no Prometheus counter for audit write failures;
  - refresh-token reuse (a stolen-token signal) is a `warning` log only;
  - failed sign-ins and throttle refusals produce neither a log nor a metric.

  An operator watching Grafana cannot see a credential-stuffing burst or a broken audit trail. Also, `CancelledError` (a `BaseException`) during the `finally` write, for example at shutdown, loses the record without even the error log.
- **Recommendation:**
  1. In `observability.py`, add these counters:
     - `gateway_audit_write_failures_total`;
     - `gateway_login_failures_total{reason="invalid_credentials|throttled"}`;
     - `gateway_refresh_rejections_total{code}`, which covers `auth.refresh_reused`.

     Increment them in `write_audit_entry` (except branch), in `login()` and in `rotate_session`/`refresh()`.
  2. Add an alert panel or rule in `infrastructure/` Grafana for `gateway_audit_write_failures_total > 0` and for a reuse rate above 0.
  3. In `write_audit_entry`, also catch `asyncio.CancelledError`: log the record, then re-raise.
  4. Making the audit row transactional with the action (refusing the action when auditing is impossible) is a design change: out of scope, owner decision.
  5. Tests: extend `TestWriteFailure` to assert the counter increments; add a test that a reused refresh token increments `gateway_refresh_rejections_total{code="auth.refresh_reused"}`.
- **Effort:** S

#### GW-AUTH-07 — The console's sign-out and a detected refresh-token reuse are audited without an actor
- **Severity:** Low
- **Category:** Security
- **Confidence:** High
- **Location:** `apps/api-gateway/src/api_gateway/routers/auth.py:216-233` (the actor is set only in the bearer branch), `:184-198` (refresh refusal: no actor); the console signs out with the cookie only, `apps/web/src/api/endpoints.ts:51` (`auth: false`).
- **About the file:** see GW-AUTH-02.
- **Problem:** `session_id_of(token)` already returns `(session_id, user_id)` from a verified refresh token, but `logout()` discards the user id. Every console sign-out is therefore stored with `user_id=NULL, username=NULL`. Likewise, when `rotate_session` closes a family for reuse, the audit row does not name the affected user, although `record.user_id` is known. These are exactly the rows an investigator filters on.
- **Recommendation:**
  1. In `logout()`, when `found` is not None, call `load_account(db, user_id=found[1])` and `set_audit_actor(...)` if the account exists.
  2. Add `user_id: int | None` to `RefreshRejectedError`, fill it in `rotate_session` once the record is known, and set the audit actor in `refresh()` from it (with a username lookup).
  3. Tests in `test_audit_log.py`: after a cookie-only `/auth/logout` from the `browser` fixture, the row's `username` is the signed-in user; after a reuse (grace 0), the `auth.refresh_reused` row names the user.
- **Effort:** S

#### GW-AUTH-08 — The "last active admin" and case-insensitive username checks are check-then-act races
- **Severity:** Low
- **Category:** Reliability
- **Confidence:** Medium
- **Location:** `apps/api-gateway/src/api_gateway/accounts.py:208-231` (`_active_admin_count` then delete/insert, no lock); `apps/api-gateway/src/api_gateway/accounts.py:164-187` (`lower(username)` pre-check; the unique index `ix_users_username` is case-sensitive).
- **About the file:** `accounts.py` holds user administration and self-service password change.
- **Problem:** Two admins demoting (or blocking) each other at the same moment both read `count == 2`, both commit, and zero active admins remain. Self-change is refused, but mutual change is not. Recovery would need direct DB access. In the same way, two concurrent creations of `Anna` and `anna` both pass the pre-check and both insert, because the database index does not fold case.
- **Recommendation:**
  1. In `update_user`, when `loses_admin` is true, lock the admin rows before counting: `select(User.id).join(...).where(Role.name == "admin", User.is_active).with_for_update()`. Alternatively take a transaction-level advisory lock, `SELECT pg_advisory_xact_lock(<const>)`, on PostgreSQL.
  2. Add a new Alembic revision with a unique index on `lower(username)`. Keep the `IntegrityError` → 409 mapping that already exists.
  3. Tests: a unit test of `update_user` that monkeypatches `_active_admin_count` is enough for the logic; note in the test docstring that real concurrency is only enforced on PostgreSQL.
- **Effort:** S

#### GW-AUTH-09 — JWT settings are not pinned or validated at start-up: algorithm, keys, issuer and audience
- **Severity:** Low
- **Category:** Security
- **Confidence:** High
- **Location:** `apps/api-gateway/src/api_gateway/config.py:44` (`jwt_algorithm: str = "RS256"`, freely configurable), `:45-48`; `apps/api-gateway/src/api_gateway/auth/jwt_handler.py:57-82` (keys are checked only when the first token is issued), `:98-106` (no `iss`/`aud`), `:195-200` (`require` has no `sid`, `iss` or `aud`).
- **About the file:** `config.py` holds pydantic settings from the environment and `.env`; for `jwt_handler.py` see GW-AUTH-05.
- **Problem:**
  - A gateway without keys starts healthy (`/health` 200) and answers every sign-in with a generic 500, `RuntimeError` in the log.
  - A mismatched pair, or a 1024-bit key pasted into `.env`, is accepted silently.
  - The algorithm comes from the environment although the rule says RS256. PyJWT refuses a PEM used as an HMAC secret and `none` with a key (verified), so this is defence in depth, not an exploit.
  - Tokens carry no `iss`/`aud`. The module docstring invites sharing the public key with downstream services, and any such verifier would accept a gateway token minted for a different purpose.
  - `sid` is required only by hand-written code (`apps/api-gateway/src/api_gateway/auth/identity.py:131-139`, `apps/api-gateway/src/api_gateway/auth/sessions.py:206-212`).
- **Recommendation:**
  1. `config.py`: `jwt_algorithm: Literal["RS256"] = "RS256"`; add `Field(ge=1, le=60)` to `jwt_access_token_expire_minutes`, `ge=0, le=30` to `refresh_reuse_grace_s`, and `ge=1` to the throttle limits.
  2. Add `validate_signing_keys()` in `jwt_handler.py`: load both PEMs, assert the key is RSA with `key_size >= 2048`, and assert `private.public_key().public_numbers() == public.public_numbers()`. Call it at the top of `lifespan` in `main.py` so a misconfiguration stops start-up.
  3. Add `"iss": "cogniboiler-gateway"` and `"aud": "cogniboiler"` in `_issue`; pass `issuer=`/`audience=` to `jwt.decode`; add `sid`, `iss` and `aud` to `require`. Note: this closes existing sessions once, at deploy time.
  4. Tests: a token signed with another key → 401 `auth.token_invalid`; an HS256 token forged with the public PEM as secret → 401; an `alg: none` token → 401; a token without `sid` → 401; an expired token → 401 `auth.token_expired`. None of these cases is tested today. `validate_signing_keys` should reject a 1024-bit key and a mismatched pair, and accept the fixture's 2048-bit pair.
- **Effort:** M

#### GW-AUTH-10 — The per-client throttle key is weak behind the OPC UA server, and any stack container may set X-Forwarded-For
- **Severity:** Low
- **Category:** Security
- **Confidence:** Medium
- **Location:** `apps/api-gateway/src/api_gateway/routers/auth.py:71-75` (the client key is `client_address(scope)`); `docker-compose.yml:267` (`--forwarded-allow-ips 172.16.0.0/12,10.0.0.0/8,192.168.0.0/16`); `apps/opcua-server/src/opcua_server/gateway.py:95` (sign-in without a forwarded address); `apps/api-gateway/src/api_gateway/auth/throttle.py:93-97`.
- **About the file:** see GW-AUTH-02.
- **Problem:** The X-Forwarded-For implementation is correct. nginx *replaces* the header (`infrastructure/nginx/proxy.inc:5`), and uvicorn honours it only from the private ranges. But:
  - (a) Every OPC UA user signs in through `opcua-server`, so they all share its container address. One misbehaving OPC UA client on port 4840 can use up the 20-failure client budget and lock every OPC UA user out for 15 minutes. The audit log also records `opcua-server` instead of the real client.
  - (b) Any container on the Compose network can send its own `X-Forwarded-For` to forge the audited address and avoid the per-client counter. This is accepted by the 2026-09-18 decision, recorded here for completeness.
  - (c) If the stack ever serves IPv6 clients, keying by the full address lets one /64 rotate addresses freely.
- **Recommendation:**
  1. (Cross-reference for the OPC UA owner.) `opcua-server` could send `X-Forwarded-For: <OPC UA peer address>` on `/auth/login`. It is inside the trusted range, so the gateway would use it. Coordinate with the OPC UA audit.
  2. In `LoginThrottle`, normalise an IPv6 client key to its /64 with `ipaddress.ip_network(f"{ip}/64", strict=False)`. Add a unit test next to `test_a_client_is_counted_across_the_names_it_tries`.
  3. Narrow `--forwarded-allow-ips` to nginx alone: give `web` a fixed address on the Compose network, or name the network's subnet. Owner decision, because it changes how the other services are recorded.
- **Effort:** S

#### GW-AUTH-11 — Token responses carry no `Cache-Control: no-store`, and the gateway sets no security headers of its own
- **Severity:** Low
- **Category:** Security
- **Confidence:** High
- **Location:** `apps/api-gateway/src/api_gateway/routers/auth.py:105-116` (`_token_response`), `:160-162`, `:199-201`, `:297-300`; `infrastructure/nginx/headers.inc` (sets CSP, nosniff and frame options, but no `Cache-Control`); `apps/api-gateway/src/api_gateway/main.py:150-161` (no header middleware).
- **About the file:** `main.py` builds the FastAPI app: middleware, CORS and routers.
- **Problem:** RFC 6749 §5.1 requires `Cache-Control: no-store` on responses that carry tokens. `/auth/login`, `/auth/refresh` and `/auth/password` return both tokens in the body with no cache directive. The same applies to `/api/v1/users` and `/api/v1/audit` (personal data). Separately, the development path (Vite proxy to a host-run gateway on :8000) has none of nginx's headers.
- **Recommendation:**
  1. In `routers/auth.py`, add `_no_store(response)` setting `Cache-Control: no-store` and `Pragma: no-cache`, and call it wherever `_set_refresh_cookie` is called.
  2. Alternatively, a tiny pure-ASGI middleware in `main.py` that adds `Cache-Control: no-store` for paths starting with `/auth` or `/api/v1/users` or `/api/v1/audit`, plus `X-Content-Type-Options: nosniff` on every response.
  3. Test in `test_auth_flow.py`: login and refresh responses have `cache-control: no-store`.
- **Effort:** S

#### GW-AUTH-12 — The gateway DB role is broader than the code needs; `scenario_runs` (an attribution trail) can be rewritten
- **Severity:** Low
- **Category:** Security
- **Confidence:** High
- **Location:** `apps/api-gateway/migrations/versions/0004_application_roles.py:37` (`GATEWAY_READ_WRITE` includes `scenario_runs`, `users`, `roles`), `:70-71` (`SELECT, INSERT, UPDATE, DELETE`).
- **About the file:** migration 0004 creates the application roles and their grants; only the `migrate` job connects as owner.
- **Problem:** The gateway code only inserts into `scenario_runs` ("who loaded which scenario, injected which fault", `apps/api-gateway/src/api_gateway/models/user.py:310-315`), yet the role may also `UPDATE` and `DELETE` it. A compromised gateway can therefore rewrite who started a fault, and the append-only guarantee of I10 does not extend to it. The role also holds `DELETE` on `users` and `roles` although "accounts are blocked, never deleted". Separately, `refresh_tokens` is never purged: a row is added per refresh (about 96 a day per open console session), so the table and the `EXISTS` subqueries in `apps/api-gateway/src/api_gateway/auth/identity.py:141-147` grow without bound. A purge job would be new functionality: owner decision.
- **Recommendation:**
  1. Add a new migration `0005_narrow_gateway_grants` (0004 is applied and must not be edited):
     - `REVOKE UPDATE, DELETE ON scenario_runs FROM cogniboiler_gateway`;
     - `REVOKE DELETE ON users, roles FROM cogniboiler_gateway`;
     - optionally, an append-only trigger on `scenario_runs` reusing the 0003 function pattern.
  2. Update `docs/architecture/invariants.md` I10 if `scenario_runs` joins the append-only set (an invariant change needs owner approval).
  3. Check with `grep` that no gateway code updates or deletes those tables before revoking. Today none does: only `db.add(ScenarioRun(...))`.
- **Effort:** S

#### GW-AUTH-13 — Working default credentials and development CORS origins live in code
- **Severity:** Low
- **Category:** Security
- **Confidence:** High
- **Location:** `apps/api-gateway/src/api_gateway/config.py:63-65` (`postgresql+asyncpg://cogniboiler:cogniboiler@localhost...`); `apps/api-gateway/alembic.ini:5` (same URL; overridden by `apps/api-gateway/migrations/env.py:45` but still committed); `apps/api-gateway/src/api_gateway/config.py:38-41` with `apps/api-gateway/src/api_gateway/main.py:151-158` (`allow_credentials=True` for `localhost:5173` in every environment).
- **About the file:** see GW-AUTH-09.
- **Problem:** A password default in code goes against the "no password in git" rule. A gateway or `alembic` started without `DATABASE_URL` silently tries the owner account with password `cogniboiler`, which works on any local PostgreSQL created with the documented defaults. The credentialed CORS allowance for the Vite origin also applies inside the stack (nothing overrides `CORS_ALLOWED_ORIGINS` in compose). Any page served on `localhost:5173` is same-site with the gateway, so the `SameSite=Strict` cookie is sent, and it could call `/auth/refresh` and read the tokens.
- **Recommendation:**
  1. `config.py`: `database_url: str = ""`, and fail fast in `dependencies.py` or `lifespan` with a clear message when it is empty.
  2. `alembic.ini`: replace the URL with a placeholder such as `driver://set-DATABASE_URL`, since `env.py` always overrides it.
  3. Default `cors_allowed_origins` to `[]` and set the Vite origins only in the development docs or in `.env.example`.
  4. Test: `Settings(_env_file=None)` has an empty `database_url` and no CORS origins.
- **Effort:** S

#### GW-AUTH-14 — No test enforces "every mutating route declares a role", and several auth branches have no test
- **Severity:** Low
- **Category:** Tests
- **Confidence:** High
- **Location:** `apps/api-gateway/tests/` (no test iterates `app.routes`); `apps/api-gateway/src/api_gateway/auth/identity.py:122-129` (the `auth.token_expired` branch is untested); `apps/api-gateway/src/api_gateway/auth/sessions.py:148-152` (refresh of a blocked account is untested); `apps/api-gateway/src/api_gateway/observability.py:46-49` and FastAPI `/docs` (public GETs without a role).
- **About the file:** the gateway test suite (SQLite in memory, fake upstreams).
- **Problem:** All 24 mutating routes require a role today, but that is enforced only by review. A route added without `OperatorUser` would pass the gate. Branches with no test:
  - an expired access token → `auth.token_expired`;
  - a refresh after the account was blocked (it should revoke the family with reason `blocked`);
  - an access token whose session an admin closed through `revoke-sessions`, used on `/auth/me`;
  - a refresh token used as a bearer token on a protected route (only the unit-level `decode_access_token` check exists);
  - on the allow side, that an `engineer` passes `/api/v1/users` refused and an `admin` passes. The admin side is covered implicitly; the engineer refusal is covered for listing only.
- **Recommendation:**
  1. Add `test_route_inventory.py`: for each `APIRoute` in `create_app().routes` with a method in `{POST, PUT, PATCH, DELETE}`, walk `route.dependant.dependencies` recursively and assert that a dependency produced by `require_role` is present. Keep an explicit allow-list of `/auth/login`, `/auth/refresh`, `/auth/logout` and `/auth/password` (`AuthenticatedUser`). Do the same for GETs, with an allow-list of `/health`, `/ready`, `/metrics`, `/docs`, `/redoc`, `/openapi.json`, `/auth/me`.
  2. Add the listed branch tests to `test_auth_flow.py` / `test_user_administration.py`. Get an expired token from `issue_access_token(..., not_after_ms=now-1000)` for an open session, so it depends on no clock.
  3. For each admin route, add one "engineer → 403" and one "admin → 2xx" case (parametrised), so both outcomes of the check are proven.
- **Effort:** M

#### GW-AUTH-15 — Stale comment on the Argon2 parameters, which are not pinned; `verify_password` swallows every exception silently
- **Severity:** Low
- **Category:** Readability
- **Confidence:** High
- **Location:** `apps/api-gateway/src/api_gateway/auth/password.py:26-28` (comment says `time_cost=2 ... parallelism=2`), `:28` (`Argon2Hasher()` with library defaults; actual hashes are `m=65536,t=3,p=4`), `:62-66` (`except Exception: return False`); `apps/api-gateway/src/api_gateway/routers/auth.py:58` (`_DUMMY_HASH` relies on the same defaults).
- **About the file:** `password.py` hashes and verifies passwords with Argon2id through pwdlib.
- **Problem:** The comment is false: it names parameters that are not the ones used. Because nothing is pinned, a `pwdlib` upgrade can change the cost silently. Stored hashes would keep the old cost while `_DUMMY_HASH` gets the new one, breaking the timing parity between unknown users and wrong passwords. The blanket `except` also turns a real fault (for example a corrupted stored hash, or a `MemoryError` under the load of GW-AUTH-02) into "wrong password" with no log line.
- **Recommendation:**
  1. `_hasher = PasswordHash([Argon2Hasher(time_cost=3, memory_cost=65536, parallelism=4)])` (the current effective values), and rewrite the comment to match.
  2. Narrow the `except` to `(argon2.exceptions.VerificationError, argon2.exceptions.InvalidHashError, ValueError)` and log `warning` "stored password hash is malformed" (without the hash) for the invalid-hash case.
  3. Test: `hash_password("x")` starts with `$argon2id$v=19$m=65536,t=3,p=4$`.
- **Effort:** S

#### GW-AUTH-16 — Duplication and dead code in the auth modules
- **Severity:** Low
- **Category:** Duplication
- **Confidence:** High
- **Location:**
  - `apps/api-gateway/src/api_gateway/auth/jwt_handler.py:172-179` (`create_access_token`/`create_refresh_token`, used only by `tests/test_api_gateway.py`);
  - `apps/api-gateway/src/api_gateway/auth/rbac.py:44-46` (`_role_level`, test-only wrapper of `identity.role_level`);
  - `apps/api-gateway/src/api_gateway/routers/auth.py:93-102` and `:239-245` (the same cookie deletion twice; `_cookie_clearing_headers` builds a throwaway `Response`);
  - `apps/api-gateway/src/api_gateway/routers/auth.py:139-147` and `:278-286` (the same throttle → 429 block);
  - `apps/api-gateway/src/api_gateway/accounts.py:36-37`, `apps/api-gateway/src/api_gateway/auth/jwt_handler.py:130,157`, `apps/api-gateway/src/api_gateway/audit.py:123`, `apps/api-gateway/src/api_gateway/db_init.py:88,104` (local `int(time.time()*1000)` while `cogniboiler_runtime.now_ms` exists and `sessions.py` uses it);
  - `apps/api-gateway/src/api_gateway/accounts.py:220-222`, `:238-240`, `:262-264` (`_account_or_404` followed by `db.get(User)` with a second, identical 404);
  - `apps/api-gateway/src/api_gateway/accounts.py:229` (unblocking a user records `revoked_reason="role_change"`).
- **About the file:** see the earlier findings.
- **Problem:** The test-only helpers let `test_api_gateway.py` exercise wrappers with a random session id instead of `issue_access_token`, which production uses. The repeated blocks invite drift: for example a fix to the cookie attributes applied in one place only. The wrong revoke reason misleads anyone reading `refresh_tokens`.
- **Recommendation:**
  1. Delete `create_access_token`, `create_refresh_token` and `_role_level`; rewrite those tests on `issue_access_token`/`issue_refresh_token` and `role_level`.
  2. Add `_delete_refresh_cookie(response: Response)` in `routers/auth.py`, used by `logout()`. For the refresh error, return the problem response and call the helper on it instead of fabricating headers.
  3. Add `_refuse_if_throttled(request, username, ip, message)` in `routers/auth.py`. It merges naturally with the atomic `begin_attempt` from GW-AUTH-02.
  4. Replace the local clocks with `cogniboiler_runtime.now_ms`.
  5. Add `_user_row_or_404(db, user_id) -> User` in `accounts.py`.
  6. Use `reason = "blocked" if request.is_active is False else ("unblocked" if activity_changes and not role_changes else "role_change")`, and extend the column comment accordingly.
- **Effort:** S

### 7.2 API gateway — operational API, WebSocket, realtime hub, upstream clients

**Area summary.** Authorization in this area is right. Every mutating route declares `OperatorUser` or `EngineerUser`, every read route declares `ViewerUser`, and the gateway never calls `PhysicsService.ApplyControlCommand`. Every request float has bounds, so pydantic rejects NaN and ±Infinity (checked by probe). Unary gRPC calls all have deadlines, and Flux names are quoted. The weak points are in the realtime path. The hub can silently drop alarm changes, which its docstring promises never to do. The telemetry and PLC-status sources die for good, with no log line, on any error that is not a gRPC error. A binary WebSocket frame raises an unhandled `KeyError`, which unauthenticated clients can use to fill the log with ERROR tracebacks. Smaller issues: the audit row can record a fault injection as a 503 failure when the fault was actually applied; `python -m api_gateway` on Windows breaks the MQTT source (wrong event loop); and routers repeat the same client accessor and gRPC-to-Problem boilerplate about 25 times.

<details><summary>Scope reviewed by the area auditor</summary>

`apps/api-gateway/src/api_gateway/routers/` (commands.py, simulation.py, alarms.py, history.py, kpi.py, plc.py, status.py, health.py, websocket.py, sensors.py), `realtime/hub.py`, `realtime/sources.py`, `clients.py`, `plant_state.py`, `plc_state.py`, `readiness.py`, `schemas/command.py`, `schemas/ops.py`, `schemas/plant.py`, `schemas/plc.py`, `schemas/sensor.py`, `__main__.py`, and as consumed context `main.py`, `audit.py`, `problems.py`, `observability.py`, `dependencies.py`, `config.py`, `auth/rbac.py`, `auth/identity.py` (resolve_access_token), `shared/runtime/.../mqtt.py` (MqttSession), `shared/observability/.../grpc_observability.py`, `infrastructure/nginx/site.inc`, the gateway service in `docker-compose.yml`. Tests: `conftest.py`, `gateway_fakes.py`, `test_websocket.py`, `test_realtime_hub.py`, `test_history_query.py`, `test_upstream_clients.py`, `test_simulation_routes.py`, `test_plant_commands.py`, `test_alarm_routes.py`, the history part of `test_api_gateway.py`, `test_gateway_startup.py` (lifespan). Findings marked "confirmed by probe" were run against throwaway tests in the scratchpad (not in the repo); `mypy` and the full suite were not run by me.

</details>

#### GW-API-01 — Realtime telemetry and PLC-status sources die permanently and silently on any non-gRPC error
- **Severity:** High
- **Category:** Reliability
- **Confidence:** High (confirmed by probe)
- **Location:** `apps/api-gateway/src/api_gateway/realtime/sources.py:56-69`, `apps/api-gateway/src/api_gateway/realtime/sources.py:72-87`, `apps/api-gateway/src/api_gateway/main.py:84-104`, `apps/api-gateway/src/api_gateway/main.py:110-112`
- **About the file:** `sources.py` holds the upstream loops that feed the WebSocket hub: the physics state stream, PLC status polling, and MQTT events. `main.py` starts them as background tasks in the lifespan.
- **Problem:** `run_telemetry` catches only `grpc.RpcError`, and so does `run_plc_status`:
  ```python
  try:
      async for message in physics.stream_system_state(interval_s=0.0):
          ...
          hub.publish(Channel.TELEMETRY, "state", plant_state(message).model_dump(mode="json"))
  except grpc.RpcError as exc:
      outage.failed(exc)
  ```
  Any other exception ends the task. That includes `ValueError` from `encode(..., allow_nan=False)` when a value is NaN or infinite, a pydantic `ValidationError` from `plant_state()` when a new `SensorQuality` value misses the `QualityName` Literal, and `ValueError` from `pb2.ControlMode.Name()` on an unknown mode in `plc_status_from_proto`. The task is stored in `sources` and not awaited until shutdown, and shutdown uses `gather(..., return_exceptions=True)`, so the exception is never logged. The probe fed one NaN pressure: the task finished with `ValueError('Out of range float values are not JSON compliant: nan')` and wrote no log record. After that, every console gets no telemetry until the gateway restarts. A diverging physics step (the Д17 class of instability) is enough to trigger this. The only sign is `gateway_telemetry_age_seconds` slowly growing.
- **Recommendation:**
  1. In `run_telemetry` and `run_plc_status`, handle a per-message conversion or encoding failure separately. Wrap `plant_state(...)` / `plc_status_from_proto(...)` and `hub.publish(...)` in `try/except (ValueError, pydantic.ValidationError)`, log once per outage through `_OutageLog` (a `warning` naming the source), skip that message and continue. Leave the `RpcError` branch as it is.
  2. Add a last-resort `except Exception` in each loop that logs with `exc_info` at `error` level (this is a platform failure), sleeps `RECONNECT_DELAY_S` and continues. `CancelledError` is not an `Exception`, so shutdown still works.
  3. In `main.py`, attach a `task.add_done_callback` to each source. If a source ends for any reason other than cancellation, log it at `error` level so nothing can die silently again.
  4. Tests in `tests/test_realtime_hub.py`: (a) a fake physics stream whose first message has `boiler.pressure_pa = nan` followed by a valid one, asserting the second one reaches the subscriber and one warning is logged; (b) a `FakePLCClient` status with `mode=99`, asserting the loop keeps running.
- **Effort:** S

#### GW-API-02 — A telemetry frame can silently evict a queued alarm change or PLC event
- **Severity:** Medium
- **Category:** Reliability
- **Confidence:** High (confirmed by probe)
- **Location:** `apps/api-gateway/src/api_gateway/realtime/hub.py:64-72` (plus the promise in the docstring at `hub.py:8-12`)
- **About the file:** `hub.py` fans live frames out to WebSocket subscribers through one bounded `asyncio.Queue` per subscriber. Telemetry is meant to be lossy, events lossless.
- **Problem:** Telemetry frames, events and control replies share one queue. When the queue is full, `offer_frame` removes the head of the queue, whatever type of frame it is:
  ```python
  if self.queue.full():
      self.queue.get_nowait()      # may be an alarm change, a PLC event or a control reply
      self.dropped_frames += 1
  self.queue.put_nowait(frame)
  ```
  The probe used queue size 2 with one alarm change followed by two telemetry frames. The queue ended up holding `["state", "state"]` with `overflowed=False`: the alarm change was gone and nothing asked the client to reload. This breaks the module's guarantee ("PLC events and alarm changes must not be lost silently"). A slow console on a busy stack can miss a trip event or an alarm transition until its next REST refresh.
- **Recommendation:**
  1. In `Subscriber.offer_frame`, stop evicting an arbitrary head frame. The simplest fix inside the current design: when the queue is full, drop the *incoming* telemetry frame and count it in `dropped_frames`. Telemetry is a stream of snapshots, so the next frame replaces it anyway.
  2. Alternative, if telemetry must stay fresh: keep telemetry in a separate one-slot "latest frame" holder that `_Connection.write` drains alongside the event queue. Only take this route if the owner agrees it is a refactor and not a feature.
  3. Add a test in `tests/test_realtime_hub.py::TestSubscriber`: queue size 2, `offer_event("alarm")`, then `offer_frame("t1")` and `offer_frame("t2")`; assert `"alarm"` is still in the queue, or that `overflowed is True`.
- **Effort:** S

#### GW-API-03 — A binary WebSocket frame causes an unhandled `KeyError`: unauthenticated ERROR tracebacks, no audit row
- **Severity:** Medium
- **Category:** Security
- **Confidence:** High (confirmed by probe)
- **Location:** `apps/api-gateway/src/api_gateway/routers/websocket.py:101-113`, `apps/api-gateway/src/api_gateway/routers/websocket.py:222-242`, `apps/api-gateway/src/api_gateway/routers/websocket.py:264-273`
- **About the file:** `websocket.py` implements `/ws`: authentication in the first frame, subscribe/unsubscribe/renew/ping, a session guard, and close handling.
- **Problem:** `_receive_json` calls `websocket.receive_text()`, which in Starlette 1.6 returns `message["text"]`. A binary frame carries `bytes` instead, so this raises `KeyError('text')`. Before authentication, the `except` at lines 230-232 does not cover `KeyError`, so the exception leaves the endpoint. uvicorn then logs an "Exception in ASGI application" traceback at ERROR level, and no refusal audit row is written. The probe saw `KeyError 'text'` and `rows: []`. After authentication, the generic branch at lines 269-273 logs `WebSocket session of viewer1 failed` at ERROR with a traceback and closes with 1011. Any anonymous client can therefore produce an ERROR traceback per connection. That floods `logs/`, makes `demo` fail (it fails on any `error` line), breaks the "error only for platform failures" rule, and skips the audit trail that exists for pre-auth refusals.
- **Recommendation:**
  1. In `_receive_json`, call `await websocket.receive()` directly, call `websocket._raise_on_disconnect`-equivalent handling (raise `WebSocketDisconnect` on `websocket.disconnect`), and if `"text"` is missing raise `_CloseConnectionError(CLOSE_BAD_REQUEST, "frames must be JSON text")`.
  2. Also catch `(TypeError, ValueError)` around `json.loads` so no decoder error reaches the generic path.
  3. Tests in `tests/test_websocket.py`: `connection.send_bytes(b"\x00")` as the first frame expects close `(4401, "ws.bad_request")` and one audit row; after `authenticate`, `send_bytes` expects close 4400 and no ERROR record in `caplog`.
- **Effort:** S

#### GW-API-04 — The audit trail can record an applied fault injection or clear as a 503 failure; timeouts are recorded as failures too
- **Severity:** Medium
- **Category:** Reliability
- **Confidence:** High
- **Location:** `apps/api-gateway/src/api_gateway/routers/simulation.py:197-213`, `apps/api-gateway/src/api_gateway/routers/simulation.py:240-252`, `apps/api-gateway/src/api_gateway/audit.py:173`, `apps/api-gateway/src/api_gateway/problems.py:50-58`
- **About the file:** `simulation.py` holds the engineer-only simulation controls. It records `scenario_runs` and sets the audit outcome from the physics acknowledgement.
- **Problem:** `inject_fault` and `_clear` make a second RPC, `get_simulation_status()`, inside the same `try` as the mutating call:
  ```python
  ack = await physics.inject_fault(...)
  status = await physics.get_simulation_status() if ack.accepted else None
  except grpc.RpcError as exc:
      raise _unavailable(exc) from exc
  ```
  If the injection succeeds and the status read fails or times out (2 s deadline), the engineer gets `503 PhysicsService is unavailable`. The audit row then says `response_status=503` with no outcome, while the fault is active in the plant, and no `scenario_runs` row exists. Separately, every command route maps `DEADLINE_EXCEEDED` to the same 503 with an empty outcome, although the PLC or physics may have applied the command. The audit shows "failed" where the truth is "unknown".
- **Recommendation:**
  1. In `inject_fault` and `_clear`, move `get_simulation_status()` into its own `try`. On `RpcError`, log a `warning` ("fault applied, run not recorded") and skip `_record_runs`, or record it with the fields that are known. Always return `_fault_ack(request, ack)` so the audit outcome reflects the real acknowledgement.
  2. In `upstream_unavailable`, or in the helper proposed in GW-API-12, set the audit outcome for `grpc.StatusCode.DEADLINE_EXCEEDED` to `"unknown: upstream deadline exceeded"` and for other codes to `"failed: <code name>"`. This needs `request`, so pass it into the helper.
  3. Tests in `tests/test_simulation_routes.py`: a fake whose `get_simulation_status` raises `UpstreamDownError` after an accepted `inject_fault` gives a 200 response with `accepted=True` and an audit outcome starting with `accepted`. In `tests/test_plant_commands.py::test_an_unreachable_plc_is_503`, also assert the audit row's status and outcome.
- **Effort:** S

#### GW-API-05 — `python -m api_gateway` on Windows runs a Proactor loop, so the realtime MQTT source cannot work
- **Severity:** Medium
- **Category:** Reliability
- **Confidence:** High
- **Location:** `apps/api-gateway/src/api_gateway/__main__.py:35-43` (compare `apps/plc-controller/src/plc_controller/__main__.py:59`, `apps/historian/src/historian/__main__.py:144`)
- **About the file:** The gateway's entry point: argument parsing and `uvicorn.run`.
- **Problem:** Project rule: "On Windows, entry points pass `loop_factory=asyncio.SelectorEventLoop`: aiomqtt needs `add_reader()`." Every other service does this; the gateway does not. uvicorn 0.41 `loops/asyncio.py` returns `asyncio.ProactorEventLoop` on win32 when there is no reload or workers subprocess. aiomqtt calls `loop.add_reader` (`aiomqtt/client.py:704`), which Proactor does not implement. A host-run gateway on the owner's Windows machine (the first documented way to run it) therefore never delivers `plc/events` or `alarms/changes` to `/ws`. It reconnects every 3 s, and asyncio reports the failing callback. The documented `uvicorn ... --reload` path happens to work only because reload uses a subprocess, which gets a Selector loop.
- **Recommendation:**
  1. In `__main__.py`, pass `loop="asyncio:SelectorEventLoop" if sys.platform == "win32" else "auto"` to `uvicorn.run`. uvicorn's `Config.get_loop_factory` imports a custom `module:attr` string as the loop factory.
  2. Add a unit test that patches `uvicorn.run`, runs the entry point's main path with `sys.platform` patched to `"win32"`, and asserts the `loop` argument. To make this testable, move the body of `if __name__ == "__main__":` into a `main()` function.
- **Effort:** S

#### GW-API-06 — WebSocket transport limits: 16 MiB frames accepted before the 8 KiB check, unauthenticated sockets accepted, no connection cap
- **Severity:** Medium
- **Category:** Security
- **Confidence:** High
- **Location:** `apps/api-gateway/src/api_gateway/routers/websocket.py:55`, `apps/api-gateway/src/api_gateway/routers/websocket.py:101-104`, `apps/api-gateway/src/api_gateway/routers/websocket.py:216`, `apps/api-gateway/src/api_gateway/routers/websocket.py:244`, `apps/api-gateway/src/api_gateway/__main__.py:35-43`, `apps/api-gateway/src/api_gateway/config.py:94-97`
- **About the file:** See GW-API-03. `__main__.py` configures uvicorn.
- **Problem:** (a) `MAX_CLIENT_FRAME_BYTES = 8192` is checked only after `receive_text()` has returned the whole frame (and it counts characters, not bytes). uvicorn's default `ws_max_size` is 16 MiB with `ws_max_queue=32`, and `__main__.py` changes neither, so the server buffers and decodes frames up to 16 MiB, including before authentication. nginx's `client_max_body_size 64k` does not apply to WebSocket frames. (b) `websocket.accept()` runs before authentication, and each unauthenticated socket is held for up to `ws_auth_timeout_s` (5 s) and then writes an audit row (`_audit_refusal`). (c) Authenticated connections have no cap per user or in total. Each `Subscriber` holds up to `ws_send_queue_size=256` telemetry frames (several KiB each) and a guard that queries the database every 30 s. A viewer account can open thousands of sockets.
- **Recommendation:**
  1. In `__main__.py`, pass `ws_max_size=65536` (or a new setting `ws_max_frame_bytes`, default 8192, shared with `MAX_CLIENT_FRAME_BYTES`) and `ws_max_queue=8` to `uvicorn.run`.
  2. In `realtime()`, reject with 1013 when `hub.subscriber_count` reaches a configured cap (`ws_max_connections`, default e.g. 200). This is a hardening limit, not a feature; if the owner wants a per-user cap too, count subscribers by `user.id` in the hub.
  3. Document the pre-auth audit write next to the REST 401 audit-flood risk (the same class of issue belongs to the auth owner). Consider `limit_conn` for `/ws` in `infrastructure/nginx/site.inc` (owner decision; outside this service).
  4. Test: with the cap set to 1, a second authenticated connection closes with 1013.
- **Effort:** M

#### GW-API-07 — Public `/ready` fans out five probes per call; its Influx ping holds a pool thread that `wait_for` cannot cancel, and that pool also runs login hashing
- **Severity:** Medium
- **Category:** Security
- **Confidence:** Medium
- **Location:** `apps/api-gateway/src/api_gateway/routers/health.py:40-50`, `apps/api-gateway/src/api_gateway/readiness.py:42-58`, `apps/api-gateway/src/api_gateway/readiness.py:66-76`, `apps/api-gateway/src/api_gateway/clients.py:81-84`, `apps/api-gateway/src/api_gateway/accounts.py:41-45`, `apps/api-gateway/src/api_gateway/routers/history.py:92`, `apps/api-gateway/src/api_gateway/routers/kpi.py:72`, `infrastructure/nginx/site.inc:15`
- **About the file:** `readiness.py` checks the database, three gRPC upstreams and InfluxDB. `health.py` serves `/health`, `/ready` and `/api/v1/platform`.
- **Problem:** `/ready` has no role and no audit (`UNAUDITED_PATHS`), and nginx proxies it publicly. The project rules name only `/health` as public; `docs/architecture/overview.md:118` documents `/ready` as public, so there is a doc-versus-rule conflict. Each call runs a DB query, three gRPC health calls and `asyncio.to_thread(historian_client.ping)`. `HistorianQueryClient` never sets an Influx timeout, so the library default of 10 s applies. `asyncio.wait_for(..., 2 s)` stops waiting but cannot stop the thread. While InfluxDB is slow, every anonymous `/ready` call therefore occupies a worker of the default executor (`min(32, cpu+4)` threads) for up to 10 s. The same executor runs Argon2 `hash_password`/`verify_password` for sign-in, plus every history and KPI query. An anonymous loop on `/ready`, or a viewer running concurrent 90-day history queries, can starve logins. The compose healthcheck uses `/health`, not `/ready`.
- **Recommendation:**
  1. Add `timeout_ms: int = 5000` to `HistorianQueryConfig` and pass it to `_InfluxDBClient(..., timeout=config.timeout_ms)`, so a stuck query or ping releases its thread.
  2. Run historian calls on a small dedicated `ThreadPoolExecutor` (e.g. 4 workers, created in the lifespan and shut down with the client) through `loop.run_in_executor`, so Influx slowness cannot starve Argon2.
  3. Cache the result of `check_readiness` for about 1-2 s (a module-level timestamp and value guarded by an `asyncio.Lock`). The public probe then costs one fan-out per interval, not one per request.
  4. Owner decision: keep `/ready` public (and correct the rule) or drop it from `site.inc` (it is only used by `smoke`, which could call `/api/v1/platform` with a token).
- **Effort:** M

#### GW-API-08 — The gateway queries InfluxDB with the all-access admin token

> **Part of theme T3** with PLAT-02, HIST-02 and ARCH-06. Fix it once.

- **Severity:** Medium
- **Category:** Security
- **Confidence:** High
- **Location:** `docker-compose.yml:286`, `apps/api-gateway/src/api_gateway/config.py:87`, `apps/api-gateway/src/api_gateway/clients.py:81-84`
- **About the file:** The gateway's historian client runs read-only Flux queries for `/api/v1/history` and `/api/v1/kpi`.
- **Problem:** `INFLUX_TOKEN: ${INFLUXDB_ADMIN_TOKEN}` gives the gateway, the internet-facing edge, the operator token. That token can read every bucket, write, delete buckets and manage users. The gateway needs read on `sensors` and `sensors_1m` only. The project demands a per-service least-privilege PostgreSQL role and MQTT account, but InfluxDB access does not follow that rule. Any future flaw in the Flux builder (see GW-API-13) or an RCE in the gateway would come with full database control.
- **Recommendation:**
  1. Have the historian's setup (which already creates `sensors_1m`) or `dev-secrets` create a read-only token scoped to the two buckets, stored as `INFLUXDB_GATEWAY_READ_TOKEN` in `.env.example` and `.env`.
  2. Set the gateway's `INFLUX_TOKEN` to that token in `docker-compose.yml`. Keep the admin token for the historian and backup only.
  3. Note it in `docs/architecture/overview.md` next to the PostgreSQL role table. (This spans the historian and infrastructure owners; coordinate.)
- **Effort:** M

#### GW-API-09 — An unknown alarm state is shown to operators as CLEARED and acknowledged (fails open)
- **Severity:** Medium
- **Category:** Reliability
- **Confidence:** High (code); Low (likelihood today)
- **Location:** `apps/api-gateway/src/api_gateway/routers/alarms.py:50-51`, `apps/api-gateway/src/api_gateway/routers/alarms.py:75-79`
- **About the file:** `alarms.py` holds the alarm REST routes and the mapping from `AlarmService` protos to REST schemas.
- **Problem:** `_state` maps any unmapped value to `"CLEARED"`: `_STATE_NAMES.get(int(value), "CLEARED")`. That covers `ALARM_STATE_UNSPECIFIED` (0, the proto default) and any state added to the contract later. Such an alarm is then reported as `cleared=True, acknowledged=True`, and consoles and OPC UA clients filter it out of the active list. For an alarm display, an unknown state should never resolve to "nothing to see".
- **Recommendation:**
  1. Map unknown values to the most conservative state, `"ACTIVE_UNACK"`, and log a `warning` with the alarm id and raw value. This is additive: the REST Literal is unchanged.
  2. Keep the explicit `from_state` UNSPECIFIED → `None` handling in `_transition_from_proto`.
  3. Test in `tests/test_alarm_routes.py`: `alarm_from_proto(alarm(state=0))` yields `state == "ACTIVE_UNACK"` and `acknowledged is False`.
- **Effort:** S

#### GW-API-10 — Out-of-range integers from query parameters cause 500 with an ERROR traceback, or a misleading 503
- **Severity:** Low
- **Category:** Security
- **Confidence:** High (confirmed by probe for `offset` and `from_ms`)
- **Location:** `apps/api-gateway/src/api_gateway/routers/alarms.py:144-147`, `apps/api-gateway/src/api_gateway/routers/alarms.py:151-160`, `apps/api-gateway/src/api_gateway/routers/alarms.py:197`, `apps/api-gateway/src/api_gateway/clients.py:250-253`, `apps/api-gateway/src/api_gateway/routers/history.py:73-74`, `apps/api-gateway/src/api_gateway/routers/kpi.py:66-67`, `apps/api-gateway/src/api_gateway/routers/simulation.py:274`
- **About the file:** The read routes that pass user integers on into protobuf messages, Flux or SQL.
- **Problem:** `offset`, `from_ms`, `to_ms` and `alarm_id` have `ge` but no `le`. The proto fields are `int32` (`offset`, `limit`) and `int64` (`from_ms`, `to_ms`, `alarm_id`). `GET /api/v1/alarms/history?offset=3000000000` and `?from_ms=99999999999999999999` raise `ValueError: Value out of range` while the request message is built. The result is HTTP 500 and `Unhandled error on GET /api/v1/alarms/history` at ERROR with a traceback (probe). `pb2.AlarmRef(alarm_id=10**20)` in the real client fails the same way. In `/history` and `/kpi`, a huge `start_ms`/`end_ms` becomes `time(v: <huge>)` in Flux. InfluxDB answers 400, which is reported as `503 Historian is unavailable` with a warning log. Any viewer can write ERROR lines, which breaks the `demo` gate and the error-level rule. An unbounded `offset` on `/simulation/runs` reaches SQL `OFFSET` as-is.
- **Recommendation:**
  1. Add upper bounds: `offset: int = Query(0, ge=0, le=2_147_483_647)` for alarm history (int32) and a sane bound for `/simulation/runs`. For time parameters, add a shared constant `MAX_EPOCH_MS = 4_102_444_800_000` (2100-01-01 UTC) in `history.py` and use `le=MAX_EPOCH_MS` for `start_ms`/`end_ms`/`from_ms`/`to_ms`. Use `alarm_id: int = Path(..., ge=1, le=2**63 - 1)`.
  2. Map `ApiException` with status 400 from Influx to a 422 `request.invalid` instead of 503 (in `history.py`, check `exc.status`).
  3. Tests: parametrized 422 cases for each bound in `test_alarm_routes.py`, `test_api_gateway.py::TestHistoryEndpoint` and `test_simulation_routes.py`.
- **Effort:** S

#### GW-API-11 — An MQTT payload containing `NaN`/`Infinity` tears down the whole realtime MQTT session
- **Severity:** Low
- **Category:** Reliability
- **Confidence:** High (mechanism confirmed by probe); Low (likelihood, since publishers are ACL-restricted services)
- **Location:** `apps/api-gateway/src/api_gateway/realtime/sources.py:90-97`, `apps/api-gateway/src/api_gateway/realtime/sources.py:128-134`, `apps/api-gateway/src/api_gateway/realtime/hub.py:39-40`
- **About the file:** See GW-API-01. `_json_object` is the gate for malformed MQTT payloads.
- **Problem:** `json.loads` accepts the non-standard literals `NaN` and `Infinity` (probe: `{'alarm_id': 7, 'value': nan}`). `hub.publish` then encodes with `allow_nan=False` and raises `ValueError` inside `consume`. `MqttSession.run` treats that as a broker failure: it drops the connection, waits 3 s and reconnects. The client id is random and the session is clean, so everything published in that window is lost, and the log says the broker failed. The publishers encode with default `json.dumps` (`apps/alert-manager/src/alert_manager/payloads.py:158`, `apps/plc-controller/src/plc_controller/events.py:112`), which emits `NaN` if a value ever becomes non-finite.
- **Recommendation:**
  1. In `_json_object`, pass `parse_constant=_reject_constant`, where the helper raises `ValueError`, and add `ValueError` to the `except` tuple. Non-finite payloads are then dropped as malformed, logged at debug like the others.
  2. Add `(b'{"value": NaN}', None)` to the parametrized `test_only_json_objects_are_forwarded`.
- **Effort:** S

#### GW-API-12 — Duplication: seven `app.state` accessors with `type: ignore` and about 25 copies of the gRPC→Problem `try/except`
- **Severity:** Medium
- **Category:** Duplication
- **Confidence:** High
- **Location:** accessors: `apps/api-gateway/src/api_gateway/routers/commands.py:36-38`, `routers/plc.py:17-19`, `routers/simulation.py:48-49`, `routers/status.py:21-23`, `routers/alarms.py:45-47`, `routers/history.py:31-33`, `routers/kpi.py:52-53`; `try/except grpc.RpcError` blocks: `routers/commands.py:66-69,80-83,94-99,110-121,139-142`, `routers/simulation.py:104-107,115-118,124-127,133-136,142-145,153-156,165-168,178-181,198-210,241-249`, `routers/alarms.py:129-134,150-162,178-183,200-203,218-223`, `routers/plc.py:28-31`, `routers/status.py:32-35`; acknowledge response: `routers/alarms.py:185-190` and `routers/alarms.py:225-230`
- **About the file:** The operational routers. Each is a thin forwarder to one upstream.
- **Problem:** Each router re-declares a `request.app.state.<client>` accessor with `# type: ignore[no-any-return]` (`history.py` and `kpi.py` even define two for the same historian client). Each route repeats `try: ... except grpc.RpcError as exc: raise upstream_unavailable("<Service>", exc) from exc`, and alarms has its own variant, `_upstream_error`. Because of this repetition, the audit fix in GW-API-04 and the status-code mapping in GW-API-15 would have to be made about 25 times. The `AcknowledgeResponse(...)` body is written out twice.
- **Recommendation:**
  1. In `dependencies.py` (or a new `api_gateway/upstreams.py`), add typed dependencies: `def get_plc_client(request: Request) -> PLCGatewayClient: return cast(PLCGatewayClient, request.app.state.plc_client)` and `PlcClient = Annotated[PLCGatewayClient, Depends(get_plc_client)]`, plus the same for Physics, Alarm and Historian. Use them as route parameters and delete the seven accessors and their `type: ignore`s.
  2. Add an async context manager in `problems.py`: `@asynccontextmanager async def upstream_call(service: str, *, not_found: ProblemError | None = None)`. It maps `RpcError` to `upstream_unavailable` (and `NOT_FOUND` to `not_found` when given). Routes then read `async with upstream_call("PLCService"): ack = await plc.set_load_demand(...)`.
  3. In `alarms.py`, extract `_acknowledge_response(request, result) -> AcknowledgeResponse` that also calls `_ack_outcome`.
  4. Existing route tests cover the behaviour; add one unit test for `upstream_call` (UNAVAILABLE → 503, NOT_FOUND → the given 404).
- **Effort:** M

#### GW-API-13 — `flux_string` is not a complete Flux escaper, although its docstring says the builder carries the guarantee
- **Severity:** Low
- **Category:** Security
- **Confidence:** Medium
- **Location:** `apps/api-gateway/src/api_gateway/clients.py:337-345`, `apps/api-gateway/tests/test_history_query.py:100-109`
- **About the file:** `clients.py` also builds the Flux queries for history and KPIs.
- **Problem:** `flux_string` is `json.dumps(value, ensure_ascii=False)`. JSON and Flux string syntax differ in two ways. Flux interpolates `${expr}` inside double-quoted strings, and JSON leaves `$` as it is. JSON writes control characters as `\u0001`, an escape Flux does not accept (Flux's escapes are `\n \r \t \\ \" \${` and `\x` hex bytes). Every value reaching the builder today is pattern-validated (measurement pattern, `FIELD_NAME` regex, bucket from the environment), so nothing is exploitable now. But the docstring's promise ("a caller added later cannot forget a check it never had to make") does not hold for `${...}`, and the tests only cover quotes, backslashes and newlines.
- **Recommendation:**
  1. Rewrite `flux_string` as an explicit escaper: replace `\` → `\\`, `"` → `\"`, `$` → `\$`, `\n`/`\r`/`\t` with their escapes, and reject (raise `ValueError`) any other character below U+0020.
  2. Add to `TestNamesCannotLeaveTheirLiteral`: `'a${r._value}b'` must appear as `"a\${r._value}b"`, and `"a\x01"` must raise.
- **Effort:** S

#### GW-API-14 — The history response reports the wrong `window_s` for ranges read from the aggregate bucket
- **Severity:** Low
- **Category:** Reliability
- **Confidence:** High
- **Location:** `apps/api-gateway/src/api_gateway/routers/history.py:90`, `apps/api-gateway/src/api_gateway/routers/history.py:114-119`, `apps/api-gateway/src/api_gateway/clients.py:473-475`
- **About the file:** `history.py` is `GET /api/v1/history`, which picks a window so the range fits into `limit` points.
- **Problem:** The route computes `window_s = history_window_s(start, end, limit)`, passes it to `fetch_history` and returns that same value. `fetch_history` quietly raises it to 60 s when `choose_source` picks `sensors_1m` (`window_s = max(window_s, AGGREGATE_WINDOW_S)`). So a 10-minute range from 8 days ago with `limit=200` is answered as `window_s: 3` while each point is really a 60 s mean. The response contract ("Each point is the mean over a window this long") is then false, and the console's trend axis and tooltip are wrong.
- **Recommendation:**
  1. Have `HistorianQueryClient.fetch_history` return the effective window with the rows, e.g. `tuple[int, list[dict[str, Any]]]` or a small dataclass `HistoryResult(window_s, rows, source)`, and use it in the response. The contract does not change, only the value becomes correct.
  2. Update `FakeHistorianClient.fetch_history` and `RecordedHistorianClient` to match. Add a route test with a fake that reports 60 and assert `window_s == 60`.
- **Effort:** S

#### GW-API-15 — Every gRPC status becomes "503 unavailable", and there is no metric for upstream failures
- **Severity:** Low
- **Category:** Logging
- **Confidence:** High
- **Location:** `apps/api-gateway/src/api_gateway/problems.py:50-58`, `apps/api-gateway/src/api_gateway/routers/alarms.py:102-108`, `apps/api-gateway/src/api_gateway/observability.py:26-39`
- **About the file:** `problems.py` builds Problem Details; `upstream_unavailable` is the single gRPC→HTTP mapping.
- **Problem:** `INVALID_ARGUMENT` (alert-manager aborts with it, `apps/alert-manager/src/alert_manager/grpc_server.py:110`), `DEADLINE_EXCEEDED`, `RESOURCE_EXHAUSTED` and `UNAVAILABLE` all become `503 upstream.unavailable`. A retrying client will then retry a request that can never succeed. Every failed request logs a warning with the full `AioRpcError` string, so during a physics outage each console poll adds a line. No counter by upstream and code exists; `http_requests_total{status="503"}` does not say which service. Detail is correctly not echoed to the client.
- **Recommendation:**
  1. In the helper from GW-API-12, map `INVALID_ARGUMENT` → 422 `request.invalid`, `DEADLINE_EXCEEDED` → 504 `upstream.timeout`, and keep 503 for `UNAVAILABLE` and the rest. This adds a status value to the error contract; name it in the report and in the OpenAPI responses.
  2. Add a Prometheus counter `gateway_upstream_failures_total{service,code}` in `observability.py`, incremented by the helper, and log `exc.code().name` rather than the whole error string.
  3. Tests: a fake raising an `AioRpcError(DEADLINE_EXCEEDED)` gives 504; the counter increments.
- **Effort:** S

#### GW-API-16 — WebSocket closes are not logged or counted; `dropped_frames` is written but never read
- **Severity:** Low
- **Category:** Logging
- **Confidence:** High
- **Location:** `apps/api-gateway/src/api_gateway/routers/websocket.py:262-273`, `apps/api-gateway/src/api_gateway/realtime/hub.py:49`, `apps/api-gateway/src/api_gateway/realtime/hub.py:69`, `apps/api-gateway/src/api_gateway/observability.py:35-39`
- **About the file:** See GW-API-03 and GW-API-02.
- **Problem:** When a session ends with `_CloseConnectionError` (1013 slow client, 4401 token expired, session revoked, account blocked, 4400 bad request), nothing is logged and no metric moves. An operator whose console keeps being disconnected leaves no trace. `Subscriber.dropped_frames` is incremented and never read. The variable `actor` at line 256 exists only for task names.
- **Recommendation:**
  1. After the `asyncio.wait`, log one `info` line per non-1000 close: user, code, reason, `dropped_frames`. Use `warning` for 1013.
  2. Add a counter `gateway_websocket_closes_total{code}` and a counter `gateway_websocket_dropped_frames_total`, incremented in `Subscriber.offer_frame`. Otherwise remove `dropped_frames`.
  3. Test: in `test_a_client_that_cannot_keep_up_is_told_to_reload`, assert a warning record with code 1013.
- **Effort:** S

#### GW-API-17 — The telemetry stream has no deadline and no keepalive: a hung connection stalls telemetry without any outage log
- **Severity:** Low
- **Category:** Reliability
- **Confidence:** Medium
- **Location:** `apps/api-gateway/src/api_gateway/clients.py:95-100`, `apps/api-gateway/src/api_gateway/clients.py:114-124`
- **About the file:** `PhysicsGatewayClient` wraps the `PhysicsService` stub. `stream_system_state` feeds the telemetry channel.
- **Problem:** `StreamSystemState` runs with `timeout=None` on a channel with no keepalive options. If the physics container is frozen or the network drops without a TCP reset, the `async for` waits forever. `run_telemetry` never reaches its `RpcError` branch, so nothing is logged and nothing reconnects. A data watchdog would give false alarms, because a paused simulation legitimately sends nothing.
- **Recommendation:**
  1. Create the physics channel with `options=[("grpc.keepalive_time_ms", 30000), ("grpc.keepalive_timeout_ms", 10000)]`. Allow the pings on the physics server (`grpc.http2.min_ping_interval_without_data_ms`, `grpc.keepalive_permit_without_calls` as needed) in the same change, or the server will answer too-frequent pings with GOAWAY. Coordinate with the physics owner.
  2. Test with an in-process server that stops responding (e.g. a servicer awaiting forever after one yield) and a short keepalive, asserting `RpcError` within a bound. If that is too timing-sensitive, cover it with a configuration assertion instead.
- **Effort:** M

#### GW-API-18 — Free-text request values reach the audit detail unsanitized
- **Severity:** Low
- **Category:** Security
- **Confidence:** High
- **Location:** `apps/api-gateway/src/api_gateway/routers/commands.py:137-138`, `apps/api-gateway/src/api_gateway/schemas/command.py:129-133`, `apps/api-gateway/src/api_gateway/routers/simulation.py:192-196`, `apps/api-gateway/src/api_gateway/schemas/plant.py:187-191`
- **About the file:** The command and simulation routes write a human-readable audit detail.
- **Problem:** `ResetRequest.operator_id` (max 128 characters, no pattern) and `FaultRequest.target` (max 64, no pattern) go straight into `set_audit_detail`, for example `f"stated operator_id={body.operator_id}"`. Newlines and control characters then end up in the audit table and on the console's audit screen. Other stated values in this area are pattern-restricted (`ScenarioRequest.name`). Impact is small (an engineer role is required and the length is capped), but the audit trail should not accept forged-looking lines.
- **Recommendation:**
  1. Add `pattern="^[A-Za-z0-9_.:-]*$"` to `FaultRequest.target` (valve names and sensor ids fit) and `pattern="^[\\w.@-]*$"` to `ResetRequest.operator_id`. This is a tightening of the request contract; check the OPC UA method (`apps/opcua-server/src/opcua_server/methods.py:153` sends `{}`) and the console.
  2. Alternatively, keep the schemas and escape the value with `repr()` in the detail string.
  3. Test: a target containing `"\n"` gives 422.
- **Effort:** S

#### GW-API-19 — Modularity: mapping code in routers, a duplicate status mapper, `clients.py` mixing gRPC and Flux, an empty module
- **Severity:** Low
- **Category:** Modularity
- **Confidence:** High
- **Location:** `apps/api-gateway/src/api_gateway/routers/alarms.py:35-99`, `apps/api-gateway/src/api_gateway/routers/status.py:37-56` (compare `apps/api-gateway/src/api_gateway/plant_state.py:32-33` and `plant_state.py:61-85`), `apps/api-gateway/src/api_gateway/clients.py:276-504`, `apps/api-gateway/src/api_gateway/routers/sensors.py` (0 bytes), `apps/api-gateway/src/api_gateway/routers/kpi.py:28-49`
- **About the file:** Proto→REST mappers live in `plant_state.py` and `plc_state.py`; `clients.py` holds three gRPC wrappers plus the InfluxDB query planner and Flux builder.
- **Problem:** The alarm mapper (`alarm_from_proto`, `_transition_from_proto`, `_STATE_NAMES`) is public code inside a router, unlike the plant and PLC mappers. `status.py` re-implements part of the plant mapping by hand, including `pb2.SensorQuality.Name(...).lower()`, which duplicates `plant_state.quality_name`. `/api/v1/status` is still used by `smoke` and `demo`, so it must stay. `clients.py` (504 lines) mixes two unrelated concerns, gRPC transport and Flux query planning. `routers/sensors.py` is an empty file from 2026-03-16T14:49:07+03:00 that nothing imports. `KpiResponse` is defined in the router, while every other schema lives in `schemas/`.
- **Recommendation:**
  1. Move the alarm mapping into `api_gateway/alarm_state.py`, mirroring `plc_state.py`, and import it in `alarms.py`.
  2. In `status.py`, build the boiler quality with `plant_state.quality_name(...)`, or derive `SystemStatusResponse` from the dicts `plant_state._scalars` produces. The response shape stays the same.
  3. Split `clients.py` into `clients.py` (gRPC wrappers) and `historian_query.py` (`HistorySource`, `history_window_s`, `choose_source`, `flux_string`, `build_*_query`, `HistorianQueryClient`), updating imports in `history.py`, `kpi.py`, `main.py` and the tests.
  4. Delete `routers/sensors.py`. Move `KpiResponse` into `schemas/ops.py` (no contract change; regenerate OpenAPI to confirm it is identical).
- **Effort:** M

#### GW-API-20 — Test gaps in this area
- **Severity:** Low
- **Category:** Tests
- **Confidence:** High
- **Location:** `apps/api-gateway/tests/test_api_gateway.py:541-574`, `apps/api-gateway/tests/test_plant_commands.py:212-226`, `apps/api-gateway/tests/test_realtime_hub.py:135-209`, `apps/api-gateway/tests/test_websocket.py:144-191`, `apps/api-gateway/tests/test_alarm_routes.py:42-59`
- **About the file:** The route, hub and WebSocket tests of the gateway.
- **Problem:** Untested branches found while reviewing:
  - `history.parse_fields`: no route test sends `fields=` at all, so the valid list, the 33-name refusal and `fields=Pressure` (bad pattern → `request.invalid_fields`) are all missing. The history route's own range errors are only covered through `/kpi`.
  - The audit row of a mutating route whose upstream fails: `test_an_unreachable_plc_is_503` checks only the status.
  - `run_telemetry`/`run_plc_status` fed a message that fails conversion (GW-API-01).
  - A binary WebSocket frame, before and after authentication (GW-API-03).
  - Mixed event and telemetry eviction in `Subscriber` (GW-API-02).
  - `alarm_from_proto` with an unknown state (GW-API-09).
  - The effective `window_s` for an aggregate source (GW-API-14).
  - Readiness when a probe exceeds `ready_check_timeout_s` (only "down" via exception is tested).
  - `_renew` with a token of the same user whose session has been revoked.
- **Recommendation:** Add the tests named in each finding above. For readiness, give `FakePLCClient.health` an `await asyncio.sleep(1)`, set `settings.ready_check_timeout_s` to 0.05 through monkeypatch, and assert that component is `down` and the status is `degraded`. For `parse_fields`, add a parametrized test in `TestHistoryEndpoint` for the three cases above.
- **Effort:** M

### 7.3 physics-engine — plant model, runtime, PhysicsService, MQTT telemetry

**Area summary.** The physics engine is in good shape. The plant is deterministic and steppable, every gRPC valve command is checked against [0, 1] again, with NaN and inf rejected (confirmed over gRPC), speed, step counts and fault ramps are bounded, and the MQTT mirror drops stale snapshots instead of queueing them and reconnects through the shared `MqttSession`. The main weakness is in the runtime's concurrency and its failure paths. When a caller cancels a `StepSimulation` or `LoadScenario`, the runtime lock is released while the worker thread is still writing to the plant, which is a confirmed data race on the only copy of process state. If the stepping loop dies, the service is frozen but still reports healthy to Docker, to Prometheus and on the retained MQTT availability topic. Refused commands are neither logged nor counted. Smaller issues: `steam_tables.py` swallows exceptions silently, about 200 lines of offline-only code sit inside the runtime model, some docstrings are stale, and a few tests depend on machine speed or assert something weak.

<details><summary>Scope reviewed by the area auditor</summary>

every module in `apps/physics-engine/src/physics_engine/` (`__main__.py`, `runtime.py`, `server.py`, `mqtt_publisher.py`, `proto_mapping.py`, `metrics.py`, `plant.py`, `boiler.py`, `turbine.py`, `condenser.py`, `heat_exchanger.py`, `combustion.py`, `emissions.py`, `equipment_health.py`, `faults.py`, `scenarios.py`, `sensors.py`, `system.py`, `models.py`, `operating_point.py`, `properties.py`, `steam_tables.py`, `constants.py`), all eleven test files in `apps/physics-engine/tests/`, plus the neighbours they depend on: `shared/runtime/src/cogniboiler_runtime/mqtt.py`, `shared/observability/src/cogniboiler_observability/grpc_observability.py`, the `physics-engine` service in `docker-compose.yml`, the gateway's physics client (`apps/api-gateway/src/api_gateway/clients.py`, `schemas/plant.py`) and the PLC's availability handling (`apps/plc-controller/src/plc_controller/events.py`, `server.py`). Three probe scripts in the scratchpad (`probe_phy_race.py`, `probe_phy_numeric.py`, `probe_phy_stream.py`) confirmed the runtime findings.

</details>

#### PHY-01 — Cancelling a step or scenario load releases the lock while the worker thread still mutates the plant
- **Severity:** Medium
- **Category:** Reliability
- **Confidence:** High (reproduced)
- **Location:** `apps/physics-engine/src/physics_engine/runtime.py:243-246`, `runtime.py:251-253`, `runtime.py:298-302`
- **About the file:** `runtime.py` paces the deterministic `PlantSimulator` against wall time. It serialises every mutation behind one `asyncio.Lock` and runs physics steps in `asyncio.to_thread`.
- **Problem:** The pattern is
  ```python
  async with self._lock:
      snapshot = await asyncio.to_thread(self._timed_step)
  ```
  `asyncio.to_thread` cannot be cancelled: cancelling the awaiting coroutine only abandons the future. When the coroutine is cancelled, `async with` releases the lock at once, but the thread keeps running `PlantSimulator.step`. The next coroutine to take the lock then mutates `_state`, `_controls`, `_step_count`, `_exhaust_pressure`, `SensorBank._held` and `_pending_faults` at the same moment. It may be `apply_command`, `load_scenario`, another `step`, or the run loop. `probe_phy_race.py` confirmed this. After cancelling `runtime.step(5)` mid-step, `lock held after cancel: False` and `worker threads still inside plant.step: 1`. `apply_command` then got the lock immediately, and a second `step` ran **two concurrent `plant.step` workers**.
  Realistic triggers:
  - the gateway's deadlines on these calls (`clients.py:157` gives StepSimulation 60 s, `clients.py:169` gives LoadScenario 30 s). 3600 transient steps took about 12 s on this machine (`probe_phy_numeric.py`: 600 transient steps in 2.0 s), so a slower host or a busy Docker VM can reach the deadline;
  - any client disconnect;
  - `stop()` cancelling `_run_loop` mid-step at shutdown.

  The likely result is a torn state: a step integrates with half-applied scenario controls, or a scenario load is overwritten by a step. The physics engine is the single owner of process state, so nothing downstream can correct it. It can interact with Д17: the documented restart path reloads a scenario on a hot plant.
- **Recommendation:**
  1. Add a private helper `PhysicsRuntime._run_locked(fn, *args)` that starts the worker with `fut = asyncio.ensure_future(asyncio.to_thread(fn, *args))` and awaits `asyncio.shield(fut)`. On `CancelledError`, it waits for `fut` to finish (still inside the lock), publishes the result if appropriate, then re-raises. Use it at lines 245, 252 and 301.
  2. Alternatively, keep a `threading.Lock` inside `PlantSimulator` around `step`, `load_scenario`, `apply_command` and the fault methods, so no two threads can ever touch the plant at once.
  3. Add a test in `test_physics_runtime.py`. Replace `runtime._plant.step` with a wrapper that blocks on a `threading.Event`, cancel `runtime.step(5)`, then assert that `runtime._lock.locked()` stays true until the event is released and that no second worker ever enters. Step simulated time only, no wall-clock sleeps.
- **Effort:** M

#### PHY-02 — A dead stepping loop freezes the plant but every health signal stays green
- **Severity:** Medium
- **Category:** Reliability / Logging
- **Confidence:** High
- **Location:** `apps/physics-engine/src/physics_engine/runtime.py:307-313`, `server.py:65-75`, `metrics.py:33-35`, `docker-compose.yml:162`, `mqtt_publisher.py:242`
- **About the file:** `_run_loop` advances the plant on the wall clock. `Health` returns the runtime status. `metrics.py` exposes gauges. The Compose healthcheck calls `Health`.
- **Problem:** Any exception in a step (for example the `ValueError: cannot convert float NaN to integer` that `turbine.py:287` raises once NaN reaches the state, see PHY-05) sets `_status = "degraded"` and ends the loop for good. Nothing restarts it: `start()` is only called once, from `serve()`. The process keeps running. Consequences:
  - The Compose healthcheck only checks that `Health` answers, not what `status` says, so the container stays `healthy` and `restart: unless-stopped` never fires.
  - `physics_simulation_running` is computed from `run_state` (line 34), so it still reads 1.0.
  - No counter records the failure.
  - The MQTT mirror treats `RuntimeUnavailableError` as fatal and stops publishing (see PHY-03 for the retained topic).
  - A paused runtime still accepts `step()` after the loop died (`runtime.py:235-247` never checks `_status`), so snapshots can resume while `status` still says "degraded".

  The operator sees a plant frozen at its last values with no alarm from the platform.
- **Recommendation:**
  1. In `docker-compose.yml:162`, make the healthcheck assert `…Health(p.Empty(), timeout=3).status == 'running'` (`assert` in the `-c` snippet), so `stack status` and `smoke` see the failure.
  2. In `metrics.py`, add `LOOP_FAILURES = Counter("physics_runtime_loop_failures_total", …)`, increment it in the `except Exception` branch of `_run_loop`, and make `RUNNING` read `runtime.status == "running" and run_state == RUNNING`.
  3. Make `step()` refuse with `RuntimeCommandError("physics runtime degraded: …")` when `_status != "running"`.
  4. Whether a failed loop should end the process, so Compose restarts it, is a behaviour change: owner decision.
  5. Extend `test_a_failing_step_degrades_the_runtime` to assert that the counter moved and that `step()` is refused afterwards.
- **Effort:** S

#### PHY-03 — `status/physics-engine` stays "online" after a clean shutdown or when the mirror stops
- **Severity:** Medium
- **Category:** Reliability / Logging
- **Confidence:** High
- **Location:** `apps/physics-engine/src/physics_engine/mqtt_publisher.py:214`, `mqtt_publisher.py:233-242`; compare `apps/plc-controller/src/plc_controller/events.py:200-204`
- **About the file:** `mqtt_publisher.py` mirrors every runtime snapshot to the broker and publishes a retained availability message.
- **Problem:** Availability relies only on the MQTT will. aiomqtt 2.5.1 disconnects gracefully in `Client.__aexit__` (checked in `.venv/Lib/site-packages/aiomqtt/client.py:798-829`), and a graceful DISCONNECT makes the broker **discard** the will. Two cases leave the retained "online" on the broker although the engine publishes nothing:
  - cancellation at shutdown (`__main__.py:77-80`);
  - the fatal `RuntimeUnavailableError` path (`session.run(..., fatal=...)`).

  The historian subscribes to `status/+` and records the wrong liveness. The PLC publisher already handles this by publishing "offline" explicitly before it stops (`events.py:200-204`).
- **Recommendation:**
  1. In `mirror()`, wrap the loop in `try/finally` and publish `"offline"` (qos 1, retain) with `publish_availability` before the client context exits. Suppress and debug-log an `MqttError` there, because the link may already be gone.
  2. Add a `TestMirror` test with the existing `FakeBroker`: stop the runtime (and, separately, cancel the mirror) and assert that the last publish is `(TOPIC_AVAILABILITY, "offline", 1, True)`.
- **Effort:** S

#### PHY-04 — `steam_tables.py` swallows every IAPWS exception and substitutes a different property without a trace
- **Severity:** Medium
- **Category:** Reliability / Logging
- **Confidence:** High
- **Location:** `apps/physics-engine/src/physics_engine/steam_tables.py:127-135`, `147-154`, `166-172`, `187-194`, `204-210`, `220-227`, `278-285`, `308-315`, `350-357`
- **About the file:** `steam_tables.py` wraps IAPWS-IF97 (the iapws package). The live runtime uses it for turbine expansion (`turbine.py:178, 191, 205`) and for operating-point solving.
- **Problem:** Nine functions use `try: IAPWS97(...) except Exception: pass` and then silently return a different quantity: saturated liquid or vapour at the same pressure, or a constant such as `4186.0` or `2010.0`. That breaks the rule that errors are never silently swallowed, and it hides bad input. `probe_phy_numeric.py` shows that `steam_enthalpy(T=nan, P=1e7)` returns a finite 2.725 MJ/kg instead of failing, so a NaN in the model is laundered into plausible numbers. The fallback in each `except` path is itself unguarded (`IAPWS97(P=..., x=...)` with an unclamped pressure), so a second failure escapes anyway. The fallback did not fire in the realistic runs I probed (cold start, hot start, fuel trip: 0 exceptions), so today it is a latent masking path rather than an active error.
- **Recommendation:**
  1. Extract one helper, `_iapws_or_saturated(prop: str, *, temp_k, pressure_pa, fallback_quality) -> float`, that does the try, falls back, and logs the first fallback per function at `warning` (then `debug`) with the inputs. Add a Prometheus counter `physics_property_fallbacks_total{function}`. Replace the nine copies with it.
  2. Reject non-finite inputs up front in `_to_float` (`if not math.isfinite(v): raise ValueError(...)`), so NaN fails loudly instead of being replaced.
  3. Narrow `except Exception` to the exception types iapws actually raises (confirm with a probe; likely `NotImplementedError`, `ValueError`, `TypeError`, `AttributeError`).
  4. Tests in `test_steam_tables.py`: NaN input raises; a fallback increments the counter and logs once.
- **Effort:** M

#### PHY-05 — Physics-engine gRPC listens on all interfaces without authentication, including when run from the host
- **Severity:** Medium
- **Category:** Security
- **Confidence:** High
- **Location:** `apps/physics-engine/src/physics_engine/server.py:262-263`, `__main__.py:96`
- **About the file:** `server.py` exposes `PhysicsService`, including `ApplyControlCommand`, `LoadScenario` and `InjectFault`, over insecure gRPC.
- **Problem:** `listen_addr = f"[::]:{port}"` is hard-coded. In Compose that is intended, because the port is not published (defence in depth; not critical). The documented host workflow (`uv run --package physics-engine python -m physics_engine`, command reference) binds the same unauthenticated service on every interface of the developer's machine. Anyone on the LAN can then drive the valves around the PLC, load scenarios, and inject faults with any `operator_id` string, which the server logs as the actor (`server.py:40-41`). The same host workflow binds `/metrics` to `127.0.0.1` by default (`__main__.py:104-106`), so the gRPC default is the inconsistent one.
- **Recommendation:**
  1. Add `--grpc-host` to `parse_args` with default `127.0.0.1`, pass it through `serve(runtime, host=..., port=...)`, and set `--grpc-host 0.0.0.0` in the `docker-compose.yml` command. The PLC server has the same pattern and should get the same change (outside this area).
  2. Record in `docs/architecture/service-boundaries.md` that `PhysicsService` trusts its network and that `operator_id` is caller-asserted.
  3. Test: `serve()` bound to `127.0.0.1`, as `test_serve_runs_until_cancelled_and_stops_the_runtime` already does, plus a `parse_args` default assertion in `TestEntryPoint`.
- **Effort:** S

#### PHY-06 — State-changing RPCs are logged inconsistently and refusals are neither logged nor counted
- **Severity:** Medium
- **Category:** Logging
- **Confidence:** High
- **Location:** `apps/physics-engine/src/physics_engine/server.py:117-121`, `155-158`, `171-175`, `194-197`, `216-217`; `runtime.py:213,221,232`; `plant.py:285,292`; `shared/observability/src/cogniboiler_observability/grpc_observability.py:126`
- **About the file:** `server.py` maps runtime refusals to `accepted=False` acknowledgements.
- **Problem:**
  - `StepSimulation` is never logged, even when accepted, and it is the call that advances an engineer-controlled simulation.
  - Every refusal (`ApplyControlCommand` out of range or NaN, speed out of range, unknown scenario, invalid fault) returns `accepted=False` without a log line. The interceptor counts these as `OK`, because they are not gRPC errors, so metrics cannot show them either.
  - A PLC bug that sends NaN valve commands is invisible on the plant side.
  - Accepted actions, by contrast, are logged two or three times: pause and resume in both runtime and server; fault injection in `plant.py:285` (warning) and `server.py:218` (warning, with operator). The runtime lines lack the operator.
- **Recommendation:**
  1. In `server.py`, log every refusal once at `warning`: `"%s refused for %s: %s", rpc, _operator(...), reason`. For `ApplyControlCommand`, limit it to the first refusal per reason per minute, because the PLC calls it at control rate.
  2. Log accepted `StepSimulation` at `info` with steps and operator.
  3. Add a counter `physics_commands_refused_total{rpc}` in `metrics.py`.
  4. Remove the operator-less duplicate logs in `runtime.pause/resume/set_speed` or demote them to `debug`, and keep `plant.py` at `debug` for injections that come through the server.
  5. Tests: `caplog` assertions in `TestPhysicsService.test_refused_requests_carry_their_reason`.
- **Effort:** S

#### PHY-07 — No guard against a non-finite state, and runtime configuration values are not range-checked
- **Severity:** Low
- **Category:** Reliability
- **Confidence:** High (probed)
- **Location:** `apps/physics-engine/src/physics_engine/boiler.py:436-444`, `plant.py:180-181`, `runtime.py:82-83`, `__main__.py:91-92`, `properties.py:83-101`
- **About the file:** `_bounded` clamps the RK4 result to the model domain. `PlantSimulator`, `PhysicsRuntime` and `__main__` validate configuration.
- **Problem:** `_bounded` uses `max`/`min`, which pass NaN straight through. `properties._clamp_temp` and `saturation_temperature` do the same (probe: `_bounded(NaN state)` and `saturation_temperature(nan)` both return NaN). Configuration checks miss several bad values:
  - `step_s <= 0.0` and `speed_factor <= 0` accept NaN. `--step-s nan` crashes later with the obscure `ValueError: cannot convert float NaN to integer`, and `--speed nan` makes `wall_step_s` NaN.
  - `step_s` has no upper bound although `boiler.py:413-416` says RK4 is only stable because the step is small. With `step_s=30` and full fuel, the probe returned a finite but wrong state: furnace gas at ambient (293 K) with the burners at 100 %, drum level 0. The engine published it without complaint.
- **Recommendation:**
  1. Add `_check_finite(state)` in `BoilerModel.step` after integration. Raise a dedicated `PlantDivergedError(ArithmeticError)` naming the non-finite component, so the runtime degrades with a clear message (see PHY-02).
  2. In `PlantConfig.__post_init__`, reject non-finite or non-positive `step_s` and `cooling_water_temp_k`, and in `PhysicsRuntime.__init__` reject non-finite `speed_factor` (`math.isfinite`).
  3. An upper bound on `step_s` is a plant-numerics decision; propose one with its stability reasoning (owner decision) and cover it with a test showing divergence above it.
  4. Tests: `PlantSimulator(PlantConfig(step_s=float("nan")))` raises `ValueError("step_s…")`, and `PhysicsRuntime(PhysicsRuntimeConfig(speed_factor=float("nan")))` raises.
- **Effort:** S

#### PHY-08 — `StreamSystemState` timed mode has no minimum interval and never ends on a failed runtime
- **Severity:** Low
- **Category:** Security / Reliability
- **Confidence:** High
- **Location:** `apps/physics-engine/src/physics_engine/server.py:89-92`; compare `apps/plc-controller/src/plc_controller/server.py:159`
- **About the file:** `StreamSystemState` streams either on every published snapshot (`interval_s == 0`) or on a timer.
- **Problem:** `interval_s` is used unchecked. `1e-9` builds and sends a full `SystemStateMsg` in a tight loop; the probe measured about 1 250 msgs/s from four streams on the service's event loop. `inf` makes the stream hang silently. The PLC's equivalent RPC floors the interval at 0.1 s. The timed branch also keeps serving stale state forever after the runtime degraded or stopped, whereas the event branch aborts with `UNAVAILABLE`. No production caller uses the timed mode (the gateway and PLC pass `0.0`).
- **Recommendation:**
  1. Clamp `interval = min(max(request.interval_s, MIN_STREAM_INTERVAL_S), MAX_STREAM_INTERVAL_S)`, with `MIN_STREAM_INTERVAL_S = 0.1` as in the PLC. Treat non-finite values as refused with `INVALID_ARGUMENT`.
  2. In the timed loop, abort with `UNAVAILABLE` when `self._runtime.status != "running"`, using the same message as the event branch.
  3. Test: a timed stream on a stopped runtime ends with `UNAVAILABLE`, and `interval_s=1e-9` yields no more than N messages over a stepped period.
- **Effort:** S

#### PHY-09 — Unexpected exceptions leak as `UNKNOWN` with type and message; `LoadScenario` catches only `ScenarioError`
- **Severity:** Low
- **Category:** Security / Reliability
- **Confidence:** Medium
- **Location:** `apps/physics-engine/src/physics_engine/server.py:171-175`, `194-197`; `scenarios.py:162-169`, `operating_point.py:216-221, 254-258`
- **About the file:** The servicer maps domain errors to acknowledgements.
- **Problem:** `load_scenario` can raise `OperatingPointError`, a `ValueError` that is not a `ScenarioError`, when an operating point cannot be solved: possible with non-default `BoilerParameters` or `cooling_water_temp_k`. `step()` can raise any model error. Both escape the handler, so grpc-aio returns `UNKNOWN` with `"Unexpected <class …>: …"`, which exposes internals to the caller. The interceptor counts them but logs nothing.
- **Recommendation:**
  1. In `LoadScenario`, catch `(ScenarioError, OperatingPointError)` and return `accepted=False`.
  2. In `StepSimulation` and `LoadScenario`, catch `Exception`, log it with `logger.exception("…failed for %s", operator)`, and `await context.abort(grpc.StatusCode.INTERNAL, "physics step failed")` with a generic message.
  3. Test: monkeypatch `_timed_step` to raise and assert `INTERNAL` with a message that does not contain the exception text.
- **Effort:** S

#### PHY-10 — Fault severity is unchecked for kinds that "do not use it": NaN and inf are accepted and published
- **Severity:** Low
- **Category:** Security
- **Confidence:** High (probed)
- **Location:** `apps/physics-engine/src/physics_engine/faults.py:89-96`, `proto_mapping.py:199-209`
- **About the file:** `FaultSpec.validated()` normalises and bounds fault requests.
- **Problem:** For `valve_stuck` and `sensor_failure`, `severity` skips validation and is stored as sent. `probe_phy_stream.py`: `InjectFault(kind=VALVE_STUCK, severity=nan)` gave `accepted=True, published severity=nan`. The value goes into every `SystemStateMsg` and `PlantStatusMsg`. The historian filters non-finite values; any JSON consumer that does not may break. The gateway's pydantic bounds block this for REST, so only direct gRPC can do it.
- **Recommendation:**
  1. In `validated()`, require `math.isfinite(self.severity)` for every kind. For kinds without a range, normalise severity to `1.0`, or keep it but bound it to [-1, 1] like the gateway schema.
  2. Change `test_severity_is_ignored_where_it_means_nothing` accordingly and add a NaN case to `test_invalid_specifications_are_refused`.
- **Effort:** S

#### PHY-11 — The offline `solve_ivp` path is dead weight in the runtime model; the RK4 path the runtime uses has no direct unit tests
- **Severity:** Low
- **Category:** Modularity / Tests
- **Confidence:** High
- **Location:** `apps/physics-engine/src/physics_engine/boiler.py:58-63, 348-399, 446-534`; `system.py:22-42, 84-150`; `apps/physics-engine/tests/test_boiler.py` (all tests call `model.simulate`)
- **About the file:** `boiler.py` holds the lumped drum model: `balance`, the RK4 `step` and `_bounded`, plus an adaptive `simulate` with alarm events.
- **Problem:** The following are used only by tests (repository-wide grep):
  - `simulate`, `_derivatives`, the five `_event_*` functions, `get_state_at` and `check_result`, about 190 lines, the reason `boiler.py` is 534 lines;
  - `BoilerTurbineSystem.steady_state`, `evaluate_at` and `SystemState`.

  `simulate` sets `terminal`/`direction` attributes on shared function objects on every call, behind six `# type: ignore[attr-defined]`. `_event_water_overflow` reads `constants.DRUM_HEIGHT` through an inline import instead of `self.params.drum_height`. The event thresholds 0.05 m and 0.1 m are magic numbers. Meanwhile the path the live plant actually runs has no direct test in `test_boiler.py`: `BoilerModel.step` (RK4 plus valve update) and `_bounded`.
- **Recommendation:**
  1. Move `simulate`, the events, `check_result`, `get_state_at` and `steady_state` into a new `physics_engine/offline.py` as free functions taking a `BoilerModel`, or delete them if the owner agrees they are unused. Replace the function-attribute mutation with small event objects or `functools.partial` wrappers that carry the attributes.
  2. Add `TestRk4Step` in `test_boiler.py`: a nominal operating point stays put over 600 steps; a closed steam valve raises pressure; `_bounded` clamps out-of-domain values.
- **Effort:** M

#### PHY-12 — Dead code and dead constants across the model
- **Severity:** Low
- **Category:** Readability
- **Confidence:** High (repository-wide grep, production code only)
- **Location:**
  - `constants.py:4-9` (`SPECIFIC_HEAT_WATER`, `SPECIFIC_HEAT_STEAM`, `LATENT_HEAT_VAPORIZATION`, `WATER_DENSITY`), `constants.py:22` (`TEMP_STEAM_MIN`), `constants.py:12,75` together with `models.py:209,220` (`drum_volume`, `BoilerParameters.min_steam_flow`)
  - `combustion.py:28-31` (gas composition fractions), `combustion.py:201-219` (`flue_gas_heat_loss`), `combustion.py:125-131` (unused `fuel_flow` and `air_flow` parameters)
  - `condenser.py:51` (`DEAERATOR_PRESSURE`), `equipment_health.py:65` (`HEALTH_ALARM_THRESHOLD`)
  - `turbine.py:339-352` (`nominal_state`, tests only)
  - `steam_tables.py` functions used only by tests: `water_specific_heat`, `steam_density`, `steam_specific_heat`, `latent_heat`, `saturated_liquid_enthalpy`, `saturated_vapor_enthalpy`
  - `mqtt_publisher.py:76` (`MQTTConfig.interval_s`, commented "publish every 100 ms" but never read), `mqtt_publisher.py:137-161` (`publish_boiler`, `publish_turbine`, tests only)
- **About the file:** These are the design-data and helper modules of the plant.
- **Problem:** Unused constants look like live design data. For example `WATER_DENSITY = 850` and `SPECIFIC_HEAT_STEAM = 2010` suggest the model uses fixed properties when it uses IF97 tables. That misleads anyone tuning the plant, and the unused paths keep test time and module size up. `test_mqtt_publisher.py:273` asserts the dead `interval_s` default.
- **Recommendation:**
  1. Delete the unused constants and parameters, the unused functions, and `MQTTConfig.interval_s` with its assertion. Keep `publish_boiler`/`publish_turbine` only if a test-only need remains; otherwise test `publish_snapshot`.
  2. Remove `HEALTH_ALARM_THRESHOLD`, or document in `EquipmentHealth` which threshold drives `maintenance_alarm`.
  3. Run `uv run pytest apps/physics-engine/tests` and `uv run mypy` afterwards.
- **Effort:** S

#### PHY-13 — Duplicated literals, aliases and construction code
- **Severity:** Low
- **Category:** Duplication
- **Confidence:** High
- **Location:**
  - `steam_tables.py:108, 237, 249, 257`: `max(611.7, min(p, 22.064e6))` four times; `steam_tables.py:125, 145, 164`: `min(temp_k, 623.0)` three times; `steam_tables.py:96`: 273.16 and 647.0
  - specific heat of water 4186: `constants.py:4`, `condenser.py:63`, `steam_tables.py:154`; steam 2010: `constants.py:5`, `steam_tables.py:227`
  - flue-gas Cp aliased three times: `heat_exchanger.py:44`, `combustion.py:40`, `boiler.py:70`
  - `equipment_health.py:257-270` and `277-289`: identical `EquipmentHealth(...)` construction
  - generating-power threshold written twice: `proto_mapping.py:134` (`power_mw > 1.0`) and `proto_mapping.py:145` (`MIN_GENERATING_POWER_W = 1.0e6`)
  - magic numbers: `combustion.py:217` (`423.15`), `condenser.py:210` (`+ 1.0`), `condenser.py:277` (`348.15`), `models.py:251` (`1273.15`)
  - request limits duplicated across services with drift: gateway `SpeedRequest gt=0.0` (`apps/api-gateway/src/api_gateway/schemas/plant.py:172-175`) against physics `SPEED_FACTOR_MIN = 0.1` (`runtime.py:29`); `steps le=3600` against `MAX_STEPS_PER_REQUEST`; `ramp_s le=3600` against `MAX_RAMP_S`
- **About the file:** These are the property and model helpers.
- **Problem:** The same physical value appears under different names in different places, so changing one leaves the others behind. The speed limit has already drifted: the gateway accepts 0.05×, which the physics engine then refuses.
- **Recommendation:**
  1. Add `IF97_P_MIN_PA`, `IF97_P_CRIT_PA`, `IF97_T_MIN_K`, `IF97_T_SAT_MAX_K` and `LIQUID_T_MAX_K` in `steam_tables.py` and a helper `_clamp_saturation_pressure(p)`.
  2. Use `constants.FLUE_GAS_CP` and `constants.SPECIFIC_HEAT_WATER` directly and delete the aliases.
  3. Add `HealthTracker._snapshot()` and call it from both `update` and `current_health`.
  4. Use `MIN_GENERATING_POWER_W` in `emissions_to_proto`.
  5. Give the other magic numbers named constants.
  6. Align the gateway's `SpeedRequest` lower bound with 0.1, and state the limits in comments on the relevant `cogniboiler.proto` fields so both sides point at one written source. Services cannot import each other.
- **Effort:** S

#### PHY-14 — Stale or misleading docstrings and comments
- **Severity:** Low
- **Category:** Readability
- **Confidence:** High
- **Location:**
  - `condenser.py:9-13` and `condenser.py:77-78`: claim the condenser produces a dynamic feedwater temperature that "replaces the fixed TEMP_FEEDWATER constant". It always returns `DEAERATOR_TEMP`, and the plant uses `boiler_params.feedwater_temp` (`plant.py:418`).
  - `condenser.py:112`: `cycle_efficiency_loss_pct` "Used as a KPI on the Grafana efficiency dashboard", but it is not in any protobuf message.
  - `condenser.py:166-173`: an exploratory comment about fouling units.
  - `combustion.py:11-12`: formula subtracts a flue-gas loss that `calculate` does not subtract.
  - `emissions.py:119`: mentions a non-existent "ScenarioRunner".
  - `emissions.py:145-146`: says "adiabatic flame temperature", while the plant passes the flame-zone temperature, as the module comment at lines 45-47 says.
  - `equipment_health.py:180-182`: a historical "BUG FIX" note.
  - `equipment_health.py:147-149`: advises passing drum water temperature, which the plant no longer does.
  - `steam_tables.py:37-62`: lint-history comments ("N802 fix", "cast fixes…", "no type: ignore needed").
  - `steam_tables.py:33`: stale `# noqa: E402`.
  - `turbine.py:343`: says "277.8 kg/s … ~220–260 MW".
  - `test_boiler.py:17-35`: `# type: ignore[misc]` on plain fixtures.
- **About the file:** These are the model docstrings, which new contributors read first.
- **Problem:** The project rules forbid stale comments, and here they misstate the physics (for example feedwater temperature and flue losses). Someone tuning the plant would reason from wrong premises.
- **Recommendation:** Correct or delete each listed comment so it describes the current code. Keep the iapws monkey-patch explanation (`steam_tables.py:19-30`), but record the patch as tracked debt: it replaces a private iapws function globally at import time.
- **Effort:** S

#### PHY-15 — Non-SI units on the wire, and one name used for two different quantities
- **Severity:** Low
- **Category:** Readability
- **Confidence:** High
- **Location:** `shared/proto/cogniboiler.proto:104` (`co2_intensity_kg_per_mwh`), `:117-123` (`turbine_hours`, `pump_hours`, `overall_health_pct`); `proto_mapping.py:131-141, 184-196`; `emissions.py:97-109`
- **About the file:** `proto_mapping.py` is the single mapping from physics types to the contract.
- **Problem:** The domain rules require SI units in contracts (seconds, not hours; no MWh). The unit is named in the field, but the values are kg/MWh, hours and percent. Separately, `EmissionsState.co2_intensity_kg_per_mwh` means kg per MWh **of fuel**, while the proto field `EmissionsMsg.co2_intensity_kg_per_mwh`, filled at `proto_mapping.py:134`, means kg per MWh **of electricity**. Same name, different quantity. `PerformanceMsg.co2_intensity_kg_per_j` already carries the SI version.
- **Recommendation:**
  1. Rename the dataclass property to `co2_intensity_kg_per_mwh_fuel` (internal only, safe).
  2. Changing the contract (adding SI fields such as `turbine_running_s` and deprecating the hour fields without renumbering) needs every consumer updated: owner decision.
- **Effort:** S (rename) / M (contract)

#### PHY-16 — Test gaps, machine-speed-dependent tests and weak assertions
- **Severity:** Low
- **Category:** Tests
- **Confidence:** High
- **Location:** `apps/physics-engine/tests/test_grpc_server.py:21-27, 56-73, 75-82`; missing coverage listed below
- **About the file:** The physics-engine test suite.
- **Problem:**
  - `test_apply_control_command_changes_live_state` runs a free-running runtime at 100× and relies on `asyncio.sleep(0.2)` producing at least one step. That depends on machine speed, which the project rules count as a defect. It also runs at 100×, a speed operators cannot request (`SPEED_FACTOR_MAX = 50`).
  - `test_stream_system_state_yields_real_updates` asserts `timestamp_ms` grows. `now_ms()` is taken at serialisation, so the test would pass if the stream sent the same snapshot twice.
  - Untested functions and cases:
    - `properties.py` has no tests: table against IF97 agreement, clamping at `T_TABLE_MIN_K/MAX_K`, NaN.
    - `heat_exchanger.counterflow_effectiveness`: the `c_min <= 0` and `ratio > 0.999` branches.
    - `combustion.CombustionModel.calculate`: `efficiency_factor` clamp and the rich/lean branches.
    - `operating_point.solve_operating_point`: out-of-range load raises `OperatingPointError`, plus an open-loop hold test (start from the steady-state scenario, step 1 800 times, assert pressure, level and power stay within tolerance), which the module docstring promises at lines 11-12.
    - `proto_mapping.performance_to_proto` and `emissions_to_proto`: the zero-power branches.
    - `proto_mapping.fault_kind_from_proto`: `FAULT_KIND_UNSPECIFIED` is covered via gRPC; an unknown value (99) is not.
    - `server.StreamSystemState` timed mode on a stopped runtime.
    - `ApplyControlCommand` with NaN or inf; `StepSimulation` rejected while degraded.
    - `PhysicsRuntime.step` cancellation (PHY-01).
    - The mirror publishing "offline" (PHY-03).
    - `runtime.load_scenario` failing with `OperatingPointError` (PHY-09).
- **Recommendation:**
  1. Rewrite `TestPhysicsGrpc` on a paused runtime: send the command, then `await runtime.step(5)`, then compare power.
  2. In the stream test, assert `second.simulation.step_count == first.simulation.step_count + 1` after an explicit `runtime.step(1)`.
  3. Add the tests listed above, all driven by `PlantSimulator.step` or `runtime.step`, with no wall-clock sleeps.
- **Effort:** M

#### PHY-17 — "Immutable" snapshots contain mutable dataclasses; stepping can overlap a resume
- **Severity:** Info
- **Category:** Reliability
- **Confidence:** High
- **Location:** `apps/physics-engine/src/physics_engine/plant.py:104-123`; `models.py:26, 81, 121`; `turbine.py:98`; `runtime.py:174-176, 241-246`
- **About the file:** `PlantSnapshot` is handed to every reader: the gRPC streams, the MQTT mirror and Prometheus callbacks running on the metrics thread.
- **Problem:**
  - `PlantSnapshot` is `frozen`, but `BoilerState`, `ControlInputs`/`ValveState` and `TurbineState` are mutable. `get_snapshot` promises "immutable, safe to keep". The plant copies them before building a snapshot, so this is safe today, but one careless reader could change what every other reader sees.
  - `step()` checks for `PAUSED` only once, before the loop. If `ResumeSimulation` arrives during a 3 600-step request, the request keeps stepping while the run loop also steps. Both take the lock, so this is not a race, but the simulation runs faster than the chosen speed.
- **Recommendation:**
  1. Make `BoilerState` and `TurbineState` `frozen=True`. Most code already uses `replace`; check the few in-place writes. Alternatively, document that snapshot members must not be mutated.
  2. In `step()`, re-check `self._run_state is RunState.PAUSED` inside the lock on every iteration and stop early otherwise.
- **Effort:** S

### 7.4 plc-controller — interlocks, E-Stop latch, commands, PLCService

**Area summary.** The PLC is carefully structured. Validation is pure and well tested, including NaN and inf on every operator input. The E-Stop latch is checked under one `asyncio.Lock` together with the scan, so no check-then-act race exists between commands and trips. Callers cannot claim the PID or SAFETY source. The MQTT publisher is bounded and reconnects through the shared session. The weakest area is what the PLC trusts from the plant side and how well its own command state survives disturbances. (1) A latched trip command is never re-sent to a plant that was reloaded or restarted, because `_forward` skips repeats (probe-confirmed). (2) A non-finite measurement passes the interlocks, permanently poisons the PID integrators and drives the fuel valve to 100 % (probe-confirmed). (3) The E-Stop latch exists only in process memory, so a PLC restart comes back in AUTO without an operator reset. The secondary themes are failure handling around the physics call (mode changes that happen even when the forward fails, an E-Stop that latches while the caller is told it failed), a plant link whose loss is invisible, an unauthenticated `PLCService`, dead code left over from the old PID and safety modules, and a thin set of safety-branch tests at unit level.

<details><summary>Scope reviewed by the area auditor</summary>

every module in `apps/plc-controller/src/plc_controller/` (`__init__.py`, `__main__.py`, `alarms.py`, `client.py`, `commands.py`, `control.py`, `events.py`, `measurements.py`, `metrics.py`, `pid.py`, `ramps.py`, `safety.py`, `safety_limits.py`, `server.py`, `service.py`, `status.py`), every test in `apps/plc-controller/tests/` (`plc_harness.py`, `test_pid.py`, `test_plc_behaviour.py`, `test_plc_commands_and_status.py`, `test_plc_publisher.py`, `test_plc_restart.py`, `test_plc_server.py`, `test_safety.py`), plus the context each finding depends on: `shared/proto/cogniboiler.proto` (the PLC messages), `shared/runtime/src/cogniboiler_runtime/mqtt.py`, `shared/observability/src/cogniboiler_observability/grpc_observability.py`, `apps/physics-engine/src/physics_engine/{plant.py,runtime.py,server.py,sensors.py}` (command validation, the state stream, scenario load), `apps/api-gateway/src/api_gateway/{clients.py,routers/commands.py,problems.py}` (how the gateway calls the PLC), and `docker-compose.yml`. Three read-only probe scripts in the scratchpad (`probe_plc_nan.py`, `probe_plc_runchange.py`) exercised the real PLC classes against a fake physics client. No repository file was changed.

</details>

#### PLC-01 — A latched trip command is never re-sent to a reloaded or restarted plant
- **Severity:** High
- **Category:** Safety
- **Confidence:** High (probe-confirmed)
- **Location:** `apps/plc-controller/src/plc_controller/service.py:635-647` (dedupe in `_forward`), `apps/plc-controller/src/plc_controller/service.py:511-530` (`_advance` on a run change does not clear `_last_sent_key`), `apps/plc-controller/src/plc_controller/service.py:489-492` (ESTOP scan path). Plant side: `apps/physics-engine/src/physics_engine/plant.py:234` (`load_scenario` replaces `_controls` with the scenario's valves).
- **About the file:** `service.py` is the PLC's business logic: the scan loop, mode and trip handling, the operator API and forwarding of commands to `PhysicsService`.
- **Problem:** `_forward` skips a command whose rounded key equals the last one sent:
  ```python
  if self._last_sent_key == key:
      self._latest_command = snapshot
      return ValidationResult(accepted=True)
  ```
  The key is never reset. When the plant's valve commands change behind the PLC's back, the PLC keeps believing its last command is in force. This happens when an engineer loads a scenario (`/api/v1/simulation/scenario` → `load_scenario` resets `_controls`) or when the physics container restarts. In ESTOP the trip command is often constant. A high-drum-level trip gives `(0, 0, 0, 0)`. A low-level trip with a failed pump saturates `hold_level` at 1.0. So after the reload the trip command is never sent again. Probe result: after a high-level trip and a new `run_id` with the plant at fuel 0.6, five scans produced **zero** new `ApplyControlCommand` calls. The PLC still reported `mode=estop` with `latest_command` = SAFETY 0/0/0/0 while the plant fired at nominal fuel. After a physics restart, `run_id` can come back with the same value (it counts from 1 again), so `_advance` may not detect the change at all.
- **Recommendation:**
  1. In `_advance`, set `self._last_sent_key = None` inside the `if self._run_id != m.run_id:` branch.
  2. In `_run_control_loop`, set `self._last_sent_key = None` each time a new stream is opened, before the `async for`, so a reconnect (physics restart) always re-sends.
  3. In the ESTOP branch of `process_state`, also clear the key when the plant's reported commands (`measurements.commands`) differ from the trip command by more than the rounding tolerance (1e-4). The plant then holds the trip whatever happened to it. This is reconciliation of an existing command, not new behaviour.
  4. Add a test in `test_plc_behaviour.py`: trip on high level (`initial_state` with `water_level=7.9`), then `runtime.load_scenario(STEADY_STATE)`, then `advance(2)`. Assert `runtime.snapshot` fuel command == 0.0. Add a unit variant with a fake physics client, as in the probe (`probe_plc_runchange.py`).
- **Effort:** S

#### PLC-02 — Non-finite measurements pass every interlock and drive the fuel valve fully open
- **Severity:** High
- **Category:** Safety
- **Confidence:** High for the mechanism (probe-confirmed). Medium for likelihood: it needs the plant to publish NaN or inf with GOOD quality, for example after a numerical blow-up. The physics engine has no finiteness check (`grep isfinite apps/physics-engine/src` finds nothing).
- **Location:** `apps/plc-controller/src/plc_controller/safety_limits.py:94-98` (`ParameterLimits.check`), `apps/plc-controller/src/plc_controller/measurements.py:88-122` (`from_proto` accepts any double), `apps/plc-controller/src/plc_controller/pid.py:197-207` (integrator), `apps/plc-controller/src/plc_controller/control.py:80-81` (`_clamp`), `apps/plc-controller/src/plc_controller/control.py:349` (fuel demand clamp), `apps/plc-controller/src/plc_controller/safety.py:134` (rate), `apps/plc-controller/src/plc_controller/safety.py:58-65` (arming timer), `apps/plc-controller/src/plc_controller/service.py:535` (`dt`).
- **About the file:** `safety_limits.py` holds the protection thresholds, `safety.py` applies them, and `measurements.py` is the boundary from the protobuf state to control and protection. `pid.py` and `control.py` compute the valves.
- **Problem:** Every comparison with NaN is False, so `ParameterLimits.check(nan)` returns `NORMAL`. The rate limiter produces a NaN rate, which also reads as `NORMAL`. `max(lo, min(hi, nan))` returns `hi`. PID `step()` then sets `integral = output - output_pd = nan`, and that stays NaN until the loop is re-primed. Probe results:
  - `SafetyInterlock.check(pressure=nan)` → `safe=True`, nothing latched.
  - `UnitController.scan` with NaN pressure → fuel valve **1.0**, and still 1.0 on the next scan with a finite 140 bar.
  - `PLCService.process_state` with NaN pressure → forwarded `fuel_valve=1.0`, source PID, mode AUTO.
  - A NaN `fuel_flow_kg_s` or `dt` makes `_firing_s` NaN, which permanently disarms the low-furnace-temperature protection.

  Only NaN drum level fails safe, and only by accident: `fuel_permitted(nan)` is False. The alarm monitor already skips non-finite values (`alarms.py:254`), so the operator gets no alarm either. The only backstop is an absolute trip once a finite over-limit value arrives.
- **Recommendation:**
  1. In `ProcessMeasurements.from_proto`, mark any sensor whose value is not `math.isfinite` as `SignalQuality.BAD` in `qualities`. Map each field to its sensor id with the table already in `alarms.ALARMED_VALUES` plus flows and power. This reuses the existing BAD-quality paths: a trip for drum pressure and level, held outputs in `control.py`.
  2. Make the protection fail safe on its own as well. In `ParameterLimits.check`, return `SafetyLevel.TRIP` when `not math.isfinite(value)` and the limit on that side is finite. In `RateOfChangeLimiter.check`, ignore a non-finite value and do not store it as `_prev_value`.
  3. Replace `_clamp` in `control.py` and the inline clamps in `pid.py:112,202,302-305` with one helper, for example `plc_controller/numeric.py::clamp()`, that raises `ValueError` on NaN. In `PIDController.step`, return `prev_output` and leave the state untouched when `setpoint`, `measurement` or `dt` is not finite.
  4. In `_advance`, treat a non-finite `simulation_time_s` as no interval (`return None`).
  5. Tests: `test_safety.py::test_non_finite_pressure_trips`, `test_pid.py::test_nan_measurement_does_not_poison_the_integrator`, and a `process_state` test with NaN pressure asserting a trip and fuel 0.0.
- **Effort:** M

#### PLC-03 — The E-Stop latch does not survive a PLC restart: the unit comes back in AUTO without a reset
- **Severity:** High
- **Category:** Safety
- **Confidence:** High (by reading; not run against Compose)
- **Location:** `apps/plc-controller/src/plc_controller/service.py:101-104` (starts in AUTO, latch clear), `apps/plc-controller/src/plc_controller/service.py:511-530` (first run primes AUTO and seeds the load demand from current power), `docker-compose.yml:22` (`restart: unless-stopped` for every service), `docs/architecture/invariants.md` I3.
- **About the file:** `service.py` owns mode and latch state. It lives only in process memory.
- **Problem:** I3 says resetting a trip is an explicit, audited action by an engineer. The latch, the trip cause and the mode exist only in the PLC process. Suppose the PLC restarts while tripped: a crash, `restart: unless-stopped`, a redeploy of one image, or an out-of-memory kill. It then starts in AUTO with the latch clear. On the first scan the controller primes on the tripped plant. The boiler master then fires toward the 140 bar pressure setpoint, at least `MIN_FIRING_FUEL_KG_S`. If the original cause has cleared, and a manual trip never has a standing cause, nothing re-trips. The plant restarts without anyone resetting it and without an audit entry. Nothing in `SystemStateMsg` (for example a last command source) lets the PLC recognise a tripped plant.
- **Recommendation:** Out of scope as a design choice — owner decision. Options, in increasing cost:
  - (a) Start the PLC in a safe hold that forwards nothing and requires an explicit operator AUTO or reset. This changes behaviour.
  - (b) Persist the latch and trip cause to a small volume file.
  - (c) Have physics report the source of its last applied command so the PLC can start latched.

  Without a fix, at minimum:
  1. Record the gap under I3 "Enforced by" in `docs/architecture/invariants.md`.
  2. Log a `warning` at startup when the first plant state shows fuel command 0 with the turbine off line, so the silent resume is at least visible.
  3. Add a test that pins today's behaviour, so any change is deliberate. Trip, close the `PLCService`, start a new one on the same plant, and assert what mode and fuel command result.
- **Effort:** M for (a) or (b)

#### PLC-04 — An operator command that fails to reach the plant still switches the PLC out of AUTO
- **Severity:** Medium
- **Category:** Reliability
- **Confidence:** High (probe-confirmed)
- **Location:** `apps/plc-controller/src/plc_controller/service.py:328-330`
- **About the file:** `send_command` validates an operator valve command, switches to MANUAL and forwards it.
- **Problem:** The mode changes before the forward:
  ```python
  self._change_mode(RuntimeMode.MANUAL, operator_id)
  self._manual_command = snapshot
  return await self._forward(snapshot)
  ```
  If `apply_command` raises (physics unavailable, `DEADLINE_EXCEEDED`), the exception propagates. The gateway returns 503, but the PLC stays in MANUAL with a MODE_CHANGED event already published. AUTO control stops, and the plant keeps the last AUTO valves with no one driving them. Probe: `send_command` raised `ConnectionError`, then `mode=manual` and `latest_command.fuel=0.5`, the default, not the operator's value. The same happens if physics returns `accepted=False`.
- **Recommendation:**
  1. Forward first, then call `_change_mode(MANUAL, ...)` only on `result.accepted`. Keep everything inside the same `async with self._lock`, so no scan can run in between.
  2. Catch `grpc.aio.AioRpcError` around `_forward` in `send_command` and return `ValidationResult(False, "plant did not acknowledge the command")`. Log it at `warning`. Physics being unreachable is not a PLC bug.
  3. Test with a fake physics client whose `apply_command` raises. Assert `mode is RuntimeMode.AUTO` and `commands_forwarded == 0`.
- **Effort:** S

#### PLC-05 — A manual E-Stop can latch while its caller is told it failed
- **Severity:** Medium
- **Category:** Reliability
- **Confidence:** High for the exception path (probe-confirmed). Medium for the timeout path (from reading).
- **Location:** `apps/plc-controller/src/plc_controller/service.py:336-344` (latch, then `_forward` that may raise), `apps/plc-controller/src/plc_controller/service.py:299`, `335` and `456` (the lock is held across the physics call), `apps/plc-controller/src/plc_controller/client.py:23` (physics timeout 2.0 s), gateway `apps/api-gateway/src/api_gateway/clients.py:34` (PLC timeout 2.0 s).
- **About the file:** `set_mode` handles AUTO, MANUAL and the manual trip.
- **Problem:** The ordering is correct: the latch is set before the trip command is sent. But if the forward raises, `set_mode` raises, gRPC returns UNKNOWN and the gateway answers "PLCService is unavailable", although the E-Stop is latched (probe: `latched=True mode=estop` after the exception). There is a second route to the same result. The lock is held for the whole physics call, and the PLC's deadline to physics (2 s) equals the gateway's deadline to the PLC (2 s). With a slow plant, an E-Stop request queued behind a scan's forward is timed out by the gateway before the PLC answers, and it may latch after the operator has been told it failed. An operator who believes the trip did not happen is the wrong outcome for an E-Stop.
- **Recommendation:**
  1. In `set_mode`'s ESTOP branch, wrap the `_forward` in `try/except grpc.aio.AioRpcError`. Return `ValidationResult(True, "E-Stop latched; the plant has not yet acknowledged the trip command")` and log a `warning`. The next scan re-sends it; together with PLC-01 the re-send is guaranteed.
  2. Keep the PLC's physics deadline clearly shorter than the gateway's. Set `PhysicsClientConfig.timeout_s` to about 1.0 s, or use `min(config.timeout_s, context.time_remaining())` passed down from the servicer.
  3. Test: a fake physics client whose `apply_command` raises. Assert that `set_mode(ESTOP)` returns accepted and that the latch is active.
- **Effort:** S

#### PLC-06 — `PLCService` authenticates no caller; host runs expose it on every interface
- **Severity:** Medium
- **Category:** Security
- **Confidence:** High
- **Location:** `apps/plc-controller/src/plc_controller/server.py:198-199` (`add_insecure_port("[::]:{port}")`), `apps/plc-controller/src/plc_controller/server.py:145-151` (`ResetEmergencyStop` trusts `operator_id`), `apps/plc-controller/src/plc_controller/__main__.py:22-24` (no host option). The same pattern appears at `apps/physics-engine/src/physics_engine/server.py:262-263`, where `ApplyControlCommand` bypasses the PLC entirely.
- **About the file:** `server.py` is the gRPC transport for `PLCService` and the process startup.
- **Problem:** Every mutating RPC (`SendCommand`, `SetControlMode`, `ResetEmergencyStop`, `UpdateSetpoints`, `SetLoadDemand`) trusts whoever can reach the port and whatever `operator_id` they send. The role check for a reset exists only at the gateway (`routers/commands.py`, `EngineerUser`). In Compose the port is unpublished, but every container on the default network reaches `plc-controller:50051`. That includes `opcua-server`, which already holds a `PLCService` stub for reads, and it also reaches `physics-engine:50052`. One compromised internal service can therefore reset trips, or skip the PLC entirely. More immediately, the documented host run (`uv run --package plc-controller python -m plc_controller`) binds `[::]:50051`, which is unauthenticated valve and reset control reachable from the LAN. By contrast, `/metrics` defaults to 127.0.0.1 on the host.
- **Recommendation:**
  1. Add a `--host` argument to `__main__.py`, defaulting to `127.0.0.1`, and pass it to `serve()` in place of the hard-coded `[::]`. Set `--host 0.0.0.0` in the `docker-compose.yml` command. Do the same for physics-engine (and alert-manager `grpc_server.py:182`).
  2. Defence in depth, mechanism to be chosen by the owner: a server interceptor that requires a per-caller shared secret in metadata for the mutating methods. Only the gateway's secret passes; `opcua-server` gets a read-only one. Alternatively mTLS, or a separate Compose network that `opcua-server` and the historian cannot join.
  3. Record the trust assumption, "only the gateway calls the mutating RPCs; enforced by network reachability only", under I2 and I3 in `invariants.md`.
- **Effort:** S (bind) / M (caller auth)

#### PLC-07 — The scan loop treats every exception as a lost stream: bugs are logged once, at `warning`, without a traceback
- **Severity:** Medium
- **Category:** Reliability
- **Confidence:** High
- **Location:** `apps/plc-controller/src/plc_controller/service.py:425-451`
- **About the file:** `_run_control_loop` subscribes to the plant stream and runs one scan per state.
- **Problem:** `except Exception as exc:` covers the stream, `process_state` and `_forward`.
  - A programming error in the scan, for example a `KeyError` in the alarm code or a `ValueError` from a helper, is logged as `"PLC scan stream failed: %s"`. It appears once for the whole outage (`_stream_failing` stays True because no scan ever completes), at `warning` and without `exc_info`. The loop then re-subscribes every 0.2 s forever. The PLC silently stops protecting the plant, and the traceback needed to fix it is never written. This breaks the rule that a platform failure is logged at `error`.
  - A single failed `ApplyControlCommand` also tears down the healthy state stream and re-subscribes, losing states in between.
  - The async generator from `stream_system_state` is left for the garbage collector instead of being closed deterministically.
- **Recommendation:**
  1. Split the handler. `except grpc.aio.AioRpcError` or `ConnectionError` stays the "link" path: `warning`, once per outage. `except Exception` becomes `logger.exception("PLC scan failed on step %d", ...)` at `error`. Log it once per distinct exception type, so a repeating bug does not flood the log.
  2. Catch forward failures inside `process_state`, as in PLC-04 and PLC-05, so a command RPC failure does not drop the stream.
  3. Wrap the stream in `contextlib.aclosing(self._physics.stream_system_state())`.
  4. Test: patch `ProcessMeasurements.from_proto` to raise `RuntimeError`, and assert one ERROR record with `exc_info`.
- **Effort:** S

#### PLC-08 — Losing the plant link is invisible: no keepalive, no staleness signal, no metric, no event
- **Severity:** Medium
- **Category:** Reliability / Logging
- **Confidence:** Medium
- **Location:** `apps/plc-controller/src/plc_controller/client.py:37-39` (channel without keepalive options), `apps/plc-controller/src/plc_controller/client.py:68-77` (stream without a liveness bound), `apps/plc-controller/src/plc_controller/service.py:191-199` (`physics_status` swallows the health error), `apps/plc-controller/src/plc_controller/metrics.py:30-68` (no link metric).
- **About the file:** `client.py` wraps the `PhysicsService` stub. `metrics.py` exports the PLC's counters.
- **Problem:** While the stream is down or hung, no interlock is evaluated, while the physics process keeps integrating with the last valves. Detection is weak:
  - A peer that dies without closing TCP, or a physics event loop that hangs, leaves `async for state in call` blocked indefinitely. HTTP/2 keepalive is not configured, and a paused plant legitimately sends nothing.
  - The PLC keeps reporting mode AUTO. `_task_error` is set only when the stream actually errors.
  - `Health` shows "degraded" only because it makes a fresh call, and `physics_status` hides the reason (`except Exception: return "degraded"`, with no log).
  - No Prometheus series reports "plant link up", reconnect count or forward failures, so Grafana cannot alert on it.

  A plant-side watchdog (a command heartbeat) would be a new feature — owner decision.
- **Recommendation:**
  1. Pass `options=[("grpc.keepalive_time_ms", 10000), ("grpc.keepalive_timeout_ms", 5000), ("grpc.keepalive_permit_without_calls", 1)]` to `insecure_channel` in `PhysicsClient._connected_stub`. The physics server must allow these pings (`grpc.http2.min_ping_interval_without_data_ms`); coordinate with the physics owner.
  2. Export `plc_plant_link_up` (0/1 from `_stream_failing`), `plc_plant_stream_failures_total` and `plc_command_forward_failures_total` from `PlcCollector`. Add counters to `PLCService.stats`.
  3. In `physics_status`, log the exception at `debug`, so the reason can be found.
  4. Test: a PLC pointed at a closed port reports `plc_plant_link_up == 0`, reusing `TestConnection`.
- **Effort:** S–M

#### PLC-09 — Commands are accepted before the first plant state; the fuel permissive fails open
- **Severity:** Medium
- **Category:** Safety
- **Confidence:** High
- **Location:** `apps/plc-controller/src/plc_controller/service.py:304-313`
- **About the file:** `send_command` applies the fuel permissive before forwarding.
- **Problem:** The check is `fuel_valve > 0.0 and measurements is not None and not fuel_permitted(...)`. Before the first scan, or after a restart (see PLC-03), `measurements is None`, so a fuel command is forwarded without the dry-drum permissive. The reset path treats the same situation as a blocker (`_reset_blockers`: "no plant state received yet"), so the two are inconsistent. The window is short, but it is exactly the moment after a restart when the plant state is unknown.
- **Recommendation:**
  1. In `send_command`, refuse with `"no plant state received yet"` when `self._latest_measurements is None`, matching `_reset_blockers`.
  2. Test: `PLCService(enable_control_loop=False)` + `send_command(fuel_valve=0.5, ...)` → not accepted, and the fake physics client receives nothing.
- **Effort:** S

#### PLC-10 — Anonymous resets and commands are accepted, and `operator_id` is normalised in some paths but not others
- **Severity:** Low
- **Category:** Security
- **Confidence:** High
- **Location:** `apps/plc-controller/src/plc_controller/server.py:150` (`request.operator_id or "unknown"`), `apps/plc-controller/src/plc_controller/service.py:285,325,328` (raw `operator_id`, possibly empty), `apps/plc-controller/src/plc_controller/service.py:227,252,334,358`, `apps/plc-controller/src/plc_controller/server.py:79,123` (the `"unknown"` literal repeated).
- **About the file:** The servicer and service carry operator attribution into logs, MQTT `plc/events` and the command sent to physics.
- **Problem:** `CommandSource` 0 is OPERATOR, so an empty `ControlCommandMsg` from any caller is accepted as an operator command with `operator_id=""`. The MODE_CHANGED event then carries an empty operator, while other paths say `"unknown"`. A reset with no `operator_id` is accepted as `"unknown"`. The gateway always sends the username, so this only matters for other callers (see PLC-06). But I3's "attributed" is not checked at the PLC at all.
- **Recommendation:**
  1. Add `require_operator(operator_id: str) -> ValidationResult` in `commands.py`, refusing empty or whitespace-only ids and ids over 64 characters.
  2. Call it in `send_command`, `set_mode`, `reset_emergency_stop`, `update_setpoints` and `set_load_demand`, and delete the `or "unknown"` fallbacks.
  3. Tests for each RPC with an empty `operator_id`. Adjust the existing tests that send commands without an `operator_id` (for example `test_plc_server.py:196-204,264-285`, `test_plc_behaviour.py:142-151`).
- **Effort:** S

#### PLC-11 — Dead code left over from the old PID and safety modules
- **Severity:** Low
- **Category:** Readability
- **Confidence:** High (grep across `apps/`, `shared/`, `scripts/`)
- **Location:**
  - `service.py:121,329,349,351,378,584`: `_manual_command` is written but never read.
  - `service.py:123,458`: `_latest_state` is written but never read.
  - `service.py:393-395`: `get_process_state` has no caller.
  - `pid.py:102-124,170-172,217-328`: `set_manual`, `set_auto`, `is_manual`, the manual branch in `step`, and all of `CascadePIDParameters` and `CascadePIDController` are used only by `test_pid.py`.
  - `safety.py:356-510`: the `SafetyStatus` returned by `check()` is ignored by `process_state`, so `trip_overrides` is computed twice per trip.
  - `safety.py:305-308` `check_count` and `safety.py:185-188` `reset_count`: tests only.
  - `measurements.py:59,72`: `step_s` and `positions` are unused.
  - `alarms.py:290-291` `active_critical`, `ramps.py:31-33` `settled`, `client.py:43-45` `connect`, `safety_limits.py:49` `SafetyAction.NONE`: unused.
- **About the file:** These are the pure control and safety modules plus the service.
- **Problem:** About 200 lines of unexercised code make the safety path harder to review. The module docstring of `pid.py:8` advertises "bumpless transfer between AUTO and MANUAL" through `set_manual`, while the PLC actually does it with `reset(initial_output=...)` in `control.py:217-232`. A reader can easily assume the wrong mechanism. The known-debt note ("PID and safety code still live in physics_engine") no longer holds: both now live in the PLC and no `physics_engine` import remains in `src/`.
- **Recommendation:** Remove the unused state and members listed above, with the tests that only exercise them (`TestCascadePID`, `test_manual_mode_*`). Make `SafetyInterlock.check` return `None` or just the level, or keep `SafetyStatus` but use it in `process_state` instead of re-reading `emergency_stop`. Update the `pid.py` module docstring. Behaviour does not change; run the full PLC suite after.
- **Effort:** S

#### PLC-12 — Duplicated conversions, mappings and constants
- **Severity:** Low
- **Category:** Duplication
- **Confidence:** High
- **Location:**
  - `CommandSnapshot → pb2.ControlCommandMsg` is written three times: `status.py:54-63` (`command_msg`), `server.py:162-170` (StreamCommands) and `service.py:650-658` (`_forward`).
  - `Setpoints → SetpointsMsg` twice: `status.py:45-51` and `server.py:93-99`.
  - The mode mapping three times: `service.py:677-684` (`_mode_to_proto`), `server.py:29-33` (`_MODES`) and `metrics.py:27` (`_MODES` strings).
  - Quality codes twice: `safety_limits.py:241-242` (`QUALITY_UNCERTAIN`, `QUALITY_BAD`) and `measurements.py:18-23` (`SignalQuality`), both mirroring `pb2.SensorQuality`.
  - The clamp idiom four times: `control.py:80-81`, `pid.py:112`, `pid.py:202`, `pid.py:302-305`.
  - Operator id literals `"plc-auto"`, `"safety-interlock"`, `"physics-engine"` and `"unknown"`: `service.py:227,252,334,358,483,506,525,620`.
  - Defaults `"localhost:50052"` and `1883`: `client.py:15`, `server.py:177-179`, `__main__.py:27,31`, `service.py:60-61`.
  - Across services: the valve range check `commands.py:137-154` versus `apps/physics-engine/src/physics_engine/plant.py:272-274`. This duplication is intentional under the "validate twice" rule; keep two implementations but align the message. The sensor ids `measurements.py:26-34` versus `apps/physics-engine/src/physics_engine/sensors.py:25-26`, and the quality enum `measurements.py:18-23` versus `sensors.py:37-41`, are string contracts not declared in the proto.
- **About the file:** `status.py` is designed as the single contract projection. Its own module docstring says a contract change should be "a change to one file".
- **Problem:** A new `ControlCommandMsg` field (as happened with `spray_valve`) has to be added in three places. Missing one silently drops the field in the stream or in forwarding.
- **Recommendation:**
  1. Use `status.command_msg()` in `server.StreamCommands` and `service._forward`, and `status.setpoints_msg()` in `server.GetSetpoints`.
  2. Move `RuntimeMode` into a new `plc_controller/modes.py` with `to_proto(mode)` and `from_proto(value) -> RuntimeMode | None`. Use it in service, server and metrics.
  3. Replace `QUALITY_*` with `SignalQuality.UNCERTAIN` and `SignalQuality.BAD`. Moving `SignalQuality` to `safety_limits.py` avoids an import cycle.
  4. Add `plc_controller/numeric.py::clamp()` (see PLC-02) and module constants `OPERATOR_AUTO`, `OPERATOR_INTERLOCK` and so on.
  5. Sensor ids: a proto enum or documented constants belong to the contract — owner decision (the contract-change skill).
- **Effort:** S

#### PLC-13 — `service.py` is at 684 of ~700 lines and mixes four responsibilities
- **Severity:** Low
- **Category:** Modularity
- **Confidence:** High
- **Location:** `apps/plc-controller/src/plc_controller/service.py` (1-684)
- **About the file:** Lifecycle, the operator API, the scan and command forwarding live in one class.
- **Problem:** The next safety fix (PLC-01, PLC-04, PLC-05) pushes the file past the project limit. The forwarding and dedupe state (`_last_sent_key`, the counters, `_latest_command`) is the part most often involved in bugs, and it is entangled with mode handling.
- **Recommendation:** After removing the dead code (PLC-11, about 30 lines):
  1. Extract `plc_controller/forwarding.py::CommandForwarder`. It owns `_last_sent_key`, `commands_forwarded`, `latest_command`, `forward(snapshot)`, `invalidate()` (used by PLC-01) and the proto conversion through `status.command_msg`.
  2. Extract `modes.py` (PLC-12).
  3. Optionally extract the scan (`process_state`, `_advance`, `_targets`, `_evaluate_alarms`, `_trip_command`) into `scan.py`, leaving `PLCService` as the operator-facing façade holding the lock.
  4. No behaviour change; the existing tests cover it.
- **Effort:** M

#### PLC-14 — Safety branches without a unit test
- **Severity:** Low
- **Category:** Tests
- **Confidence:** High (grep over `apps/plc-controller/tests`: no test mentions `arming`, `steam_temp=`, `sensor_qualities`, `UnitController`, `hold_level`, `constrain`, `trip_overrides` or `"Fuel not permitted"`)
- **Location:** `safety.py:446-452` (steam temperature trip, armed on line), `safety.py:419-427` (low and high arming), `safety.py:467-474` (a BAD drum pressure or level trips; UNCERTAIN warns), `safety_limits.py:245-252` (`trip_overrides`: vent on high pressure, feedwater 0 on high level), `service.py:305-313` (fuel permissive refusal in `send_command`), `service.py:346-347` (MANUAL refused while latched; only AUTO is tested), `service.py:476-485` (interlock trip while in MANUAL), `control.py:314-331` (turbine pressure guard hold and close), `control.py:341-343,361-363,378-384` (BAD-sensor hold paths), `pid.py:139-149` (`constrain` back-calculation, used by every loop), `safety.py:58-66` (`ArmingTracker` proving time and reset).
- **About the file:** `test_safety.py` exercises `SafetyInterlock` only with the four positional values and the default `ALL_ARMED`. `UnitController` is covered only indirectly through the live plant.
- **Problem:** The arming logic and the instrument-quality trip are the two protections most likely to be broken by a refactor. A regression in either (for example `low_side` computed against the wrong limit) would pass the current suite, because the lockstep tests do not reach those states.
- **Recommendation:** Add pure unit tests. Each is a few lines using the existing `measurements()` helper in `test_plc_commands_and_status.py`:
  - `test_high_steam_temp_trips_only_on_line`
  - `test_low_pressure_is_not_armed_off_line`
  - `test_low_flue_gas_needs_proven_flame`
  - `test_bad_drum_pressure_quality_trips` / `test_uncertain_quality_warns`
  - `test_trip_overrides_vent_on_high_pressure_and_stop_feed_on_high_level`
  - `test_arming_tracker_proves_flame_after_10_s_and_resets_on_flame_loss`
  - `test_pressure_guard_closes_turbine_valve_on_large_deficit`
  - `test_bad_level_holds_the_level_trim`
  - `test_constrain_moves_integral_by_the_limited_delta`
  - service-level: `send_command` refused on a low drum (fake physics), `set_mode(MANUAL)` refused while latched, and an interlock trip in MANUAL forwarding a SAFETY command.
- **Effort:** M

#### PLC-15 — `StreamCommands` has no input validation, no bound on concurrent streams and no production caller
- **Severity:** Low
- **Category:** Security
- **Confidence:** High (probe: `asyncio.sleep(nan)` raises `ValueError` on 3.14)
- **Location:** `apps/plc-controller/src/plc_controller/server.py:153-171`
- **About the file:** This is the server-streaming RPC that repeats `latest_command`.
- **Problem:** `interval = max(request.interval_s, 0.1)`:
  - NaN passes through `max`, so `asyncio.sleep(nan)` raises after the first message. The result is an unhandled exception, logged by grpc at `error`, and an UNKNOWN status.
  - `inf` parks a server coroutine forever after one message.
  - The number of concurrent streams is unbounded.

  The only caller is `test_plc_behaviour.py:286`. No gateway, OPC UA or console code uses it.
- **Recommendation:**
  1. Validate: `if not math.isfinite(request.interval_s): await context.abort(grpc.StatusCode.INVALID_ARGUMENT, ...)`, and clamp to `[0.1, 60.0]`.
  2. Build the message with `status.command_msg` (PLC-12).
  3. Removing the RPC is a contract change (I6) — owner decision; if it is kept, add a test for the NaN interval.
- **Effort:** S

#### PLC-16 — Refusals are not logged, the publisher's losses are not exported, and one counter is mislabelled
- **Severity:** Low
- **Category:** Logging
- **Confidence:** High
- **Location:**
  - `service.py:626-628` (`_reject` logs nothing): refused E-Stop-time commands, refused mode changes, and refusals of reserved sources (`service.py:294-297`) leave no line from the service. Only `SendCommand` has an `info` line at `server.py:77-85`. A refused reset (`service.py:365-373`) publishes an event but logs nothing, and `UpdateSetpoints` and `SetControlMode` log nothing at the server either.
  - `service.py:197-198`: the swallowed health error.
  - `events.py:166-169`: `dropped` is never exported, and `MqttSession.failures` is not exported either.
  - `metrics.py:49`: `plc_warnings` "Interlock warnings raised." actually counts scans with any warning (`safety.py:506-507`).
- **About the file:** `metrics.py` is the PLC's Prometheus collector. `events.py` is the MQTT publisher.
- **Problem:** A caller that sends `source=PID` or `SAFETY` is trying to pass for the interlock, yet leaves only an `info` line. A refused reset is invisible in the service log. Lost `plc/events` or alarm messages during a broker outage appear only as an `error` log line, never on the Platform dashboard.
- **Recommendation:**
  1. In `_reject`, log `logger.info("Refused: %s", reason)`. Log the reserved-source refusal at `warning` with the operator_id.
  2. Add `plc_publish_dropped_total` (from `publisher.dropped`) and `plc_mqtt_connected` to `PlcCollector`, exposing `PlcPublisher.connected`.
  3. Rename or re-document `plc_warnings` as "Scans that raised an interlock warning". Renaming a metric affects Grafana, so update the Platform dashboard in the same change.
  4. Add a test asserting the collector reports `plc_publish_dropped_total`.
- **Effort:** S

#### PLC-17 — A failed non-trip instrument silently blinds its protection; missing qualities default to GOOD
- **Severity:** Low
- **Category:** Safety
- **Confidence:** Medium
- **Location:** `apps/plc-controller/src/plc_controller/safety.py:438-452` (the interlock evaluates water temperature, furnace gas and steam temperature whatever their quality), `apps/plc-controller/src/plc_controller/alarms.py:309-321` (the alarm monitor drops BAD readings), `apps/physics-engine/src/physics_engine/sensors.py:7` ("A failed sensor holds its last value and reports BAD"), `apps/plc-controller/src/plc_controller/safety.py:468` and `apps/plc-controller/src/plc_controller/measurements.py:75-77` (a sensor that is not reported counts as GOOD).
- **About the file:** `safety.py` applies the thresholds. `alarms.py` decides what is an alarm.
- **Problem:** When the steam temperature instrument fails, physics holds its last value. The high-steam-temperature trip then compares a frozen number, and the only indication is a quality *warning*. The alarm layer and the interlock treat a BAD reading in opposite ways, and neither policy is written down: the `safety_limits.py` docstring covers only drum pressure and level. And if the plant stops reporting `sensors` altogether (version skew), every instrument reads GOOD, so even the drum-level quality trip cannot fire.
- **Recommendation:** Tripping on a BAD steam-temperature instrument is a policy change — owner decision. Without changing behaviour:
  1. State the policy in the `safety_limits.py` docstring: a BAD non-trip instrument degrades its protection to a warning.
  2. In `SafetyInterlock.check`, treat a TRIP_SENSOR missing from `sensor_qualities` as UNCERTAIN (a warning), not GOOD. This is fail-safe hardening.
  3. Add a test pinning both behaviours.
- **Effort:** S

#### PLC-18 — Stale docstrings, a misleading name and unnamed thresholds
- **Severity:** Low
- **Category:** Readability
- **Confidence:** High
- **Location:**
  - `safety.py:87`: the usage example `RateOfChangeLimiter("pressure_pa", warn=5e5, trip=10e5)` uses keyword names that do not exist (`warn_rate`, `trip_rate`).
  - `safety.py:489`: `worst = trips[0]` is the first trip in evaluation order, not the worst. Only its overrides apply (`trip_overrides`), so the pressure vent wins over a simultaneous high-level feed stop by ordering alone.
  - `service.py:59,96`: `DEFAULT_CONTROL_INTERVAL_S` is only the stream retry delay, since the scan is event-driven.
  - `service.py:612`: `max(dt, 1.0e-3)` duplicates the clamp in `control.hold_level` (`control.py:240`).
  - `alarms.py:204-215`: the alarm deadbands (2 bar, 0.1 m, 2 K, 20 K, 3 K, 1 bar/s) are inline literals with no stated reasoning, outside `safety_limits.py`.
  - `safety_limits.py:235`: `ON_LINE_STEAM_FLOW_KG_S = 24.5`, "10 % of rated", is a literal, while `control.py:53` derives its threshold from `RATED_STEAM_FLOW_KG_S`.
  - `safety.py:351`: `check(dt=1.0)` defaults a safety rate check to a made-up 1 s interval.
- **About the file:** These are the protection modules and the service.
- **Problem:** Each item is small. Together they mislead a reviewer of the safety path: the wrong API in an example, "worst" implying severity ranking, and thresholds with no reasoning attached, where the house rule requires physical reasoning for thresholds.
- **Recommendation:**
  1. Fix the example.
  2. Rename `worst` to `first_trip` and add one comment explaining why evaluation order decides the cause. Combining the overrides of several causes would be a control change — owner decision.
  3. Rename the constant to `STREAM_RETRY_DELAY_S`.
  4. Drop the redundant `max`.
  5. Move the deadbands into named constants in `safety_limits.py` with a one-line reason each, and derive `ON_LINE_STEAM_FLOW_KG_S` from a shared rated-flow constant. A small `plant_design.py` avoids the `safety_limits → control` import.
  6. Make `dt` a required keyword argument in `check`.
- **Effort:** S

### 7.5 historian and alert-manager

**Area summary.** Both services are small, typed and readable. They use the shared `MqttSession`, and the alarm lifecycle is correct and well tested. A missed "cleared" message is recovered by the PLC snapshot, the NaN filtering is deliberate, and Flux literals are escaped for quotes. The weakest points are at the edges. (1) In the historian, a payload that fails with anything other than `DecodeError`/`ValueError` escapes the message handler and tears down the whole MQTT session, and the log calls it an MQTT error. (2) In the alert manager, a database outage longer than about 2 s loses alarm activations for good, because the snapshot only clears alarms and never raises them. (3) `AlarmService` write RPCs accept any caller on the Compose network with an operator name the caller picks. (4) No service handles SIGTERM, so cleanup code never runs in containers. Logging during outages is one line per failed write or message, not one line per outage.

<details><summary>Scope reviewed by the area auditor</summary>

`apps/historian/src/historian/` (`__main__.py`, `points.py`, `storage.py`, `subscriber.py`, `writer.py`, `metrics.py`), `apps/historian/tests/` (all three files); `apps/alert-manager/src/alert_manager/` (`__main__.py`, `db.py`, `grpc_server.py`, `lifecycle.py`, `metrics.py`, `models.py`, `payloads.py`, `processor.py`, `publisher.py`, `subscriber.py`, `views.py`), `apps/alert-manager/tests/` (all files). For context: `shared/runtime/src/cogniboiler_runtime/{mqtt,liveness,clock}.py`, `shared/observability/.../grpc_observability.py`, `apps/api-gateway/migrations/versions/0002_alarm_lifecycle.py` and `0004_application_roles.py`, `apps/api-gateway/src/api_gateway/routers/alarms.py` and `clients.py` (the AlarmService callers), `apps/plc-controller/src/plc_controller/events.py` (the alarm publisher), `infrastructure/docker/mosquitto/{acl,mosquitto.conf}`, `docker-compose.yml`, `Dockerfile`, `docs/architecture/{invariants,service-boundaries}.md`.

</details>

#### HIST-01 — One malformed JSON payload tears down the whole MQTT session (RecursionError / OverflowError escape the handler)
- **Severity:** Medium
- **Category:** Reliability
- **Confidence:** High (confirmed by probe)
- **Location:** `apps/historian/src/historian/subscriber.py:192-196` (only `DecodeError, ValueError` caught), `apps/historian/src/historian/subscriber.py:159-165` (`_json` catches only `UnicodeDecodeError, JSONDecodeError`), `apps/historian/src/historian/points.py:55-59` (`_finite` → `float(value)`), `apps/historian/src/historian/points.py:195`
- **About the file:** `subscriber.py` turns every MQTT message into InfluxDB points and batches them. `points.py` builds points for plant status, alarm changes, PLC events and availability.
- **Problem:** `json.loads(b"[" * 100000)` raises `RecursionError`, and `float(10**400)` raises `OverflowError` (reached by `{"alarm": {"id": 1000…0}}`). Neither is a `ValueError`, so the exception leaves `_handle_message` and `_consume` and is caught by `MqttSession.run` as a broker failure. The session is torn down, the probe logs `"Historian: MQTT error: int too large to convert to float — retrying every 5 s"` (misleading), and the service waits 5 s before reconnecting. During that time all QoS 0 telemetry is lost, any messages already queued in the aiomqtt client are discarded, and the liveness flag goes false. The `skipped` counter is not incremented. Probe output: `alarms/changes ESCAPED RecursionError`, `plc/events ESCAPED RecursionError`, `alarms/changes ESCAPED OverflowError`. Only alert-manager and plc-controller may publish these topics (broker ACL), so this takes a buggy or compromised publisher. Still, one bad message should never cost 5 s of history.
- **Recommendation:**
  1. In `HistorianSubscriber._json`, catch `(UnicodeDecodeError, ValueError, RecursionError)`. `JSONDecodeError` is already a `ValueError`.
  2. In `points._finite`, wrap `float(value)` in `try/except OverflowError: return None`.
  3. In `_handle_message`, add a final `except Exception` around `handler(raw_payload)` that logs a warning with topic and exception type, increments `_skipped` and `MESSAGES_SKIPPED`, and continues. Never let a payload error reach `MqttSession`.
  4. Tests in `test_historian_runtime.py::TestSubscriber`: `sub._handle_message("alarms/changes", b"[" * 100000)` and a payload with a 400-digit `alarm.id`. Both must return normally with `stats["skipped"] == 1`.
- **Effort:** S

#### HIST-02 — The historian authenticates to InfluxDB with the operator (all-access) token

> **Part of theme T3** with PLAT-02, GW-API-08 and ARCH-06. Fix it once.

- **Severity:** Medium
- **Category:** Security
- **Confidence:** High
- **Location:** `docker-compose.yml:198` (`INFLUXDB_TOKEN: ${INFLUXDB_ADMIN_TOKEN}`); also `docker-compose.yml:286` (gateway) and `:117` (Grafana) share it
- **About the file:** Compose wiring for the historian. The token is read in `apps/historian/src/historian/__main__.py:142`.
- **Problem:** The historian needs write access to the raw bucket. Only its one-off storage setup (`storage.apply_policy`) needs bucket and task management. Running it with the operator token means that code execution in the historian container, or a leak of its environment, gives full control of InfluxDB: delete every bucket, read everything, create users and tokens. This goes against the project's per-service-credential principle, which is already applied to PostgreSQL roles and MQTT accounts.
- **Recommendation:**
  1. Have `dev-secrets` (or an init step of the `influxdb` service) create a historian token with write on `sensors` and `sensors_1m`, read on `sensors`, and write on tasks in the org. Pass it as `INFLUXDB_HISTORIAN_TOKEN`.
  2. If splitting setup from ingestion is preferred (setup with the admin token in a one-shot job, ingestion with a write-only token), that is a structural change and needs an owner decision.
  3. Name the gateway and Grafana admin-token use in the same change: the gateway needs only read.
- **Effort:** M

#### HIST-03 — No graceful shutdown: buffered points are dropped and `InfluxWriter.close()` is never called
- **Severity:** Low
- **Category:** Reliability
- **Confidence:** High
- **Location:** `apps/historian/src/historian/__main__.py:97` (`await asyncio.gather(*tasks)` with no `finally`), `apps/historian/src/historian/writer.py:236-238` (`close()` only called in tests), `Dockerfile:78` (python is PID 1, no init, no SIGTERM handler anywhere: `grep add_signal_handler` finds nothing in `apps/` or `shared/`)
- **About the file:** The entry point wires the writer, subscriber, stats loop and storage policy.
- **Problem:** Python installs no SIGTERM handler, and PID 1 in a container ignores signals it has no handler for. `docker stop` / `stack down` therefore waits 10 s and then SIGKILLs the historian. Up to `batch_size - 1` (49) points, about 2 s of data, are lost on every restart, and the InfluxDB client is never flushed or closed. Even on Ctrl+C, cancellation skips `_flush()` because there is no `finally`. The same root cause affects ALM-05 and every other Python service.
- **Recommendation:**
  1. In `main()`, wrap the gather in `try/finally`: `await subscriber.flush()` (make `_flush` public or add `aclose()`), then `await asyncio.to_thread(writer.close)`.
  2. Install a SIGTERM handler once for all services, for example a helper `run_service(main_coro)` in `cogniboiler_runtime` that uses `loop.add_signal_handler(SIGTERM, task.cancel)` on POSIX and also holds the Windows `loop_factory` line (see HIST-08). Alternatively add `init: true` to the Compose `python-service` anchor, but that alone still needs a handler to run the `finally`.
  3. Test: cancel `entry.main` and assert that a `RecordingWriter` got the partial batch.
- **Effort:** S

#### HIST-04 — InfluxDB failures log once per write, not once per outage; storage-policy retries warn every minute
- **Severity:** Low
- **Category:** Logging
- **Confidence:** High
- **Location:** `apps/historian/src/historian/writer.py:216` and `:232`, `apps/historian/src/historian/storage.py:148-153`, `apps/historian/src/historian/__main__.py:66-67`
- **About the file:** `writer.py` wraps the synchronous InfluxDB write API. `storage.py` makes the buckets and the downsampling task match the policy.
- **Problem:** With InfluxDB down or the token wrong, every flush (every 2 s at the stack defaults) and every stats write (30 s) logs a warning with the full `ApiException` text: about 1,800 lines per hour. `ensure_storage` adds one more every 60 s. The rule "network clients log once per failure" is followed by `MqttSession`, but not here. An empty token is only warned about once at start, after which every write fails loudly.
- **Recommendation:**
  1. Give `InfluxWriter` a `_failing: bool` flag like `MqttSession._announce_failure`: warn on the first failure, log at debug while it keeps failing, and log info "InfluxDB writes succeed again (N points lost)" on the first success.
  2. Apply the same pattern in `ensure_storage`.
  3. Extend `test_failures_are_counted_and_logged_not_raised` to three failures that must produce one warning, then a success that must produce one recovery line.
- **Effort:** S

#### HIST-05 — `flux_string` does not escape Flux interpolation `${` and emits JSON `\uXXXX` escapes Flux does not accept

> **Same escaper defect as GW-API-13** (section 7.2) in a second copy of `flux_string`; see theme T8.

- **Severity:** Low
- **Category:** Security
- **Confidence:** High (probe: `flux_string('a${b}') == '"a${b}"'`, `flux_string('a\x01') == '"a\u0001"'`)
- **Location:** `apps/historian/src/historian/storage.py:47-54`, used at `:59`, `:63`, `:65`, `:76`
- **About the file:** Builds and installs the downsampling Flux task.
- **Problem:** The docstring says the function prevents a configured name from changing the task. However, Flux string literals interpolate `${expr}`, and `json.dumps` leaves `${` untouched. A bucket or org value like `x${...}` is therefore evaluated as a Flux expression inside the installed task. Flux accepts only `\n \r \t \\ \" \${` and `\xNN`, so the `\u0001` form that JSON produces for control characters breaks the task. The values come from trusted configuration (argparse / env), so the practical risk is low. The problem is that a security control is claimed but only partly implemented, and the existing test (`test_a_bucket_name_with_a_quote_cannot_rewrite_the_task`) checks quotes only.
- **Recommendation:**
  1. Rewrite `flux_string` as an explicit escaper: `\\` → `\\\\`, `"` → `\"`, `${` → `\${`, `\n`/`\r`/`\t` → their escapes. Reject other control characters with `ValueError`, or better, validate bucket and org names at argument parsing against the InfluxDB naming rules.
  2. Add tests: `downsample_flux` with `raw_bucket='a${r}'` must contain `\${`, and a name with `\x01` must raise.
- **Effort:** S

#### HIST-06 — No plausibility check on message timestamps; one out-of-range timestamp can poison a whole batch
- **Severity:** Low
- **Category:** Reliability
- **Confidence:** Medium (InfluxDB 2 rejecting the whole request on a line with an unparsable timestamp matches the project's own reasoning in `writer.py:114-117`; confirming needs a write against the stack)
- **Location:** `apps/historian/src/historian/points.py:186-188` and `:207` (alarm `at_ms` accepted as any int), `apps/historian/src/historian/points.py:213-216` (PLC event), `apps/historian/src/historian/writer.py:101-102` (`timestamp_ns` unbounded), `apps/historian/src/historian/writer.py:153` and `:164` (boiler/turbine: `timestamp_ms == 0` written at the epoch, while plant status falls back to `now_ms()` at `points.py:63`)
- **About the file:** Point builders.
- **Problem:** `timestamp_ns(at_ms)` can produce a value outside int64 nanoseconds (after 2262 or negative). The client passes integers through unchanged (`influxdb_client ... _convert_timestamp` returns `Integral` as is), so InfluxDB rejects the line. The failure is counted against all 50 points of the batch (`writer.py:230`). A zero timestamp on boiler/turbine lands at 1970, outside retention. Plant status and boiler/turbine treat a missing timestamp inconsistently.
- **Recommendation:**
  1. Add `plausible_timestamp_ms(value) -> int | None` in `writer.py` (for example between 2000-01-01 and now + 1 day, with the bounds as named constants).
  2. Use it in every builder. Return `None` (skip, counted) for JSON events, and use one explicit rule for protobuf messages (either fall back to `now_ms()` like `build_plant_point`, or skip).
  3. Add parametrized tests: `at_ms` of `-1`, `0`, `10**16` must be dropped; boiler with `timestamp_ms=0` must follow the chosen rule.
- **Effort:** S

#### HIST-07 — Ingestion is serialized behind InfluxDB latency, and the aiomqtt incoming queue is unbounded
- **Severity:** Low
- **Category:** Performance
- **Confidence:** Medium
- **Location:** `apps/historian/src/historian/subscriber.py:206-218` (each message awaits `_store_point` → `_flush` → `to_thread(write)` inline in the consume loop), `apps/historian/src/historian/subscriber.py:240-248` (`Client(...)` without `max_queued_incoming_messages`, so aiomqtt 2.5.1 uses an unbounded queue), `apps/historian/src/historian/writer.py:193` (client default timeout 10 s)
- **About the file:** The subscriber consume loop.
- **Problem:** When InfluxDB is slow rather than down (compaction, disk pressure), each flush can block the consume loop for up to 10 s. Messages keep arriving into aiomqtt's unbounded in-memory queue, so memory grows as long as the publish rate exceeds 50 points per write latency. This is not a crash, but it is unbounded growth with no metric. A second, smaller point: `flush_periodically` and `_store_point` can run two `to_thread` writes at once, and `InfluxWriter._written/_errors += n` run in worker threads without a lock.
- **Recommendation:**
  1. Pass `max_queued_incoming_messages` (for example 10,000) to `Client` in `_open_client`, and the same in the alert-manager subscriber. aiomqtt then drops the newest messages and logs, which bounds memory.
  2. Pass an explicit `timeout=` to `InfluxDBClient` in `_new_client`, taken from a CLI option with a named default.
  3. Serialize flushes with an `asyncio.Lock` in `_flush`, or update the counters on the event loop from the result of `to_thread`.
- **Effort:** S

#### HIST-08 — Duplicated plumbing across services: topic constants, JSON-object decoding, protobuf handler pattern, entry-point boilerplate
- **Severity:** Low
- **Category:** Duplication
- **Confidence:** High
- **Location:**
  - Topic constants defined six times: `apps/historian/src/historian/subscriber.py:59-65`, `apps/opcua-server/src/opcua_server/subscriber.py:44-48`, `apps/physics-engine/src/physics_engine/mqtt_publisher.py:58-61`, `apps/plc-controller/src/plc_controller/events.py:41-45`, `apps/api-gateway/src/api_gateway/realtime/sources.py:32-33`, `apps/alert-manager/src/alert_manager/payloads.py:25-27`.
  - JSON-object decode written three times with the same gap as HIST-01: `historian/subscriber.py:159-165`, `api_gateway/realtime/sources.py:89-97`, `alert_manager/payloads.py:75-82`.
  - Protobuf parse and handling duplicated almost line for line: `historian/subscriber.py:141-157,177-207` and `opcua_server/subscriber.py:119-155`.
  - Entry-point boilerplate: `sys.path.insert(... parents[4] / "shared" / "generated")` in six `__main__.py` files (for example `apps/historian/src/historian/__main__.py:23`, `alert_manager/__main__.py:14`), the Windows `loop_factory` line in five (`apps/historian/src/historian/__main__.py:141-145`, `alert_manager/__main__.py:99-105`), and the `--liveness-file/--metrics-port/--metrics-host` arguments plus `MQTT_USERNAME/MQTT_PASSWORD` env reading in each.
- **About the file:** The subscribers and entry points of every MQTT service.
- **Problem:** Topics are a contract (`12-domain-rules.md`), yet six literal copies can drift apart. The JSON gap in HIST-01 exists three times because the helper was copied. Entry-point copies make a cross-cutting fix (SIGTERM, HIST-03) a six-file change.
- **Recommendation:**
  1. Add `shared/runtime/src/cogniboiler_runtime/topics.py` with the topic constants and import it everywhere.
  2. Add `cogniboiler_runtime.payloads.decode_json_object(raw: bytes, *, max_bytes: int) -> dict[str, Any] | None`. It checks size, catches `UnicodeDecodeError, ValueError, RecursionError`, and returns a dict or `None`. Use it in the historian and gateway. Alert-manager wraps `None` into `PayloadError`.
  3. Add `cogniboiler_runtime.entry`: `add_runtime_arguments(parser, default_metrics_port)`, `mqtt_credentials(default_user)`, and `run(main_coro)`, which sets the selector loop on Windows and installs the SIGTERM handler.
  4. Drop the `sys.path.insert` hacks: the container sets `PYTHONPATH` (`Dockerfile:40`). For host runs, rely on a workspace `.pth` or document `PYTHONPATH`.
- **Effort:** M

#### HIST-09 — `stats["stored"]` counts attempted points, not stored ones; small writer and entry-point cohesion issues
- **Severity:** Low
- **Category:** Readability
- **Confidence:** High
- **Location:** `apps/historian/src/historian/subscriber.py:231` (`self._stored += len(batch)` whatever the write outcome), `apps/historian/src/historian/writer.py:206-234` (`write_point` and `write_points` are the same body), `apps/historian/src/historian/subscriber.py:227-230` (special case for one point), `apps/historian/src/historian/__main__.py:70-79` (the stats loop lives in the entry point), `apps/historian/src/historian/writer.py:198-200` (`written` used only by tests)
- **About the file:** Subscriber, writer and entry point.
- **Problem:** The 30-second "Stats: … stored=N" log line and the `historian_stats.stored` field report points as stored even when InfluxDB refused them, so they look healthy during an outage. The duplicated write methods and the `record_stats` business loop in `__main__.py` break the rule that `__main__.py` holds argument parsing and wiring only.
- **Recommendation:**
  1. Rename the counter to `flushed`, or have `write_points` return the accepted count and add only that.
  2. Implement `write_point(p)` as `self.write_points([p])` and remove the special case in `_flush`.
  3. Move `record_stats` into `historian/stats.py` (for example `async def report_stats(subscriber, writer, interval_s)`) and test it there.
- **Effort:** S

#### HIST-10 — Test gaps and weak tests in the historian suite
- **Severity:** Low
- **Category:** Tests
- **Confidence:** High
- **Location:** `apps/historian/tests/test_historian.py:100-388` (one field per test, about 30 near-identical tests; redundant `@pytest.mark.asyncio  # type: ignore[misc]`), `apps/historian/tests/test_historian_runtime.py:521-526` (`sleep(0.05)` then a negative assertion), `test_historian_runtime.py:124-139`
- **About the file:** Historian unit tests.
- **Problem:** Missing cases:
  - `_handle_message` with a deeply nested JSON payload, and with an integer too large for a float (HIST-01);
  - `flux_string` with `${` or a control character (HIST-05);
  - out-of-range `at_ms` / `timestamp_ms` (HIST-06);
  - concurrent `flush_periodically` and `_store_point` (HIST-07);
  - a failing write followed by a successful one, for log-once behaviour (HIST-04).

  `test_an_empty_aggregate_bucket_disables_the_policy` would still pass if the policy were wrongly scheduled on a slow machine, because it only waits 50 ms.
- **Recommendation:**
  1. Merge the field-presence tests into one parametrized test per message.
  2. Add the cases above.
  3. In the aggregate-bucket test, have the fake `Sub.run` set an event and await it instead of sleeping, so the assertion runs after `gather` has started every task.
- **Effort:** S

#### ALM-01 — A database outage longer than about 2 s loses alarm activations permanently
- **Severity:** High
- **Category:** Reliability
- **Confidence:** High
- **Location:** `apps/alert-manager/src/alert_manager/subscriber.py:118-136` (`STORE_ATTEMPTS = 3`, `STORE_RETRY_DELAY_S = 1.0` at `:35-36`), `apps/alert-manager/src/alert_manager/subscriber.py:111-115` (message dropped), `apps/alert-manager/src/alert_manager/processor.py:118-140` (the snapshot only clears), `apps/plc-controller/src/plc_controller/events.py:88-118` (PLC sends condition details only on transitions; the snapshot carries only keys)
- **About the file:** `subscriber.py` consumes `alerts/#` and hands conditions to the processor with retries. `processor.py` owns the lifecycle.
- **Problem:** Paho acknowledges QoS 1 messages as soon as they are queued, so redelivery cannot save a dropped message. If PostgreSQL is unreachable for more than the three attempts (restart, failover, pool exhaustion), an `alerts/critical` "active" message is counted in `alarm_messages_failed_total`, logged, and discarded. When the database returns, the PLC snapshot still lists the key every 10 s, but `handle_snapshot` only schedules clears for rows it finds and ignores keys it has no alarm for. The critical alarm never appears in the console, `AlarmService` or history until the condition clears and comes back. A lost clear is repaired by the snapshot; a lost activation is not. Plant safety still holds (interlocks are in the PLC), but operator alarm awareness does not. In addition:
  - during the outage every message produces two warnings and one ERROR with a full traceback, not one line per outage;
  - non-transient errors (`IntegrityError`, `DataError`, for example ALM-04's out-of-range timestamp) are retried as if transient, stalling the only consumer for 2 s each.
- **Recommendation:**
  1. In `_with_retries`, retry only transient errors (`OperationalError`, `InterfaceError`, or `DBAPIError` with `connection_invalidated`). Retry them with capped backoff (1 s up to 30 s) until they succeed, keeping the message and blocking the consumer. Order is preserved and the broker session holds the rest. Fail fast on other `SQLAlchemyError`s.
  2. Log the first failure as a warning, repeats at debug, and one info line on recovery (reuse the `_failing` pattern from `MqttSession`).
  3. Bound the aiomqtt incoming queue (`max_queued_incoming_messages`) so a long outage cannot grow memory without limit.
  4. Observability without new logic: in `handle_snapshot`, count and warn once per key when `report.active_keys` contains a key with no open alarm (a new counter `alarm_snapshot_unmatched_keys_total`). This makes a lost activation visible.
  5. Raising a missing alarm from the snapshot would be new business logic and needs richer snapshot content: out of scope, owner decision.
  6. Tests: a `RecordingHandler` that fails `OperationalError` five times must still end with one condition stored; `IntegrityError` must not be retried; a snapshot with an unmatched key must log once.
- **Effort:** M

#### ALM-02 — `AlarmService` write RPCs trust any caller on the network and a caller-chosen `operator_id`
- **Severity:** Medium
- **Category:** Security
- **Confidence:** High
- **Location:** `apps/alert-manager/src/alert_manager/grpc_server.py:178-185` (`add_insecure_port("[::]:…")`, no auth interceptor), `apps/alert-manager/src/alert_manager/grpc_server.py:142-175` (`AcknowledgeAlarm` / `AcknowledgeAll`), `apps/alert-manager/src/alert_manager/processor.py:148` and `:165` (`operator_id.strip()[:128] or "unknown"`), `docs/architecture/invariants.md:127-141` (I11, "enforced by" `opcua-server` not holding a command client)
- **About the file:** The gRPC bridge from the gateway to the alarm processor.
- **Problem:** Any container on the Compose network can call `AcknowledgeAll(operator_id="admin")` and silence every open alarm, with no RBAC check and no gateway audit row. That includes `opcua-server`, which is published on host port 4840 with anonymous read (Д14). The transition row then attributes the action to a user who did not do it. I11 is enforced only by the convention that `opcua-server` does not call these RPCs. `04-architecture-boundaries.md` says such a boundary is "not enforced". An empty `operator_id` is recorded as `"unknown"` instead of refused. `PLCService` has the same shape (another area).
- **Recommendation:**
  1. Refuse an empty `operator_id` in both acknowledge RPCs with `INVALID_ARGUMENT`, and log `context.peer()` with each acknowledgement.
  2. Add a server interceptor in `cogniboiler_observability` or `cogniboiler_runtime` that requires a shared secret in metadata (for example `x-cogniboiler-caller-token`, from `.env` via `dev-secrets`) on write methods. Only the gateway gets that secret, not `opcua-server`. Update the gateway client in the same task. mTLS would be heavier and is an owner decision.
  3. Update I11 "Enforced by" to name the gate.
  4. Tests: an ack without the token gets `PERMISSION_DENIED`, an ack with it is accepted, an empty operator gets `INVALID_ARGUMENT`.
- **Effort:** M

#### ALM-03 — No database timeouts; a hung database blocks the processor lock and the intake while liveness stays green
- **Severity:** Medium
- **Category:** Reliability
- **Confidence:** Medium
- **Location:** `apps/alert-manager/src/alert_manager/db.py:34-37` (`create_async_engine(..., pool_pre_ping=True)` with no `connect_args`), `apps/alert-manager/src/alert_manager/processor.py:150`, `:167`, `:246`, `:316` (`self._lock` held across DB I/O), `apps/alert-manager/src/alert_manager/__main__.py:86` (liveness follows only `subscriber.connected`)
- **About the file:** Engine and session factory.
- **Problem:** asyncpg has no command timeout by default. A database that accepts connections but stops answering (lock wait, network partition without RST) blocks the awaiting coroutine forever while it holds `_lock`. Every acknowledgement, clear and activation then queues behind it, and the MQTT consumer stops. The liveness file is refreshed because the broker session is still up, so the healthcheck passes while the service does nothing.
- **Recommendation:**
  1. `create_async_engine(url, pool_pre_ping=True, hide_parameters=True, pool_timeout=10, connect_args={"timeout": 5, "command_timeout": 10})`, with the numbers as named constants in `db.py`. Optionally also set `server_settings={"statement_timeout": "10000"}`.
  2. Feed the liveness callback with "connected and the last message finished processing less than N s ago": expose a `last_progress_monotonic` on `AlertSubscriber`.
  3. Test `create_engine` passes the options (inspect `engine.dialect` / `engine.pool`).
- **Effort:** S

#### ALM-04 — Payload validation gaps: huge integers escape as OverflowError, deep JSON as RecursionError, timestamps unchecked
- **Severity:** Medium
- **Category:** Security
- **Confidence:** High (probe: `huge int value OTHER OverflowError`, `deep json OTHER RecursionError`, `ts 1e300 ACCEPTED ts=100000000000000005250476025520`, `ts -5 ACCEPTED`)
- **Location:** `apps/alert-manager/src/alert_manager/payloads.py:78` (only `UnicodeDecodeError, JSONDecodeError`), `apps/alert-manager/src/alert_manager/payloads.py:94-101` (`float(value)` unguarded), `apps/alert-manager/src/alert_manager/payloads.py:136` and `:152` (`int(_number(...))` with no range), `apps/alert-manager/src/alert_manager/subscriber.py:111-115` (non-`PayloadError` handled as a platform failure), `apps/alert-manager/src/alert_manager/processor.py:126` (snapshot filter `raised_at_ms <= report.timestamp_ms`)
- **About the file:** `payloads.py` decodes `alerts/*` messages into condition and snapshot reports.
- **Problem:** Malformed input that is not reported as `PayloadError` is counted in `alarm_messages_failed_total`, which is documented as "could not be stored after retries", and logged at ERROR with a traceback. That sends a payload problem to the platform-failure channel and makes the `demo` run fail. A `timestamp_ms` of `1e300` passes validation, fails at the PostgreSQL `BIGINT` insert, and is retried three times (ALM-01). A future `raised_at_ms` (for example `1e13`) opens an alarm that the snapshot filter will never clear. `active_keys` has no length limit, and Mosquitto sets no `message_size_limit` (default 256 MB), so `json.loads` of a huge payload runs on the event loop.
- **Recommendation:**
  1. In `_decode`, catch `(UnicodeDecodeError, ValueError, RecursionError)` and reject payloads over a named `MAX_PAYLOAD_BYTES` (for example 64 KiB) before decoding (or use the shared helper from HIST-08).
  2. In `_number`, catch `OverflowError` → `PayloadError`.
  3. Add `_timestamp(payload, field)`: an integral number between 0 and `now_ms() + 1 day` (named constant), else `PayloadError`.
  4. Cap `active_keys` (for example 1,000 entries, each ≤ 200 characters to match `key String(200)`).
  5. Parametrized tests in `test_alarm_messages.py` for each case.
- **Effort:** S

#### ALM-05 — Shutdown cleanup never runs in containers; committed alarm changes can be lost unpublished
- **Severity:** Medium
- **Category:** Reliability
- **Confidence:** High
- **Location:** `apps/alert-manager/src/alert_manager/__main__.py:87-93` (`finally` cleanup), `apps/alert-manager/src/alert_manager/publisher.py:71-80` (`aclose` cancels without draining), `Dockerfile:84` (python as PID 1, no SIGTERM handler)
- **About the file:** Entry point and the `alarms/changes` publisher.
- **Problem:** Same root cause as HIST-03: SIGTERM is ignored and SIGKILL follows after 10 s, so `server.stop(grace=5)`, `processor.close()` and `publisher.aclose()` never run. In-flight acknowledgements are cut off. Changes already committed to PostgreSQL but still in the in-memory publisher queue are never published, so the historian's `alarm_changes` history gets a permanent gap (the gateway re-reads `AlarmService` and recovers). Even on SIGINT, `aclose()` cancels without the bounded drain that the PLC publisher has (`plc_controller/events.py:199-213`, `CLOSE_DRAIN_TIMEOUT_S`).
- **Recommendation:**
  1. Use the shared SIGTERM-aware runner from HIST-03 / HIST-08.
  2. Give `AlarmChangePublisher.aclose()` the same bounded drain as the PLC publisher, ideally by extracting the shared outbox (ALM-06).
  3. Test: with a connected fake broker, `alarm_changed` followed at once by `aclose()` publishes the change.
- **Effort:** S

#### ALM-06 — The queue-backed MQTT publisher is duplicated between alert-manager and plc-controller, and the alert-manager copy has no drop metric
- **Severity:** Low
- **Category:** Duplication
- **Confidence:** High
- **Location:** `apps/alert-manager/src/alert_manager/publisher.py:24-105` versus `apps/plc-controller/src/plc_controller/events.py:47-50,134-262` (same `QUEUE_LIMIT = 1000`, the same drop-oldest code, the same "queue full: %d … dropped so far" error at 1 and every 100, the same `start`/`aclose`/wakeup-event drain loop)
- **About the file:** The publisher of `alarms/changes`.
- **Problem:** The two copies have already drifted: only the PLC copy has a close drain and a `dropped` property. Dropped alarm changes are visible only as log lines; there is no Prometheus counter, so the Platform dashboard cannot show them.
- **Recommendation:**
  1. Extract `class MqttOutbox` into `shared/runtime/src/cogniboiler_runtime/mqtt.py`: a bounded deque of `(topic, payload, qos, retain)`, `put()`, `run(session, on_connect=None)`, `aclose(drain_timeout_s)`, and a `dropped` property.
  2. Rebuild `AlarmChangePublisher` and `PlcPublisher` on it. The PLC keeps its snapshot timer as an `on_idle` hook.
  3. Add `alarm_changes_dropped_total` in `alert_manager/metrics.py`, or a labelled `mqtt_outbox_dropped_total{topic}` in `cogniboiler_observability`.
  4. Move the existing publisher tests to `shared/runtime/tests/test_mqtt_outbox.py`, and add the missing case of a publish failing mid-drain (the message stays at the head and is resent first).
- **Effort:** M

#### ALM-07 — Out-of-range `alarm_id` turns into an unhandled database error (gateway answers "upstream unavailable")
- **Severity:** Low
- **Category:** Reliability
- **Confidence:** Medium (asyncpg refuses an int4 parameter above 2^31-1 with `DataError`; not reproducible on the SQLite test database)
- **Location:** `apps/alert-manager/src/alert_manager/grpc_server.py:129-158` (no range check), `apps/alert-manager/src/alert_manager/processor.py:151` and `:229` (`session.get(AlarmEvent, alarm_id)`), `apps/api-gateway/migrations/versions/0002_alarm_lifecycle.py` (`id` is `Integer`, while `AlarmRef.alarm_id` is `int64`), `apps/api-gateway/src/api_gateway/routers/alarms.py:197` and `:215` (`ge=1`, no upper bound)
- **About the file:** The AlarmService servicer.
- **Problem:** A viewer can request `GET /api/v1/alarms/2147483648`. The alert manager raises a `DBAPIError`, gRPC returns `UNKNOWN` and logs an error with a traceback, and the gateway maps it to "AlarmService unavailable" instead of 404. The same happens on `/ack` for operators. Any authenticated user can generate ERROR lines on demand, which also fails `demo`.
- **Recommendation:**
  1. In `GetAlarm` and `AcknowledgeAlarm`, return `NOT_FOUND` when `not 1 <= request.alarm_id <= 2**31 - 1` (a named constant `MAX_ALARM_ID`).
  2. Optionally add `le=2**31-1` in the gateway path parameters. That is gateway-owned; coordinate it.
  3. Test: `GetAlarm(alarm_id=2**40)` gets `NOT_FOUND`.
- **Effort:** S

#### ALM-08 — The alarm role may DELETE and UPDATE its audit-like transition history
- **Severity:** Low
- **Category:** Security
- **Confidence:** High
- **Location:** `apps/api-gateway/migrations/versions/0004_application_roles.py:73` (`GRANT SELECT, INSERT, UPDATE, DELETE ON alarm_events, alarm_transitions TO cogniboiler_alarms`), `apps/api-gateway/migrations/versions/0002_alarm_lifecycle.py:141` (`ON DELETE CASCADE`); `grep delete apps/alert-manager/src` finds no deletes or updates of transitions
- **About the file:** Database roles migration.
- **Problem:** `alarm_transitions` records who acknowledged what and when. The service only ever inserts into it and updates `alarm_events`, yet its role may rewrite or delete both tables. With the cascade, deleting an alarm also erases its trail. A compromised alert-manager could remove evidence of an acknowledgement. `audit_log` is protected append-only; the alarm trail is not.
- **Recommendation:**
  1. Add a new Alembic revision `0005` (never edit `0004`): `REVOKE UPDATE, DELETE ON alarm_transitions FROM cogniboiler_alarms; REVOKE DELETE ON alarm_events FROM cogniboiler_alarms`, with a downgrade that re-grants.
  2. Record the least-privilege intent in the migration docstring and in `service-boundaries.md`.
  3. Verify on a clean and on an existing database per `06-quality-and-testing.md`.
- **Effort:** S

#### ALM-09 — Unused in-memory stats, no metric for rejected payloads, and the retry/loop helpers sit outside shared code
- **Severity:** Low
- **Category:** Logging
- **Confidence:** High
- **Location:** `apps/alert-manager/src/alert_manager/subscriber.py:64-67,80-87` (`stats` read only by tests; nothing logs or exports it), `apps/alert-manager/src/alert_manager/subscriber.py:107-110` (rejected payload: warning per message, no counter), `apps/alert-manager/src/alert_manager/subscriber.py:163-166` (non-bytes payload skipped silently), `apps/alert-manager/src/alert_manager/subscriber.py:154` (direct `client.subscribe` instead of the shared `subscribe_all`), `apps/alert-manager/src/alert_manager/metrics.py:12-15`
- **About the file:** MQTT intake.
- **Problem:** A publisher sending bad payloads shows up only as a warning per message, with no counter to alert on and no rate limit. The `received/processed/skipped/failed` counters are production dead code. The historian logs and stores its equivalents every 30 s.
- **Recommendation:**
  1. Add `alarm_messages_rejected_total{reason}` and increment it in the `PayloadError` branch and the non-bytes branch.
  2. Either delete `stats` and the counters (update the tests to assert on metrics), or log them periodically like the historian.
  3. Use `subscribe_all(client, ((SUBSCRIBE_TOPIC, 1),))` for consistency.
- **Effort:** S

#### ALM-10 — Alert-manager test gaps and timing-based negative assertions
- **Severity:** Low
- **Category:** Tests
- **Confidence:** High
- **Location:** `apps/alert-manager/tests/test_alarm_processor.py:155-160`, `:306-313`, `:315-321` (`await asyncio.sleep(0.1)` then assert nothing happened), `apps/alert-manager/tests/test_alarm_runtime.py:218-219` (`FakeBroker.publish` never fails), `apps/alert-manager/tests/test_alarm_messages.py:115-123` (bounds not checked for `key`, `source_service`, `direction`, `action`)
- **About the file:** Alert-manager unit tests.
- **Problem:** The negative snapshot and clear tests would still pass on a slow machine if a clear were wrongly scheduled but had not run within 100 ms. This is the machine-speed dependence the rules forbid. Missing cases:
  - a publish failing mid-drain keeps the message and resends it first, as `publisher.py:95-97` promises;
  - `_with_retries` with `IntegrityError` (should not retry) and a long `OperationalError` streak (ALM-01);
  - a snapshot arriving while a clear for the same key is pending (`processor.py:131`);
  - a listener that raises inside `_notify` after commit;
  - the payload edge cases of ALM-04;
  - an out-of-range `alarm_id` (ALM-07).
- **Recommendation:**
  1. Replace the sleeps with a deterministic check: assert `key not in processor._pending_clears` right after `handle_snapshot` / `handle_condition`, or add a test-only `await processor.drain()` that gathers pending clear tasks.
  2. Add the cases listed. For the mid-drain failure, give `FakeBroker.publish` a `fail_next` flag that raises `MqttError`.
- **Effort:** S

#### ALM-11 — `processor.py` mixes intake handling, clear-hold scheduling, acknowledgement and queries
- **Severity:** Info
- **Category:** Modularity
- **Confidence:** High
- **Location:** `apps/alert-manager/src/alert_manager/processor.py:72-412` (one 340-line class: reports `:103-140`, acknowledgement `:144-181`, queries `:185-241`, state changes and timers `:245-412`); also `apps/alert-manager/src/alert_manager/payloads.py:8-10,124` and `tests/test_alarm_messages.py:63-77` (legacy "format before lifecycles" fallback key without `direction`, while the PLC always sends `key` and `state`, `plc_controller/events.py:94-95`)
- **About the file:** The alarm processor, the only writer of alarm state.
- **Problem:** At 412 lines it is under the limit and cohesive enough, but read-only queries (`list_alarms`, `get_alarm`) share the class with the lock-guarded writers and the clear-timer bookkeeping. That makes the locking rules harder to see. The legacy fallback is dead compatibility code, and its derived key (`source:parameter:severity`) differs from the real key format (`source:parameter:direction:severity`, migration comment), so an old-format message would open a second alarm for the same condition.
- **Recommendation:**
  1. When the file is next touched, move queries into `alert_manager/queries.py` (`AlarmQueries(sessions)`) and keep the lock-guarded mutations in `processor.py`. `AlarmServicer` takes both.
  2. With owner agreement, remove the legacy fallback and make `key` and `state` required fields, since no publisher uses the old format.
- **Effort:** S

### 7.6 opcua-server

**Area summary.** The write path follows I11. The server holds no command client, methods refuse any session that is not a gateway-verified `GatewayUser`, the gateway decides the role and writes the audit row, and every variable is read-only. Anonymous callers get only User-level service rights, and asyncua refuses AddNodes, DeleteNodes and non-Value writes for anyone who is not Admin. Passwords are never logged, the HTTP calls run in a worker thread with a timeout, and outages are logged once. The weak spots are all in the identity lifecycle and in edge handling. Re-activating a session orphans the previous gateway session. Every normal disconnect signs out twice, which writes a spurious audit row. All OPC UA sign-ins share one gateway throttle bucket, so one bad client can lock out every OPC UA user. An anonymous client can make the service write ERROR tracebacks (which fails `demo`). The asyncua transport limits are left at 100 MiB per message and 1000 sessions. The tests cover allow and deny for methods well, but they miss the identity edge cases, read-only coverage beyond one node, and the clear-password refusal end to end.

<details><summary>Scope reviewed by the area auditor</summary>

every module in `apps/opcua-server/src/opcua_server/` (`__init__.py`, `__main__.py`, `address_space.py`, `client.py`, `gateway.py`, `identity.py`, `methods.py`, `metrics.py`, `projection.py`, `security.py`, `server.py`, `subscriber.py`, `ua_types.py`, `units.py`, `upstreams.py`), all of `apps/opcua-server/tests/` (`opcua_fakes.py`, `test_address_space.py`, `test_opcua_end_to_end.py`, `test_opcua_gateway_client.py`, `test_opcua_methods.py`, `test_security.py`), `apps/opcua-server/pyproject.toml`, and the opcua-server entry in `docker-compose.yml`. For context I also read the invariants (I11), the overview and the OPC UA decisions of 2026-09-16 and 2026-09-18; the asyncua 2.0.1 server internals (`internal_session.py`, `internal_server.py`, `uaprocessor.py`, `address_space.py`, `server.py`, `crypto/permission_rules.py`); the gateway routes, schemas, login throttle and logout that the OPC UA methods call; `shared/runtime` (`MqttSession`); and `shared/observability` (`client_interceptors`). I confirmed the findings with a throwaway probe in the scratchpad (real asyncua server and client against the stand-in gateway from `opcua_fakes.py`). I did not change anything in the repository.

</details>

#### OPC-01 — Re-activating a session orphans its gateway session, and each activation starts a fresh gateway login
- **Severity:** Medium
- **Category:** Security
- **Confidence:** High (confirmed with the probe)
- **Location:** `apps/opcua-server/src/opcua_server/identity.py:61-68`, `apps/opcua-server/src/opcua_server/identity.py:85-93`; asyncua `server/internal_session.py:193-198` replaces `self.user` on every ActivateSession
- **About the file:** `identity.py` maps OPC UA identity tokens to users. It starts the gateway sign-in in the background and passes the session user to method callbacks through a context variable.
- **Problem:** OPC UA allows ActivateSession again on a live session (to change the user, or to reconnect). Each time, `GatewayUserManager.get_user` creates a new `GatewaySession` with a new login task, and asyncua overwrites `self.user`. The previous `GatewayUser` is dropped without `GatewaySession.close()`. Its refresh-token family stays valid in the gateway database until it expires, and its login task is never awaited (if it failed, asyncio logs "Task exception was never retrieved"). Probe: one connect plus two `activate_session` calls produced `/auth/login` ×3 and `/auth/logout` ×2, and both logouts belong to the last session (see OPC-02). Two gateway sessions were left open. Nothing limits how often a client may re-activate, so one connection can create unlimited gateway sessions and login tasks.
- **Recommendation:**
  1. In `_UserAwareSession.activate_session`, keep `previous = self.user` before calling `super().activate_session(...)`. After it returns, if `previous` is a `GatewayUser` with a session and `previous is not self.user`, schedule `previous.session.close()` exactly as `close_session` does (put the task in `_CLOSING`).
  2. Extract that scheduling into one helper, `_schedule_sign_out(user: User) -> None`, used by both `activate_session` and `close_session`.
  3. Add a test in `test_opcua_end_to_end.py`: connect with a username, call `client.activate_session(username=..., password=...)` twice, disconnect, and assert that the number of `/auth/logout` calls equals the number of `/auth/login` calls, each with a distinct refresh token.
- **Effort:** S

#### OPC-02 — A normal disconnect signs out twice and writes a spurious "no open session" audit row
- **Severity:** Medium
- **Category:** Reliability
- **Confidence:** High (confirmed: one session produced `['/auth/login', '/api/v1/commands/load', '/auth/logout', '/auth/logout']`)
- **Location:** `apps/opcua-server/src/opcua_server/identity.py:102-108`, `apps/opcua-server/src/opcua_server/gateway.py:183-198`; asyncua `server/uaprocessor.py:610-624` calls `close_session` again on transport loss
- **About the file:** `GatewaySession` in `gateway.py` holds one OPC UA user's tokens and refreshes them. Its `close()` signs the user out at the gateway.
- **Problem:** A client sends CloseSession (first `close_session`), then drops the TCP connection. asyncua's `_should_close_session_on_transport_loss()` returns True for a session that is no longer activated, so it calls `close_session` a second time. The base class returns early, but the override still schedules `user.session.close()` again. In the second `close()`, `_tokens` is already `None`, so the code falls back to `self._login.result()`. That is the original login's refresh token, which is stale if the session had refreshed in the meantime. The gateway's `/auth/logout` (`apps/api-gateway/src/api_gateway/routers/auth.py:204-247`) is a mutating, audited route. Every OPC UA session therefore leaves a second audit row with outcome "no open session", so the audit log over-reports sign-outs, and every disconnect costs an extra HTTP call.
- **Recommendation:**
  1. In `GatewaySession.close()`, return at once when `self._closed` is already True, before touching tokens. A session closed by a failed refresh or a refused sign-in needs no logout either.
  2. In `_UserAwareSession.close_session`, read `was_open = self.state is not SessionState.Closed` before calling `super()`, and schedule the sign-out only when `was_open`.
  3. Drop the `self._login.result()` fallback when `_tokens` was cleared by an earlier close. Keep it only for "login finished but was never awaited".
  4. Tests: in `test_opcua_gateway_client.py`, add `test_closing_twice_signs_out_once` (call `close()` twice, assert `client.logouts == ["r1"]`). In `TestMethods.test_a_signed_in_user_commands_through_the_gateway` (`apps/opcua-server/tests/test_opcua_end_to_end.py:170-200`), assert exactly one `/auth/logout` after a short settle.
- **Effort:** S

#### OPC-03 — An anonymous client can make the service write ERROR tracebacks (NaN or infinite AlarmId)
- **Severity:** Medium
- **Category:** Reliability
- **Confidence:** High (confirmed: anonymous call gets `BadUnexpectedError`, and asyncua logs "Error executing method call … ValueError: cannot convert float NaN to integer")
- **Location:** `apps/opcua-server/src/opcua_server/methods.py:53-57`, `apps/opcua-server/src/opcua_server/methods.py:171-177`
- **About the file:** `methods.py` holds the OPC UA method callbacks. They validate arguments and forward the call to the gateway as the session user.
- **Problem:** asyncua does not enforce the declared input types, so a client may send a Double for `AlarmId`. `_number` accepts NaN and ±inf. In `acknowledge_alarm`, `number < 1` is False for NaN, so `int(number)` runs and raises `ValueError` (NaN) or `OverflowError` (inf). Argument validation runs before the user check, so any anonymous client can do this. Each call writes an ERROR-level traceback. `demo` fails when a service writes an `error` line into `logs/`, and the house rule keeps `error` for platform failures. Separately, NaN reaches the gateway for authenticated `SetLoadDemand` and `ApplyValveCommand` calls as non-standard JSON (`json.dumps` emits `NaN`). The gateway rejects it with 422, but only after the round trip.
- **Recommendation:**
  1. In `_number`, return `None` unless `math.isfinite(value)`.
  2. Add `_integer(variant) -> int | None` that accepts `int` (not `bool`) and integral finite floats, and use it for `AlarmId` instead of `_number` plus `int()`. This also avoids the loss of precision for ids above 2**53.
  3. Wrap the body of each callback, or `_forward`, so an unexpected exception is logged once at warning with the method name and returned as `BadInternalError`. It should never reach asyncua's traceback log.
  4. Tests: extend `test_bad_arguments_never_reach_the_gateway` (`apps/opcua-server/tests/test_opcua_methods.py:133-153`) with `acknowledge_alarm(None, math.nan, "x")`, `acknowledge_alarm(None, math.inf, "x")`, `set_load_demand(None, math.nan)` and `apply_valve_command(None, math.inf, 0, 0)`. Each should return `BadInvalidArgument`.
- **Effort:** S

#### OPC-04 — All OPC UA sign-ins share one gateway throttle bucket and one audit address, and logins are unbounded
- **Severity:** Medium
- **Category:** Security
- **Confidence:** High
- **Location:** `apps/opcua-server/src/opcua_server/identity.py:61-63`, `apps/opcua-server/src/opcua_server/gateway.py:60-91`; gateway throttle `apps/api-gateway/src/api_gateway/auth/throttle.py:95-110` and `apps/api-gateway/src/api_gateway/config.py:58-60` (20 failures per client address per 900 s)
- **About the file:** `gateway.py` is the urllib-based HTTP client that signs OPC UA users in at the gateway and forwards their method calls.
- **Problem:** Every OPC UA ActivateSession with a username starts a gateway login immediately, even when the session will only read. Nothing limits how many run at once, and each takes a thread from the default executor for up to `REQUEST_TIMEOUT_S` = 10 s. The requests carry no `X-Forwarded-For`, so the gateway sees every OPC UA user as the opcua-server container address. After 20 wrong passwords from any OPC UA client within 15 minutes, the per-client bucket refuses every OPC UA sign-in, including valid ones. The gateway's audit `client_ip` for every OPC UA action is also the container, not the OPC UA peer. This is a new angle on accepted Д16, which is about throttle state living in memory, not about the key. A flood of activations also fills the small default thread pool, so legitimate method calls queue behind it.
- **Recommendation:**
  1. Pass the OPC UA peer address to the gateway. `_UserAwareSession` has `self.name` (asyncua's peer name). Give `GatewayClient.login` an optional `client_address` argument that is sent as `X-Forwarded-For`. The gateway already trusts that header from private addresses.
  2. Give `GatewayClient` its own bounded `concurrent.futures.ThreadPoolExecutor` (for example 8 workers) and run `loop.run_in_executor(self._executor, ...)` instead of `asyncio.to_thread`, so login floods cannot starve other `to_thread` users. Add a module-level `asyncio.Semaphore` (for example 4) around `login`.
  3. Record the residual risk in the report or overview: behind Docker's published port, all host clients may still appear as the bridge gateway address.
  4. Test: with the stand-in gateway, assert that a login request carries `X-Forwarded-For` equal to the session peer.
- **Effort:** M

#### OPC-05 — Transport and session limits left at asyncua defaults: 100 MiB messages, 1000 connections, 1000 subscriptions
- **Severity:** Medium
- **Category:** Security
- **Confidence:** High (probe printed `TransportLimits(max_recv_buffer=65535, max_send_buffer=65535, max_chunk_count=1601, max_message_size=104857600)`)
- **Location:** `apps/opcua-server/src/opcua_server/server.py:121`, `apps/opcua-server/src/opcua_server/server.py:129-138`; asyncua `server/server.py:124-131`, `server/internal_server.py:60-84`
- **About the file:** `server.py` builds the asyncua server and its address space, and writes value updates into it.
- **Problem:** An anonymous client may send requests of up to 100 MiB. Examples are a `Comment` string that is later forwarded to the gateway, or a Read of millions of node ids. It may also open up to 1000 connections and sessions, and create up to 1000 subscriptions with unbounded monitored items. The whole address space is 84 variables and 6 methods. Message size alone is enough to exhaust memory with a few parallel connections. The port is published on 127.0.0.1 only, which reduces exposure but does not remove it inside the Compose network.
- **Recommendation:**
  1. In `CogniBoilerOPCServer.start()`, before `self._server.start()`, set `self._server.limits = TransportLimits(max_recv_buffer=65535, max_send_buffer=65535, max_chunk_count=64, max_message_size=4 * 1024 * 1024)`. Also set `iserver.max_connections`, `iserver.max_subscriptions` and `InternalSession.max_connections` to modest named constants (for example 50) in a small `limits` section of `server.py`.
  2. Add module constants with a one-line reason each (no magic numbers).
  3. Test: start the server and assert the revised limits on `server._server.limits` and `iserver`.
- **Effort:** S

#### OPC-06 — The AuthenticationToken is a guessable counter, and a new channel can rebind another user's session
- **Severity:** Medium
- **Category:** Security
- **Confidence:** Medium (read in asyncua source, not exploited in the probe)
- **Location:** asyncua `server/internal_session.py:59-60` (`auth_token = ua.NodeId(self._auth_counter)`, incremented from 1000) and `server/uaprocessor.py:262-276` (ActivateSession on a new SecureChannel looks the session up by token); project factory `apps/opcua-server/src/opcua_server/identity.py:114-131`
- **About the file:** `install_identity` replaces asyncua's session factory with `_UserAwareSession`, so the project already owns session construction.
- **Problem:** OPC UA Part 4 requires the AuthenticationToken to be a random secret. asyncua uses sequential integers and accepts an ActivateSession for a live session from any new SecureChannel. On the `None` endpoint the client signature check is a no-op. Another client can therefore guess `NodeId(1000+n)`, re-activate that session with its own (anonymous or own-credential) token, and replace the victim's `GatewayUser`. The victim's later method calls are then refused, or run as the attacker's user, and the victim's gateway session is orphaned (OPC-01). The attacker cannot gain the victim's identity, because `get_user` always builds a new user, but it can hijack and disrupt sessions.
- **Recommendation:**
  1. In `_UserAwareSession.__init__`, after `super().__init__`, call `iserver.unregister_external_session(self)` if `external`, set `self.auth_token = ua.NodeId(secrets.token_bytes(32), 0, ua.NodeIdType.ByteString)`, then register again.
  2. Test: create two sessions through the factory and assert that their tokens are ByteString NodeIds, are distinct, and are not sequential.
  3. Note it as upstream debt so it can be revisited on an asyncua upgrade (the decision of 2026-09-16 already names `identity.py` as a workaround site).
- **Effort:** S

#### OPC-07 — Unexpected HTTP failures escape the error mapping, and a failed login breaks the session until reconnect
- **Severity:** Medium
- **Category:** Reliability
- **Confidence:** Medium
- **Location:** `apps/opcua-server/src/opcua_server/gateway.py:76-82`, `apps/opcua-server/src/opcua_server/gateway.py:155-163`, `apps/opcua-server/src/opcua_server/methods.py:91-95`
- **About the file:** See OPC-04.
- **Problem:** `_request` catches `URLError`, `TimeoutError` and `OSError` only. `response.read()` can raise `http.client.IncompleteRead`, and a malformed status line raises `BadStatusLine`/`LineTooLong`; these are `HTTPException`, not `OSError`, and both can happen when the gateway restarts mid-response. `Request(...)` raises `ValueError` on a bad `--gateway-url`, and `json.dumps` raises on unexpected types. These pass through `GatewaySession.tokens()`, which catches `GatewayUnavailableError` only, and through `_forward_as_user`. The client gets `BadUnexpectedError`, asyncua logs an ERROR traceback, and `METHOD_CALLS` is not incremented. If the login task itself failed that way, `self._tokens` stays `None` and every later call re-awaits the same failed task and re-raises. The session never recovers and is never marked closed.
- **Recommendation:**
  1. In `_request`, also catch `http.client.HTTPException` and `ValueError`, and map them to `GatewayUnavailableError(f"{method} {path}: {type(error).__name__}")`.
  2. Serialize with `json.dumps(payload, allow_nan=False)` so NaN fails locally (see OPC-03).
  3. In `GatewaySession.tokens()`, when the awaited login raised anything, set `self._closed = True` after logging, so the state is explicit.
  4. Tests: a stand-in gateway handler that sends a short body with a larger `Content-Length` (IncompleteRead). Assert that `GatewayClient.request` raises `GatewayUnavailableError` and that the method returns `BadCommunicationError`.
- **Effort:** S

#### OPC-08 — An empty or non-bytes MQTT payload is written as an all-zero plant state with Good quality
- **Severity:** Low
- **Category:** Reliability
- **Confidence:** High (confirmed: `_handle_message("sensors/boiler", b"")` wrote 16 updates such as `(2100, 0.0, GOOD)`)
- **Location:** `apps/opcua-server/src/opcua_server/subscriber.py:204-211`, `apps/opcua-server/src/opcua_server/subscriber.py:114-128`, `apps/opcua-server/src/opcua_server/subscriber.py:177`
- **About the file:** `subscriber.py` is the MQTT-to-OPC UA bridge. It parses protobuf telemetry, thins bursts, and writes values with instrument quality.
- **Problem:** Proto3 parses `b""` as a valid message with every field at its default. `_consume` turns any non-bytes payload into `b""`. A zero-length publish, for example clearing a retained message, therefore sets drum pressure 0 Pa, level 0 m and so on, all with Good status, and the source timestamp falls back to "now" because `timestamp_ms` is 0. The test `test_the_bridge_subscribes_and_reconnects` (`apps/opcua-server/tests/test_opcua_methods.py:383-424`) feeds `"not bytes"` and asserts only `received >= 1`, so it passes while this happens.
- **Recommendation:**
  1. In `_handle_message`, skip (count as skipped, debug log) payloads that are empty or not `bytes`.
  2. In each parser, treat `msg.timestamp_ms == 0` as malformed: raise `ValueError` so the existing skip path handles it.
  3. Tests: `_handle_message(TOPIC_BOILER, b"")` writes nothing and increments `skipped`. In the reconnect test, assert `RecordingOPC().updates == []`.
- **Effort:** S

#### OPC-09 — Failed address-space writes are silently dropped
- **Severity:** Low
- **Category:** Reliability
- **Confidence:** High
- **Location:** `apps/opcua-server/src/opcua_server/server.py:307-329`; asyncua `server/internal_server.py:395-402` discards the StatusCode returned by asyncua `server/address_space.py:807-837` (`BadTypeMismatch`, `BadNodeIdUnknown`)
- **About the file:** See OPC-05.
- **Problem:** `Server.write_attribute_value` returns `None`, because asyncua throws away the per-write StatusCode. If a projection ever produces a value whose variant type does not match the node (for example an `Int64` node fed a float, or a list for a scalar), the write is refused and nobody learns about it. The node silently keeps a stale value with Good status. The docstring of `update_variable` says only `KeyError` can happen.
- **Recommendation:**
  1. In `_write`, call `self._server.iserver.aspace.write_attribute_value(node.nodeid, AttributeIds.Value, data_value)` directly and check the returned code. On a bad code, log a warning with the node id, browse name and status name. Rate-limit it to once per node, for example with a `set[int]` of nodes already reported.
  2. Test: `update_variable(<Int64 node>, ["a"])` logs once and does not raise.
- **Effort:** S

#### OPC-10 — Bugs in the bridge are reported as "MQTT error" and trigger silent reconnect loops
- **Severity:** Low
- **Category:** Logging
- **Confidence:** High
- **Location:** `apps/opcua-server/src/opcua_server/subscriber.py:154-159`, `apps/opcua-server/src/opcua_server/subscriber.py:171-180`, `apps/opcua-server/src/opcua_server/subscriber.py:183-194`; `shared/runtime/src/cogniboiler_runtime/mqtt.py` (`MqttSession.run` catches every `Exception`)
- **About the file:** See OPC-08.
- **Problem:** `_handle_message` catches only `DecodeError`/`ValueError`, and `_apply` catches only `KeyError`. Any other exception from a projection or an asyncua write (`AttributeError`, `TypeError`, `UaStatusCodeError`) leaves `_consume`. `MqttSession` then logs it once as "MQTT error: … retrying" at warning, afterwards at debug, drops the broker connection and reconnects every 5 s. A code defect then looks like a broker outage and repeats forever almost silently. The same exception inside `flush_pending` kills that task, which brings down the whole `asyncio.gather` in `apps/opcua-server/src/opcua_server/__main__.py:76-84`.
- **Recommendation:**
  1. Wrap the per-message work in `_handle_message` and the loop body in `flush_pending` with `except Exception: logger.exception("OPC UA bridge failed on %s", topic)`, then count the message as skipped. Keep MQTT failures for the session loop.
  2. Test: a `RecordingOPC` whose `update_variable` raises `RuntimeError` makes `_handle_message` log and return, without raising.
- **Effort:** S

#### OPC-11 — urllib's default opener follows redirects with the bearer token, honours proxy variables, and reads unbounded bodies
- **Severity:** Low
- **Category:** Security
- **Confidence:** High
- **Location:** `apps/opcua-server/src/opcua_server/gateway.py:63-80`
- **About the file:** See OPC-04.
- **Problem:** `urllib.request.urlopen` uses the global opener. It follows 30x redirects, keeping the `Authorization` header and turning POST into GET, and it routes through `HTTP_PROXY`/`http_proxy` when set. A misconfigured or compromised address at `--gateway-url` could make the user's access token and password go elsewhere. `response.read()` and `error.read()` have no size cap. TLS verification is not disabled anywhere (no `verify=False`). The default URL is plain `http://api-gateway:8000`, so passwords and tokens cross the Compose network unencrypted. That is acceptable for the single-host demo, but it should be stated.
- **Recommendation:**
  1. Build one opener in `GatewayClient.__init__` with `urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())`. `_NoRedirect` is a subclass of `HTTPRedirectHandler` whose `redirect_request` returns `None`. Use `self._opener.open(request, timeout=...)`.
  2. Refuse any scheme other than `http`/`https` in `__init__` (raise `ValueError` at startup).
  3. Read at most `MAX_REPLY_BYTES = 1_048_576` (`response.read(MAX_REPLY_BYTES + 1)`) and treat anything longer as an empty body with a warning.
  4. Test: a stand-in gateway answering 307 to `/api/v1/commands/load` gives `BadCommunicationError`, and no second request carries `Authorization`.
- **Effort:** S

#### OPC-12 — OpenAlarmCount and the other alarm counts are computed from a list capped at 200
- **Severity:** Low
- **Category:** Reliability
- **Confidence:** High
- **Location:** `apps/opcua-server/src/opcua_server/client.py:36-39`, `apps/opcua-server/src/opcua_server/projection.py:88-115`; `AlarmListMsg.total` is the full count (`apps/alert-manager/src/alert_manager/grpc_server.py:114-127`)
- **About the file:** `projection.py` turns upstream protobuf messages into `(node id, value)` updates.
- **Problem:** `alarm_updates` uses `len(open_alarms)` for node 2800 and counts unacknowledged and critical alarms over the returned page only. With more than 200 open alarms, which is possible in an alarm flood, the counts cap silently while `total` holds the real number.
- **Recommendation:**
  1. Use `alarms.total or len(open_alarms)` for 2800.
  2. Document that 2801 and 2802 cover the listed page, or request them separately. Do not add new RPCs without an owner decision.
  3. Test: `AlarmListMsg(total=250, alarms=[...3...])` gives 2800 == 250.
- **Effort:** S

#### OPC-13 — Node ids of the Simulation, PLC and Alarms folders are magic numbers repeated outside the catalogue
- **Severity:** Low
- **Category:** Modularity
- **Confidence:** High
- **Location:** `apps/opcua-server/src/opcua_server/projection.py:45-51`, `apps/opcua-server/src/opcua_server/projection.py:72-84`, `apps/opcua-server/src/opcua_server/projection.py:104-115`, `apps/opcua-server/src/opcua_server/upstreams.py:25-26`; catalogue `apps/opcua-server/src/opcua_server/address_space.py:259-291`
- **About the file:** `address_space.py` is the single catalogue of nodes and field mappings. `upstreams.py` polls PLCService and AlarmService.
- **Problem:** The other folders map through `*_FIELD_TO_NODEID` dictionaries, but 2600–2606, 2700–2712 and 2800–2804 are literals in `projection.py`. 2712 and 2804 are defined a second time in `upstreams.py`. A renumbering in the catalogue compiles fine and fails at runtime with `KeyError`, which `upstreams` does not catch, so the task dies.
- **Recommendation:**
  1. Add named constants in `address_space.py` (`NODEID_PLC_COMMUNICATION`, `NODEID_ALARM_COMMUNICATION`, `NODEID_SIM_SCENARIO`, …) and use them in the `_d(...)` rows, in `projection.py` and in `upstreams.py`.
  2. Add a test: every id returned by `plant_updates`, `plc_updates` and `alarm_updates` for a default message is in `VARIABLES_BY_NODE_ID`. The probe showed this holds today; the test pins it.
- **Effort:** S

#### OPC-14 — The read-only gRPC channels skip the shared observability interceptors
- **Severity:** Low
- **Category:** Logging
- **Confidence:** High
- **Location:** `apps/opcua-server/src/opcua_server/client.py:16`, `apps/opcua-server/src/opcua_server/client.py:30`; the convention is followed in `apps/api-gateway/src/api_gateway/clients.py:97-98` and `apps/plc-controller/src/plc_controller/client.py:37-38`
- **About the file:** `client.py` holds the read-only PLCService and AlarmService clients.
- **Problem:** The domain rules require gRPC channels to use `cogniboiler_observability` interceptors. Today they would only add a correlation id, and the polling loops set none. So the effect is small, but the service is the one exception to the rule, and a correlation id added later would not propagate.
- **Recommendation:**
  1. Pass `interceptors=client_interceptors()` to both `grpc.aio.insecure_channel` calls.
  2. Optionally wrap each poll in `correlation_scope(None)` in `upstreams.py`, so a slow or failing read can be matched in the PLC and alert-manager logs.
- **Effort:** S

#### OPC-15 — Refusals are logged and counted unevenly, and logs carry the typed username, not the verified one
- **Severity:** Low
- **Category:** Logging
- **Confidence:** High
- **Location:** `apps/opcua-server/src/opcua_server/methods.py:99-100`, `apps/opcua-server/src/opcua_server/methods.py:136-190`, `apps/opcua-server/src/opcua_server/methods.py:94`, `apps/opcua-server/src/opcua_server/methods.py:97`, `apps/opcua-server/src/opcua_server/methods.py:112-114`, `apps/opcua-server/src/opcua_server/gateway.py:107-111`, `apps/opcua-server/src/opcua_server/identity.py:62`
- **About the file:** See OPC-03.
- **Problem:**
  - A gateway 400/404/409/422 becomes `BadInvalidArgument` with no log line at all.
  - Invalid arguments return before `_forward`, so `opcua_method_calls_total` never counts them.
  - `refresh()` drops the status and `code`, so "revoked", "expired" and "account blocked" all look the same.
  - Method logs use `user.name`, which is the name the client typed and unverified on the failure paths, instead of `tokens.username`, the gateway-verified name that the audit shows.
  - The login task name embeds the unbounded, client-supplied username (`name=f"opcua-login-{username}"`).
  - The gateway's 429 (throttled) becomes `BadCommunicationError`, which reads as an outage.
- **Recommendation:**
  1. Log `reply.status` and `reply.body.get("code")` at info for the 4xx mapping.
  2. Count invalid arguments with the label `outcome="invalid"`. Move validation behind one helper `_invalid(action)` that increments `METHOD_CALLS`.
  3. In `refresh()`, log a warning with the status and `code` on failure. It must never include the token.
  4. Log `tokens.username` after a successful sign-in. On failures, log the typed name with `%r` and mark it `unverified`.
  5. Name the login task `"opcua-login"` without the username.
  6. Map 429 to `BadTooManyOperations`.
  7. Tests: assert the `METHOD_CALLS` samples (`prometheus_client.REGISTRY.get_sample_value`) for accepted, refused, failed and invalid outcomes. No test references `METHOD_CALLS` today.
- **Effort:** S

#### OPC-16 — The legacy RSA-1_5 password encryption is accepted on the None endpoint
- **Severity:** Low
- **Category:** Security
- **Confidence:** Medium
- **Location:** `apps/opcua-server/src/opcua_server/identity.py:71-93`; asyncua `server/internal_server.py:445-470` ("TODO check if algorithm is allowed")
- **About the file:** See OPC-01.
- **Problem:** The advertised token policy is Basic256Sha256, whose password algorithm is RSA-OAEP. asyncua also decrypts `rsa-1_5` (PKCS#1 v1.5). A successfully decrypted token always activates, because the password is checked later at the gateway, while a padding failure returns `BadIdentityTokenInvalid`. That is the shape of a Bleichenbacher oracle. The linked OpenSSL is 4.0.2, whose implicit rejection largely neutralises it, so this is hygiene rather than an open hole.
- **Recommendation:**
  1. In `_UserAwareSession.activate_session`, refuse a `UserNameIdentityToken` whose `EncryptionAlgorithm` is set and is not in `{"http://www.w3.org/2001/04/xmlenc#rsa-oaep", "http://opcfoundation.org/UA/security/rsa-oaep-sha2-256"}`, with `BadIdentityTokenRejected`. Extend `sends_password_in_clear` into `token_is_acceptable(token, peer_certificate)`.
  2. Unit test in `test_security.py` for an `rsa-1_5` token.
- **Effort:** S

#### OPC-17 — Test gaps: read-only coverage, identity lifecycle, clear-password refusal end to end, and one timing-dependent test
- **Severity:** Low
- **Category:** Tests
- **Confidence:** High
- **Location:** `apps/opcua-server/tests/test_opcua_end_to_end.py:146-152`, `apps/opcua-server/tests/test_security.py:19-33`, `apps/opcua-server/tests/test_opcua_methods.py:350-364`, `apps/opcua-server/tests/test_opcua_methods.py:383-424`
- **About the file:** The tests drive a real asyncua server and client against a stand-in HTTP gateway, plus unit tests of projections, the bridge and the gateway client.
- **Problem:** Gaps and weak tests:
  - `test_values_are_not_writable_by_clients` checks one node (`Pressure`), the Value attribute, and an anonymous session only.
  - Clear-password refusal is tested only through the pure helper `sends_password_in_clear`. Nothing proves that `_UserAwareSession.activate_session` is actually installed and refuses with `BadIdentityTokenRejected`.
  - Not tested: re-activation (OPC-01), double close (OPC-02), NaN/inf (OPC-03), empty payload (OPC-08), `METHOD_CALLS` (OPC-15), and `secure(server, None)` (the self-signed fallback and its warning).
  - `test_a_burst_is_thinned_to_the_latest_values` assumes three `_handle_message` calls finish within 200 ms of wall-clock time (`time.monotonic` in `apps/opcua-server/src/opcua_server/subscriber.py:161`). On a slow runner the second message is applied at once and the assertion `== [4.1]` fails. The rules call a speed-dependent test a defect.
- **Recommendation:**
  1. `test_no_variable_or_property_is_writable`: browse all 84 variables and their `EngineeringUnits` properties and assert that `AccessLevel` and `UserAccessLevel` lack `CurrentWrite`. Write as both an anonymous and a signed-in session, expect `BadUserAccessDenied`, and also expect AddNodes and DeleteNodes to be refused for a signed-in user. Pair it with the existing allowed read so both outcomes are covered.
  2. `test_a_clear_password_on_the_open_endpoint_is_refused`: call `_UserAwareSession.activate_session` through a session built by the installed factory, with an unencrypted `UserNameIdentityToken`, and expect `ServiceError(BadIdentityTokenRejected)`. Keep a matching allow case with `EncryptionAlgorithm` set.
  3. Inject a clock into `MQTTOPCBridge` (`clock: Callable[[], float] = time.monotonic`) and drive the burst test with a fake clock.
  4. Add the tests named in OPC-01, 02, 03, 07, 08, 09, 12 and 15.
- **Effort:** M

#### OPC-18 — Duplicated logic inside the service and across services
- **Severity:** Low
- **Category:** Duplication
- **Confidence:** High
- **Location:** `apps/opcua-server/src/opcua_server/subscriber.py:218-221` and `apps/opcua-server/src/opcua_server/projection.py:30-31` (two `_fields` helpers, one coercing to float); `apps/opcua-server/src/opcua_server/subscriber.py:114-159` and `apps/historian/src/historian/subscriber.py:143-204` (same parse-by-topic and `DecodeError`/`ValueError` skip); `apps/opcua-server/src/opcua_server/security.py:60-117` and `scripts/dev_secrets/__main__.py:42-100` (the same self-signed certificate builder); `apps/opcua-server/src/opcua_server/__main__.py:18` and the same `sys.path.insert(... "shared" / "generated")` in all six `apps/*/src/*/__main__.py`; `apps/opcua-server/src/opcua_server/server.py:60` and `apps/opcua-server/src/opcua_server/__main__.py:34` (endpoint path repeated)
- **About the file:** Cross-cutting.
- **Problem:** Each pair can drift. For example, one `_fields` coerces to float and the other does not. The scripts copy is deliberate, because scripts may not import packages, but it should be named as such. The `sys.path` hack depends on the source-tree depth (`parents[4]`).
- **Recommendation:**
  1. Keep one `_fields(message, mapping) -> list[Update]` in `projection.py`, and import it in `subscriber.py`. Coercion already happens in `server._coerce`.
  2. Move the "decode protobuf by topic, count skips" helper into `shared/runtime` (for example `decode_by_topic(parsers, topic, raw) -> T | None`) and use it from historian and opcua-server.
  3. Build the endpoint in `__main__` from a shared `endpoint_url(port)` in `server.py`.
  4. Replace the per-service `sys.path` hack with the generated stubs installed as a workspace package. This is cross-service, so record it as debt.
  5. Unit conversion: none is duplicated. The OPC UA server publishes SI with `EngineeringUnits` only (`units.py`), while bar/°C/MW conversion lives only in `apps/web/src/units.ts`. That is correct under I4.
- **Effort:** M

#### OPC-19 — The entry point exceeds the size rule and carries behaviour and misleading log text
- **Severity:** Low
- **Category:** Modularity
- **Confidence:** High
- **Location:** `apps/opcua-server/src/opcua_server/__main__.py:53-62`, `apps/opcua-server/src/opcua_server/__main__.py:64-73`, `apps/opcua-server/src/opcua_server/__main__.py:85-88`
- **About the file:** `__main__.py` parses arguments and wires the server, the bridge and the projections.
- **Problem:**
  - It is 120 lines, over the roughly 100-line rule for entry points.
  - It holds the `log_stats` loop, which is behaviour: an INFO line every 30 s, about 2,900 lines a day, duplicating what `MQTT_RECEIVED` already counts.
  - It logs `opc.tcp://localhost:%d` while binding `0.0.0.0`.
  - On shutdown the remaining gather tasks keep running while `opc_server.stop()` executes, and pending `_CLOSING` sign-outs are not drained.
- **Recommendation:**
  1. Move `log_stats` into `MQTTOPCBridge` as a metric (a `Counter` for skipped messages in `metrics.py`), or drop it. Log the actual endpoint.
  2. Before `opc_server.stop()`, cancel and await the gather's child tasks (use `asyncio.TaskGroup`).
  3. Add `identity.drain_sign_outs(timeout_s)` that awaits `_CLOSING`, called in `finally`.
- **Effort:** S

#### OPC-20 — A certificate and key pair from the environment is not checked at startup
- **Severity:** Info
- **Category:** Reliability
- **Confidence:** Medium
- **Location:** `apps/opcua-server/src/opcua_server/security.py:52-57`, `apps/opcua-server/src/opcua_server/security.py:120-132`
- **About the file:** `security.py` provides the application certificate and the endpoint policies.
- **Problem:** A mismatched pair, or an expired certificate from an old `.env`, is loaded without complaint. The failure only shows per connection: SignAndEncrypt handshakes fail and password decryption on the `None` endpoint fails. The key is never written to disk (good), RSA-2048 and 365 days fit Basic256Sha256, and the SAN carries the application URI, `localhost`, the hostname and 127.0.0.1.
- **Recommendation:**
  1. In `secure()`, parse both PEMs, compare public keys, and fail fast with a clear message (no key material in it) on a mismatch.
  2. Warn when `not_valid_after` is within 30 days, and refuse when it has passed.
  3. Unit tests for a mismatched pair and an expired certificate.
- **Effort:** S

### 7.7 Web console (apps/web)

**Area summary.** The console is small, strictly typed (strict + `noUncheckedIndexedAccess`, no `any`, no `console.*`, no `dangerouslySetInnerHTML`/`innerHTML`, no `href`/`src` built from server data), and keeps the security architecture it promises: the access token stays in memory only, the refresh token stays in the httpOnly cookie and is never read from the response body, the WebSocket token goes in the first frame, and every mutating route the console calls is role-checked on the gateway. The weak points are in failure handling rather than in the design. An empty or locale-comma numeric field turns into a valid `0` command, and a unit test locks that in. Any 5xx from nginx or the Vite proxy during a token refresh signs the operator out. The live channel can stop for good while the session stays signed in. A failed sign-out silently leaves the refresh cookie valid. The safety-critical Control screen has no component test, and the confirm-then-run pattern is copied across three screens.

<details><summary>Scope reviewed by the area auditor</summary>

every file under `apps/web/src/` read in full (`App.tsx`, `main.tsx`, `api/http.ts`, `api/endpoints.ts`, `api/realtime.ts`, `api/queryKeys.ts`, `api/types.ts`, `session/*`, `live/*`, `alarms/*`, `components/**` incl. `ui/*`, `screens/*`, `trends/*`, `insights/*`, `units.ts`, `theme.ts`, `test/*`, and all `*.test.ts(x)`; `schema.gen.ts` used as a reference only), `styles.css` (skimmed, dead-selector scan), `index.html`, `vite.config.ts`, `eslint.config.js`, `tsconfig.json`, `package.json`, `playwright.config.ts`, `playwright.readme.config.ts`, `e2e/*` (support, roles, session read; others skimmed), `readme/readme.spec.ts` (skimmed), `Dockerfile`, `.dockerignore`, `.gitignore`. Cross-checks (read only): `apps/api-gateway/src/api_gateway/routers/{websocket,auth,commands,simulation,alarms,users,audit}.py`, `auth/rbac.py`, `schemas/{auth,command,plant}.py`, `problems.py`, `infrastructure/nginx/{headers,site,proxy}.inc`, `shared/openapi/api-gateway.json`, `.github/workflows/ci.yml`, Vite's dev-proxy error handler in `node_modules/vite`. Ran `pnpm --dir apps/web run lint` (exit 0) and `pnpm --dir apps/web run typecheck` (exit 0).

</details>

#### WEB-01 — An empty or unparsable number field becomes a valid `0` command (valves, load, fault severity/ramp)
- **Severity:** Medium
- **Category:** Reliability
- **Confidence:** High
- **Location:** `apps/web/src/screens/ControlScreen.tsx:83`, `:110`, `:130`, `:237-239`, `:315-318`; `apps/web/src/screens/EngineerScreen.tsx:245-255`, `:346-347`; the bug is locked in by `apps/web/src/screens/ControlScreen.test.tsx:19-20`
- **About the file:** `ControlScreen.tsx` holds the operator/engineer forms for load demand, PLC mode, E-Stop, manual valves and setpoints. `EngineerScreen.tsx` holds the simulation, scenario and fault-injection forms.
- **Problem:** Validity is checked with `inRange(Number(value), range)`, and `Number("")` and `Number("  ")` are `0`. A `type="number"` input also reports `""` for text the browser cannot parse, such as `12,5` in an en-US locale. So a cleared or comma-typed field is "valid" wherever `0` is inside the range: load (0–300 MW), every valve (0–100 %), fault severity (min 0 for pump failure, fouling and leak) and fault ramp. Scenario: the operator clears the Feedwater field to retype it, or types `12,5`, and presses "Send positions…". The dialog says `Feedwater 0.0 %` and "Send and switch to MANUAL" sends `feedwater_valve: 0`, which starves the drum and trips on low level. The dialog does show the value, but `0.0` next to three other numbers is easy to miss. The existing test names this case "refuse what is not a number" and then asserts `true`:
  ```ts
  it("refuse what is not a number", () => {
    expect(inRange(Number(""), LIMITS.valvePct)).toBe(true);
  ```
  The gateway accepts `0` as a legitimate value, so the server cannot catch this.
- **Recommendation:**
  1. Add `parseDecimal(text: string): number | null` next to the formatters in `apps/web/src/units.ts` (it is presentation-edge parsing). Return `null` for an empty or whitespace-only string, otherwise `Number(text)`, and `null` if that result is not finite.
  2. Change `inRange` in `ControlScreen.tsx:64` to take `number | null` and return `false` for `null`. Use `parseDecimal` in `NumberField`, `LoadSection`, `ValvesSection` (`fraction()`), `SetpointsSection`, and in `EngineerScreen` for steps, severity and ramp.
  3. Fix the test at `ControlScreen.test.tsx:20` to expect `false`. Add cases for `"  "` and for a form whose field is cleared: the send button must be disabled.
- **Effort:** S

#### WEB-02 — Any 5xx during a token refresh signs the operator out, because only `network.unreachable` counts as transient
- **Severity:** Medium
- **Category:** Reliability
- **Confidence:** High
- **Location:** `apps/web/src/session/session.ts:168-176` (`fail`), `:124-136` (`restoreOnce`); `apps/web/src/api/http.ts:101-113`, `:148`
- **About the file:** `session.ts` is the `SessionManager`: the in-memory access token, single-flight renewal, the proactive renew timer, and the signed-in/out state.
- **Problem:** `fail()` keeps the session only when the error code is `network.unreachable`, which happens only when `fetch` itself rejects. The browser never talks to the gateway directly, though; it talks to nginx (stack) or the Vite proxy (dev). When the gateway restarts, nginx answers 502/504 with HTML and Vite's proxy answers `502 text/plain` (checked in `node_modules/vite/dist/node/chunks/node.js`, the `proxy.on("error")` handler). `readProblem` turns that into `code: "http.502"`, so `fail()` runs `forget({status:"signed_out", notice:"Your session has ended. Sign in again."})`. Scenario: `docker compose restart api-gateway` while the 60-second-ahead renew timer fires, and every open console drops to the sign-in screen with a false "session has ended". Yet the refresh cookie is still valid, and a reload would restore the session. At start-up (`restoreOnce`) a 502 gives a sign-in form with no notice at all, instead of "The gateway cannot be reached". The existing test (`session.test.ts:126`) covers status 0 only, so the path that happens in production is not tested.
- **Recommendation:**
  1. In `session.ts`, add `function transient(error: unknown): boolean` that returns true for `network.unreachable` and for `ApiError` with `status >= 500` or `status === 429`.
  2. Use it in `fail()` (keep the session and return `null`) and in `restoreOnce()` (show the "gateway cannot be reached" notice).
  3. After a transient failure, re-arm the renewal: call `scheduleRenewal`-style logic with a short retry delay (e.g. `MIN_RENEW_DELAY_MS`) so the session is not left with an expiring token and no timer. Today `fail()` clears nothing but also schedules nothing.
  4. Add tests in `session.test.ts`: a refresh rejected with `apiError(502, "http.502")` keeps `signed_in` and schedules a retry; `restore()` with a 503 shows the unreachable notice.
- **Effort:** S

#### WEB-03 — The live channel can stop for good while the session stays signed in, and the 4401 path reconnects without backoff
- **Severity:** Medium
- **Category:** Reliability
- **Confidence:** High
- **Location:** `apps/web/src/api/realtime.ts:194-205` (`handleUnauthorized`), `:115-124`; `apps/web/src/session/session.ts:170-173`; `apps/web/src/live/LiveProvider.tsx:47`, `:28-61`
- **About the file:** `realtime.ts` is the framework-free WebSocket client (auth frame, subscribe, backoff, close-code handling). `LiveProvider.tsx` owns one instance per signed-in session.
- **Problem:** The `unauthorized()` handler is `manager.renew()`, and it returns `null` in two very different cases: "the session is over" (the manager is then `signed_out`) and "the refresh failed transiently, session kept" (`session.ts:170-173`, and more cases once WEB-02 is fixed). `handleUnauthorized` treats both as final, `this.stop()` at `:201`, and nothing restarts it: the `LiveProvider` effect depends only on `[manager, queryClient, store]`. Result: the header shows "Offline", and the mimic, PLC state and trends freeze until sign-out or reload, while REST polling keeps running. The Control screen keeps offering commands based on the last `live.plc` snapshot (e.g. `ResetSection` hidden because a stale `emergency_stop_active` is false). Separately, the 4401 path calls `connect()` immediately after a successful renewal and never goes through `scheduleReconnect`. If the server keeps refusing a freshly renewed token (it closes 4401 for `ws.auth_timeout`, `ws.bad_request` or "token belongs to another user"; see `routers/websocket.py:228-241`), the client loops refresh → connect → 4401 with no delay, and rotates the refresh cookie each time.
- **Recommendation:**
  1. In `LiveProvider.tsx`, change the handler to `unauthorized: async () => { const token = await manager.renew(); return token ?? (manager.snapshot().status === "signed_in" ? manager.accessToken() : null); }`. Or, more cleanly, give `RealtimeHandlers.unauthorized` a result type `{ token: string } | "retry" | "ended"`.
  2. In `RealtimeClient.handleUnauthorized`, call `stop()` only for "ended". For "retry", call `scheduleReconnect()` so backoff applies.
  3. After a successful renewal, reconnect through `scheduleReconnect()` rather than `connect()`, or at least count the attempt, so repeated 4401s back off.
  4. Tests in `realtime.test.ts`: a "retry" answer reconnects after the backoff delay and does not reach `stopped`; three consecutive 4401 closes produce increasing delays. In `shell.test.tsx`: a transient renewal failure keeps the connection state out of `stopped`.
- **Effort:** M

#### WEB-04 — A failed sign-out looks like success, leaves the refresh cookie valid, and raises an unhandled rejection
- **Severity:** Medium
- **Category:** Security
- **Confidence:** High
- **Location:** `apps/web/src/components/Layout.tsx:141`; `apps/web/src/session/session.ts:142-148`; `apps/web/src/api/endpoints.ts:50-52`
- **About the file:** `Layout.tsx` is the signed-in shell (navigation, connection badge, theme, Sign out). `endpoints.ts` holds the typed route functions.
- **Problem:** `signOut()` forgets the in-memory state in `finally` and rethrows. `Layout` calls `void manager.signOut()`, so a failure becomes an "Uncaught (in promise)" error and the operator sees the normal sign-in screen. Only `POST /auth/logout` can clear the httpOnly refresh cookie and revoke the session. If that request fails (gateway restarting behind nginx → 502, or the network is down), the cookie and the server session both survive, and the next page load restores the session silently through `restoreOnce`. Scenario on a shared control-room workstation: operator A clicks Sign out during a gateway blip and walks away; B reloads the console and is signed in as A, with A's role. Also, `signOut` sends `auth: false`, so no bearer token goes with it. The gateway's logout also revokes by bearer (`routers/auth.py:223-233`), so sending the access token would close the session even when the cookie is missing or already rotated (see WEB-07's renew race).
- **Recommendation:**
  1. In `SessionManager.signOut`, catch the failure. When `signOut()` did not succeed, forget with a notice such as "Sign-out could not reach the gateway; the session may still be open. Sign in and sign out again once it answers." Do not rethrow, so nothing is left unhandled.
  2. In `endpoints.ts`, let `signOut(accessToken: string | null)` send `Authorization: Bearer …` without the renew-on-401 logic. For example, add a `RequestOptions.auth` value `"send-only"` in `http.ts` that attaches the token but skips renewal. `SessionManager.signOut` passes `this.token`.
  3. Tests: in `session.test.ts`, update "forgets the token on sign-out even if the gateway does not answer" to expect a resolved promise and the new notice. In `endpoints.test.ts`, assert that logout carries the bearer header.
- **Effort:** S

#### WEB-05 — The Control screen, where the plant is actually commanded, has no component test; its test file tests other screens
- **Severity:** Medium
- **Category:** Tests
- **Confidence:** High
- **Location:** `apps/web/src/screens/ControlScreen.test.tsx:1-52` (only `LIMITS`, `inRange`, then `FAULT_KINDS` at `:26` from `EngineerScreen` and `USERNAME_PATTERN` at `:45` from `UsersScreen`); `apps/web/src/screens/ControlScreen.tsx:418-476`; also misplaced: `apps/web/src/theme.test.ts:120` (horn), `:162` (trend parameters)
- **About the file:** `ControlScreen.tsx` renders load, mode/E-Stop, E-Stop reset, manual valves and setpoints per role.
- **Problem:** No test renders `ControlScreen`. Untested: which sections each role sees (operator: no setpoints or reset; engineer: all); the SI payloads (MW→W, bar→Pa, °C→K, %→fraction) actually sent to `setLoadDemand`, `sendValveCommand` and `updateSetpoints`; that every command goes through `ConfirmDialog` and Cancel sends nothing; the E-Stop and trip flow; disabled states while `emergency_stop_active`; and the "Last command" result and error display. A unit-conversion or wiring regression here would reach the PLC unnoticed until e2e runs against a stack. The role e2e (`e2e/roles.spec.ts:22-36`) checks region visibility only. The misplaced tests make it hard to see which modules are covered.
- **Recommendation:**
  1. Create a render test for `ControlScreen` in `ControlScreen.test.tsx`, following the setup of `EngineerScreen.test.tsx`: mock `../live/LiveProvider` `useLive` with a `PlcStatus` fixture from `src/test/fixtures.ts`, mock `../api/endpoints`, and wrap in `SessionProvider` with a manager signed in as operator or engineer.
  2. Cases: (a) operator sees Load, Mode, Manual valves, and no Setpoints or Emergency stop reset; (b) set 250 MW, confirm → `setLoadDemand(250e6)`; Cancel → not called; (c) valves 50/40/60/0 → `sendValveCommand({fuel_valve:0.5,…})`; (d) setpoints 160 bar, 5 m, 540 °C → `pressure_pa:16e6, steam_temp_k:813.15`; (e) engineer with `emergency_stop_active` sees "Reset E-Stop…", confirm → `resetEmergencyStop` called; (f) a clearing field disables the send button (with WEB-01); (g) a refused `CommandAck` shows "Refused: <reason>".
  3. Move the `FAULT_KINDS` test to `EngineerScreen.test.tsx`, the username/password test to `AdminScreens.test.tsx`, the horn test to a new `alarms/horn.test.ts`, and the trend-parameter test to `trends/series.test.ts`.
- **Effort:** M

#### WEB-06 — The "confirm, run, show the last result" logic is copied in three screens
- **Severity:** Medium
- **Category:** Duplication
- **Confidence:** High
- **Location:** `apps/web/src/screens/ControlScreen.tsx:56-62` (`PendingCommand`), `:418-476`; `apps/web/src/screens/EngineerScreen.tsx:44-50` (`PendingAction`), `:461-517`; `apps/web/src/screens/UsersScreen.tsx:28-35` (`PendingChange`), `:235-315`
- **About the file:** The three command screens of the console.
- **Problem:** Each screen declares the same `{title, body, confirmLabel, danger?, run}` interface under a different name. Each has `useState<Pending | null>` plus `useState<string | null>` for the last label, a `useMutation({ mutationFn: (run) => run(), onSettled: invalidate(key) })`, a `ConfirmDialog` wired as `busy={mutation.isPending}` / `onConfirm={() => { setLast(title); mutate(pending.run, { onSettled: () => setPending(null) }) }}`, and a "Last command"/"Last action" `Panel` with `CommandResult`. About 60 near-identical lines per screen. Fixes such as WEB-09 (cancel while busy) or showing a failure glyph instead of `OkIcon` (`ControlScreen.tsx:442`, `EngineerScreen.tsx:479`, shown even when the command failed) would have to be made three times.
- **Recommendation:**
  1. Create `apps/web/src/components/ConfirmedAction.tsx` exporting `interface PendingAction<T> { title; body; confirmLabel; danger?; run: () => Promise<T>; done?: string }`.
  2. Add a hook `useConfirmedAction<T>(invalidate: QueryKey)` returning `{ ask(action), runNow(label, run), dialog: ReactNode, last: { label, data, error } | null }`. It owns the pending state, the mutation, the `ConfirmDialog` and the invalidation.
  3. Add `<LastActionPanel title="Last command" last={…} />`, which picks `OkIcon` or `RefusedIcon` from the error or `accepted` state.
  4. Replace the three blocks and delete the three interfaces. Add one test file for the hook (confirm runs once and invalidates; cancel runs nothing; an error shows in the panel).
- **Effort:** M

#### WEB-07 — Cache and session state can outlive a user switch or a sign-out that races a renewal
- **Severity:** Low
- **Category:** Security
- **Confidence:** Medium
- **Location:** `apps/web/src/session/SessionProvider.tsx:42-46`; `apps/web/src/session/session.ts:150-166` (`exchange`), `:178-191` (`accept`), `:193-198` (`forget`)
- **About the file:** `SessionProvider` bridges `SessionManager` to React and clears the TanStack Query cache.
- **Problem:** (a) The cache is cleared only when the status becomes `signed_out`. The refresh cookie is shared by all tabs. If tab 2 signs out and signs in as user B, tab 1's timer-driven `renew()` exchanges B's cookie and `accept()` moves tab 1 straight from `signed_in(A)` to `signed_in(B)`, with no `signed_out` in between. Tab 1 then shows A's cached queries (users page, audit filter results, alarm details) under B's identity until they refetch. The WebSocket reconnects because the server closes "token belongs to another user", but the REST cache is not reset. (b) There is no guard against a stale renewal: if a renewal is in flight when the user clicks Sign out, `forget()` runs, then the refresh response calls `accept()` and the UI comes back as signed in until the next request fails with `auth.session_invalid`.
- **Recommendation:**
  1. In `SessionProvider`, clear on identity change, not on status. Keep `const identity = state.status === "signed_in" ? state.user.username : null` and call `queryClient.clear()` in an effect keyed on `identity` whenever it changes.
  2. In `SessionManager`, add `private generation = 0`, increment it in `forget()`, capture it at the start of `exchange()`, and skip `accept()` (return `null`) if it changed.
  3. Tests: `accept` of a different username clears the cache (`shell.test.tsx`); `signOut()` during a pending `renew()` leaves `signed_out` (`session.test.ts`).
- **Effort:** S

#### WEB-08 — A 401 that arrives after a finished renewal triggers another refresh
- **Severity:** Low
- **Category:** Performance
- **Confidence:** High
- **Location:** `apps/web/src/api/http.ts:163-175`; `apps/web/src/session/session.ts:76-81`
- **About the file:** `http.ts` is the single HTTP layer (token attach, renew-once, Problem Details decoding).
- **Problem:** Renewal is single-flight only while it is in flight (`renewing` is reset in `finally`). A request sent with the old token whose 401 arrives after the first refresh finished calls `session.renew()` again and starts a second refresh, which rotates the cookie again. It is harmless for correctness (the new cookie is used), but it is wasted round-trips and extra audit rows during a burst, e.g. five queries fired together after the laptop wakes from sleep.
- **Recommendation:**
  1. In `request()`, remember the token actually sent (`const sent = session?.accessToken() ?? null`). On a renewable 401, if `session.accessToken()` is non-null and differs from `sent`, retry with the current token without calling `renew()`.
  2. Test in `http.test.ts`: credentials whose `accessToken()` changes between send and 401 → `renew` not called, and the retry uses the new token.
- **Effort:** S

#### WEB-09 — The confirmation dialog can be dismissed while a command is on its way, and it does not trap focus
- **Severity:** Low
- **Category:** Reliability
- **Confidence:** High
- **Location:** `apps/web/src/components/ConfirmDialog.tsx:30-41`, `:44`, `:57`
- **About the file:** The modal every plant command goes through.
- **Problem:** When `busy`, the buttons are disabled, but Escape (`:33-35`) and a backdrop click (`:44`) still call `onCancel`, which closes the dialog mid-command. It is `aria-modal` but has no focus trap and the background is not `inert`, so a keyboard user can Tab to controls behind the backdrop, e.g. "Emergency stop…" or another "Load…". Pressing Enter there calls `ask()`/`setPending()` and replaces the pending command while the dialog is open.
- **Recommendation:**
  1. Ignore Escape and backdrop clicks while `busy`. Read `busy` through `useEffectEvent`, like `cancel`.
  2. Mark the page behind the dialog `inert` while it is open, or render it with the native `<dialog>` and `showModal()`, which gives inertness and a top layer. The unused `dialog` CSS at `styles.css:808-818` already exists. Keep the same markup and classes so nothing changes visually.
  3. Tests in `ConfirmDialog.test.tsx`: Escape while `busy` does not call `onCancel`; focus stays inside the dialog after Tab.
- **Effort:** S

#### WEB-10 — Live frames invalidate too much: every alarm query per alarm frame, and all queries on every reconnect attempt
- **Severity:** Low
- **Category:** Performance
- **Confidence:** High
- **Location:** `apps/web/src/live/LiveProvider.tsx:37-39`, `:48-50`; `apps/web/src/api/realtime.ts:189`, `:213`
- **About the file:** `LiveProvider` feeds WebSocket frames into `LiveStore` and invalidates REST caches.
- **Problem:** Each `alarms` frame invalidates `queryKeys.alarms.all`, which also refetches the open history page and the alarm detail, once per frame. An alarm flood (a trip raises several conditions at once) gives one refetch burst per change. `resync()` runs `queryClient.invalidateQueries()` on every reconnect attempt (`:213`), so every 10 s during an outage all active queries are refetched, including immutable history ranges marked `staleTime: Infinity`. After a 1013 close it runs twice (`:189` and again in the timer at `:213`).
- **Recommendation:**
  1. Invalidate only `queryKeys.alarms.active` on a frame (the banner, overview and active tab). Invalidate the history and detail keys only if the frame's `alarm.id` matches the selected detail, or leave them to the user.
  2. Coalesce frame invalidations with a microtask or `setTimeout(…, 250)` flag, so a burst causes one refetch.
  3. In `RealtimeClient`, call `resync()` once when the state first becomes `live` after a `reconnecting` period (in the `subscribed` branch), not before each attempt. Remove the duplicate call at `:189`.
  4. Update `shell.test.tsx:125` and `realtime.test.ts:171` to assert the new counts.
- **Effort:** S

#### WEB-11 — Server protocol frames and close code 4400 are ignored silently; a protocol bug reconnects forever
- **Severity:** Low
- **Category:** Logging
- **Confidence:** High
- **Location:** `apps/web/src/api/realtime.ts:12` (unused `CLOSE_BAD_REQUEST`), `:146-178` (`receive`), `:180-192` (`closed`)
- **About the file:** The WebSocket client.
- **Problem:** The gateway sends `{"type":"error","code":…}` (e.g. `ws.unknown_message`), `{"type":"closing",code,reason}` and `{"type":"renewed"}` (`routers/websocket.py:150-157`, `:208-210`, `:283-287`). `receive()` drops them, and malformed JSON is dropped by an empty `catch` at `:153`. A 4400 close (the client sent a malformed frame) is treated like a network drop and reconnects with backoff forever. A contract drift between console and gateway would show only as a permanently "Reconnecting…" badge, with nothing in the browser console.
- **Recommendation:**
  1. In `receive()`, handle `type === "error"` and `type === "closing"` with one `console.error("realtime:", frame.code ?? frame.reason)` each. This is a real diagnostic, not debug output.
  2. In `closed()`, treat `CLOSE_BAD_REQUEST` as fatal: `console.error` once and `stop()`. Otherwise delete the constant.
  3. Tests in `realtime.test.ts`: an `error` frame is logged (spy on `console.error`); a 4400 close ends in `stopped` without scheduling a timer.
- **Effort:** S

#### WEB-12 — Gateway limits and shapes are hand-copied into the console and can drift
- **Severity:** Low
- **Category:** Modularity
- **Confidence:** High
- **Location:** `apps/web/src/screens/ControlScreen.tsx:48-54` (`LIMITS`); `apps/web/src/screens/EngineerScreen.tsx:40-42`, `:61-93`; `apps/web/src/screens/UsersScreen.tsx:25-26`; `apps/web/src/api/types.ts:8` (`Role`), `:27` (`HistoryMeasurement`); role list declared twice, `apps/web/src/api/endpoints.ts:274` and `apps/web/src/session/session.ts:25`
- **About the file:** `types.ts` is meant to alias only generated gateway shapes.
- **Problem:** The project rule is "the console never declares a gateway shape by hand". `Role` is spelled out even though `components["schemas"]["UserCreateRequest"]["role"]` in `schema.gen.ts` already carries the union. The command bounds (`50e5…185e5` Pa, `400…848` K, steps `1…3600`, ramp `≤3600`), the password length and the username pattern are copied from Python schemas. They are present in `shared/openapi/api-gateway.json` as `minimum`/`maximum`/`minLength`/`pattern`, but nothing checks the copies. The existing "limits" test compares against hard-coded numbers (`ControlScreen.test.tsx:9-16`), so it would pass after a gateway change. The WebSocket frame types (`types.ts:58-93`) are hand-written by necessity (no schema), which the file already documents.
- **Recommendation:**
  1. Replace `export type Role = …` with `export type Role = Schemas["UserCreateRequest"]["role"]`. Keep one `ROLES` array (in `session/roles.ts`, next to `LEVEL`) and import it in `session.ts` and `UsersScreen.tsx`. Remove it from `endpoints.ts`.
  2. Add `apps/web/src/api/contract.test.ts`. It imports `../../../../shared/openapi/api-gateway.json` and asserts `barToPascals(LIMITS.pressureBar[0]) === SetpointRequest.properties.pressure_pa.minimum` and the same for every bound, `STEPS_MAX`, `FAULT_RAMP_MAX_S`, `PASSWORD_MIN_LENGTH` and `USERNAME_PATTERN.source`.
  3. Derive `LIMITS.steamTempC` from `kelvinToCelsius(400)` and `kelvinToCelsius(848)` instead of the literals `126.85`/`574.85`.
- **Effort:** S

#### WEB-13 — Unit conversions done outside `units.ts`, and a repeated null-guard pattern
- **Severity:** Low
- **Category:** Readability
- **Confidence:** High
- **Location:** `apps/web/src/screens/ControlScreen.tsx:239` (`/ 100`); `apps/web/src/screens/EngineerScreen.tsx:375` (`fault.intensity * 100`); `apps/web/src/screens/ControlScreen.tsx:52` (pre-converted °C literals); `apps/web/src/screens/TrendsScreen.tsx:65-106`
- **About the file:** `units.ts` is declared "the one place" for SI↔display conversion.
- **Problem:** Percent→fraction for the valve command and fraction→percent for the fault intensity are written inline. There is no `percentToFraction` in `units.ts`, and `fractionToPercent` exists but is not used at `:375`. `KpiPanel` repeats `x === null ? null : convert(x)` six times inside template literals of about 120 characters, which is hard to read and easy to get wrong.
- **Recommendation:**
  1. Add `percentToFraction` to `units.ts` with a test in `units.test.ts`, and use it at `ControlScreen.tsx:239`. Use `fractionToPercent` at `EngineerScreen.tsx:375`.
  2. Add `formatConverted(value: number | null | undefined, convert: (v: number) => number, digits: number): string` to `units.ts` and use it in `KpiPanel`, one line per item.
- **Effort:** S

#### WEB-14 — Dead code: unused endpoint wrappers, types, a constant and stylesheet rules
- **Severity:** Low
- **Category:** Readability
- **Confidence:** High
- **Location:** `apps/web/src/api/endpoints.ts:54-56` (`fetchProfile`), `:58-65` (`changePassword`), `:69-75` (`fetchPlant`, `fetchPlcStatus`), `:169-171` (`fetchSimulation`); `apps/web/src/api/types.ts:16` (`Fault`), `:52` (`Readiness`); `apps/web/src/api/realtime.ts:12`; `apps/web/src/styles.css:808-818` (`dialog`), `:866-873` (`.visually-hidden`), `textarea` selectors at `:123-126`, `:197-211`; redundant `.mimic .pipe.in-alarm-*` at `:736-746` (already covered by `.mimic .in-alarm-*`)
- **About the file:** The client layer and the single stylesheet.
- **Problem:** These exports are referenced only by `endpoints.test.ts` (the "covers every exported request function" test keeps them alive). `changePassword` is the risky one: it returns a new session and closes every other session, including the caller's. A future caller that forgets to feed the result to `SessionManager.accept` signs the user out. No `<dialog>`, `<textarea>` or `.visually-hidden` element exists in `src/`.
- **Recommendation:**
  1. Delete the unused wrappers and types, and update `endpoints.test.ts:307-338`. If `changePassword` is kept for a planned screen, move it into `SessionManager` as `changePassword()` that calls `accept()`. Otherwise remove it (a new screen is an owner decision).
  2. Delete the dead CSS rules. Keep `dialog` only if WEB-09 adopts the native element.
- **Effort:** S

#### WEB-15 — Raw exception text can reach the operator
- **Severity:** Low
- **Category:** Reliability
- **Confidence:** High
- **Location:** `apps/web/src/api/http.ts:189`, `:198-200`
- **About the file:** `describeError` is documented as "A sentence for the operator; never the raw exception text."
- **Problem:** `describeError` returns `error.message` for any non-`ApiError`. `request()` calls `response.json()` on a 2xx without a guard. A 200 with an HTML body (a mis-routed path answered by nginx's `try_files … /index.html`, or a captive proxy) throws `SyntaxError: Unexpected token '<'…`, and that text is shown in `ErrorOf`. Server-side 500s are already generic (`problems.py:146`), so there is no leak of internals, but the operator sees parser output.
- **Recommendation:**
  1. Wrap the `response.json()` at `:189` in try/catch and throw `new ApiError({status: response.status, code: "response.malformed", title: "Bad response", detail: "The gateway sent an answer the console cannot read.", errors: [], retryAfterS: null})`.
  2. In `describeError`, return a fixed sentence for non-`ApiError` values ("Something went wrong in the console.") and `console.error(error)` once, so the detail stays available to a developer.
  3. Tests in `http.test.ts`: a 200 with `"<html>"` rejects with `response.malformed`; `describeError(new TypeError("x"))` does not contain `"x"`.
- **Effort:** S

#### WEB-16 — The horn's AudioContext is never closed; each sign-in creates another
- **Severity:** Low
- **Category:** Reliability
- **Confidence:** High
- **Location:** `apps/web/src/alarms/horn.ts:7`, `:25-30`, `:38`; `apps/web/src/components/AlarmBanner.tsx:20`, `:35-40`
- **About the file:** `Horn` beeps with Web Audio every 2 s while an unacknowledged critical alarm stands. `AlarmBanner` owns one per mounted shell.
- **Problem:** `stop()` clears the interval but keeps the `AudioContext`. `AlarmBanner` remounts on every sign-in (the whole `SignedIn` tree is recreated), so a long-running console creates a new, never-closed context per session. Browsers cap or warn about live audio contexts, and each one holds an audio thread.
- **Recommendation:**
  1. Add `dispose(): void { this.stop(); void this.context?.close(); this.context = null; }` to `Horn`.
  2. Call `horn.dispose()` in the unmount effect of `AlarmBanner` (`:35-40`).
  3. Test (next to the existing horn test): `dispose()` calls `close()` on the fake context.
- **Effort:** S

#### WEB-17 — Untested branches in the client layer (HTTP, realtime, session)
- **Severity:** Low
- **Category:** Tests
- **Confidence:** High
- **Location:** `apps/web/src/api/http.test.ts`, `apps/web/src/api/realtime.test.ts`, `apps/web/src/session/session.test.ts`
- **About the file:** The unit tests of the three modules that own authentication and the live channel.
- **Problem:** Well-tested happy paths, but these branches have no test: `http.ts:171-174` (`renew()` → `null` rethrows the original 401), `:176-180` (a second 401 after renewal ends the session), `:186-188` (204 → `undefined`), `:101-104` (an aborted signal rethrows the `AbortError`, not `network.unreachable`), `auth.token_missing` renewal. In realtime: `connect()` with `token() === null` (`:121-124`), `stop()` while `unauthorized()` is pending (`:196-199`, the `running` check), `renew()` while the socket is not open (no-op), a close event from a replaced socket being ignored (`:136-139`). In session: the renewal timer firing calls `renew()`, `detach()` clears the timer, and the superseded retry failing twice.
- **Recommendation:** Add one `it(...)` per branch above to the matching file, using the existing `credentials()`, `FakeSocket` and `ManualTimers` helpers. No new infrastructure is needed.
- **Effort:** M

#### WEB-18 — Small state-display bugs on the screens
- **Severity:** Low
- **Category:** Reliability
- **Confidence:** High
- **Location:** `apps/web/src/screens/AlarmsScreen.tsx:169`, `:191`; `apps/web/src/components/TrendChart.tsx:33-34`, `:49`, `:67`; `apps/web/src/screens/UsersScreen.tsx:114`, `:137`; `apps/web/src/screens/ControlScreen.tsx:433-436`
- **About the file:** Alarm list, trend chart, user row and control screen.
- **Problem:** (a) `failure = acknowledge.error ?? acknowledgeAll.error`: an old single-ack error keeps showing after a successful "Acknowledge all", and the ack-all result is never shown (`:191` shows only `acknowledge.data`). (b) `TrendChart` reads `--text`, `--border` and `--accent` once when it is created, and its effect depends only on `[parameter, syncKey]`, so after the theme toggle the axes and line keep the old theme's colours until the screen is reopened. (c) A user with `role: null` initialises the select to `"viewer"`, so `role !== user.role` is true and "Apply…" appears without any change. (d) The Control screen uses the last `live.plc` even when `live.connection` is `reconnecting` or `stopped`. It does not say the state may be stale, and the enabled/disabled states follow stale data. The PLC still enforces everything; showing a staleness note would be a UI change, so that part is an owner decision.
- **Recommendation:**
  1. (a) Keep one "last acknowledgement" state (`{data, error}`) set in each mutation's `onSettled`, and render that.
  2. (b) Pass the resolved theme (from `Layout`'s theme state, via a small context or a `data-theme` read) into the `TrendChart` effect dependencies so the chart is rebuilt on change.
  3. (c) Initialise with `useState<Role | "">(user.role ?? "")`, add an empty option, and show "Apply…" only when a role is chosen.
  4. (d) Owner decision: disable "Send…/Set…/Reset…" while `live.connection !== "live"`, or show a note.
- **Effort:** S

#### WEB-19 — The console does not need `style-src 'unsafe-inline'`; the CSP could be tightened
- **Severity:** Info
- **Category:** Security
- **Confidence:** Medium
- **Location:** `infrastructure/nginx/headers.inc:5` (see also section 7.8); console side: `apps/web/src/components/Mimic.tsx:107-109`, `:359`, `apps/web/index.html:1-15`
- **About the file:** `headers.inc` sets the CSP for the console. `index.html` is the Vite entry.
- **Problem:** `index.html` has no inline script or style. The only inline styles in the app are React `style={…}` props (applied through CSSOM) and uPlot's `el.style[...] =` writes (checked in `node_modules/uplot/dist/uPlot.esm.js:98`; uPlot uses only `textContent`, never `innerHTML`). CSP `style-src` does not govern CSSOM writes, and the production build extracts CSS into files. So `'unsafe-inline'` is probably unnecessary for the console. Dropping it removes a CSS-injection vector.
- **Recommendation:**
  1. The nginx owner tries `style-src 'self'` and runs `console-e2e` against the stack, watching for CSP violations in the browser console (add a Playwright `page.on("console")` check for `Content Security Policy` messages).
  2. Keep `'unsafe-inline'` only if a violation appears, and name its source.
- **Effort:** S

#### WEB-20 — Playwright failure traces contain the demo passwords, and the readme config uses `localhost`
- **Severity:** Info
- **Category:** Security
- **Confidence:** High
- **Location:** `apps/web/playwright.config.ts:25`; `apps/web/e2e/support.ts:39`; `.github/workflows/ci.yml:161-168`; `apps/web/playwright.readme.config.ts:6`
- **About the file:** The e2e configuration and sign-in helper.
- **Problem:** `trace: "retain-on-failure"` records the value of every `fill()`, including `demoPassword(role)`. CI uploads `apps/web/e2e-results/` as an artifact. The CI secrets are throwaway (`dev-secrets` per run) and local `e2e-results` is gitignored, so the impact is minimal. It does contradict "the checks never print them" (`14-command-reference.md`). Separately, the readme config defaults to `http://localhost:8080`, while the platform note requires IPv4 loopback (`127.0.0.1`), as `playwright.config.ts:6` uses.
- **Recommendation:**
  1. Document in `e2e/support.ts` that traces contain the demo passwords. Or sign in through `page.request.post("/auth/login")` inside a `test.step` with tracing paused (`context.tracing` group) if the owner wants them excluded.
  2. Change `playwright.readme.config.ts:6` to `http://127.0.0.1:8080`.
- **Effort:** S

### 7.8 Platform — Docker, Compose, nginx, Mosquitto, Grafana, CI, supply chain

**Area summary.** The platform layer is disciplined for a portfolio project. Every published port binds 127.0.0.1, the gRPC ports are never published, every secret is interpolated with `:?` so no service starts on a known default, the Python runtime images are multi-stage, non-root and contain neither pip nor uv, the broker refuses anonymous clients and has a per-account ACL, and CI runs the gate verbatim with least-privilege `permissions:` and a Trivy gate before publishing. The weakest points are these. (1) The Docker build context and layer order break the "demos without internet" invariant after any local run, and they send database backups into the build context. (2) One all-powerful InfluxDB operator token is shared by three consumers. (3) Containers have no runtime hardening at all (capabilities, no-new-privileges, read-only filesystems, limits). (4) The supply chain uses mutable tags everywhere, and CI publishes a rebuilt image rather than the one it scanned. Since Q7 the GHCR packages are public, so I rate severity for **"someone deploys the published images"** unless a finding says otherwise. Locally, most items drop by one level.

<details><summary>Scope reviewed by the area auditor</summary>

`Dockerfile`, `apps/web/Dockerfile`, `.dockerignore`, `apps/web/.dockerignore`, `docker-compose.yml`, `.env.example`, `infrastructure/nginx/{default.conf,site.inc,proxy.inc,headers.inc,40-cogniboiler-tls.sh}`, `infrastructure/docker/mosquitto/{mosquitto.conf,acl,start-broker.sh}`, `infrastructure/grafana/provisioning/**`, `infrastructure/grafana/dashboards/*.json` (skimmed: queries, templating, credentials), `infrastructure/prometheus/prometheus.yml`, `.github/workflows/ci.yml`, `.github/PULL_REQUEST_TEMPLATE.md`, `.github/ISSUE_TEMPLATE/*`, root `pyproject.toml`, every `apps/*/pyproject.toml` and `shared/*/pyproject.toml`, `apps/web/package.json`, `.pre-commit-config.yaml`, `.gitignore`, `apps/web/.gitignore`, `.gitattributes`, `.claude/settings.json`, `apps/infrastructure/**`, `apps/ai-predictor/**`, `Makefile`, `SECURITY.md`, `CONTRIBUTING.md`, `CHANGELOG.md`, `scripts/{stack,dev_secrets,backup,restore,audit_deps}/__main__.py` and their `config/*.json` (non-secret parts), `scripts/stack/tests/test_compose.py`, `scripts/quality_gate/config/steps.json`; for cross-checks `.ai/project/14-command-reference.md`, `docs/architecture/overview.md`, `docs/architecture/invariants.md`, `docs/__arch__/open-questions.md` (decisions of 2026-09-18, Q7). `.env` and `certs/` were not opened. `audit-deps` was **not run**: `scripts/audit_deps/__main__.py:76-80` installs pip-audit through `uvx` (a change to the uv tool cache), and `:144-145` writes reports into `audit-reports/` inside the repository. Both break the read-only rule.

</details>

#### PLAT-01 — `stack up` needs the internet after any local run; backups and logs enter the build context
- **Severity:** Medium
- **Category:** Reliability
- **Confidence:** High
- **Location:** `.dockerignore:1-27`, `Dockerfile:16-20` (and `:22-35`), `scripts/stack/__main__.py:111-112`, `docker-compose.yml:28-29`, `docs/architecture/invariants.md` (I8)
- **About the file:** `.dockerignore` limits the root build context used by all six Python service images. The `Dockerfile` builds one uv environment per service.
- **Problem:** Three things combine. (a) `.dockerignore` does not exclude `logs/` or `backups/`. The running stack keeps writing `logs/` through the bind mount at `docker-compose.yml:29`, and `backups/<stamp>/` holds `postgres.sql` plus an InfluxDB backup. (b) The build does `COPY . .` (`Dockerfile:17`) *before* `uv sync --frozen` (`:20` … `:35`), with no uv cache mount. Any change anywhere in the context therefore invalidates all six `uv sync` layers and makes them download every wheel from PyPI again. (c) `stack up` always adds `--build` (`scripts/stack/__main__.py:111-112`). The result: once the stack has written one log line, the next `stack up` on a machine without internet fails in `uv sync`. That breaks invariant I8 ("runs without internet access once images are built"). The demo-day scenario is the owner running the demo, then restarting it offline. It also makes every local rebuild a full re-download (about six minutes plus bandwidth). Separately, each build copies the DB dump (Argon2 hashes, sessions, audit log) and the InfluxDB backup into the builder's context and the `build` stage cache. They never reach the final images, because the runtime stages copy only `.venv`, `shared/generated` and the migrations.
- **Recommendation:**
  1. Add to `.dockerignore`: `logs`, `backups`, `audit-reports`, `image-scan`, `.env.*`, `**/.env`, `apps/infrastructure`, `scripts`, `task-checklist.md` (already present), `*.log`.
  2. Split the build stage so dependencies are a separate, stable layer. Copy only `pyproject.toml`, `uv.lock`, `apps/*/pyproject.toml` and `shared/*/pyproject.toml` first, run `uv sync --frozen --no-dev --no-install-workspace --package <svc>`, then `COPY` the sources and run the existing `uv sync --frozen --no-dev --no-editable --package <svc>`. Add `RUN --mount=type=cache,target=/root/.cache/uv` to both sync steps. This is the documented uv Docker pattern.
  3. Verify: `stack up`, run `demo`, disconnect the network, `stack down`, `stack up`. Every service must reach healthy, and the build log should show `CACHED` for the dependency layers.
- **Effort:** M

#### PLAT-02 — One InfluxDB operator token for the historian, the gateway and Grafana
- **Severity:** Medium
- **Category:** Security
- **Confidence:** High
- **Location:** `docker-compose.yml:74`, `:117`, `:198`, `:286`; `infrastructure/grafana/provisioning/datasources/influxdb.yaml:22-23`
- **About the file:** Compose wires InfluxDB credentials into each consumer. The Grafana datasource provisioning gives Grafana its query credentials.
- **Problem:** `DOCKER_INFLUXDB_INIT_ADMIN_TOKEN` is the *operator* token. It has all rights over all orgs: create or delete buckets, users and tokens, and read and write everything. The same value goes to the historian (`INFLUXDB_TOKEN`, `:198`), the gateway (`INFLUX_TOKEN`, `:286`) and Grafana (`:117`, `influxdb.yaml:23`). The gateway only queries (`apps/api-gateway/src/api_gateway/clients.py:440-450`), and Grafana only queries. A compromise of the internet-facing gateway process therefore gives full control of the time-series store. So does an injected Flux query: the dashboards interpolate `${bucket}` from a URL-settable variable, `alarms.json:48`, and Flux has `to()`, `buckets()` and HTTP functions. The attacker could delete buckets, mint tokens or rewrite history. This breaks the project's own least-privilege rule ("each service has its own PostgreSQL role and MQTT account"), which InfluxDB does not follow. It also makes backup restores fragile (`14-command-reference.md`: "the backup's tokens must match the `.env`").
- **Recommendation:**
  1. Keep the operator token only in the `influxdb` service and in `backup`/`restore`, which already run inside that container.
  2. Create two scoped tokens. The historian needs read/write on the org's buckets plus tasks, because it creates `sensors_1m` and the downsampling task in `apps/historian/src/historian/storage.py:89-114`. The gateway and Grafana need read-only on the two buckets. With the official image this is a one-shot init script under `/docker-entrypoint-initdb.d/` that runs `influx auth create --read-bucket … --write-bucket …` using token values from `.env`. Add `INFLUXDB_HISTORIAN_TOKEN` and `INFLUXDB_READ_TOKEN` to `.env.example` and `scripts/dev_secrets/config/secrets.json` as `token` kinds.
  3. Point `docker-compose.yml:198`, `:286`, `:117` and `influxdb.yaml:23` at the scoped tokens.
  4. Verify with `stack up`, `smoke`, and the Grafana dashboards showing data. From the gateway container, a write with its token must return 403.
- **Effort:** M

#### PLAT-03 — `/docs` and `/redoc` load a floating-version CDN script, with no SRI, on the console's origin
- **Severity:** Medium
- **Category:** Security
- **Confidence:** Medium
- **Location:** `infrastructure/nginx/site.inc:28-34`; `apps/api-gateway/src/api_gateway/main.py:135-136`; FastAPI defaults `swagger-ui-dist@5` / `redoc@2` on `cdn.jsdelivr.net` (`.venv/Lib/site-packages/fastapi/openapi/docs.py:79,236`)
- **About the file:** `site.inc` holds the nginx locations shared by the HTTP (8080) and HTTPS (8443) servers.
- **Problem:** nginx publishes `/docs`, `/redoc` and `/openapi.json` on the same origin as the console. The CSP allows `script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net`. FastAPI's default pages load `swagger-ui-dist@5` and `redoc@2`: *major-version ranges* with no Subresource Integrity. Any new 5.x release (for example, after an npm account takeover) runs on the console origin with no code change here. From that page a script can `fetch('/auth/refresh', {method:'POST', credentials:'include'})`. The refresh cookie is `SameSite=Strict` and scoped to `/auth`, but it is same-origin, and the refresh response carries the tokens in its body (accepted item Q1). The script then holds a valid session of whoever opened `/docs`, admin included. A separate issue: `/docs` does not work offline, which conflicts with "demos without internet".
- **Recommendation:**
  1. Simplest: stop proxying `/docs`, `/redoc` and `/openapi.json` through nginx (delete `site.inc:28-34`). The contract is already committed as `shared/openapi/api-gateway.json`. Developers can use a host-run gateway at :8000.
  2. If the docs must stay: in `create_app` set `docs_url=None` and `redoc_url=None`, and serve Swagger UI through `get_swagger_ui_html(swagger_js_url=…, swagger_css_url=…)` pinned to an exact version (`swagger-ui-dist@5.x.y`). Better still, vendor the files into the image so the page works offline. Then drop `https://cdn.jsdelivr.net` from the CSP.
  3. Verify: `curl -I https://localhost:8443/docs` shows the new policy, and the console-e2e run stays green.
- **Effort:** S

#### PLAT-04 — No runtime hardening on any container
- **Severity:** Medium
- **Category:** Security
- **Confidence:** High
- **Location:** `docker-compose.yml:21-29` (the `x-service` / `x-python-service` anchors), each infra service `:34-107`, `:334-349`
- **About the file:** Compose defines how every container runs.
- **Problem:** No service sets `cap_drop`, `security_opt: ["no-new-privileges:true"]`, `read_only`, `pids_limit`, or memory/CPU limits. The Python images run as uid 10001 (good), but they keep Docker's default capability set (`NET_RAW`, `CHOWN`, `SETUID`, etc.). The infra containers (Postgres, InfluxDB, Mosquitto's start script, which runs as root: `infrastructure/docker/mosquitto/start-broker.sh:3`) are unconstrained. A runaway process can exhaust host memory, or a fork loop the pid table, and take down the whole demo machine. A leak in the physics runtime, or a WebSocket flood on the gateway, are plausible triggers. An RCE in the gateway, which faces the internet in a deployment, gets more kernel surface than it needs.
- **Recommendation:**
  1. Add to the `x-service` anchor (`docker-compose.yml:21`):
     ```yaml
     security_opt: ["no-new-privileges:true"]
     cap_drop: [ALL]
     pids_limit: 256
     read_only: true
     tmpfs: ["/tmp"]
     ```
     Python services write only `/tmp` (liveness file) and `/app/logs` (bind mount), so `read_only` fits. Check `opcua-server` for a temporary certificate directory first.
  2. Add `mem_limit` per service (for example 512m for Python services, 1g for InfluxDB and Postgres) through a second anchor.
  3. `web`: `cap_drop: [ALL]` works for nginx-unprivileged on 8080/8443. `read_only` needs `tmpfs` for `/tmp`, `/var/cache/nginx` and `/etc/nginx/conf.d`, because `40-cogniboiler-tls.sh:16` writes `tls.conf` there.
  4. Infra images: add `no-new-privileges` and limits. Apply `cap_drop: [ALL]` with the minimal `cap_add` each needs (Postgres and Mosquitto need `CHOWN`, `SETUID`, `SETGID`, `DAC_OVERRIDE`, `FOWNER` at start), verified one at a time.
  5. Verify with `stack up` (all healthy), `smoke`, `console-e2e`, and `docker inspect --format '{{.HostConfig.CapDrop}}'`.
- **Effort:** M

#### PLAT-05 — CI publishes a rebuilt image, not the one Trivy scanned; third-party images are never scanned
- **Severity:** Medium
- **Category:** Supply chain
- **Confidence:** High
- **Location:** `.github/workflows/ci.yml:171-184` (scan in `stack`), `:241-267` (rebuild in `publish` with `pull: true`, `no-cache-filters: runtime`)
- **About the file:** The single CI workflow: gate, audit, stack (build, e2e, Trivy), publish, release.
- **Problem:** The `stack` job builds `cogniboiler/<svc>:dev` through Compose and scans those. The `publish` job then builds from scratch on another runner, pulling fresh base images and deliberately rebuilding `runtime` without cache. The pushed digest is therefore not the scanned digest. It may contain a base-image update published between the two jobs, or a hijacked tag (see PLAT-06), and nothing scanned it. The decision of 2026-09-18 ("a fixable HIGH/CRITICAL … never pushed") holds only by timing. The Trivy loop (`:178`) also covers only the seven project images. The demo runs `grafana/grafana:10.4.0`, `influxdb:2.7`, `postgres:16-alpine`, `eclipse-mosquitto:2.0` and `prom/prometheus:v3.8.0` unscanned, and the release instructions tell users to run them.
- **Recommendation:**
  1. In `publish`, build with `push: false` and `load: true`, run the same Trivy command against the local tag, then push. An alternative is to push by digest and scan `ghcr.io/…@sha256:…` before adding the tags. Either way, the scanned and pushed artifacts become identical.
  2. Add a non-blocking (report-only, `--exit-code 0`) Trivy pass over the five third-party images in `stack`, kept as an artifact, so a stale Grafana (PLAT-07) becomes visible.
  3. Verify in a CI run on a branch with `publish` temporarily enabled by a `workflow_dispatch` dry run.
- **Effort:** M

#### PLAT-06 — Mutable tags across the supply chain: GitHub Actions, Trivy with the Docker socket, base images, and uv itself
- **Severity:** Medium
- **Category:** Supply chain
- **Confidence:** High
- **Location:** `.github/workflows/ci.yml:32,36,43,47,94,218,220,231,243` (actions by tag), `:179-181` (`aquasec/trivy:0.74.0` with `/var/run/docker.sock`); `Dockerfile:9-10`; `apps/web/Dockerfile:5,14`; `docker-compose.yml:36,64,88,107,336`
- **About the file:** CI workflow, image definitions, Compose.
- **Problem:** Every action is referenced by a movable tag (`actions/checkout@v7`, `astral-sh/setup-uv@v10.1.0`, `pnpm/action-setup@v6`, `docker/*@v4…v7`). The `publish` job holds `packages: write`, so a re-pointed tag in `docker/build-push-action` or `docker/login-action` could push arbitrary images under the project's name. The Trivy step runs a tag-pinned third-party image with the host Docker socket mounted. That is root on the runner, able to rewrite the uv and pnpm caches that `setup-uv`/`setup-node` save at job end for later runs on `main` (cache poisoning). Base images are tags, not digests. `ghcr.io/astral-sh/uv:python3.14-bookworm-slim` (`Dockerfile:9`) doesn't even pin the uv *version*, so each build can resolve and install with a different uv. Reproducibility ("images are reproducible, pinned base image tags") holds only loosely.
- **Recommendation:**
  1. Pin every `uses:` to a full commit SHA with the tag as a comment (`actions/checkout@<sha> # v7.x.y`). Add Dependabot or Renovate for `github-actions` and `docker` so pins stay fresh. This is config only, not a feature.
  2. Pin the uv image to a version and digest: `ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.x.y-python3.14-bookworm-slim@sha256:…`. Pin `PYTHON_IMAGE`, `node:24-alpine` and `nginxinc/nginx-unprivileged:1.30-alpine` by digest too. The Trivy gate plus Dependabot keep them current, which matches the "rebuild on new CVE" decision.
  3. Pin `aquasec/trivy` by digest. Prefer `aquasecurity/trivy-action@<sha>` or scanning image tarballs (`docker save` then `trivy image --input`) over mounting the socket.
  4. Pin the infra images in Compose by digest, or at least by patch (`influxdb:2.7.x`, `eclipse-mosquitto:2.0.x`, `postgres:16.x-alpine`).
  5. Optional cheap check: run `zizmor` or `actionlint` in the `audit` job.
- **Effort:** M

#### PLAT-07 — Grafana 10.4.0 is outdated, phones home, and its provisioned datasource is editable
- **Severity:** Medium (deployment) / Low (local)
- **Category:** Security
- **Confidence:** Medium
- **Location:** `docker-compose.yml:107-117`; `infrastructure/grafana/provisioning/datasources/influxdb.yaml:21,24`; `infrastructure/grafana/provisioning/datasources/prometheus.yaml:18`; `infrastructure/grafana/provisioning/dashboards/dashboards.yaml:8-9`
- **About the file:** The Grafana service and its provisioning.
- **Problem:** `grafana/grafana:10.4.0` is a March 2024 release of a line that no longer gets security fixes, and PLAT-05 means it is never scanned. The service sets only `GF_USERS_ALLOW_SIGN_UP=false`. It relies on defaults for anonymous access (off, but implicit), and leaves analytics reporting, update checks and the news feed on. Those are outbound calls from an "offline" demo. The InfluxDB datasource carries the operator token (PLAT-02) and is `editable: true`. Any Grafana editor can therefore repoint it or reuse it for arbitrary Flux, and dashboards are `editable: true` with `disableDeletion: false`. `tlsSkipVerify: true` (`influxdb.yaml:21`) is meaningless on an `http://` URL, and it becomes dangerous if someone switches to https.
- **Recommendation:**
  1. Move to a supported Grafana tag (current 12.x, pinned per PLAT-06). Verify the four dashboards load, and that the Flux `${bucket}` variable still resolves.
  2. Add explicit env: `GF_AUTH_ANONYMOUS_ENABLED: "false"`, `GF_ANALYTICS_REPORTING_ENABLED: "false"`, `GF_ANALYTICS_CHECK_FOR_UPDATES: "false"`, `GF_NEWS_NEWS_FEED_ENABLED: "false"`, `GF_SECURITY_DISABLE_GRAVATAR: "true"`, `GF_SECURITY_COOKIE_SAMESITE: strict`.
  3. Set `editable: false` on both datasources, `allowUiUpdates: false` on the dashboard provider, and delete `tlsSkipVerify: true`.
- **Effort:** S

#### PLAT-08 — Backups hold the InfluxDB tokens, contrary to their docstring, and secret files get default permissions on POSIX
- **Severity:** Low (local, Windows) / Medium (shared Linux host)
- **Category:** Security
- **Confidence:** Medium
- **Location:** `scripts/backup/__main__.py:5-6`, `:115`, `:137`, `:180`; `scripts/dev_secrets/__main__.py:187-189`; `scripts/stack/__main__.py:45-48`
- **About the file:** `backup` writes DB dumps into `backups/<UTC stamp>/`. `dev-secrets` writes `.env`. `stack` prepares `logs/`.
- **Problem:** The backup docstring says no token is "written into the backup folder". But `influx backup` copies InfluxDB's metadata store, which holds the API tokens (not hashed unless hashed tokens are enabled), including the operator token. `14-command-reference.md` itself says "the backup's tokens must match the `.env`". `postgres.sql` holds password hashes and session rows. Both are created with the process umask (typically `0644` on Linux), as is `.env` with the JWT private key and TLS keys (`write_text`, `dev_secrets/__main__.py:188`). `stack` makes `logs/` world-writable (`0o777`, `stack/__main__.py:48`). On a shared Linux host, any local user can read every secret and plant files in `logs/`. PLAT-01 also sends `backups/` into the Docker build context.
- **Recommendation:**
  1. Correct the docstring at `backup/__main__.py:5-6`: the folder contains credentials; treat it like `.env`.
  2. On POSIX, create the backup folder with `target.mkdir(mode=0o700, …)` and `os.chmod(dump, 0o600)` after writing. In `dev-secrets`, write the temp file through `os.open(path, O_WRONLY|O_CREAT|O_TRUNC, 0o600)` before `replace`.
  3. Replace `chmod(0o777)` with `chmod(0o1777)` (sticky bit), or with `chown` to uid 10001 when running as root. Or document `0o770` with a group, and adjust `scripts/stack/tests/test_compose.py:76-77` to match.
  4. Add unit tests asserting the modes on POSIX (skipped on Windows).
- **Effort:** S

#### PLAT-09 — `.claude/settings.json` auto-allows a command that prints every secret, and the deny list has easy bypasses
- **Severity:** Low
- **Category:** Security
- **Confidence:** High
- **Location:** `.claude/settings.json:30`, `:38-52`
- **About the file:** The shared Claude Code permission lists (mirrored by project rules in `12-domain-rules.md`).
- **Problem:** `Bash(docker compose config:*)` is allowlisted. Without `--no-interpolate`, that command renders every `.env` value (DB passwords, JWT private key, TLS keys, MQTT passwords) into the assistant's transcript, and possibly into logs and remote model context, without a prompt. This conflicts with the project's secrets rule. The deny entries match prefixes only, so these all pass: `docker compose --profile full down -v` (profile flag before `down`), `git push origin +main` and `git push origin main -f`, `git clean -xdf` / `git clean -d -f`, and `git checkout -- .` / `git restore .`.
- **Recommendation:**
  1. Replace line 30 with `Bash(docker compose config --no-interpolate:*)` and `Bash(docker compose config --services:*)`.
  2. Add deny entries: `Bash(docker compose * down -v*)`, `Bash(docker compose * down --volumes*)`, `Bash(git push * +*)`, `Bash(git push * -f*)`, `Bash(git push * --force*)`, `Bash(git clean -*f*)`, `Bash(git checkout -- .*)`, `Bash(git restore .*)`.
  3. Mirror in `.codex/` if it has an equivalent. Record in `.ai/CHANGELOG.md`. Removing an allow entry is a tightening, so no owner approval is needed per `08-rules-evolution.md`.
- **Effort:** S

#### PLAT-10 — Mosquitto: ACL broader than the topic contract, unbounded file log, healthcheck password on argv
- **Severity:** Low
- **Category:** Security
- **Confidence:** High
- **Location:** `infrastructure/docker/mosquitto/acl:5-6,9,14,18,21,28`; `infrastructure/docker/mosquitto/mosquitto.conf:6,15-24`; `docker-compose.yml:54,57`
- **About the file:** Broker configuration, the per-account ACL, and its Compose service.
- **Problem:** Line-by-line against the topic table in `14-command-reference.md`:
  - `physics-engine` gets `write sensors/#`, but it publishes exactly `sensors/plant|boiler|turbine|system/heartbeat`. It also has `readwrite status/physics-engine` where only write (its will) is needed.
  - `plc-controller` gets `write alerts/#` for `alerts/warning|critical|snapshot`, again with `readwrite` where write suffices.
  - `alert-manager` `read alerts/#`, `historian` `read sensors/#` and `status/+`, and `opcua-server` `read sensors/#` are wider than needed. That is harmless today, but a new topic under `sensors/` or `alerts/` gets rights silently, with no ACL review, which contradicts "a new topic gets its broker ACL entry".
  - No account has unexpected cross-service rights. The ACL is otherwise correct.

  `log_dest file /mosquitto/log/mosquitto.log` duplicates stdout (which is already rotated by the `docker-logs` anchor) into a volume with no rotation, so it grows without bound at `log_type information` (one line per connect and disconnect). There is no `max_packet_size`, `max_connections` or `max_keepalive`: an authenticated client can send 256 MB packets. The healthcheck passes `-P "$MQTT_MONITOR_PASSWORD"` on the argv, visible to `ps` inside the container. That is low impact because `monitor` reads only `$SYS/#`. The listener binds `0.0.0.0` inside the container, but the host mapping is loopback-only, so that is fine. MQTT without TLS is defence-in-depth only; enabling it is out of scope and an owner decision.
- **Recommendation:**
  1. Replace wildcards with the exact topics from the contract, and `readwrite` with `write` on the two status topics. Test with `smoke`, and watch `stack logs mosquitto` for `denied` lines.
  2. Delete `log_dest file …` (`mosquitto.conf:15`) and the `mosquitto_log` volume (`docker-compose.yml:54,353`), or keep it with `log_type warning`/`error` only.
  3. Add `max_packet_size 65536`, `max_connections 64` and `max_keepalive 120`. Telemetry frames are protobuf and small; confirm the largest payload first.
- **Effort:** S

#### PLAT-11 — nginx: the `/docs` location drops server-level headers; HSTS pinned on `localhost`; default TLS 1.2 ciphers
- **Severity:** Low
- **Category:** Security
- **Confidence:** High (inheritance) / Medium (HSTS effect)
- **Location:** `infrastructure/nginx/site.inc:29-34`, `infrastructure/nginx/headers.inc:1-5`, `infrastructure/nginx/40-cogniboiler-tls.sh:23-25`
- **About the file:** The shared locations, the header set, and the entrypoint hook that adds the 8443 server.
- **Problem:** nginx does not inherit `add_header` into a location that declares its own. The `/docs` location re-adds three headers but loses `Referrer-Policy`, `Permissions-Policy` and, on 8443, `Strict-Transport-Security`. `Strict-Transport-Security: max-age=31536000` on host `localhost` applies to every port of `localhost`. Once a developer trusts the self-signed certificate (the usual way to silence the warning), the browser forces HTTPS for a year on `http://localhost:3000` (Grafana), `:9090` and `:5173` (Vite). Browsers ignore HSTS while the certificate error persists, hence Medium confidence on the effect. `ssl_ciphers` is left at nginx's default `HIGH:!aNULL:!MD5`, which still offers CBC suites under TLS 1.2. Port 8080 stays plain HTTP when TLS is configured. That is fine locally; a deployer gets no redirect.
- **Recommendation:**
  1. In the `/docs` block, add `add_header Referrer-Policy no-referrer always;` and the `Permissions-Policy` line. Better: move the per-location CSP into a `map $uri $csp` in `default.conf`, so `headers.inc` is the only `add_header` site. Or drop the block entirely (PLAT-03).
  2. In `40-cogniboiler-tls.sh`, lower HSTS to `max-age=300`, or omit it when `server_name` is localhost. Deployers behind a real hostname can raise it.
  3. Add `ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384:ECDHE-ECDSA-CHACHA20-POLY1305:ECDHE-RSA-CHACHA20-POLY1305;` and `ssl_session_tickets off;` (Mozilla intermediate).
  4. Verify with `curl -skI https://localhost:8443/docs` and `curl -sI http://localhost:8080/`.
- **Effort:** S

#### PLAT-12 — Unused production dependencies inflate images and attack surface
- **Severity:** Low
- **Category:** Supply chain
- **Confidence:** High
- **Location:** `apps/api-gateway/pyproject.toml:14` (`bcrypt`), `:15` (`cryptography`, only transitive through `pyjwt[crypto]`), `:18` (`structlog`); `apps/alert-manager/pyproject.toml:11` (`alembic`), `:13-14` (`pydantic`, `structlog`); `apps/plc-controller/pyproject.toml:9` (`numpy`), `:10` (`pydantic`), `:11` (`pyyaml`), `:12` (`structlog`); `pydantic` and `structlog` in `apps/historian`, `apps/opcua-server`, `apps/physics-engine`
- **About the file:** The per-service manifests. Each `uv sync --package <svc>` in the `Dockerfile` installs exactly these into that image.
- **Problem:** A grep over `apps/*/src` finds no import of `bcrypt` (passwords are Argon2id through `pwdlib[argon2]`), none of `alembic` in alert-manager (the Alembic chain lives in the gateway), and none of `yaml` or `numpy` in plc-controller. `structlog` is imported only by `shared/observability`, which declares it. `pydantic` is imported only by the gateway. Every unused package is shipped, scanned, and a potential Trivy/pip-audit blocker. numpy alone adds about 25 MB to the PLC image.
- **Recommendation:**
  1. Remove `bcrypt` from api-gateway, `alembic` from alert-manager, and `numpy` and `pyyaml` from plc-controller. Remove the direct `structlog` and `pydantic` entries where the service never imports them (keep `pydantic` in api-gateway).
  2. Run `uv lock`, then the full gate: mypy strict and the tests will catch a hidden import.
  3. Optional: add `deptry` as a dev tool, run as `python -m deptry` in a gate section so the problem cannot return.
- **Effort:** S

#### PLAT-13 — No automated check of the Compose exposure rules
- **Severity:** Low
- **Category:** Tests
- **Confidence:** High
- **Location:** `docker-compose.yml:39-40,66-67,90-91,109-110,240-241,318-320,339-340`; `scripts/stack/tests/test_compose.py` (tests only argv building)
- **About the file:** Compose is the only place ports are published. The stack tests cover the command builder.
- **Problem:** The rules "every published port binds 127.0.0.1", "gRPC 50051-50053 are never published", "every long-running service has a healthcheck" and "no `${VAR:-default}` for a secret" are enforced by review only. One edit like `- "8080:8080"` exposes the console on every interface, and the gate stays green.
- **Recommendation:**
  1. Add a stdlib-only test class in `scripts/stack/tests/test_compose.py`. It reads `docker-compose.yml` line by line, collects list items under each `ports:` key, and asserts each matches `^- "127\.0\.0\.1:\d+:\d+"$`. It also asserts that no item contains `5005[123]`, and that every service block other than `migrate` contains `healthcheck:`. The gate runs it through the existing "developer scripts' own tests" step.
  2. Optionally, in the CI `stack` job, run `docker compose config --no-interpolate --format json` (no secrets rendered) and check the same with `python -c`.
- **Effort:** S

#### PLAT-14 — CI hygiene: no job timeouts, setup steps copied three times, double runs, no explicit SBOM or provenance
- **Severity:** Low
- **Category:** Reliability
- **Confidence:** High
- **Location:** `.github/workflows/ci.yml:13-17`, `:27-60`, `:62-98`, `:100-131`, `:241-267`
- **About the file:** The CI workflow.
- **Problem:** No job sets `timeout-minutes`, so a hung `stack up --wait` (300 s cap), a stuck Playwright run or a stuck Trivy pull holds a runner for the 6-hour default. Checkout, setup-uv, `uv python install`, pnpm setup, setup-node and `pnpm install` are repeated verbatim in `gate`, `audit` and `stack`, and edits drift: `stack` syncs Python before Node, the others after. `on: push: branches ["**"]` plus `pull_request` runs everything twice for any same-repo PR, although the owner decided on no PRs. `docker/build-push-action` gets neither `sbom: true` nor `provenance: mode=max`, so published images carry only the default minimal attestation.
- **Recommendation:**
  1. Add `timeout-minutes:` of 20 for `gate`, 15 for `audit`, 45 for `stack` and 30 for `publish`.
  2. Move the shared setup into `.github/actions/setup/action.yml` (a composite action with a `node: true/false` input) and call it from the three jobs.
  3. Drop the `pull_request:` trigger, or restrict `push` to `main` and tags `v*`, matching the no-PR policy.
  4. Add `sbom: true` and `provenance: mode=max` to both build-push steps (`:243`, `:257`).
- **Effort:** S

#### PLAT-15 — Duplication and drift in Compose and the Dockerfile
- **Severity:** Low
- **Category:** Duplication
- **Confidence:** High
- **Location:** `docker-compose.yml:56-60,80-84,99-103,126-130,161-166,182-187,204-209,226-231,253-258,302-307,327-331,345-349` (healthcheck timing ×12); `:35,55` … (`logging` and `restart` repeated on six infra services); `:140,155,174,195,217,243,269` (`LOG_DIR`); `Dockerfile:58,64,72,78,84,90` versus `docker-compose.yml:153,172,267,193,215,239` (CMD and `command` both set ports and metrics flags)
- **About the file:** Compose and the per-service image targets.
- **Problem:** `interval: 10s / timeout: 5s / retries: 5 (/ start_period: 20s)` appears twelve times. `LOG_DIR: /app/logs` appears in each Python service, although the `x-python-service` anchor already owns the log volume. The infra services repeat `logging: *docker-logs` and `restart: unless-stopped` instead of using an anchor. Every Python service's argv is written twice, in the image `CMD` and the Compose `command`, which already differ: Compose adds the MQTT host and the forwarded-IPs flag. A port change must be made in both places. The `web` healthcheck has no `start_period`, unlike the other services.
- **Recommendation:**
  1. Add `x-healthcheck: &healthcheck {interval: 10s, timeout: 5s, retries: 5, start_period: 20s}` and use `healthcheck: {<<: *healthcheck, test: [...]}`.
  2. Move `LOG_DIR: /app/logs` into `x-python-service.environment`. Services merge their own `environment` map with it, which works with YAML merge only if every service uses map syntax (they already do).
  3. Add `x-infra: &infra {restart: unless-stopped, logging: *docker-logs}` for the six infra services.
  4. Keep the argv in one place. Either drop `command:` where it only repeats `CMD`, and pass hosts through environment variables the services already read, or reduce `CMD` to `["python","-m","<pkg>"]`.
  5. Verify: `docker compose config --no-interpolate` before and after gives the same effective config.
- **Effort:** S

#### PLAT-16 — Every secret is a container environment variable, including private keys
- **Severity:** Low
- **Category:** Security
- **Confidence:** High
- **Location:** `docker-compose.yml:42-48,70-74,93-95,112-117,141-143,220,246-247,272-279,286,322-323`
- **About the file:** Compose passes `.env` values to containers.
- **Problem:** The JWT private key, the TLS and OPC UA private keys, the DB and MQTT passwords and the Influx token are all environment variables. They show in `docker inspect`, in `/proc/1/environ`, in child processes (the `migrate` `sh -c`, the Mosquitto start script) and in any crash report that dumps the environment. Docker access is root-equivalent anyway, so the local impact is small. For deployers it is the weaker pattern. Positives: no `${VAR:-default}` secret defaults exist (only empty defaults for the optional certificates), and the published images contain no secret.
- **Recommendation:**
  1. Short term, no code change: document in `14-command-reference.md` that `docker inspect` reveals secrets.
  2. Proper fix: Compose `secrets:` with `environment:` sources, mounted at `/run/secrets/*`, plus `*_FILE` support in the services' settings. This changes how services read configuration: out of scope, owner decision.
- **Effort:** L (proper fix)

#### PLAT-17 — Ignore files: over-broad `lib/`, `.env.*` variants unignored, web build context too wide
- **Severity:** Low
- **Category:** Security
- **Confidence:** High
- **Location:** `.gitignore:17` (`lib/`), `:138` (`.env` only), `:356` (`tags`), `:422-441` (no `image-scan/`); `apps/web/.dockerignore:1-6`
- **About the file:** The git and Docker ignore lists.
- **Problem:** `lib/` (Python packaging boilerplate) matches any directory named `lib`. `git check-ignore` confirms that a future `apps/web/src/lib/` or `shared/lib/` would be silently untracked, and a reviewer would never see the code. `.env.local`, `.env.backup` or `apps/web/.env.local`, which Vite loads by convention, are not ignored, so a secret copied into one can be committed. `image-scan/` (CI Trivy output) is not ignored. `apps/web/.dockerignore` does not exclude `readme-results`, `e2e`, `.env*`, `playwright*.config.ts` or `readme/`, so these enter the console build stage. Vite would inline `VITE_*` variables from an `apps/web/.env*` into the public bundle.
- **Recommendation:**
  1. `.gitignore`: change `lib/` to `/lib/` and `lib64/` to `/lib64/`, and `tags` to `/tags`. Add `.env.*`, then `!.env.example`, then `image-scan/`.
  2. `apps/web/.dockerignore`: add `.env*`, `readme-results`, `readme`, `e2e`, `playwright*.config.ts`.
  3. Verify with `git check-ignore -v apps/web/src/lib/x.ts` (must print nothing) and `git status` (no change in tracked files).
- **Effort:** S

#### PLAT-18 — Pre-commit hook spawns the `ruff` executable and has no private-key guard
- **Severity:** Low
- **Category:** Tests
- **Confidence:** High
- **Location:** `.pre-commit-config.yaml:9-14`, `:16-31`
- **About the file:** Commit-time formatting and hygiene hooks.
- **Problem:** `entry: uv run --no-sync ruff format` launches `ruff.exe`. `14-command-reference.md` records that Windows App Control blocks that executable at random and requires `python -m <tool>`. The gate follows this rule; the hook does not. Separately, the project generates PEM keys into `.env` and `certs/` every time, yet the hook set lacks `detect-private-key`, the cheapest last line against committing one.
- **Recommendation:**
  1. Change line 11 to `entry: uv run --no-sync python -m ruff format`.
  2. Add `- id: detect-private-key` under `pre-commit-hooks`.
  3. Verify with `uv run --no-sync python -m pre_commit run --all-files`.
- **Effort:** S

#### PLAT-19 — Dead and stale files in the repository hygiene layer
- **Severity:** Low
- **Category:** Readability
- **Confidence:** High
- **Location:** `apps/infrastructure/db/env.py` (0 bytes), `apps/infrastructure/db/versions/alembic.ini` (0 bytes); `SECURITY.md:5`; `CONTRIBUTING.md:16-17`; `CHANGELOG.md`; `.github/PULL_REQUEST_TEMPLATE.md` (checklist); `docs/architecture/overview.md:23`; `docker-compose.yml:75`; `pyproject.toml:4,9`; `apps/ai-predictor/pyproject.toml:6-18`
- **About the file:** Various repository-level files.
- **Problem:**
  - `apps/infrastructure/` holds two empty files that nothing references (grep finds no match). They look like a second Alembic chain.
  - `SECURITY.md` says "report to: [твой email]": no contact, and Russian in an external document, which breaks the language policy. Anyone following the policy cannot report a vulnerability.
  - `CONTRIBUTING.md` still says PRs go into `develop` and `main` is PR-protected, which contradicts the 2026-09-15 no-PR decision. `CHANGELOG.md` holds only "Initial project structure". The PR template tells contributors to run `pytest`, `ruff check .` and `mypy .` instead of `quality-gate`.
  - `overview.md:23` still shows Mosquitto `:9001` (the WebSocket listener, removed 2026-09-18).
  - `DOCKER_INFLUXDB_INIT_RETENTION: 90d` (`docker-compose.yml:75`) contradicts the documented 7-day raw bucket. The historian overrides it (`apps/historian/src/historian/storage.py:43`), but anyone reading Compose is misled.
  - The root project is described as "AI-Driven" with keyword `machine-learning`, although AI is deferred.
  - `apps/ai-predictor/pyproject.toml` lists `torch`, `mlflow` etc. with open lower bounds. It is not locked and not audited. That is harmless while it is not a member, but a trap if someone adds it.
- **Recommendation:**
  1. Delete `apps/infrastructure/`.
  2. Rewrite `SECURITY.md` in English with GitHub private vulnerability reporting ("Security → Report a vulnerability") as the channel. It needs no personal email; the owner may choose.
  3. Align `CONTRIBUTING.md` and the PR template with `11-commands.md` (branch naming `type/…`, local ff-merge, `quality-gate`). Either keep `CHANGELOG.md` current or delete it and point to GitHub releases.
  4. Fix `overview.md:23` to `(:1883)`. Remove `DOCKER_INFLUXDB_INIT_RETENTION`, or set it to `7d` to match. Update `pyproject.toml:4,9`.
  5. Add a comment-free guard: in `apps/ai-predictor/pyproject.toml`, keep only the package metadata and empty `dependencies` until the stage reopens.
- **Effort:** S

### 7.9 Shared packages, developer scripts, cross-service duplication

**Area summary.** The shared layer is small and mostly careful. The correlation ids from outside are validated, the gRPC metric labels are bounded, the proto has never renumbered or reused a field, and the scripts never use `shell=True`, validate the catalog, and default `confirm()` to "no". The weakest point is the shared MQTT loop. It classifies every exception as a broker outage, and a failure that follows a successful connect defeats its "log once" guarantee: I measured one WARNING plus one INFO per retry and no traceback. On the ops side, none of the five asyncio services handles SIGTERM, so their carefully written graceful-shutdown code never runs in Docker. The biggest duplication is in the MQTT and entry-point wiring: a queue-backed publisher copied between plc-controller and alert-manager, and argparse, credential and event-loop boilerplate in five `__main__.py`. In the scripts, the smoke/demo HTTP client and the `compose()` builders are copied too. Security gaps are defence-in-depth: `.env` is written with default permissions, the gateway HTTP metrics take a method label straight from the client, and an InfluxDB token is expanded onto a CLI argv inside the container.

<details><summary>Scope reviewed by the area auditor</summary>

- (A) `shared/observability/src/cogniboiler_observability/{__init__,logs,correlation,grpc_observability,metrics}.py` and its three test modules; `shared/runtime/src/cogniboiler_runtime/{__init__,mqtt,liveness,clock}.py` and its three test modules; `shared/proto/cogniboiler.proto` (plus its git history for removed or reused field numbers).
- (B) `dev_tools_scripts_runner.py`; `scripts/runner/*` (main, config_loader, execution, config/*.json); `scripts/_toolkit/*` (config, console, envfile, processes, reexec, steps); every script: `backup`, `restore`, `dev_secrets`, `smoke`, `demo`, `stack`, `clean_caches`, `console_e2e`, `readme_media`, `quality_gate`, `selftest`, and each one's `config/*.json`. I skimmed `audit_deps`, `doctor`, `format_code`, `generate_*`, `install_hooks` and `sync_agents` for subprocess use and exit codes.
- (C) All six `__main__.py`, every `metrics.py`, `api_gateway/{observability,clients,plant_state,plc_state,main,config}.py`, `api_gateway/realtime/sources.py`, `api_gateway/routers/alarms.py`, `alert_manager/{subscriber,publisher,payloads,grpc_server}.py`, `historian/{subscriber,writer,points}.py`, `opcua_server/{subscriber,client,upstreams,projection,units,gateway}.py`, `physics_engine/{mqtt_publisher,server}.py`, `plc_controller/{events,client,server,service}.py`; plus `Dockerfile`, `docker-compose.yml` and the Mosquitto ACL for context.
- Throwaway probes, kept outside the repository: the `MqttSession` failure and logging behaviour, the gateway HTTP metric labels, and discovery of the scripts' unittest suite.

</details>

#### SHR-01 — The shared MQTT loop treats every bug as a broker outage, logs no traceback, and breaks "log once" after a successful connect
- **Severity:** Medium
- **Category:** Reliability
- **Confidence:** High (confirmed with a probe)
- **Location:** `shared/runtime/src/cogniboiler_runtime/mqtt.py:114` (catch-all `except Exception`), `:126-131` (`_announce_connected` resets `_failing` on every connect), `:133-139` (`_announce_failure` logs `%s` of the exception, no `exc_info`)
- **About the file:** `MqttSession` is the one reconnect loop that every MQTT-speaking service uses (physics, PLC, alert-manager ×2, historian, OPC UA, gateway).
- **Problem:** Anything raised by `work` counts as "the broker's fault": a `TypeError` in a payload handler, an asyncua error in `opcua_server/subscriber.py:_apply`, a bug in the gateway's `consume`. Such an exception is logged as `"<name>: MQTT error: <message>"` with no stack trace, and the session reconnects. Because `_failing` is cleared on every successful connect, a failure that comes after the connect logs at WARNING again on every cycle. I ran a probe with a context manager that connects and then raises: **1117 cycles gave 1117 WARNING lines, 1117 INFO lines ("MQTT connection is back"), and not one traceback.** At the default 5 s delay that is about 17,000 warnings plus 17,000 infos a day. A realistic trigger is a retained message that some handler cannot process, for example on `status/+`: the broker redelivers it after every reconnect. The operator sees an "MQTT error" that is really a code defect, with nothing that points at the line. The rule "log once per failure" is broken, and so is the rule "errors are never silently swallowed" (the traceback is lost).
- **Recommendation:**
  1. Split the handling in `MqttSession.run`: `except (aiomqtt.MqttError, OSError, TimeoutError)` is the reconnectable broker failure and keeps today's once-per-outage logging. Any other `Exception` gets `self._logger.error("%s: session work failed", self._name, exc_info=True)` and still reconnects, so the plant keeps running. To keep `cogniboiler_runtime` free of an aiomqtt import, add a constructor argument `broker_errors: tuple[type[BaseException], ...] = (OSError, TimeoutError)` and have callers pass `aiomqtt.MqttError`.
  2. Clear `_failing` only after the session has been healthy for a while, not at connect. For example, clear it in `run` after `work` has been running for at least `reconnect_delay_s`, or when `work` returns normally.
  3. Add tests to `shared/runtime/tests/test_mqtt_session.py`: `test_a_failure_right_after_connect_warns_once` (asserts one WARNING over three cycles) and `test_a_non_broker_error_is_logged_with_its_traceback`.
- **Effort:** M

#### SHR-02 — The gateway's HTTP metrics take the method label from the client, so an unauthenticated caller can grow them without bound
- **Severity:** Medium
- **Category:** Security
- **Confidence:** High (confirmed with a probe)
- **Location:** `apps/api-gateway/src/api_gateway/observability.py:93-97` (the observability pattern the shared package would own; the file lives in the gateway)
- **About the file:** The ASGI middleware that stamps the correlation id and counts and times every HTTP request.
- **Problem:** `method = str(scope["method"])` goes straight into `HTTP_REQUESTS.labels(method, route, status)` and `HTTP_SECONDS.labels(method, route)`. uvicorn/h11 and nginx (`infrastructure/nginx/site.inc` has no `limit_except`) pass any method token through. My probe sent `JUNK0`…`JUNK4` to `/health`, and the label set became `['GET', 'JUNK0', 'JUNK1', 'JUNK2', 'JUNK3', 'JUNK4']`. Each new method creates one counter series and about 12 histogram series, in gateway memory and in Prometheus. A loop of random methods, with no login, is a slow memory denial of service. Route labels are already bounded; the method label is not. By contrast, the gRPC interceptor is safe: `grpc_observability.py:108` only labels methods that have a registered handler.
- **Recommendation:**
  1. Add `_KNOWN_METHODS = frozenset({"GET","POST","PUT","PATCH","DELETE","HEAD","OPTIONS"})` and label `method if method in _KNOWN_METHODS else "other"`.
  2. Better, put a reusable helper in `shared/observability/metrics.py`: `def bounded_label(value: str, allowed: frozenset[str], *, other: str = "other") -> str`, so every service uses the same guard.
  3. Add a test that sends two unknown methods and asserts that only an `other` series appears.
- **Effort:** S

#### SHR-03 — An exception in the `on_failure` hook escapes `MqttSession.run` and ends the session for good
- **Severity:** Low
- **Category:** Reliability
- **Confidence:** High (confirmed with a probe)
- **Location:** `shared/runtime/src/cogniboiler_runtime/mqtt.py:117-118`; the one caller is `apps/historian/src/historian/subscriber.py:265` (`on_failure=self._flush`)
- **About the file:** As SHR-01.
- **Problem:** `await on_failure()` runs inside the `except` block with no guard. In my probe a hook that raised made `run()` propagate `RuntimeError: hook failed`. That contradicts the docstring ("Only cancellation ends it"). In the historian, `subscriber.run()` would then finish, `asyncio.gather` in `historian/__main__.py` would fail, and the process would exit. Today `_flush` cannot raise because `InfluxWriter` swallows its own errors, so this is a latent trap for the next caller.
- **Recommendation:** Wrap the hook in `try: await on_failure() except Exception: self._logger.exception("%s: on_failure hook failed", self._name)`. Add the test `test_a_failing_hook_does_not_end_the_session`.
- **Effort:** S

#### SHR-04 — Logging has no redaction step, and console mode prints control characters as they are
- **Severity:** Low
- **Category:** Logging
- **Confidence:** High
- **Location:** `shared/observability/src/cogniboiler_observability/logs.py:98-103` (processor chain), `:106` (`ConsoleRenderer(colors=False)`)
- **About the file:** `configure_logging` turns every stdlib and structlog record into one JSON line (or one console line) with `service`, `correlation_id` and the exception text.
- **Problem:** No processor masks sensitive keys. Today no call site logs a password, token or cookie (I grepped every service), but nothing stops a future `logger.info("login %s", body)` or `log.bind(authorization=...)` from reaching `logs/*.log`, which `demo` reads and which are world-writable (SCR-06). Exceptions are rendered by `format_exc_info` (`:57`) without local variables, which is good. In `LOG_FORMAT=console` mode, `ConsoleRenderer` prints the event and string values unescaped, so a newline in a user-controlled value (a username, a comment) can forge a fake log line. The JSON renderer escapes correctly and the log files are always JSON, so the exposure is limited to someone running a service by hand.
- **Recommendation:**
  1. Add `_redact(_, __, event)` to `shared` in `logs.py`: replace the value of any key matching `(?i)password|secret|token|authorization|cookie|api_key` with `"***"`. Test it in `test_shared_observability.py`.
  2. For console mode, add a small processor that replaces `\r`, `\n` and other C0 control characters in string values with `\\n`-style escapes before `ConsoleRenderer`.
- **Effort:** S

#### SHR-05 — The reconnect delay is fixed, with no jitter
- **Severity:** Low
- **Category:** Reliability
- **Confidence:** High
- **Location:** `shared/runtime/src/cogniboiler_runtime/mqtt.py:23`, `:124`
- **About the file:** As SHR-01.
- **Problem:** After a broker restart, seven sessions in six services all retry on exact 5 s multiples, three of them with persistent sessions. For one machine this is a mild thundering herd, but the shared helper is the one place a fix would help everyone.
- **Recommendation:** Keep the 5 s base and add ±20 % jitter: `await asyncio.sleep(self._reconnect_delay_s * random.uniform(0.8, 1.2))`, through an injectable `jitter: Callable[[], float]` so tests stay deterministic. No exponential backoff is needed.
- **Effort:** S

#### SHR-06 — The liveness loop can die on `mkdir`, and a file from the future counts as fresh
- **Severity:** Low
- **Category:** Reliability
- **Confidence:** Medium
- **Location:** `shared/runtime/src/cogniboiler_runtime/liveness.py:58-59`, `:65-72`
- **About the file:** The healthcheck file that historian and alert-manager refresh while they are connected; `python -m cogniboiler_runtime.liveness` checks its mtime.
- **Problem:** (a) `run()` calls `mkdir` outside any `try`. An `OSError` there, such as a read-only mount, ends the task, and because it runs under `asyncio.gather` it ends the whole service. That is louder than the "keep beating and say once" design in `beat()`. (b) `is_fresh` returns `True` when `st_mtime` is in the future (clock skew, or a restored container file), so a dead service can look healthy for as long as the skew lasts. The write at `:44` is not atomic, but only the mtime is read, so that does not matter.
- **Recommendation:** Move the `mkdir` into the same `try/except OSError` path as `beat` (log once, keep looping). In `is_fresh`, use `0 <= time.time() - st_mtime <= max_age_s`. Add one test for each.
- **Effort:** S

#### SHR-07 — Four proto enums use a meaningful zero value, so an unset field reads as GOOD, AUTO, OPERATOR or RUNNING
- **Severity:** Low
- **Category:** Readability
- **Confidence:** High
- **Location:** `shared/proto/cogniboiler.proto:7-11` (`SensorQuality GOOD = 0`), `:13-18` (`CommandSource OPERATOR = 0`), `:20-24` (`ControlMode AUTO = 0`), `:26-29` (`SimulationRunState SIMULATION_RUNNING = 0`)
- **About the file:** The gRPC and MQTT telemetry contract.
- **Problem:** In proto3 an unset enum field decodes as 0. A publisher that forgets `quality` reports every instrument GOOD, which fails open. A `ControlModeRequest` built without `mode` asks for AUTO. `FaultKind` and `AlarmState` do this correctly with `*_UNSPECIFIED = 0`. Renumbering is forbidden, so the contract itself cannot be fixed without a breaking change (owner decision).
- **Recommendation:**
  1. Add comments on these four enums stating that 0 is a real value, never "unknown".
  2. Add tests at the edges: physics `proto_mapping` always sets `quality` explicitly, and the PLC's `SetControlMode` test covers a request without an explicit mode.
  3. Record the trade-off in `docs/architecture/invariants.md`. A v2 enum with UNSPECIFIED is out of scope (owner decision).
- **Effort:** S

#### SHR-08 — The proto contract has gaps in unit naming and field comments
- **Severity:** Info
- **Category:** Readability
- **Confidence:** High
- **Location:** `shared/proto/cogniboiler.proto:49` (`uptime_seconds`, while the rest of the file uses `_s`), `:104` (`co2_intensity_kg_per_mwh`, next to the SI `co2_intensity_kg_per_j` at `:135`), `:117`, `:120`, `:122` (`*_hours`), `:123` (`overall_health_pct`), `:225-232` (`SafetyEventMsg.value/threshold`: no unit, and `level`/`action` have no allowed values), `:235-241` (the unit is in a string field), `:244-253` (`AlarmConditionMsg` has no `unit`, unlike `AlarmMsg.unit` at `:360`); many fields have no comment at all (for example `:88-97`, `:147-156`)
- **About the file:** As SHR-07.
- **Problem:** The rule is "SI units and the unit in the field name". Renaming is a breaking change, but readers need to know the unit of each non-SI field.
- **Recommendation:** Add trailing comments only: `[h]`, `[%]`, `[kg/MWh]`, "SI unit of `parameter`", and the allowed values of `level` and `action`. There are no field or number changes, so this needs no `reserved`. I verified in git history that no field was ever removed or renumbered.
- **Effort:** S

#### SHR-09 — Test gaps in the shared packages
- **Severity:** Low
- **Category:** Tests
- **Confidence:** High
- **Location:** `shared/runtime/tests/test_mqtt_session.py` (no connect-then-fail case, no failing-hook case), `shared/observability/tests/test_shared_observability.py` (no console-mode escaping, no redaction), `shared/runtime/tests/test_liveness.py` (no future-mtime case, no mkdir failure)
- **About the file:** The unit tests of the shared packages.
- **Problem:** Each defect in SHR-01, SHR-03, SHR-04 and SHR-06 went unnoticed because its scenario has no test.
- **Recommendation:** Add the tests named in those findings, one per behaviour, in the existing style.
- **Effort:** S

#### SCR-01 — `.env` (JWT private key, TLS keys, every password) is written with default permissions, and a re-run resets a tightened mode
- **Severity:** Medium
- **Category:** Security
- **Confidence:** High
- **Location:** `scripts/dev_secrets/__main__.py:187-189`; similarly `scripts/backup/__main__.py:137`, `:143`, `:180` (the backup folder holds password hashes, session records and the audit log)
- **About the file:** `dev-secrets` creates or completes `.env` from `.env.example`; `backup` dumps the databases into `backups/<stamp>/`.
- **Problem:** `temporary.write_text(...)` followed by `temporary.replace(target_path)` creates a new file with the process umask, usually 0644 on POSIX, so every local user can read it. A developer who ran `chmod 600 .env` gets 0644 back the next time `dev-secrets` runs, because `replace` swaps in the new inode. The same applies to `postgres.sql`. On Windows, NTFS inheritance makes this a non-issue.
- **Recommendation:**
  1. Add `write_private(path: Path, text: str) -> None` to `scripts/_toolkit` (a new `files.py`). It creates the temp file with `os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)`, writes it, and then calls `os.replace`. On `os.name == "nt"` it falls back to `write_text`.
  2. Use it in `dev_secrets`. In `backup`, `chmod(0o700)` the target folder right after `mkdir`.
  3. Add a POSIX-only unittest (`skipUnless(os.name == "posix")`) that asserts `stat.S_IMODE(...) == 0o600`.
- **Effort:** S

#### SCR-02 — The dev-secrets tests never run: the directory is not a package, and the gate's interpreter has no `cryptography`
- **Severity:** Medium
- **Category:** Tests
- **Confidence:** High (confirmed by discovery)
- **Location:** `scripts/dev_secrets/tests/` (contains only `test_certificates.py`, no `__init__.py`); `scripts/quality_gate/config/steps.json` (the "developer scripts' own tests" step runs `python -W error -m unittest discover -s scripts -t .`); `scripts/selftest/config/steps.json`
- **About the file:** The certificate and key-pair generation tests of `dev-secrets`.
- **Problem:** `unittest discover` does not descend into directories without `__init__.py`. `python -m unittest discover -s scripts -t . -v` runs 134 tests, and none of them is `test_certificates`. Even with `__init__.py`, the step runs under the orchestrator's `sys.executable`, where `cryptography` is usually absent, so `@skipUnless(HAS_CRYPTOGRAPHY)` would skip the class quietly. The code that generates the JWT and TLS key material has no test in the gate.
- **Recommendation:**
  1. Add an empty `scripts/dev_secrets/tests/__init__.py`.
  2. Add a gate step `uv run --no-sync python -W error -m unittest scripts.dev_secrets.tests.test_certificates`, or make the skip fail under `CI` (`if os.environ.get("CI") and not HAS_CRYPTOGRAPHY: raise`).
  3. Add a `selftest` check that counts the test modules found against the `tests/` directories on disk, so the next missing `__init__.py` cannot hide.
- **Effort:** S

#### SCR-03 — `restore` is not atomic: a failing `psql` leaves a half-restored database, and the run carries on
- **Severity:** Medium
- **Category:** Reliability
- **Confidence:** High
- **Location:** `scripts/restore/__main__.py:75-86` (psql without `--single-transaction`), `:176-200` (continues after a failure), `:51-56` (any folder, including an absolute path anywhere, is accepted if it has `postgres.sql`); dump flags at `scripts/backup/__main__.py:58-59` (`--clean --if-exists`)
- **About the file:** Puts a `backup` folder back into the live PostgreSQL and InfluxDB, after asking first.
- **Problem:** The dump starts with `DROP ... IF EXISTS` statements. `psql --set ON_ERROR_STOP=1` stops at the first error but does not roll back what has already run. If a statement fails midway (version mismatch, a truncated or foreign dump), the tables are already dropped and only part of the data is back: that is data loss. The script then still restores InfluxDB and restarts the services (`:185-196`) on top of an inconsistent PostgreSQL. The manifest is never checked, so a folder from another project is accepted.
- **Recommendation:**
  1. Add `--single-transaction` to the psql command (`pg_dump --clean` output is transaction-safe), so a failure rolls back to the pre-restore state.
  2. If `postgres_ok` is false, skip the InfluxDB restore, start the services again, and return 1 with a clear message.
  3. Before confirming, read `manifest.json` and refuse (unless `--force`) when it is missing or `project` differs from the config.
  4. Add unit tests on the command builders: `--single-transaction` is present, and a folder without a manifest is refused. An automatic pre-restore backup would be a new feature (owner decision).
- **Effort:** M

#### SCR-04 — `smoke` and `demo` carry two copies of the same HTTP client, login and `.env` loading, and the copies have drifted
- **Severity:** Medium
- **Category:** Duplication
- **Confidence:** High
- **Location:** `scripts/smoke/__main__.py:27-36` (`GatewayUnreachableError`, `Reply`), `:52-58` (`dig`), `:60-88` (`call`), `:112-135` (login), `:259-266` (reading `.env`); `scripts/demo/__main__.py:37-54`, `:56-62`, `:64-91`, `:176-186` (`sign_in`), `:473-481`
- **About the file:** `smoke` checks a running stack end to end; `demo` plays the VISION §7 scenario. Both talk to the gateway with urllib.
- **Problem:** About 60 lines are copied. They have already drifted: smoke's `call` drops the error body on `HTTPError` (`smoke:80-81`), while demo reads it and shows the refusal reason (`demo:84-89`). A fix to timeouts, TLS or headers has to be made twice. Sharing through `scripts/_toolkit` is the sanctioned way (scripts never import each other).

  ```python
  # smoke/__main__.py:80        # demo/__main__.py:84
  except urllib.error.HTTPError as error:      except urllib.error.HTTPError as error:
      return Reply(error.code, None)                raw = error.read() ...
  ```
- **Recommendation:**
  1. Create `scripts/_toolkit/gateway.py` with `class GatewayUnreachableError(RuntimeError)`, `@dataclass(frozen=True) class Reply` (with `accepted` and `refusal`), `dig(body, *keys) -> Any`, `call(base_url, method, path, *, token=None, payload=None, timeout=10.0) -> Reply` (demo's variant, which keeps error bodies), `sign_in(base_url, username, password) -> str | None`, and `load_env(root: Path, env_file: str) -> dict[str, str]`, which raises one `EnvFileError` with the filename.
  2. Make both scripts import from it, and move the tests of `dig` and `call` into `scripts/_toolkit/tests/test_gateway.py`.
- **Effort:** M

#### SCR-05 — backup and restore expand the InfluxDB admin token onto the `influx` CLI's argv inside the container, contrary to their docstrings
- **Severity:** Low
- **Category:** Security
- **Confidence:** Medium
- **Location:** `scripts/backup/__main__.py:72-73`, `scripts/restore/__main__.py:106-107`; the claims at `scripts/backup/__main__.py:5-6` and `scripts/restore/__main__.py:7-8`
- **About the file:** As SCR-03.
- **Problem:** `sh -c '... influx backup ... --token "$DOCKER_INFLUXDB_INIT_ADMIN_TOKEN"'` keeps the token off the host's `docker compose` command line, but the container's shell expands it into the `influx` process arguments. On a Linux host, container processes show up in the host's `ps` for as long as the backup runs (inside the VM with Docker Desktop). The docstrings say "no password or token is ever passed on a command line", which is not quite true. The pg_dump path is correct: its password comes from the environment.
- **Recommendation:** Use the CLI's environment variable: `INFLUX_TOKEN="$DOCKER_INFLUXDB_INIT_ADMIN_TOKEN" influx backup {dir}` (the same for restore). Update the command-builder tests so they assert that `--token` is absent.
- **Effort:** S

#### SCR-06 — `stack up` makes `logs/` world-writable (0777, no sticky bit) on POSIX
- **Severity:** Low
- **Category:** Security
- **Confidence:** High
- **Location:** `scripts/stack/__main__.py:48`
- **About the file:** `stack` wraps `docker compose` and prepares the bind-mounted `logs/` for services running as uid 10001.
- **Problem:** With mode 0777 and no sticky bit, any local user can delete or replace `logs/<service>.log`, for example with a symlink, or plant fake lines. `demo` treats those lines as evidence ("no service logged an error"), and they may hold usernames and addresses from the audit trail.
- **Recommendation:** Use `0o1777` (sticky), so only the owner of a file can remove or rename it. Or better, `0o770` plus a group that uid 10001 can join, documented in `14-command-reference.md`. Add a POSIX-only test for the mode.
- **Effort:** S

#### SCR-07 — The Compose argv builder is copied three times, and backup/restore bypass `_toolkit.processes`
- **Severity:** Low
- **Category:** Duplication
- **Confidence:** High
- **Location:** `scripts/backup/__main__.py:32-41`, `scripts/restore/__main__.py:28-37`, `scripts/stack/__main__.py:27-38` (`compose()`); `scripts/backup/__main__.py:112-116` and `scripts/restore/__main__.py:112-116` (private `run()` on raw `subprocess.run`, with `FileNotFoundError` handling at `backup:146`, `restore:171`)
- **About the file:** The Docker-facing scripts.
- **Problem:** Three identical `["docker","compose","--project-name",…,"--file",…]` builders, plus two private `run()` helpers. They do not echo the command, they bypass `find_tool` (Windows shims), and they report "docker missing" differently from `stack`.
- **Recommendation:** Add `scripts/_toolkit/compose.py`: `compose_argv(config: Mapping[str, Any], *args: str, profile: str | None = None) -> list[str]`, `exec_sh(config, service: str, script: str) -> list[str]`, and `run_piped(argv, cwd, *, stdin: Path | None = None, stdout: Path | None = None) -> CommandResult`, built on `processes.find_tool` and returning `NOT_FOUND` like `processes.run`. Replace the three copies and move their tests.
- **Effort:** S

#### SCR-08 — Argument edge cases: `--into` outside the repo crashes, and a `0` flag is silently ignored
- **Severity:** Low
- **Category:** Reliability
- **Confidence:** High
- **Location:** `scripts/backup/__main__.py:135`, `scripts/restore/__main__.py:145` (`relative_to(root)`); `scripts/smoke/__main__.py:257` (`args.history_wait_s or config[...]`), `scripts/demo/__main__.py:471` (`args.speed or config[...]`)
- **About the file:** CLI parsing in these scripts.
- **Problem:** `backup --into D:\backups` (or `/mnt/backups`) raises an uncaught `ValueError` traceback from `Path.relative_to` before any work, and so does `restore --list --into ...`, although the help text invites a custom location. `smoke --history-wait-s 0` and `demo --speed 0` fall back to the config value, because `0 or x` is `x`.
- **Recommendation:** Print `target.as_posix()` when it is not relative to the root (a helper `display_path(path, root) -> str` in `_toolkit/console.py`). Replace `a or b` with `a if a is not None else b`. Add a unittest for each.
- **Effort:** S

#### SCR-09 — `scripts/` is outside `mypy --strict`, and five scripts have no tests
- **Severity:** Low
- **Category:** Tests
- **Confidence:** High
- **Location:** `pyproject.toml:71` (`files = ["apps", "shared"]`); scripts without a `tests/` directory: `scripts/doctor`, `scripts/format_code`, `scripts/generate_proto`, `scripts/install_hooks`, `scripts/quality_gate`
- **About the file:** The mypy configuration, and the script catalog.
- **Problem:** The scripts are fully annotated but never type-checked, so a wrong annotation passes the gate. `quality_gate`, the path CI runs, has no test of its step selection beyond `_toolkit/tests/test_step_selection.py`, and `generate_proto --check` has no test at all.
- **Recommendation:** Add a gate step `uv run --no-sync python -m mypy --strict scripts dev_tools_scripts_runner.py` (the orchestrator is stdlib-only, so this adds no dependency), and fix what it reports. Add small tests for `generate_proto`'s check comparison and for the loading of `quality_gate`'s `steps.json`/`quick_steps`.
- **Effort:** M

#### SCR-10 — The catalog loader does not type-check list and text fields
- **Severity:** Low
- **Category:** Reliability
- **Confidence:** High
- **Location:** `scripts/runner/config_loader.py:131-134` (`aliases`, `examples`, `platforms`, `destructive`), `:160-165` (`_text` does not check that `en`/`ru` are strings), `:125` (`title` unchecked)
- **About the file:** Validates the hand-edited `scripts/runner/config/*.json` before anything is launched.
- **Problem:** `"aliases": "qg"` becomes `('q', 'g')` through `tuple(str)`, which registers the one-letter identifiers `q` and `g`. A typo then launches that script instead of the "unknown script" error the code is careful to give (`main.py:136-142`). `"destructive": "false"` becomes `True` (harmless). A numeric `title` breaks rendering deep inside.
- **Recommendation:** Add `_string_list(entry, key, where) -> tuple[str, ...]`, which raises `ConfigValidationError` unless the value is a list of non-empty strings. Check `title`, `en` and `ru` with `isinstance(str)`, and require `destructive` to be a `bool`. Add three negative tests in `scripts/runner/tests/`.
- **Effort:** S

#### SCR-11 — The `.env` parser differs from Compose and python-dotenv on common hand edits
- **Severity:** Low
- **Category:** Reliability
- **Confidence:** High
- **Location:** `scripts/_toolkit/envfile.py:48-58` (quoted value), `:60` (unquoted value)
- **About the file:** Parses and merges `.env` for `dev-secrets` (write) and for `smoke`, `demo` and `console-e2e`'s checks (read).
- **Problem:** (a) `KEY="x" # note` never matches `endswith('"')`, so the loop swallows the following lines until one ends in `"`. The later keys vanish, and the merge fails with a misleading "double quote not supported". (b) `KEY='x'` keeps the single quotes, while Compose strips them. (c) `KEY=x # note` keeps `"x # note"`, while Compose reads `x`. In cases (b) and (c) the stack starts with one password and smoke or demo logs in with another, which shows up as a confusing login failure.
- **Recommendation:** In `parse`, close a double-quoted value at the first unescaped `"` and allow only whitespace or a `#` comment after it. Strip matching single quotes. For unquoted values, cut at ` #`. Add a test for each of the three cases in `scripts/_toolkit/tests/test_envfile.py`.
- **Effort:** S

#### SCR-12 — `readme-media` deletes a config-supplied path without checking that it stays inside the repository
- **Severity:** Low
- **Category:** Security
- **Confidence:** High
- **Location:** `scripts/readme_media/__main__.py:69` (`shutil.rmtree(raw, ignore_errors=True)` with `raw = root / config["raw_dir"]`)
- **About the file:** Captures the README screenshots and GIF.
- **Problem:** `clean_caches/plan.py:37-38` resolves each target and refuses anything outside the root. This script does not. A config edit such as `"raw_dir": ".."` or an absolute path would wipe a directory outside the repo, silently (`ignore_errors=True`). The config is trusted repository content, so this is defence in depth.
- **Recommendation:** Move `_inside(root, path)` from `clean_caches/plan.py` into `scripts/_toolkit/config.py` as `resolve_inside(root: Path, relative: str) -> Path`, which raises `ScriptConfigError` on escape. Use it in `readme_media` and `clean_caches`, and drop `ignore_errors=True` in favour of reporting the error.
- **Effort:** S

#### DUP-01 — None of the five asyncio services handles SIGTERM, so their graceful-shutdown code never runs in Docker
- **Severity:** Medium
- **Category:** Reliability
- **Confidence:** High
- **Location:** `apps/alert-manager/src/alert_manager/__main__.py:97-105`, `apps/historian/src/historian/__main__.py:139-145`, `apps/opcua-server/src/opcua_server/__main__.py:113-120`, `apps/physics-engine/src/physics_engine/__main__.py:109-116`, `apps/plc-controller/src/plc_controller/__main__.py:44-60` (no `signal`/`add_signal_handler` anywhere in `apps/*/src`); exec-form `CMD` in `Dockerfile:58-90` and `docker-compose.yml:153,172,193,215,239` with no `init: true`
- **About the file:** The service entry points.
- **Problem:** Each container runs `python -m <service>` as PID 1. Linux does not deliver a signal whose disposition is default to PID 1, and Python installs no SIGTERM handler. So `docker compose stop`/`down`, and `stack down`, wait the full 10 s and then SIGKILL. None of the `finally` blocks runs: PLC `server.stop(grace=5)` and `service.close()` (`plc_controller/server.py:205-207`), alert-manager `processor.close()` and `publisher.aclose()` (`__main__.py:89-93`, which also drops queued alarm changes), OPC UA `opc_server.stop()`, physics `runtime.stop()`. The historian has no `finally` at all and loses up to one batch (50 points or 2 s) at every stop, even on Ctrl+C from the host. The gateway is not affected: uvicorn installs its own handlers.
- **Recommendation:**
  1. Add to `shared/runtime` (a new `service.py`): `def run_service(main: Coroutine[Any, Any, int | None]) -> int`. It picks `asyncio.SelectorEventLoop` on win32, and on POSIX registers `loop.add_signal_handler(signal.SIGTERM, main_task.cancel)` inside a small wrapper coroutine, then returns the exit code. Each `__main__` becomes `configure_logging("x"); sys.exit(run_service(main(parse_args())))`. That also removes the duplicated `asyncio.run(..., loop_factory=...)` (see DUP-03).
  2. Give historian's `main` a `finally: await subscriber.flush()` (make `_flush` public or add `aclose()`).
  3. Test it by running `run_service` on a coroutine, sending `os.kill(os.getpid(), SIGTERM)` (POSIX only), and asserting that the `finally` ran.
- **Effort:** M

#### DUP-02 — The queue-backed MQTT publisher is written twice (plc-controller and alert-manager)

> **Same duplication as ALM-06** (section 7.5).

- **Severity:** Medium
- **Category:** Duplication
- **Confidence:** High
- **Location:** `apps/plc-controller/src/plc_controller/events.py:134-262` (`PlcPublisher`: `_enqueue` 179-190, `start` 194-197, `aclose` 199-213, `_open_client` 215-223, `_drain` 232-262); `apps/alert-manager/src/alert_manager/publisher.py:28-105` (`AlarmChangePublisher`: `alarm_changed` 56-65, `start` 67-69, `aclose` 71-80, `_open_client` 82-89, `_publish_queued` 94-105); the constants `QUEUE_LIMIT = 1000` at `events.py:47` and `publisher.py:24`
- **About the file:** The two services that publish ordered, must-not-lose-silently JSON messages over one persistent connection.
- **Problem:** The same bounded deque, drop-oldest policy, "log at the first drop and every 100th", wake-up `Event`, task start and cancel, and publish-then-`popleft` drain loop are copied nearly line for line. They have already drifted: the PLC drains with a timeout and a close deadline, alert-manager does not, and neither counts failures in the shared `MQTT_PUBLISH_ERRORS` counter, which only physics uses (`physics_engine/mqtt_publisher.py:132`).

  ```python
  # events.py:182                                  # publisher.py:57
  if len(self._queue) >= QUEUE_LIMIT:              if len(self._queue) >= QUEUE_LIMIT:
      self._queue.popleft(); self._dropped += 1        self._queue.popleft(); self._dropped += 1
      if self._dropped == 1 or self._dropped % 100 == 0:   if self._dropped == 1 or ...
  ```
- **Recommendation:** Add to `shared/runtime` (`mqtt_queue.py`) `class QueuedMqttPublisher[ClientT]` with `__init__(self, session: MqttSession[ClientT], *, name: str, limit: int = 1000, publish: Callable[[ClientT, QueuedMessage], Awaitable[None]], on_connect: Callable[[ClientT], Awaitable[None]] | None = None, idle: Callable[[ClientT], Awaitable[None]] | None = None, idle_interval_s: float | None = None)`, plus `enqueue(message: QueuedMessage) -> None`, `start() -> None`, `aclose(drain_timeout_s: float = 0.0) -> None` and `dropped: int`. Here `QueuedMessage` is `dataclass(topic, payload, qos=1, retain=False)`. The PLC passes `on_connect` (announce "online") and `idle` (the snapshot every 10 s); alert-manager passes neither. Increment `MQTT_PUBLISHED` and `MQTT_PUBLISH_ERRORS` inside the helper. Move the queue tests into `shared/runtime/tests`.
- **Effort:** M

#### DUP-03 — The entry-point boilerplate is repeated across five `__main__.py` files
- **Severity:** Medium
- **Category:** Duplication
- **Confidence:** High
- **Location:**
  - `sys.path.insert(... parents[4] / "shared" / "generated")`: `apps/alert-manager/src/alert_manager/__main__.py:14`, `apps/api-gateway/src/api_gateway/__main__.py:14`, `apps/historian/src/historian/__main__.py:23`, `apps/opcua-server/src/opcua_server/__main__.py:18`, `apps/physics-engine/src/physics_engine/__main__.py:21`, `apps/plc-controller/src/plc_controller/__main__.py:11` (redundant in containers, where `Dockerfile:41` sets `PYTHONPATH`)
  - `--mqtt-host`/`--mqtt-port`: alert 31-32, historian 102-103, opcua 93-94, physics 94-95, plc 30-31
  - `--metrics-port`/`--metrics-host`: alert 40-48, historian 127-135, opcua 102-110, physics 98-106, plc 32-40
  - `--liveness-file` and its wiring: alert 34-39 / 85-86, historian 121-126 / 95-96
  - MQTT credentials from the environment: alert 64-65, historian 54-55, opcua 45-46, physics 66-67, plc 53-54
  - `asyncio.run(..., loop_factory=SelectorEventLoop if win32)`: alert 99-105, historian 141-145, opcua 116-120, physics 112-116, plc 47-60
  - periodic stats loops: `apps/historian/src/historian/__main__.py:70-77`, `apps/opcua-server/src/opcua_server/__main__.py:64-73`
- **About the file:** Each service's argument parsing and wiring.
- **Problem:** Five copies of the same argument, credential and loop code. It drifts: the PLC calls `start_metrics_server` inside `server.serve()` (`plc_controller/server.py:195`) rather than in its entry point. The gateway uses pydantic-settings for the same `MQTT_*` names (`api_gateway/config.py:75-78`). And the `sys.path` insert creates import-order coupling: any module importing `cogniboiler_pb2` works only if `__main__` or a conftest ran first.

  ```python
  # historian/__main__.py:54                         # opcua-server/__main__.py:45
  mqtt_username=os.environ.get("MQTT_USERNAME", "historian"),   mqtt_username=os.environ.get("MQTT_USERNAME", "opcua-server"),
  mqtt_password=os.environ.get("MQTT_PASSWORD") or None,        mqtt_password=os.environ.get("MQTT_PASSWORD") or None,
  ```
- **Recommendation:**
  1. `shared/runtime/mqtt.py`: `@dataclass(frozen=True) class MqttCredentials(username: str, password: str | None = field(repr=False))` with `@classmethod from_environment(cls, default_username: str) -> MqttCredentials`, and `def add_mqtt_arguments(parser: argparse.ArgumentParser) -> None`.
  2. `shared/observability/metrics.py`: `def add_metrics_arguments(parser, default_port: int) -> None`.
  3. `shared/runtime/liveness.py`: `def add_liveness_argument(parser) -> None`.
  4. `shared/runtime/service.py`: `run_service` (see DUP-01) and `async def log_periodically(logger, interval_s: float, describe: Callable[[], str]) -> None`.
  5. Replace the six `sys.path.insert` lines with one `cogniboiler_runtime.use_generated_stubs()`. Better, make `shared/generated` a workspace package; that is a structural change the owner should approve.
  6. `__main__.py` then holds only wiring, as the rule requires.
- **Effort:** M

#### DUP-04 — The gateway's entry point does not select the selector event loop on Windows, so its MQTT source cannot work when the gateway runs on the host

> **Same defect as GW-API-05** (section 7.2). Fix it once.

- **Severity:** Medium
- **Category:** Reliability
- **Confidence:** Medium (read from the installed libraries; I did not run the gateway)
- **Location:** `apps/api-gateway/src/api_gateway/__main__.py:35-43` (`uvicorn.run` without `loop=`); aiomqtt user `apps/api-gateway/src/api_gateway/realtime/sources.py:100-142`; uvicorn 0.41 `loops/asyncio.py` returns `asyncio.ProactorEventLoop` on win32 (`uv.lock:1543-1544`); aiomqtt 2.5.1 (installed library) `client.py:704` calls `loop.add_reader`
- **About the file:** The gateway entry point: it configures logging and runs uvicorn.
- **Problem:** The rule says Windows entry points must use `SelectorEventLoop` because aiomqtt needs `add_reader()`. The other five services comply; the gateway does not. When it runs from the host on Windows (the documented `uv run --package api-gateway python -m api_gateway`), uvicorn builds a Proactor loop. The `add_reader` callback then fails inside the loop, CONNACK is never read, the connect times out, and `MqttSession` retries forever. The `plc/events` and `alarms/changes` WebSocket channels stay empty, with one warning in the log. The Compose stack (Linux) is not affected.
- **Recommendation:** Pass a custom factory: add `def selector_loop_factory(use_subprocess: bool = False) -> Callable[[], asyncio.AbstractEventLoop]: return asyncio.SelectorEventLoop` to `api_gateway/__main__.py` and call `uvicorn.run(..., loop="api_gateway.__main__:selector_loop_factory")` on win32. (uvicorn 0.41 accepts an import string, see `config.py:488-498`.) Add a unit test that the factory returns `SelectorEventLoop`. Confirm by starting the gateway on Windows and checking that `realtime-mqtt-events` logs "connected to MQTT".
- **Effort:** S

#### DUP-05 — The MQTT client factories and message-consume loops are copied in every service
- **Severity:** Low
- **Category:** Duplication
- **Confidence:** High
- **Location:** `_open_client`: `alert_manager/subscriber.py:138-146`, `alert_manager/publisher.py:82-89`, `plc_controller/events.py:215-223`, `historian/subscriber.py:240-248`, `opcua_server/subscriber.py:196-202`, `api_gateway/realtime/sources.py:117-124`, `physics_engine/mqtt_publisher.py:199-215`. Consume loops: `historian/subscriber.py:250-257` and `opcua_server/subscriber.py:204-211` (identical), `alert_manager/subscriber.py:148-167`, `api_gateway/realtime/sources.py:126-133`. `RECONNECT_DELAY_S = 5.0` is redefined in `alert_manager/subscriber.py:34`, `alert_manager/publisher.py:25`, `historian/subscriber.py:73`, `opcua_server/subscriber.py:54`, `physics_engine/mqtt_publisher.py:54` and `plc_controller/events.py:48`, although the shared `DEFAULT_RECONNECT_DELAY_S` (`mqtt.py:23`) is already the default.
- **About the file:** Each service's MQTT adapter.
- **Problem:** Each adapter repeats `Client(hostname=…, port=…, identifier=…, username=…, password=…)` and the `subscribe_all` → `async for message in client.messages` → bytes-or-empty → `_handle_message(str(topic), payload)` loop. The payload handling has drifted: historian and OPC UA accept only `bytes`, alert-manager accepts `bytes | bytearray`. The six local copies of the delay constant hide the shared value they are meant to follow.
- **Recommendation:** In `shared/runtime/mqtt.py` add `def client_factory(host: str, port: int, credentials: MqttCredentials, *, identifier: str | None = None, persistent: bool = False, will: Will | None = None) -> Callable[[], Client]` (this module may import aiomqtt, a dependency every service already has), and `async def consume(client, subscriptions: Sequence[tuple[str, int]], handle: Callable[[str, bytes], Awaitable[None]]) -> None`, which normalises `bytes | bytearray` and passes `b""` otherwise. Delete the local `RECONNECT_DELAY_S` constants except the gateway's deliberate 3.0.
- **Effort:** M

#### DUP-06 — The "log once per outage" state machine is written five times
- **Severity:** Low
- **Category:** Duplication
- **Confidence:** High
- **Location:** `shared/runtime/src/cogniboiler_runtime/mqtt.py:69`, `:126-139`; `apps/api-gateway/src/api_gateway/realtime/sources.py:38-53` (`_OutageLog`); `apps/opcua-server/src/opcua_server/upstreams.py:45-57` and `:69-84` (`down` flags, twice); `apps/plc-controller/src/plc_controller/service.py:433-451` (`_stream_failing`). Missing where it is needed: `apps/historian/src/historian/writer.py:213-216`, `:227-230` log a WARNING on every failed InfluxDB write (one per batch, every 2 s, during an InfluxDB outage).
- **About the file:** The reconnect and poll loops of gateway, OPC UA, PLC and historian.
- **Problem:** Five hand-rolled flags with slightly different wording, and the historian writer has none, so an InfluxDB outage costs about 1800 warnings an hour, against the rule "log once per failure".

  ```python
  # sources.py:46                                   # upstreams.py:50
  if not self._down:                                if not down:
      logger.warning("Realtime source %s unavailable: %s", ...)   logger.warning("PLCService unreachable ...: %s", exc)
      self._down = True                                 ...; down = True
  ```
- **Recommendation:** Add to `shared/observability` (a new `outage.py`): `class OutageLog` with `__init__(self, logger: logging.Logger, what: str)`, `failed(self, error: BaseException) -> bool` (WARNING on the first failure, DEBUG after that; returns `True` on the first), and `recovered(self) -> bool` (INFO once). Use it in all five places and in `InfluxWriter.write_point(s)`. `MqttSession` can keep its own flags or delegate to it.
- **Effort:** S

#### DUP-07 — gRPC channel and server setup is copied, and the OPC UA clients skip the observability interceptors
- **Severity:** Low
- **Category:** Duplication
- **Confidence:** High
- **Location:** Channels with interceptors: `api_gateway/clients.py:97-99`, `:184-186`, `:236-238`; `plc_controller/client.py:37-39`. **Without** interceptors: `opcua_server/client.py:16`, `:30`. Servers: `physics_engine/server.py:260-271` and `plc_controller/server.py:196-207` (identical, down to the shield comment), `alert_manager/grpc_server.py:180-185`.
- **About the file:** The gRPC clients and servers of four services.
- **Problem:** The rule requires every channel to use `client_interceptors()`, and the OPC UA read clients break it. The practical effect is small today (their background polls run without a correlation id), but it shows why a helper that enforces the rule beats a convention. The server bootstrap (interceptor, `[::]:{port}`, start, shielded `wait_for_termination`, `stop(grace=5)`) is triplicated.

  ```python
  # physics_engine/server.py:260                        # plc_controller/server.py:196
  server = grpc.aio.server(interceptors=[ServerObservability()])   server = grpc.aio.server(interceptors=[ServerObservability()])
  listen_addr = f"[::]:{port}"                         listen_addr = f"[::]:{port}"
  ```
- **Recommendation:** In `shared/observability/grpc_observability.py` add `def observed_channel(target: str, *, options: Sequence[tuple[str, Any]] = ()) -> grpc.aio.Channel`, `async def start_observed_server(register: Callable[[grpc.aio.Server], None], port: int, *, name: str) -> grpc.aio.Server`, and `async def serve_until_cancelled(server: grpc.aio.Server, *, grace_s: float = 5.0) -> None`. Switch `opcua_server/client.py` to `observed_channel`. Add a test that `observed_channel` forwards the correlation id.
- **Effort:** S

#### DUP-08 — Services keep their own copies of shared clock helpers and constants
- **Severity:** Low
- **Category:** Duplication
- **Confidence:** High
- **Location:**
  - `int(time.time() * 1000)` / local `_now_ms`: `api_gateway/accounts.py:36-37`, `api_gateway/clients.py:430-431`, `api_gateway/audit.py:123`, `api_gateway/auth/jwt_handler.py:130`, `:157`, `api_gateway/db_init.py:88`, `:104`, `api_gateway/readiness.py:86`, `api_gateway/realtime/hub.py:135`, `api_gateway/routers/history.py:38`, `api_gateway/routers/simulation.py:64`, `api_gateway/routers/websocket.py:83`, `:149`, `:171`, `:181`, `opcua_server/gateway.py:164`, `physics_engine/mqtt_publisher.py:165`
  - `NANOSECONDS_PER_MILLISECOND = 1_000_000` redefined at `api_gateway/clients.py:278`, although the same file imports `MILLISECONDS_PER_DAY` from `cogniboiler_runtime` (`:17`); `historian/writer.py:101-102` (`timestamp_ms * 1_000_000`)
- **About the file:** Timestamp creation across services. `cogniboiler_runtime.clock` exists precisely to name this once (its docstring says so).
- **Problem:** About 20 hand-written copies of the helper the shared package introduced. `now_ms` is already used by alert-manager, gateway `auth/sessions.py`, historian, physics and PLC, so the gateway alone is inconsistent with itself.
- **Recommendation:** Replace every occurrence with `cogniboiler_runtime.now_ms()` and `NANOSECONDS_PER_MILLISECOND`. Delete the two `_now_ms` functions. Add a ruff `flake8-tidy-imports`/`banned-api` rule or a grep test that forbids `time.time() * 1000` under `apps/`.
- **Effort:** S

#### DUP-09 — Protobuf enum→name mappings and scalar-field reflection are repeated in gateway, historian and OPC UA
- **Severity:** Low
- **Category:** Duplication
- **Confidence:** High
- **Location:**
  - `AlarmState` → `"ACTIVE_UNACK"…`: `api_gateway/routers/alarms.py:35-40` and `opcua_server/projection.py:22-27` (identical dicts)
  - `SensorQuality` → lower-case name: `api_gateway/plant_state.py:32-33`, `api_gateway/routers/status.py:44`, `historian/writer.py:55-59`
  - `ControlMode` → lower-case name: `api_gateway/plc_state.py:31-32`, `opcua_server/projection.py:72`
  - the `SIMULATION_PAUSED` check: `api_gateway/plant_state.py:47-51`, `historian/points.py:81`, `opcua_server/projection.py:49`
  - "every scalar field of a flat message": `api_gateway/plant_state.py:36-42`, `historian/writer.py:105-130`, `opcua_server/projection.py:30-31`, `opcua_server/subscriber.py:218-221`
- **About the file:** The protobuf→REST, protobuf→InfluxDB and protobuf→OPC UA projections.
- **Problem:** Each consumer re-derives the contract's names. They can disagree, and `SensorQuality.Name(value)` in the gateway raises `ValueError` on an enum value it does not know (proto3 enums are open). That exception is not a `grpc.RpcError`, so `run_telemetry` (`realtime/sources.py:56-69`) would die. The task's end would also be hidden at shutdown by `gather(..., return_exceptions=True)` (`api_gateway/main.py:112`).
- **Recommendation:** Add a `cogniboiler_runtime/contract.py` that takes the generated enum wrapper as a parameter, so the shared package does not import the stubs: `def enum_label(enum: EnumTypeWrapper, value: int, *, strip_prefix: str = "", lower: bool = False, unknown: str = "unknown") -> str` and `def scalar_fields(message: Message, *, skip: frozenset[str] = frozenset()) -> dict[str, object]`. Use them at the locations above. Add a test that an unknown enum value maps to `"unknown"`. Separately, give each gateway source task a done-callback that logs any unexpected exit (`logger.exception`).
- **Effort:** M

#### DUP-10 — JSON payload decoding and finite-number checks are re-implemented in four services
- **Severity:** Low
- **Category:** Duplication
- **Confidence:** High
- **Location:** `historian/subscriber.py:159-165` (`_json`), `api_gateway/realtime/sources.py:90-97` (`_json_object`), `opcua_server/gateway.py:117-122` (`_json`, returns `{}`), `alert_manager/payloads.py:75-82` (`_decode`, raises); finite checks at `historian/points.py:54-59` (`_finite`) and `alert_manager/payloads.py:94-101` (`_number`)
- **About the file:** Decoding of the JSON MQTT payloads (`alarms/changes`, `plc/events`, `alerts/*`) and gateway HTTP bodies.
- **Problem:** Four variants of "UTF-8 → `json.loads` → must be an object", with different failure shapes (`None`, `{}`, an exception), plus two copies of "a number, not a bool, and finite".

  ```python
  # historian/subscriber.py:161                  # api_gateway/realtime/sources.py:93
  try: value = json.loads(raw.decode("utf-8"))   try: value = json.loads(payload)
  except UnicodeDecodeError, json.JSONDecodeError: return None   (same)
  return value if isinstance(value, dict) else None   (same)
  ```
- **Recommendation:** In `shared/runtime` (a new `payloads.py`) add `def json_object(raw: bytes | bytearray) -> dict[str, Any] | None` and `def finite_number(value: object) -> float | None`. Alert-manager keeps its raising wrapper on top (`json_object(raw) or raise PayloadError`). Add unit tests for non-UTF-8, arrays, `NaN`, `True` and `inf`.
- **Effort:** S

### 7.10 Test coverage and test quality

**Area summary.** Python service line coverage is 97.4 % (7 737 of 7 944 statements, six services) and branch coverage is 91.2 %. That matches the ROADMAP Д11 claim of 97.6 % to within 0.2 points. All 1 085 Python tests, the 134 script tests and the 212 console tests pass, with no skip or xfail markers in the service suites. The remaining gaps are few but concentrated where they matter most. The PLC's instrument-quality trip, its arming gates, its high-level trip response on the feedwater valve and its bad-measurement controller holds never run in any test. The gateway's expired and malformed access-token branches and the OPC UA clear-text-password refusal are also untested. No coverage floor is enforced anywhere: the gate runs `pytest --no-cov`. Test quality is generally high: stepped rather than timed, fakes rather than brokers. A handful of real-time sleeps and port-probe races remain, and near-identical helpers are duplicated across five services.

<details><summary>Scope reviewed by the area auditor</summary>

measured line and branch coverage for all eight Python workspace packages and the developer scripts. The runs happened one after another between 2026-09-26T22:13:35-03:00 and 22:18:02-03:00, using `uv run --no-sync python -m pytest <tests> -p no:cacheprovider -o addopts= --cov=<pkg> --cov-branch`, with `COVERAGE_FILE` and JSON reports kept outside the repository. For the scripts: `coverage run --branch --source=scripts -m unittest discover -s scripts -t .`. For the console: `pnpm --dir apps/web run test --no-cache`, run at 2026-09-27T09:15:55-03:00. `--no-cache` kept Vitest from rewriting `node_modules/.vite/vitest/.../results.json`, and its mtime stayed at 2026-09-26T02:22:45-03:00. I then read every uncovered line range in the coverage reports and all Python test modules under `apps/*/tests`, `shared/*/tests` and `scripts/*/tests`, plus the console `src/**/*.test.ts(x)` files, `apps/web/e2e/*.spec.ts` (names only), `scripts/quality_gate/config`, `.github/workflows/ci.yml` and the root `pyproject.toml`. I also ran a static AST scan for tests with missing or weak assertions and for duplicated helpers. No file in the repository was written. No `.coverage` or `htmlcov/` appeared in the repo root, and `git status` stayed clean.

</details>

#### Measured coverage

| Package | Tests (pass/fail/skip) | Line % | Branch % | Runtime | Notes |
|---|---|---|---|---|---|
| physics-engine (`physics_engine`) | 197 / 0 / 0 | 97.4 | 88.8 | 16.9 s (20 s wall) | Gaps: `steam_tables.py` fallbacks, spray command path, startup classification |
| plc-controller (`plc_controller`) | 152 / 0 / 0 | 95.8 | 89.5 | 54.7 s (57 s wall) | Lowest service. Safety gaps in `safety.py`, `control.py`, `service.py`; `__main__.py` 0 % |
| api-gateway (`api_gateway`) | 372 / 0 / 0 | 96.9 | 92.2 | 89.7 s (98 s wall) | Token-rejection branches, simulation 503 paths, WS closes; `__main__.py` 0 %; 1 Starlette deprecation warning |
| historian (`historian`) | 91 / 0 / 0 | 99.4 | 95.6 | 3.9 s (7 s wall) | Only `__main__` guard and client constructor uncovered |
| alert-manager (`alert_manager`) | 99 / 0 / 0 | 99.0 | 93.8 | 12.8 s (16 s wall) | Race branches in `processor.py` |
| opcua-server (`opcua_server`) | 123 / 0 / 0 | 99.0 | 91.1 | 49.0 s (51 s wall) | Clear-text password refusal (`identity.py:88-92`) uncovered |
| shared/observability (`cogniboiler_observability`) | 24 / 0 / 0 | 99.5 | 97.4 | 1.2 s (3 s wall) | |
| shared/runtime (`cogniboiler_runtime`) | 27 / 0 / 0 | 98.4 | 90.9 | 0.6 s (2 s wall) | |
| developer scripts (`scripts`, production code only) | 134 / 0 / 0 (unittest, Windows) | 53.6 | 43.3 | 4.0 s (10 s wall) | 62.6 % if the test files are counted. Most `__main__.py` 12–48 %; `confirm()` untested |
| **Six services combined** | 1 034 / 0 / 0 | **97.4** | **91.2** | | ROADMAP Д11: 97.6 % |

No run hung and none hit Windows App Control (`os error 4551`); each passed on its first attempt. One process issue: my first attempt put `COVERAGE_FILE=<cov>/physics` next to `physics.log`. Coverage's erase step deletes `<data_file>.*`, which removed the log. I reran every package with the data files in `cov/data/`; the figures above come from that clean rerun.

Console: 23 test files and 212 tests pass (Vitest 5.0.0, 11.2 s). No `.skip`, `.only` or `.todo` markers.
- Modules with no test file and no test that renders or imports them: `src/main.tsx` (bootstrap), `src/api/queryKeys.ts`, `src/components/TrendChart.tsx` (replaced by a `vi.mock` in `LiveScreens.test.tsx:37`), and `src/api/types.ts` / `src/api/schema.gen.ts` (types only).
- Tested only through pure helpers: `src/screens/ControlScreen.tsx`. `ControlScreen.test.tsx` imports `LIMITS` and `inRange` and never renders `<ControlScreen>` (see TST-18).
- Exported functions no test touches: `useAcknowledge` and `useAcknowledgeAll` (`alarms/queries.ts:20,29`) are exercised only indirectly through `AlarmsScreen.test.tsx`; `siUnitOf` and `localInputToMs` (`units.ts`) are never named in a test; `readProblem` (`api/http.ts:125`) is exercised only indirectly. In `request()`, the "renewal returns null" and "second 401 after renewal" branches (`api/http.ts:170-178`) are untested (see TST-24).

#### TST-01 — PLC instrument-quality trip never runs in any test
- **Severity:** High
- **Category:** Tests
- **Confidence:** High
- **Location:** `apps/plc-controller/src/plc_controller/safety.py:467-474` (lines 470 and 472 uncovered); `apps/plc-controller/src/plc_controller/measurements.py:37-41` (lines 40-41 uncovered); caller `apps/plc-controller/src/plc_controller/service.py:474`
- **About the file:** `safety.py` is the PLC's interlock layer. Each scan it compares measurements against trip and warning limits and latches the E-Stop. `measurements.py` turns the protobuf state into typed readings with a quality code for each instrument.
- **Problem:** A BAD drum-pressure or drum-level instrument is supposed to trip the unit (`TRIP_SENSORS`, `safety_limits.py:240`), and an UNCERTAIN one to raise a warning. No test passes `sensor_qualities` to `SafetyInterlock.check()`, and no scan runs with a bad trip instrument. A regression would slip through unnoticed: a renamed sensor key, an inverted comparison, or dropping the `sensor_qualities=` argument at `service.py:474`. Each of these would silently remove the "cannot protect without instruments" trip. The fail-safe mapping of an unknown quality code to BAD (`measurements.py:40-41`) is also unexercised.
- **Recommendation:**
  1. In `apps/plc-controller/tests/test_safety.py`, add `@pytest.mark.parametrize("sensor", ["drum_pressure", "drum_level"])` `test_a_bad_trip_instrument_trips(sensor)`. Arrange: a fresh interlock and nominal values. Act: `check(..., sensor_qualities={sensor: 2})`. Assert: `status.level is SafetyLevel.TRIP`, the event parameter is `f"{sensor}_quality"`, and `interlock.emergency_stop.is_active`.
  2. Same file: `test_an_uncertain_trip_instrument_warns_only`. With `sensor_qualities={"drum_level": 1}`, assert `level is SafetyLevel.WARNING` and that the E-Stop is not active.
  3. Same file: `test_a_bad_non_trip_instrument_does_not_trip`. With `{"steam_temp": 2}`, assert `SafetyLevel.NORMAL`. This pins the scope of `TRIP_SENSORS`.
  4. In `apps/plc-controller/tests/test_plc_commands_and_status.py`, add `test_an_unknown_quality_code_reads_as_bad`. Build a `SystemStateMsg` whose drum-level sensor quality is 99, convert it with the measurements builder, and assert `m.is_bad("drum_level")`.
  5. In `apps/plc-controller/tests/test_plc_behaviour.py` (the `plc_harness.rig` lockstep), add `test_a_failed_drum_level_transmitter_trips_the_unit`. Inject the physics `sensor_failure` fault on the drum level through `runtime.inject_fault`, call `rig.advance(3)`, then assert the PLC status reports mode ESTOP with a trip cause of `drum_level_quality`.
- **Effort:** S

#### TST-02 — Interlock arming gates are never exercised; every interlock test is fully armed
- **Severity:** High
- **Category:** Tests
- **Confidence:** High
- **Location:** `apps/plc-controller/src/plc_controller/safety.py:426-427` (the early `return` for a disarmed side is uncovered), `safety.py:440-452` (arming arguments), `safety.py:51-53` (`ArmingTracker.state` uncovered); tests `apps/plc-controller/tests/test_safety.py:55-64` (`nominal_check` never passes `arming=`)
- **About the file:** See TST-01. `evaluate()` lets low pressure trip only when the unit is on line, low flue-gas temperature only once firing is proven, and high steam temperature only when on line.
- **Problem:** All interlock tests use the default `ALL_ARMED`. Nothing verifies that a disarmed side is ignored, or that disarming one side leaves the other side protected. Two regressions would pass the whole suite: swapping `low_armed` and `high_armed`, and computing `low_side` from the wrong threshold. The second would suppress high-side trips while the unit is off line.
- **Recommendation:**
  1. In `test_safety.py`, add class `TestArming` with `test_low_pressure_does_not_trip_while_off_line`. Act: `check(pressure=10e5, ..., arming=ArmingState(on_line=False, firing_proven=False))`. Assert: `SafetyLevel.NORMAL`.
  2. `test_low_pressure_trips_once_on_line`: same values with `ArmingState(on_line=True, firing_proven=False)`. Assert: `SafetyLevel.TRIP`.
  3. `test_high_pressure_trips_even_while_off_line`: `pressure=PRESSURE_LIMITS.trip_high + 1e5` with both flags False. Assert TRIP. This proves disarming only affects the low side.
  4. `test_high_steam_temperature_is_armed_only_on_line`: pass `steam_temp` above `trip_high`. Assert NORMAL when off line and TRIP when on line.
  5. `test_low_flue_gas_temperature_needs_proven_firing`: assert NORMAL with `firing_proven=False` and TRIP or WARNING with `True`.
  6. Add `test_the_tracker_reports_its_last_state` for `ArmingTracker.update(...)` followed by `.state`.
- **Effort:** S

#### TST-03 — High drum-level trip response (feedwater forced shut) is never asserted
- **Severity:** High
- **Category:** Tests
- **Confidence:** High
- **Location:** `apps/plc-controller/src/plc_controller/service.py:605-609` (line 609 uncovered); `apps/plc-controller/src/plc_controller/safety_limits.py:245-252`; weak test `apps/plc-controller/tests/test_safety.py:332-342`
- **About the file:** `service.py` is the PLC scan and command service. `_trip_command` computes the valve positions forced while the E-Stop is latched, according to the trip cause. `trip_overrides` maps a trip cause to those positions.
- **Problem:** An overflow trip must close feedwater (`feedwater=0.0`); every other trip keeps the drum wet through `hold_level`. `test_scenario_4_drum_overflow_trips` only asserts `level == TRIP`, and no test anywhere mentions `feedwater_valve_override`. The scan-level branch that applies the override (`service.py:609`) never runs. If the override is dropped, or applied to the wrong cause, carry-over into the turbine is not caught. Separately, no interlock test passes `steam_temp=`, so the steam-temperature limit is covered only indirectly.
- **Recommendation:**
  1. In `test_safety.py::test_scenario_4_drum_overflow_trips`, add `assert status.feedwater_valve_override == pytest.approx(0.0)` and `assert status.fuel_valve_override == pytest.approx(0.0)`.
  2. Add `test_a_low_level_trip_leaves_feedwater_to_the_level_hold`. With `water_level=0.2`, assert `status.feedwater_valve_override is None`.
  3. In `test_plc_behaviour.py`, add `test_an_overflow_trip_shuts_feedwater_while_latched`. Use `rig(initial_state=<BoilerState with drum level above 7.8 m>)`, call `rig.advance(2)`, and assert the last forwarded command has `feedwater_valve == 0.0` and `fuel_valve == 0.0`.
  4. Add `test_scenario_11_high_steam_temperature_trips` in `test_safety.py` with `steam_temp` above `STEAM_TEMP_LIMITS.trip_high`.
- **Effort:** S

#### TST-04 — Gateway access-token rejection branches untested (expired, malformed subject, missing session, deleted user)
- **Severity:** High
- **Category:** Tests
- **Confidence:** High
- **Location:** `apps/api-gateway/src/api_gateway/auth/identity.py:122-125, 134-139, 162-163`; `apps/api-gateway/src/api_gateway/auth/jwt_handler.py:63-64, 78-79`
- **About the file:** `identity.py` resolves a bearer token to the current user and is the choke point for every authenticated REST route. `jwt_handler.py` signs and verifies RS256 tokens.
- **Problem:** No HTTP test sends an expired access token, so the stable code `auth.token_expired`, which the console's renew logic keys on (`apps/web/src/api/http.ts:165-167`), is unverified. The WebSocket guard timer is tested (`test_websocket.py:285`), but that is a different path. A token with a non-numeric `sub`, a token without `sid`, and a token for a deleted user are also never presented. A regression that let such a token through, or returned 500 instead of 401, would pass.
- **Recommendation:**
  1. In `apps/api-gateway/tests/test_auth_flow.py`, add `test_an_expired_access_token_is_401_token_expired`. Arrange: `issue_access_token(...)` with `monkeypatch` on the clock used by `_issue` so that `exp` lies in the past. Act: `GET /api/v1/status` with that bearer. Assert: 401 and `response.json()["code"] == "auth.token_expired"`.
  2. Add `test_a_token_with_a_non_numeric_subject_is_invalid`. Sign a payload with `jwt.encode` using the ephemeral key from `settings.jwt_private_key`, `sub="abc"` and all required claims. Assert 401 and `auth.token_invalid`.
  3. Add `test_a_token_without_a_session_id_is_invalid`. Use `create_access_token(user_id=1, role="viewer")`, which has an empty `sid`. Assert 401 and `auth.token_invalid`.
  4. Add `test_a_token_of_a_deleted_user_is_session_invalid`. Arrange: `operator_tokens`, then delete the user row through the `get_db` override. Assert 401 and `auth.session_invalid`.
  5. In `test_api_gateway.py::TestJWTHandler`, add `test_signing_without_a_private_key_fails_loudly`. Set `settings.jwt_private_key = ""` with `monkeypatch` and assert `pytest.raises(RuntimeError, match="jwt_private_key")`. Add the same test for the public key on decode.
- **Effort:** S

#### TST-05 — Refresh-token defence branches untested (unknown jti, blocked account, garbage logout cookie)
- **Severity:** High
- **Category:** Tests
- **Confidence:** High
- **Location:** `apps/api-gateway/src/api_gateway/auth/sessions.py:119-120, 148-152, 208-212`
- **About the file:** `sessions.py` owns refresh-token rotation, reuse detection and family revocation for gateway sessions.
- **Problem:** Rotation and reuse are tested. The following are not:
  - a validly signed refresh token whose `jti` is not in the table, or whose `sub` does not match the row (a forged or foreign token);
  - the branch that revokes the whole family when the account was blocked or deleted between refreshes (lines 150-152);
  - `session_id_of` returning `None` for a garbage cookie.
  A regression in any of these reopens sessions for blocked users or crashes logout.
- **Recommendation:**
  1. In `test_auth_flow.py`, add `test_a_refresh_token_unknown_to_the_database_is_401`. Sign a refresh token with `create_refresh_token(1, "viewer", "fam")`, which is never stored. POST `/auth/refresh` with it as the cookie. Assert 401 and `auth.refresh_invalid`.
  2. Add `test_refresh_for_a_blocked_account_closes_the_family`. Arrange: `operator_tokens`, then set `User.is_active = False` directly in the DB (bypassing `update_user`, so `revoke_user_sessions` does not run). Act: refresh. Assert 401 and that every `RefreshToken` row of that family now has `revoked_reason == "blocked"`.
  3. Add `test_logout_with_a_garbage_cookie_still_succeeds`. Send the cookie `refresh_token=not-a-jwt` to `/auth/logout` and assert the documented status (200 or 204) plus a cleared cookie.
- **Effort:** S

#### TST-06 — OPC UA refusal of a clear-text password is never executed
- **Severity:** High
- **Category:** Tests
- **Confidence:** High
- **Location:** `apps/opcua-server/src/opcua_server/identity.py:84-93` (lines 89-92 uncovered); predicate tests `apps/opcua-server/tests/test_security.py:19-33`
- **About the file:** `identity.py` wires gateway sign-in into asyncua sessions. `_UserAwareSession.activate_session` must reject a username token sent unencrypted on the `None` endpoint.
- **Problem:** The pure predicate `sends_password_in_clear` is tested, but the subclass that raises `BadIdentityTokenRejected` never runs. The end-to-end tests use asyncua's client, which always encrypts the password. If the override is not installed, or its condition is inverted, operator passwords could travel in clear and no test would fail.
- **Recommendation:**
  1. In `apps/opcua-server/tests/test_security.py`, add `test_a_session_refuses_a_password_in_clear`. Arrange: `params = ua.ActivateSessionParameters()` with `params.UserIdentityToken = ua.UserNameIdentityToken(UserName="operator1", Password=b"pw", EncryptionAlgorithm=None)`. Create `session = object.__new__(_UserAwareSession)`; the check runs before `super()`. Act and assert: `with pytest.raises(ServiceError)` on `session.activate_session(params, None)`, then check that the raised status code equals `StatusCodes.BadIdentityTokenRejected`.
  2. Add a test that the server really uses `_UserAwareSession`. In `test_opcua_end_to_end.py`, after the server starts, assert that the server's session class, or its patched `InternalSession` factory, is `_UserAwareSession`. This follows the project's "test that the gate is installed" rule.
- **Effort:** S

#### TST-07 — PLC controller hold paths for a bad measurement are untested
- **Severity:** Medium
- **Category:** Tests
- **Confidence:** High
- **Location:** `apps/plc-controller/src/plc_controller/control.py:341-342` (pressure), `361-362` (drum level), `381-384` (steam temperature)
- **About the file:** `control.py` holds `UnitController`, the coordinated pressure, level and steam-temperature cascade the PLC runs in AUTO.
- **Problem:** When a transmitter is BAD, each loop must freeze at its last output (or at the current spray command) instead of integrating garbage. None of the three freeze branches runs. A regression that fed a failed sensor into the PID would drive the feedwater or spray valve to an end stop on an instrument fault.
- **Recommendation:**
  1. In `apps/plc-controller/tests/test_plc_commands_and_status.py`, reuse the `measurements(**overrides)` helper (line 54) in a new class `TestBadMeasurementHold`.
  2. `test_a_bad_level_transmitter_freezes_the_level_trim`. Prime a `UnitController()` with `track(measurements(), targets, dt=1)`. Call `track` again with `measurements(qualities={"drum_level": SignalQuality.BAD}, water_level_m=0.1)`. Assert the level loop output equals the previous one.
  3. `test_a_bad_steam_temperature_holds_the_spray_command`. With `qualities={"steam_temp": BAD}` and `commands.spray=0.3`, assert the spray output is `0.3`.
  4. `test_a_bad_pressure_transmitter_freezes_the_pressure_trim`, analogous for drum pressure.
- **Effort:** S

#### TST-08 — Command-level dry-drum fuel permissive is covered by no test
- **Severity:** Medium
- **Category:** Tests
- **Confidence:** Medium
- **Location:** `apps/plc-controller/src/plc_controller/service.py:304-313` (line 310 uncovered); the permissive itself is tested at `apps/plc-controller/tests/test_safety.py:393-400`
- **About the file:** See TST-03. `send_command` validates an operator's manual valve command before switching to MANUAL and forwarding it.
- **Problem:** Refusing `fuel_valve > 0` while the last measured drum level is at or below trip-low is a hard interlock, and it has no test. Through the public API the branch is mostly shadowed: a low-level scan also latches the E-Stop (`safety.py:476-483`), so the E-Stop message wins. The branch therefore matters for a race, or after a future change to trip latching, and nothing would notice if it were deleted.
- **Recommendation:**
  1. In `apps/plc-controller/tests/test_plc_server.py`, add a white-box `test_fuel_is_refused_while_the_last_level_is_below_trip_low`. Arrange: a `PLCService` with the control loop disabled and `svc._latest_measurements = <ProcessMeasurements with water_level_m=0.4>`; the E-Stop is not active. Act: `await svc.send_command(fuel_valve=0.2, feedwater_valve=0.5, steam_valve=0.5, source=pb2.CommandSource.OPERATOR, operator_id="op")`. Assert: `not result.accepted` and `"Fuel not permitted" in result.reason`.
  2. Add the counterpart `test_zero_fuel_is_accepted_with_a_low_level`, which asserts the command is accepted with `fuel_valve=0.0`.
- **Effort:** S

#### TST-09 — Physics gRPC tests run a real-time plant and sleep 0.2 s (machine-speed dependent)
- **Severity:** Medium
- **Category:** Tests
- **Confidence:** High
- **Location:** `apps/physics-engine/tests/test_grpc_server.py:21-28` (`speed_factor=100.0`, running), `:70-72` (`await asyncio.sleep(0.2)` then `assert after.turbine.electrical_power_w < before...`)
- **About the file:** An integration test of `PhysicsServicer` over an in-process gRPC server.
- **Problem:** Every other runtime test is "stepped rather than timed" (`test_physics_runtime.py:1`), but this class runs the plant on the wall clock. `test_apply_control_command_changes_live_state` passes only if roughly 20 plant steps (IAPWS-97 work in `to_thread`) finish within 200 ms. The project's rules name machine-speed dependence as a defect (`.ai/project/12-domain-rules.md`, Tests). Under coverage tracing or CI load, this test can fail with no real regression.
- **Recommendation:**
  1. In the fixture at `test_grpc_server.py:21`, use `PhysicsRuntimeConfig(start_paused=True, dt=1.0)`.
  2. Replace the sleep with `await self.runtime.step(20)` after the command, then read `GetSystemState`.
  3. In `test_stream_system_state_yields_real_updates`, call `await self.runtime.step(1)` between the two `stream.read()` calls instead of relying on the running loop.
  4. Drop the redundant `@pytest.mark.asyncio`, since `asyncio_mode = "auto"` in the root `pyproject.toml`.
- **Effort:** S

#### TST-10 — Negative assertions after short real-time sleeps can pass vacuously
- **Severity:** Medium
- **Category:** Tests
- **Confidence:** High
- **Location:** `apps/alert-manager/tests/test_alarm_processor.py:138, 159, 311, 320` (`await asyncio.sleep(0.1)` against a real `CLEAR_HOLD_S = 0.02` in `apps/alert-manager/tests/conftest.py:14`); `apps/physics-engine/tests/test_physics_runtime.py:51, 90-92`; `apps/historian/tests/test_historian_runtime.py:522-526`; `apps/api-gateway/tests/test_realtime_hub.py:179`
- **About the file:** These are tests of the alarm clear-hold, runtime pausing, the historian entry point and the gateway's PLC status source.
- **Problem:** Each test waits a fixed wall time and then asserts that something did not happen: no clear, no step, `ensure_storage` not called. On a slow machine the forbidden event may simply not have happened yet, so a real regression passes. That is a machine-speed dependence that hides failures. For example, `test_an_empty_aggregate_bucket_disables_the_policy` (`test_historian_runtime.py:477-526`) asserts `called == []` after 50 ms, before `main` has necessarily reached the storage step.
- **Recommendation:**
  1. Alert-manager: replace each `sleep(0.1)` with an ordering barrier. After the negative action, send a condition on a different key that does produce a change, `await wait_for_state(...)` on it, and only then assert the first key did not change. The processor serialises on one lock, so the barrier proves the earlier work has finished.
  2. Physics `test_resuming_advances_on_the_wall_clock_and_pausing_stops_it`: after `pause()`, call `await paused.wait_for_update(...)` with `asyncio.wait_for(..., 0.2)` inside `pytest.raises(TimeoutError)`, or assert `status.step_count` right after `pause()` returns under the lock.
  3. Historian: make the fake `Sub.run` set an `asyncio.Event` and await it before cancelling. Then `called == []` is checked after `main` has passed the storage step.
  4. Gateway hub: replace `sleep(0.05)` with a wait on the `caplog` record that proves the outage was logged.
- **Effort:** M

#### TST-11 — PLC service error, reconnect and run-change paths partially untested
- **Severity:** Medium
- **Category:** Tests
- **Confidence:** High
- **Location:** `apps/plc-controller/src/plc_controller/service.py:195-198` (health failure means degraded), `444->450` (repeated stream failure logged once), `520-522` (run change clears active alarms and publishes transitions), `660-662` (PhysicsService refuses a forwarded command)
- **About the file:** See TST-03.
- **Problem:** Four behaviours can regress unseen:
  - `physics_status()` reporting "degraded" when Health raises;
  - the "log once per failure" rule for the scan stream (`.ai/project/12-domain-rules.md`, Async services);
  - alarms cleared and published when the plant jumps to a new run;
  - a command refused by physics being turned into a rejection instead of being counted as forwarded.
- **Recommendation:**
  1. In `test_plc_server.py`, add `test_physics_status_is_degraded_when_health_fails` with a fake physics client whose `health()` raises `grpc.aio.AioRpcError`. Assert the result is `"degraded"`.
  2. Add `test_a_refused_forward_is_rejected_and_logged`. The fake client's `apply_control_command` returns `ControlAck(accepted=False, reason="valve range")`. Assert `not result.accepted`, `reason == "valve range"`, `commands_forwarded` unchanged, and the warning present in `caplog`.
  3. In `test_plc_behaviour.py`, add `test_two_stream_failures_log_one_warning`. Use a fake stream that raises twice and then yields. Assert exactly one "PLC scan stream failed" record and one "restored" record.
  4. In `test_plc_restart.py`, add `test_a_new_run_clears_active_alarms_and_announces_them`. Raise a low-level alarm, load a new scenario through the rig, and assert a cleared transition plus a `RUN_CHANGED` event were published through the recording publisher.
- **Effort:** M

#### TST-12 — Gateway upstream-failure paths (simulation, status, realtime reconnect) untested
- **Severity:** Medium
- **Category:** Tests
- **Confidence:** High
- **Location:** `apps/api-gateway/src/api_gateway/routers/simulation.py:144-145, 155-156, 167-168, 180-181, 209-210, 248-249`; `apps/api-gateway/src/api_gateway/routers/status.py:34-35`; `apps/api-gateway/src/api_gateway/realtime/sources.py:67-68`
- **About the file:** `simulation.py` holds the engineer-only simulation control routes, `status.py` the viewer status route, and `sources.py` the background tasks that feed the WebSocket hub from PhysicsService.
- **Problem:** Only `pause` has a 503 test. For resume, speed, step, scenario, fault inject, fault clear and `/status`, nothing verifies that a gRPC error becomes a stable 503 Problem rather than a 500. For scenario and faults, nothing verifies that no `scenario_runs` row is written on failure. The telemetry source's reconnect-after-`RpcError` loop never runs, so the "reconnect with a delay, log once" rule is unverified there.
- **Recommendation:**
  1. In `apps/api-gateway/tests/test_simulation_routes.py`, add one test parametrised over `("resume", {}), ("speed", {"speed_factor": 2}), ("step", {"steps": 1}), ("scenario", {"name": "nominal"}), ("faults", {...}), ("faults/clear", {...})`: `test_an_unreachable_physics_service_is_503(path, body)`. Set `app.state.physics_client.down = True`, POST as engineer, and assert 503, the stable `code`, and zero new `scenario_runs` rows.
  2. In `test_api_gateway.py::TestStatusEndpoint`, add `test_status_is_503_when_physics_is_down`.
  3. In `test_realtime_hub.py`, add `test_the_telemetry_source_reconnects_after_a_stream_error`. The fake physics stream raises `grpc.RpcError` once and then yields a state. Assert that one telemetry frame arrives and that the outage is logged once.
- **Effort:** S

#### TST-13 — WebSocket slow-client, early-disconnect and internal-error closes untested
- **Severity:** Medium
- **Category:** Tests
- **Confidence:** High
- **Location:** `apps/api-gateway/src/api_gateway/routers/websocket.py:177-180` (overflow means close 1013), `230-231` (disconnect during auth), `269-273` (unexpected task error means close 1011)
- **About the file:** The authenticated WebSocket endpoint: first-message auth, per-user guard and revalidation, channel subscriptions.
- **Problem:** Three behaviours are untested:
  - a client that cannot keep up must be told to reload (`CLOSE_TRY_AGAIN_LATER`) instead of silently losing alarm frames;
  - a disconnect before auth must not raise;
  - an internal failure must close with 1011 and log an error.
- **Recommendation:**
  1. In `apps/api-gateway/tests/test_websocket.py`, add `test_a_client_that_falls_behind_is_closed_1013`. Build a hub with `queue_size=1`, authenticate, publish several alarm frames without reading, and assert the close code is 1013 with reason "client too slow; reload state".
  2. Add `test_a_disconnect_before_auth_is_quiet`: open, close immediately, and assert no error record in `caplog`.
  3. Add `test_an_internal_failure_closes_1011`. `monkeypatch` the connection's sender to raise `RuntimeError`, then assert close code 1011 and an error-level log.
- **Effort:** M

#### TST-14 — History `fields` validation (Flux-injection guard) never exercised
- **Severity:** Medium
- **Category:** Tests
- **Confidence:** High
- **Location:** `apps/api-gateway/src/api_gateway/routers/history.py:52-62` (lines 55-62 uncovered); the escaping layer is tested at `apps/api-gateway/tests/test_history_query.py:77`
- **About the file:** `history.py` is the REST route for recorded telemetry. `parse_fields` checks that each requested field name matches `^[a-z][a-z0-9_]{0,63}$` and allows at most 32 names before they reach the Flux query builder.
- **Problem:** No request passes `fields=`, so neither the allow-list nor the count cap is tested. Escaping in `clients.py` is a second layer, but a relaxed regex or a lost 422 would go unnoticed.
- **Recommendation:**
  1. In `test_history_query.py` (or `test_api_gateway.py::TestHistoryEndpoint`), add `@pytest.mark.parametrize("fields", ['pressure_pa") |> drop(', "Pressure", ",".join(f"f{i}" for i in range(33)), "1abc"])` `test_bad_field_lists_are_422(fields)`. GET `/api/v1/history?...&fields=<value>` as viewer and assert 422 with `code == "request.invalid_fields"`.
  2. Add `test_valid_fields_reach_the_historian`. With `fields=pressure_pa, steam_temp_k`, assert that `RecordedHistorianClient` received `("pressure_pa", "steam_temp_k")`, stripped and ordered.
- **Effort:** S

#### TST-15 — No coverage floor anywhere; the root `--cov` default silently overrides `--cov=<pkg>`
- **Severity:** Medium
- **Category:** Tests
- **Confidence:** High
- **Location:** `scripts/quality_gate/config/*.json` (step "service test suites (pytest)", argv `... pytest -q --no-cov`); `pyproject.toml` `[tool.pytest.ini_options] addopts = "--cov --cov-report=term-missing"`; `.venv/Lib/site-packages/pytest_cov/plugin.py:184` (`return None if True in cov_source`)
- **About the file:** The quality gate is the one verification path CI runs. The root pytest options apply to every local pytest call.
- **Problem:** Coverage is measured by nobody. The gate disables it, and CI runs the gate. The 97.6 % in ROADMAP Д11 is a one-time snapshot (97.4 % today), and a drop would go unnoticed. The bare `--cov` in `addopts` also makes any `--cov=physics_engine` on the command line equivalent to measuring everything (pytest-cov merges them into `source=None`). Anyone reproducing per-package numbers gets wrong figures unless they pass `-o addopts=`.
- **Recommendation:**
  1. Add a gate step, or a separate CI job so the quick gate stays fast, that runs `pytest -q -p no:cacheprovider -o addopts= --cov=physics_engine --cov=plc_controller ... --cov-branch --cov-fail-under=95`. Also add `[tool.coverage.report] fail_under` plus per-package floors: at least 95 % line for `plc_controller` and `api_gateway`, at least 97 % for the rest.
  2. Replace the bare `--cov` in `addopts` with explicit sources, or remove it from `addopts` and put coverage options only in the new step. That keeps `--cov=<pkg>` meaningful.
  3. Record the chosen floor in `docs/__arch__/ROADMAP.md` next to Д11 so the claim is checked rather than remembered.
- **Effort:** S

#### TST-16 — Developer scripts: the destructive-operation guards are untested
- **Severity:** Medium
- **Category:** Tests
- **Confidence:** High
- **Location:** `scripts/_toolkit/console.py:45-67` (all of `confirm()` uncovered); `scripts/clean_caches/__main__.py:33-62` (dry-run gate, 43 % file coverage); `scripts/restore/__main__.py:119-200` (confirmation and restore sequence, 37 %)
- **About the file:** `confirm()` is the shared "ask before doing something irreversible" prompt. `clean-caches` deletes files only with `--apply`. `restore` overwrites both databases after asking first.
- **Problem:** The documented contract (no terminal and no flag means no, and EOF means no) and the `--apply` gate protect against data loss, yet neither has a test. Scripts production code is at 53.6 % line and 43.3 % branch coverage. A refactor that made `confirm()` return `True` on EOF, or inverted `if not args.apply`, would delete data with the suite green.
- **Recommendation:**
  1. In `scripts/_toolkit/tests/`, add `test_console.py` with `ConfirmTest`:
     - `test_assume_yes_short_circuits`;
     - `test_no_terminal_means_no`: patch `sys.stdin` with an `io.StringIO` whose `isatty()` returns False and assert `False`;
     - `test_eof_means_no`: patch `isatty` to True and `builtins.input` to raise `EOFError`;
     - `test_only_y_or_yes_is_consent`: parametrise via `subTest` over "y", "YES", "n", "", "maybe".
  2. In `scripts/clean_caches/tests/`, add `test_main.py::test_without_apply_nothing_is_deleted`. Patch `repo_root` to a `TemporaryDirectory` holding a `__pycache__`, run `main([])`, and assert the directory still exists and the return code is 0. Add `test_apply_deletes_the_plan`.
  3. In `scripts/restore/tests/test_restore.py`, add `test_a_refused_confirmation_runs_nothing`. Patch `confirm` to return False and `run` with a recorder. Assert exit 1 and that `run` was never called.
- **Effort:** M

#### TST-17 — Physics spray-valve command never reaches the controls in any test
- **Severity:** Medium
- **Category:** Tests
- **Confidence:** High
- **Location:** `apps/physics-engine/src/physics_engine/plant.py:279-280` (line 280 uncovered); `apps/physics-engine/src/physics_engine/server.py:112-115` (the `HasField("spray_valve")` path is never used with a value)
- **About the file:** `plant.py` is the plant model's command and fault surface. `apply_command` validates and stores the valve commands.
- **Problem:** Only an out-of-range spray value is tested; it raises before line 280. A valid spray command from the PLC is never applied in the physics suite, so a regression that dropped it would leave steam-temperature control without an actuator, and no physics test would fail.
- **Recommendation:**
  1. In `apps/physics-engine/tests/test_physics_plant.py`, add `test_a_valid_spray_command_is_stored`. Call `plant.apply_command(fuel_valve=0.5, feedwater_valve=0.5, steam_valve=0.5, spray_valve=0.3)` and assert `plant.snapshot` or the controls show `spray_valve_command == 0.3` after one step.
  2. In `test_physics_runtime.py`, add `test_the_grpc_command_forwards_the_spray_valve`. Send `ControlCommandMsg(..., spray_valve=0.25)` through `PhysicsServicer` and assert the next `GetSystemState` reports `spray_valve_command == 0.25`.
  3. Add the counterpart: without the field set, the previous spray command is kept.
- **Effort:** S

#### TST-18 — The console's ControlScreen is never rendered in unit tests
- **Severity:** Medium
- **Category:** Tests
- **Confidence:** High
- **Location:** `apps/web/src/screens/ControlScreen.tsx:108-476` (components `LoadSection`, `ModeSection`, `ValvesSection`, `SetpointsSection`, `ResetSection`, `ControlScreen`); `apps/web/src/screens/ControlScreen.test.tsx:1-52` (imports only `LIMITS`, `inRange`)
- **About the file:** The operator's command screen: load, mode, manual valves, setpoints and E-Stop reset, each behind a confirmation dialog, with buttons disabled while the E-Stop is latched.
- **Problem:** Several guard rules are checked only by Playwright against a running stack (`apps/web/e2e/control.spec.ts`, `roles.spec.ts`), not by the Vitest gate:
  - AUTO, MANUAL, trip and valve buttons are disabled while `emergency_stop_active` (lines 173, 182, 192, 270);
  - an out-of-range value disables send;
  - Cancel sends nothing;
  - the setpoint and reset sections are hidden below engineer.
  A local change can break them without a failing unit test.
- **Recommendation:**
  1. In `ControlScreen.test.tsx`, add `describe("ControlScreen")` with a render helper that follows `EngineerScreen.test.tsx` (providers plus `src/test/fixtures.ts`).
  2. `it("disables mode, trip and valve buttons while the E-Stop is latched")`: render with `plc.emergency_stop_active = true` and assert each button is `toBeDisabled()`.
  3. `it("asks before sending and a cancel sends nothing")`: click "Set … MW", click Cancel in the dialog, and assert the mocked `endpoints.setLoad` was not called.
  4. `it("hides setpoints and reset from an operator")`: render with the operator role and assert `queryByRole("button", {name: /Apply setpoints/})` is null.
- **Effort:** M

#### TST-19 — Silent `except Exception: pass` fallbacks in the steam tables are untested
- **Severity:** Medium
- **Category:** Reliability
- **Confidence:** Medium
- **Location:** `apps/physics-engine/src/physics_engine/steam_tables.py:127-134` (lines 131-132 uncovered), `278-285` (281-285 uncovered), and the similar blocks at `311-315` and `353-357`
- **About the file:** `steam_tables.py` wraps IAPWS-97 water and steam property calls for the boiler and turbine models.
- **Problem:** When IAPWS-97 raises, the code silently falls back to saturated-state values, with no log and no test. `.ai/universal/06-quality-and-testing.md` forbids silently swallowed errors. A property-call bug, such as a unit mix-up that pushes the state outside a region, would quietly return saturated values and distort the simulation without any signal. This finding asks only for tests; any decision to log is for the owner.
- **Recommendation:**
  1. In `apps/physics-engine/tests/test_steam_tables.py`, add a parametrised `test_an_iapws_failure_falls_back_to_saturation(func, sat_kwargs)`. `monkeypatch` `steam_tables.IAPWS97` with a wrapper that raises `NotImplementedError` for the `(T, P)` signature and delegates otherwise. Assert the result equals `IAPWS97(P=..., x=0 or 1)` in SI units (density, entropy and so on).
  2. Report the silent swallow to the owner as a Reliability debt. The test pins current behaviour so that any later logging change is deliberate.
- **Effort:** S

#### TST-20 — Weak assertions on type or not-None only
- **Severity:** Low
- **Category:** Tests
- **Confidence:** High
- **Location:**
  - `apps/api-gateway/tests/test_api_gateway.py:40` (`isinstance(result, str)`), `:82`;
  - `apps/api-gateway/tests/test_models.py:239, 278, 338` (`id is not None`);
  - `apps/historian/tests/test_historian.py:101, 178, 249` (`isinstance(point, Point)`);
  - `apps/physics-engine/tests/test_mqtt_publisher.py:63, 100`;
  - `apps/physics-engine/tests/test_turbine.py:207`;
  - `apps/opcua-server/tests/test_address_space.py:143, 167` (`hasattr`);
  - `apps/opcua-server/tests/test_opcua_gateway_client.py:314`;
  - `apps/physics-engine/tests/test_grpc_server.py:52-53` (`pressure_pa > 0`);
  - `scripts/restore/tests/test_restore.py:42`.
- **About the file:** Unit tests of password hashing, ORM models, point builders, protobuf mappers and address-space field keys.
- **Problem:** These tests pass for almost any output of the right type. Most have stronger siblings; the `test_models.py` and `test_opcua_gateway_client.py:314` cases do not. In the latter, `tokens() is not None` does not prove that "invalidating before sign-in changes nothing".
- **Recommendation:**
  1. Delete the pure type tests whose siblings already assert content (`test_hash_returns_string`, `test_access_token_is_string`), or merge them into the content tests.
  2. `test_models.py:239` and similar: re-read the row in a new session and assert the foreign keys, `granted_at_ms` and the role name.
  3. `test_opcua_gateway_client.py:314`: assert `await session.tokens() == <the original tokens>` and `client.refreshes == []`.
  4. `test_grpc_server.py:52`: compare with `runtime.snapshot` values using `pytest.approx`.
- **Effort:** S

#### TST-21 — Near-identical test helpers duplicated across services
- **Severity:** Low
- **Category:** Tests
- **Confidence:** High
- **Location:**
  - `until(predicate)`: `apps/alert-manager/tests/test_alarm_runtime.py:249`, `apps/opcua-server/tests/test_opcua_methods.py:275`, `apps/physics-engine/tests/test_physics_plant.py:177`, `apps/physics-engine/tests/test_physics_runtime.py:39`, `apps/plc-controller/tests/test_plc_publisher.py:82`;
  - `FakeBroker`: `apps/alert-manager/tests/test_alarm_runtime.py:198`, `apps/historian/tests/test_historian_runtime.py:319`, `apps/physics-engine/tests/test_physics_plant.py:148`;
  - the `broker` fixture in three of those files;
  - `FakeInflux`: `apps/api-gateway/tests/test_upstream_clients.py`, `apps/historian/tests/test_historian_runtime.py`;
  - the probe-a-free-port idiom (TST-23) in five files;
  - `bearer(tokens)` defined six times inside api-gateway (`test_alarm_routes.py:15`, `test_audit_log.py:23`, `test_auth_flow.py:20`, `test_plant_commands.py:19`, `test_simulation_routes.py:18`, `test_user_administration.py`).
- **About the file:** Per-service test modules.
- **Problem:** The copies have already drifted: `until` has timeouts of 5 s and 10 s and poll intervals of 1 ms and 5 ms, and one version has `# type: ignore`. A fix to one fake broker (for example, reconnect semantics) does not reach the others.
- **Recommendation:**
  1. Move `bearer` into `apps/api-gateway/tests/gateway_fakes.py` and import it everywhere. This touches only the gateway.
  2. Create a development-only test-support package, for example `shared/testing/src/cogniboiler_testing/` with `async def until(predicate, timeout_s=5.0, poll_s=0.001)`, `FakeMqttClient` and `free_tcp_port()`. Add it as a dev dependency of each service, in line with the "through a development dependency only" rule in `.ai/project/12-domain-rules.md`, then replace the copies.
  3. Keep service-specific fakes (`gateway_fakes.py`, `opcua_fakes.py`, `plc_harness.py`) where they are.
- **Effort:** M

#### TST-22 — Many near-identical tests that should be parametrised
- **Severity:** Low
- **Category:** Tests
- **Confidence:** High
- **Location:**
  - `apps/plc-controller/tests/test_safety.py:259-457`: about 20 scenario tests repeating the same four-argument `check()` call;
  - `apps/api-gateway/tests/test_api_gateway.py:147-170` (`TestRBAC`, 8 one-line tests);
  - `test_api_gateway.py:176-205` (`TestHealthEndpoint`, 5 tests of one GET);
  - `test_api_gateway.py:407-490` (bound checks).
- **About the file:** The interlock unit tests and the gateway's original catch-all test module.
- **Problem:** Repetition hides which limits are covered. For example, the overflow trip checks the level but not the overrides (TST-03), and steam temperature is missing entirely. Adding the missing cases means copying more boilerplate.
- **Recommendation:**
  1. In `test_safety.py`, add a table `TRIPS = [("pressure_pa", dict(pressure=190e5), 0.0, 1.0, None), ...]` holding the parameter, the overrides, and the expected fuel, steam and feedwater positions. Write one `@pytest.mark.parametrize` test `test_each_trip_latches_with_its_valve_response` that asserts the level, the event parameter and all three overrides.
  2. Collapse `TestRBAC` into one parametrised `test_role_levels(role, level)` plus one ordering test.
  3. Collapse `TestHealthEndpoint` into one test asserting the full JSON body.
- **Effort:** S

#### TST-23 — Port probing then release (TOCTOU) in six tests
- **Severity:** Low
- **Category:** Tests
- **Confidence:** High
- **Location:** `apps/physics-engine/tests/test_physics_runtime.py:326-329`; `apps/plc-controller/tests/test_plc_behaviour.py:245-247, 318-320`; `apps/alert-manager/tests/test_alarm_runtime.py:148-150`; `apps/opcua-server/tests/test_opcua_end_to_end.py:45-48`; `shared/observability/tests/test_observability_streams.py:206-208`
- **About the file:** Tests of `serve()` entry points and the OPC UA end-to-end client.
- **Problem:** Each test binds port 0, reads the port, closes the socket, then asks the server to bind the same port. Another process can take the port in between, a classic source of rare CI flakes. The "port 0 only" rule is followed in letter but not in effect. The harnesses that call `add_insecure_port("127.0.0.1:0")` directly (`plc_harness.py`, `test_grpc_server.py`) are fine.
- **Recommendation:**
  1. Put one `free_tcp_port()` helper in the shared test-support module (TST-21).
  2. Wrap each `serve(port=...)` start in a small retry: up to 3 attempts, re-probing if the first health call fails with `UNAVAILABLE` because the bind lost the race. Alternatively, where the entry point already returns the bound server, use the port it actually bound.
  3. `test_grpc_server.py:34` binds `[::]:0` and then dials `localhost`. Use `127.0.0.1:0` as `plc_harness.py` does, because this environment reaches only IPv4 loopback (`.ai/project/14-command-reference.md`, Platform notes).
- **Effort:** S

#### TST-24 — Console session-renewal failure branches untested
- **Severity:** Low
- **Category:** Tests
- **Confidence:** High
- **Location:** `apps/web/src/api/http.ts:170-178`; `apps/web/src/api/http.test.ts:48-69`
- **About the file:** `http.ts` is the console's single REST client; it renews an expired access token once and repeats the request.
- **Problem:** Only a successful renewal and a non-renewable 401 are tested. Two branches have no test: `renew()` returning `null` (the refresh cookie is gone), and a second 401 after renewal, which must end the session. A regression here could put the console into a renew loop or leave a dead session looking signed in.
- **Recommendation:**
  1. In `http.test.ts`, add `it("gives up when renewal fails")`. The fetch mock answers 401 `auth.token_expired`, and `credentials.renew` resolves `null`. Assert `request()` rejects with `ApiError` and that fetch was called once.
  2. Add `it("ends the session when the repeated request is refused again")`. Mock 401 twice with renew returning `"t2"`. Assert `session.ended` was called with the second problem's code and fetch was called twice.
- **Effort:** S

#### TST-25 — Two service entry points at 0 % coverage
- **Severity:** Low
- **Category:** Tests
- **Confidence:** High
- **Location:** `apps/plc-controller/src/plc_controller/__main__.py:3-47` (23 statements, 0 %); `apps/api-gateway/src/api_gateway/__main__.py:8-35` (17 statements, 0 %)
- **About the file:** Argument parsing and wiring for the PLC controller and the gateway's uvicorn launcher.
- **Problem:** Every other service has entry-point tests (physics `test_physics_plant.py:290-333`, historian, alert-manager, OPC UA). For these two, a wrong default (for example `--physics-target`), a missing `configure_logging`, or the Windows `SelectorEventLoop` factory required by `.ai/project/12-domain-rules.md` can regress unseen.
- **Recommendation:**
  1. In `apps/plc-controller/tests/test_plc_server.py`, add `test_the_command_line_defaults`, patching `sys.argv` as `test_physics_plant.py:291` does. Assert the physics target, gRPC port, metrics port and MQTT host defaults.
  2. Add `test_main_wires_service_and_server`. `monkeypatch` `serve`, `PLCService.start` and `start_metrics_server` with recorders, run `main(args)` to completion, and assert each was called with the parsed values.
  3. In `apps/api-gateway/tests/test_gateway_startup.py`, add `test_the_launcher_passes_host_and_port_to_uvicorn` with `uvicorn.run` patched.
- **Effort:** S

#### TST-26 — Uncovered model branches in physics (startup classification, fuel-rich combustion)
- **Severity:** Low
- **Category:** Tests
- **Confidence:** High
- **Location:** `apps/physics-engine/src/physics_engine/plant.py:440-446` (hot, warm or cold start classification); `apps/physics-engine/src/physics_engine/combustion.py:111-117` (air ratio below 1.0 and near-stoichiometric); `combustion.py:217-219` (stack loss helper); `apps/physics-engine/src/physics_engine/operating_point.py:197-201`
- **About the file:** Equipment-health start counting and the combustion efficiency model.
- **Problem:** These plausible regressions have no test: miscounted start types, which feed equipment-health wear, and wrong efficiency when the air ratio falls below 1.0.
- **Recommendation:**
  1. In `apps/physics-engine/tests/test_equipment_health.py`, add a parametrised `test_a_start_is_classified_by_inlet_temperature(temp_k, expected)` covering values above `HOT_START_STEAM_TEMP_K`, between the warm and hot thresholds, and below the warm threshold. Drive `plant._update_health` or run a scenario from cold to online, and assert the start counter for `expected` rose by one.
  2. In the combustion tests, add `test_fuel_rich_combustion_loses_efficiency` (ratio 0.9 gives `COMBUSTION_EFFICIENCY * 0.95`) and `test_the_optimal_band_is_flat` (ratio 1.03).
- **Effort:** S

#### TST-27 — Platform-conditional skips mean two script tests never run in CI
- **Severity:** Info
- **Category:** Tests
- **Confidence:** High
- **Location:** `scripts/runner/tests/test_execution.py:33-40, 42-50` (`skipUnless(os.name == "nt", ...)`); `scripts/dev_secrets/tests/test_certificates.py:34` (`skipUnless(HAS_CRYPTOGRAPHY, ...)`); CI runs on `ubuntu-latest` (`.github/workflows/ci.yml:28`)
- **About the file:** Tests of the orchestrator's argument splitting and of dev-certificate generation.
- **Problem:** The two Windows path-splitting tests are skipped on every CI run, so they guard only local Windows runs. The reasons are legitimate. There are no `pytest.mark.skip`, `skipif` or `xfail` markers in any service or shared suite, which complies with `.ai/project/11-commands.md`.
- **Recommendation:**
  1. Keep the skips. Optionally add a `windows-latest` job that runs only `python -W error -m unittest discover -s scripts -t .` so the two tests execute somewhere automated.
  2. In the gate's "dev scripts" step, assert that `cryptography` is importable, so `test_certificates.py` cannot be skipped silently in CI.
- **Effort:** S

### 7.11 Architecture conformance — invariants, boundaries, contracts

**Area summary.** Service boundaries at the import level are clean — no service imports another's package, the only cross-service dependency (plc-controller → physics-engine) is a dev group, and PostgreSQL table ownership and the MQTT ACL match the documents. The weak spot is how the two actuator-related invariants (I2 "actuators only through the PLC" and I11 "edge writes only through the gateway") are enforced: purely by which methods the client wrappers happen to expose, on unauthenticated gRPC services that sit on one flat Compose network with every other container — and no test would fail if a new code path called a write RPC. The append-only audit log (I10) is well designed (triggers plus a non-owner role) but neither the triggers nor the grants are exercised by any automated test, because the suites build the schema with `create_all`. MQTT JSON producers and consumers currently agree field-for-field, but there is no shared schema or version, the gateway forwards the payloads to the browser unvalidated, and the console declares those shapes by hand. The rest is documentation drift and hygiene.

<details><summary>Scope reviewed by the area auditor</summary>

`docs/architecture/invariants.md`, `docs/architecture/service-boundaries.md`, `docs/architecture/overview.md`, `.ai/project/10-project-map.md`, `.ai/project/12-domain-rules.md`, `.ai/project/14-command-reference.md`, `.ai/universal/04-architecture-boundaries.md`; every `apps/*/pyproject.toml` and the root `pyproject.toml`; `docker-compose.yml`, `infrastructure/docker/mosquitto/{acl,mosquitto.conf,start-broker.sh}`, `infrastructure/nginx/*.inc`, `.gitignore`, `.pre-commit-config.yaml`; `shared/proto/cogniboiler.proto` (units); physics-engine `server.py`, `runtime.py`, `plant.py` (command path), `mqtt_publisher.py` (topics); plc-controller `service.py`, `server.py`, `client.py`, `commands.py`, `safety.py`, `safety_limits.py`, `alarms.py`, `events.py`, `__main__.py`; api-gateway `clients.py`, `realtime/sources.py`, `routers/{commands,simulation,alarms,users,kpi}.py`, `db_roles.py`, `db_init.py`, `models/user.py` (table list), migrations `0001`, `0003`, `0004`; alert-manager `payloads.py`, `publisher.py`, `views.py`, `models.py` (tables); historian `points.py`, `subscriber.py` (topics); opcua-server `client.py`, `methods.py`, `subscriber.py` (topics); `apps/web/src/api/types.ts`; the tests named by or relevant to each invariant (`apps/api-gateway/tests/{test_upstream_clients,test_plant_commands,test_audit_log,test_db_roles,conftest}.py`, `apps/plc-controller/tests/{test_plc_behaviour,test_plc_server,plc_harness}.py`, the opcua-server test list). Two read-only interpreter checks of `SafetyInterlock` with NaN inputs (no files written).

</details>

#### Invariant verification table

| Invariant | Enforced in code? | Bypass found? | Test pins it? | Notes |
|---|---|---|---|---|
| I1 physics single owner of state | Yes (only physics-engine holds `PlantSimulator`; others hold no plant model) | No code path in services; `ml/preprocessing/generate_dataset.py` runs its own simulator offline (deferred AI, not a service) | No (review only — as the doc honestly says) | Import check clean. |
| I2 actuators only via PLC | Partly — `PhysicsGatewayClient` exposes no valve method (`apps/api-gateway/src/api_gateway/clients.py:87-176`) | Yes, structurally: gateway holds a full `PhysicsServiceStub` (`clients.py:100`); physics `ApplyControlCommand` accepts any caller (`apps/physics-engine/src/physics_engine/server.py:103-121`) on a flat network | No — no test asserts the gateway/opcua never reference `ApplyControlCommand` | ARCH-01, ARCH-02. "Enforced by" wording ("only holds a PLC command client") is inaccurate. |
| I3 interlocks cannot be disabled | Yes: no config/env/CLI flag; thresholds are module constants; PID/SAFETY sources refused (`apps/plc-controller/src/plc_controller/service.py:65-67,294-297`); reset refused with blockers (`service.py:364-373`); gateway reset needs engineer | NaN measurement silently passes pressure/temperature trips (ARCH-05); `ParameterLimits` mutable (ARCH-13) | Yes: `test_plc_behaviour.py:109-119` (reserved sources), `:204-209` (reset refused), `test_plant_commands.py:141-175` (engineer accepted, operator 403) | Acceptance and rejection both pinned — good. |
| I4 SI + `timestamp_ms` | Mostly | Contract fields in hours, percent, ppmv, kg/MWh (`shared/proto/cogniboiler.proto:103-123`) | No (naming review) | ARCH-10. Timestamps consistently `*_ms`. |
| I5 no committed secret | `.gitignore:138,430-431` | No secret files tracked (`git ls-files` check) | No scanner (no `detect-private-key` hook, no CI secret scan) | ARCH-14 (Info). Doc is honest ("and review"). |
| I6 additive contracts | `generate-proto --check` / `generate-openapi --check` in gate | MQTT/WebSocket JSON has no schema or check (ARCH-04) | Only proto/OpenAPI | |
| I7 AI optional | Yes: `ai-predictor` not in `[tool.uv.workspace].members` (`pyproject.toml:18-30`) | No importer found | Implicitly (gate runs without it) | |
| I8 one machine, no internet | Compose + provisioning from repo | Base images pinned only to moving minor tags (`docker-compose.yml:36,64,88`), not digests | CI `stack` job | Not reported separately (supply-chain audit area). |
| I9 tests need no infra | Yes (in-process gRPC on port 0, lockstep harness) | — | Gate | Text of I9 is stale: Д2 closed 2026-09-19 (ARCH-09). |
| I10 audit append-only | Yes: triggers (`migrations/versions/0003_sessions_and_append_only_audit.py:33-53`), role limited to SELECT/INSERT (`0004_application_roles.py:70-72`), gateway connects as `cogniboiler_gateway` (`docker-compose.yml:272`) | No application UPDATE/DELETE of `audit_log` found | **No** — suites use `Base.metadata.create_all` (`apps/api-gateway/tests/conftest.py:63`), so triggers and grants are never exercised | ARCH-03. FK `ON DELETE SET NULL` latent contradiction (ARCH-15). |
| I11 edge writes via gateway | Yes: OPC UA methods POST to gateway routes with the session token (`apps/opcua-server/src/opcua_server/methods.py:132-190`) | Structurally possible: `PLCStatusClient`/`AlarmReadClient` wrap full command-capable stubs (`apps/opcua-server/src/opcua_server/client.py:17,31`) | Acceptance yes (`test_opcua_end_to_end.py:156-218`); no test that opcua never uses a write RPC | ARCH-01, ARCH-02. |

#### ARCH-01 — Actuator and PLC-write RPCs are unauthenticated and reachable from every container on one flat network
- **Severity:** High
- **Category:** Security
- **Confidence:** High
- **Location:** `docker-compose.yml` (no `networks:` key anywhere: all 13 services share the default network); `apps/physics-engine/src/physics_engine/server.py:103-121` (`ApplyControlCommand`, no caller check), `:262-263` (insecure port on all interfaces); `apps/plc-controller/src/plc_controller/server.py:62-165,194-199` (`SendCommand`, `SetControlMode`, `ResetEmergencyStop`, `UpdateSetpoints` with caller-supplied `operator_id`); `apps/api-gateway/src/api_gateway/clients.py:100`; `apps/opcua-server/src/opcua_server/client.py:17,31`; `docs/architecture/invariants.md:27-28`
- **About the file:** `server.py` in physics-engine is the gRPC transport for the plant; `server.py` in plc-controller is the PLC's gRPC transport; `docker-compose.yml` wires the whole local stack.
- **Problem:** I2 says actuators are reached only through the PLC, and I3/I11 say resets and edge writes are role-checked and audited by the gateway. In code, the only thing keeping other callers off `PhysicsService.ApplyControlCommand` and `PLCService.ResetEmergencyStop` is that the physics port is not published to the host. Inside Compose, every container — including third-party images (Grafana, Prometheus, InfluxDB, Mosquitto, nginx) — can open `physics-engine:50052` and `plc-controller:50051`. Physics applies any in-range valve command regardless of the PLC's E-Stop latch; the PLC accepts a reset or mode change with any `operator_id` and without the gateway's role check or audit row. Failure scenario: a compromised Grafana plugin (or a future service with a bug) calls `ApplyControlCommand(fuel_valve=1.0)` while the unit is tripped; in MANUAL mode the PLC does not re-send on each scan (`service.py:493-495`), so the command stands, and nothing records who did it. The "Enforced by" line ("The gateway only holds a PLC command client") is also inaccurate: the gateway holds a full `PhysicsServiceStub`, it merely does not wrap `ApplyControlCommand`.
- **Recommendation:**
  1. Segment the Compose network (hardening, no new logic): define a `plant` network containing only `physics-engine`, `plc-controller` and `api-gateway` (the gateway needs physics for reads and simulation control), an `edge` network for `api-gateway`, `opcua-server`, `plc-controller`, `alert-manager`, and keep Grafana, Prometheus, InfluxDB, Mosquitto and `web` off the `plant` network. Prometheus then needs a route to `:9100` of each service — bind metrics on a separate `metrics` network.
  2. Correct I2's "Enforced by" text to say what is true today (wrapper convention + unpublished port), and add the network segmentation and the test from ARCH-02 as the enforcement once they exist (registry change — owner decision).
  3. Authenticating gRPC callers (a per-service token in metadata checked by a server interceptor, or mTLS) is the proper fix but touches secrets and the contract — out of scope, owner decision; record it as debt in the decision log.
  4. Optional follow-up, owner decision (it is new PLC logic): the PLC could compare the plant's reported valve commands with its own last command and raise a discrepancy event.
- **Effort:** M (network segmentation and doc), L (caller authentication)

#### ARCH-02 — No test pins I2 or I11: a new code path to a write RPC would pass the gate
- **Severity:** Medium
- **Category:** Tests
- **Confidence:** High
- **Location:** `apps/api-gateway/src/api_gateway/clients.py:100`; `apps/opcua-server/src/opcua_server/client.py:12-39`; `apps/api-gateway/tests/test_upstream_clients.py:43-103` (fake servicer simply omits `ApplyControlCommand`); `docs/architecture/invariants.md:27-28,139-142`
- **About the file:** `clients.py` (gateway) holds every upstream gRPC wrapper; `client.py` (opcua-server) holds the "read-only" PLC and alarm wrappers.
- **Problem:** Both invariants are enforced only by which methods the wrappers expose, while `self._stub` underneath can call any RPC. A single added line such as `await self._stub.ApplyControlCommand(...)` in the gateway, or `self._stub.SendCommand(...)` in opcua-server, compiles, type-checks and passes every existing test (the gateway fake would answer UNIMPLEMENTED only in the test that happens to exercise that path). `04-architecture-boundaries.md` states that a doc-comment-only boundary is not enforced; this is that case.
- **Recommendation:**
  1. Add an architecture test (e.g. `apps/api-gateway/tests/test_boundaries.py` and one in opcua-server) that walks the package source with `ast` and fails if any attribute access named `ApplyControlCommand` appears in `api_gateway`, and if any of `SendCommand`, `UpdateSetpoints`, `ResetEmergencyStop`, `SetLoadDemand`, `SetControlMode`, `AcknowledgeAlarm`, `AcknowledgeAll`, or any `PhysicsServiceStub` usage appears in `opcua_server`.
  2. Make the test prove it can accept: assert that the gateway's allowed RPCs (`SendCommand`, `ResetEmergencyStop`, …) are found, so the scanner is known to see the code.
  3. Optionally narrow the attribute type: store the stub as a `Protocol` listing only the permitted RPCs so mypy refuses the forbidden call.
  4. Name the test in the "Enforced by" lines of I2 and I11 (owner decision on the registry text).
- **Effort:** S

#### ARCH-03 — The append-only audit triggers and the restricted grants are never exercised by any test
- **Severity:** Medium
- **Category:** Tests
- **Confidence:** High
- **Location:** `apps/api-gateway/migrations/versions/0003_sessions_and_append_only_audit.py:33-75,206-209`; `apps/api-gateway/migrations/versions/0004_application_roles.py:37-77`; `apps/api-gateway/tests/conftest.py:63`, `apps/api-gateway/tests/test_models.py:44`, `apps/api-gateway/tests/test_gateway_startup.py:40` (all `Base.metadata.create_all`); `docs/architecture/invariants.md:121-126`
- **About the file:** migration 0003 creates the triggers that refuse UPDATE/DELETE/TRUNCATE on `audit_log`; 0004 creates the two non-owner application roles and their grants.
- **Problem:** I10 is the strongest security invariant in the registry, and its "Enforced by" line names exactly these two migrations. But no test runs the Alembic chain: the suites create tables from the ORM models, which carry no trigger (`models/user.py:12-13` says so), and `test_db_roles.py` covers only password provisioning. Nothing in CI's `smoke` checks it either. A future migration that drops or re-creates `audit_log` (e.g. a batch alter on SQLite, or a table rebuild) could silently lose the triggers, or a grant list edit could add `UPDATE`, and the gate would stay green.
- **Recommendation:**
  1. Add a test that runs `alembic upgrade head` against a temporary SQLite file (the 0003 SQLite branch already creates triggers), inserts one `audit_log` row (acceptance), then asserts `UPDATE` and `DELETE` raise with "append-only".
  2. Add a static test over `0004_application_roles.GATEWAY_APPEND_ONLY` and the generated statements asserting `audit_log` is granted exactly `SELECT, INSERT` and never appears in `GATEWAY_READ_WRITE`, and that no statement grants `TRIGGER`, `TRUNCATE` or role membership.
  3. Optionally add a `smoke` step that, as `cogniboiler_gateway`, attempts `UPDATE audit_log` and expects `insufficient_privilege` (PostgreSQL-only proof of the grants).
  4. Consider `alembic check` (models vs. chain drift) in the gate — also catches a model change without a migration.
- **Effort:** M

#### ARCH-04 — MQTT JSON contracts have no schema or version, the gateway forwards them to browsers unvalidated, and the console declares them by hand
- **Severity:** Medium
- **Category:** Reliability
- **Confidence:** High
- **Location:** producers `apps/plc-controller/src/plc_controller/events.py:88-131`, `apps/alert-manager/src/alert_manager/payloads.py:156-164`; consumers `apps/alert-manager/src/alert_manager/payloads.py:104-153`, `apps/historian/src/historian/points.py:180-227`, `apps/api-gateway/src/api_gateway/realtime/sources.py:90-139`; console `apps/web/src/api/types.ts:54-88`; rule `.ai/project/14-command-reference.md:113`
- **About the file:** `events.py` builds the PLC's `alerts/*` and `plc/events` payloads; `sources.py` relays `plc/events` and `alarms/changes` to the WebSocket; `types.ts` names the console's API shapes.
- **Problem:** I verified field-by-field that today's producers and consumers agree (`alerts/*`: key/state/source_service/severity/parameter/direction/unit/value/threshold/action/message/timestamp_ms; `alarms/changes`: alarm.id/severity/parameter/message/value/threshold, transition.at_ms/to_state/actor; `plc/events`: kind/operator_id/detail/timestamp_ms). But each side re-types the keys as string literals, there is no `schema_version`, and the only guard against drift is review. The gateway passes any JSON object straight to every signed-in browser (`sources.py:131-137`), and `types.ts:54-56` admits the WebSocket frames are declared by hand — contrary to "The console never declares a gateway shape by hand." Renaming `to_state` in alert-manager would break the console alarm list and the historian's `alarm_changes` tag with no failing test. Smaller mismatches: the PLC's `alarm_id` is a string `key:timestamp_ms` that no consumer reads, while `alarms/changes` uses `alarm_id` for the integer database id — same name, two meanings; alert-manager still accepts a pre-lifecycle payload without `key`/`state` (`payloads.py:8-10,119,124`) that no producer sends anymore.
- **Recommendation:**
  1. Define the JSON payloads once as typed models (pydantic or `TypedDict`) in a shared contract module (e.g. `shared/runtime` or a new `shared/contracts` package) and have producers build and consumers parse through them.
  2. Have the gateway validate `plc/events` and `alarms/changes` against those models before publishing to the hub, and add the WebSocket frame models to the OpenAPI schema (as components) so `types.ts` aliases generated types.
  3. Add a round-trip test: producer function output → every consumer parser, in one test file, so a rename fails the gate.
  4. Add a `schema_version` field only if the owner wants versioned MQTT contracts (contract change — owner decision); otherwise document "additive only" per I6.
  5. Remove the legacy no-`key` branch of `parse_condition` in a separate change once the owner confirms no old producer exists; rename or drop the unused `alarm_id` in `alerts/*` via the contract-change process.
- **Effort:** M

#### ARCH-05 — A non-finite measurement silently passes the pressure and temperature interlocks

> **Same root cause as PLC-02** (section 7.4). Fix it there; this entry records the invariant-level view.

- **Severity:** Medium
- **Category:** Safety
- **Confidence:** Medium (behaviour confirmed; whether physics can emit NaN in practice is unconfirmed — a failed sensor holds its last value, `apps/physics-engine/src/physics_engine/sensors.py:7`)
- **Location:** `apps/plc-controller/src/plc_controller/safety_limits.py:84-99`; `apps/plc-controller/src/plc_controller/safety.py:414-452`; contrast `apps/plc-controller/src/plc_controller/alarms.py:254`
- **About the file:** `safety_limits.py` holds the trip/warn limits and their comparison; `safety.py` runs the interlock check every scan.
- **Problem:** `ParameterLimits.check` uses `value <= trip_low or value >= trip_high`; NaN fails every comparison and returns NORMAL. A read-only check confirmed `SafetyInterlock().check(nan, 4.5, nan, nan, 1.0, steam_temp=nan)` returns `normal` with no E-Stop (a NaN drum level does trip, via the fuel permissive). The alarm monitor explicitly skips non-finite values, so an operator would see neither an alarm nor a trip. I3 promises that nothing turns an interlock off; a numerical blow-up in the model (or a malformed message on the PLC's input) would. The valve path already refuses NaN (`commands.py:143-153`, `plant.py:272-274`) — the measurement path does not.
- **Recommendation:**
  1. Treat a non-finite value of an interlocked parameter as a trip (fail-safe), in `SafetyInterlock.check` or `ParameterLimits.check`; state the physical reasoning in the commit body as `12-domain-rules.md` requires.
  2. Add tests: NaN and ±inf pressure, water temperature, flue-gas and steam temperature each latch the E-Stop; a finite in-range value does not (acceptance).
- **Effort:** S

#### ARCH-06 — The historian's ownership of InfluxDB is enforced only by convention: gateway and Grafana hold the all-access admin token

> **Same root cause as PLAT-02** (section 7.8), also GW-API-08 and HIST-02. Fix it once.

- **Severity:** Medium
- **Category:** Security
- **Confidence:** High
- **Location:** `docker-compose.yml:117` (Grafana), `:198` (historian), `:286` (api-gateway), all `${INFLUXDB_ADMIN_TOKEN}`; `docs/architecture/service-boundaries.md:13`; `.ai/project/12-domain-rules.md:56`
- **About the file:** `docker-compose.yml` distributes credentials to each container.
- **Problem:** `service-boundaries.md` makes the historian the owner of recording, retention and downsampling, and the domain rules give each service its own PostgreSQL role and MQTT account. InfluxDB has no such split: the gateway (which only reads history and KPIs) and Grafana (read-only dashboards) receive the operator token that can write points, delete buckets and change retention. A bug or compromise in the internet-facing gateway can erase or forge the plant history the audit and KPIs rely on.
- **Recommendation:**
  1. Have `dev-secrets` (or a one-shot setup like `migrate`) create read-only tokens scoped to the raw and `sensors_1m` buckets for the gateway and Grafana, and a write token for the historian (the historian still needs rights for its bucket/task setup — keep the admin token there, or move setup to the one-shot job).
  2. Extend the domain rule "own PostgreSQL role and MQTT account" to InfluxDB tokens (rule change — owner decision) and record it in `service-boundaries.md`.
- **Effort:** M

#### ARCH-07 — opcua-server's gRPC channels skip the shared client interceptors

> **Same finding as OPC-14** (section 7.6) and DUP-07.

- **Severity:** Low
- **Category:** Readability
- **Confidence:** High
- **Location:** `apps/opcua-server/src/opcua_server/client.py:16,30`; rule `.ai/project/12-domain-rules.md:46`; `docs/architecture/overview.md:175-182`
- **About the file:** the PLC-status and alarm-read clients of the OPC UA projection.
- **Problem:** Every other gRPC channel passes `client_interceptors()`; these two do not, so their calls carry no correlation id and are not counted by the client metrics, contrary to the domain rule. Security relevance is indirect: PLC and alarm logs for OPC UA reads cannot be tied to a request.
- **Recommendation:**
  1. Pass `interceptors=client_interceptors()` to both `insecure_channel` calls.
  2. Add a test mirroring `test_upstream_clients.py:237` that the correlation id reaches the fake servicer's metadata.
- **Effort:** S

#### ARCH-08 — MQTT topic names are re-declared in six services and some subscription filters are bare literals
- **Severity:** Low
- **Category:** Duplication
- **Confidence:** High
- **Location:** `apps/physics-engine/src/physics_engine/mqtt_publisher.py:58-62`; `apps/plc-controller/src/plc_controller/events.py:41-45`; `apps/alert-manager/src/alert_manager/payloads.py:25-27`; `apps/historian/src/historian/subscriber.py:59-71`; `apps/opcua-server/src/opcua_server/subscriber.py:44-52`; `apps/api-gateway/src/api_gateway/realtime/sources.py:32-33`; ACL `infrastructure/docker/mosquitto/acl`
- **About the file:** each module is the MQTT edge of its service.
- **Problem:** Each service defines its topics as constants (good), but the same strings live in up to four places (`alarms/changes` ×4, `sensors/boiler` ×3, `plc/events` ×3) plus the ACL and two docs; the wildcard filters `sensors/#` and `status/+` are literals inside subscription lists. `12-domain-rules.md` requires a topic change to update every publisher and subscriber in one task — nothing helps find them, and nothing checks the ACL matches.
- **Recommendation:**
  1. Move the topic names into one shared module (e.g. `cogniboiler_runtime.topics`, since `shared/runtime` already holds what "every service does the same way and no service owns").
  2. Import them in each service; derive the wildcard filters from the constants.
  3. Add a test that parses `infrastructure/docker/mosquitto/acl` and asserts each account's write grants cover exactly the topics its service publishes.
- **Effort:** S

#### ARCH-09 — Documentation drift in the invariants registry and the overview
- **Severity:** Low
- **Category:** Documentation
- **Confidence:** High
- **Location:** `docs/architecture/invariants.md:108-111` vs `docs/__arch__/ROADMAP.md:661` and `docs/architecture/overview.md:317-319`; `docs/architecture/overview.md:23` vs `infrastructure/docker/mosquitto/mosquitto.conf:4-7` and `overview.md:76-77`; `docs/architecture/invariants.md:27-28` (see ARCH-01); `apps/api-gateway/migrations/versions/0003_sessions_and_append_only_audit.py:13-15`
- **About the file:** the registry of binding invariants and the "what exists today" overview.
- **Problem:** (a) I9 says the remaining PLC integration tests still run on the wall clock (debt Д2), but Д2 was closed 2026-09-19 and every PLC plant test uses the lockstep harness; the overview already says so. (b) The overview diagram shows Mosquitto on `:9001` (a WebSocket port) while the same document and the broker config state there is no WebSocket listener — a reader could believe the browser can reach MQTT. (c) I2's "Enforced by" overstates the enforcement. (d) Migration 0003's docstring still says the application connects as the table owner; this was fixed by 0004 — an applied migration must not be edited, so only note it.
- **Recommendation:**
  1. Propose to the owner an update of I9's "Enforced by" sentence (registry changes need owner approval) to drop the Д2 remark.
  2. Remove `, :9001` from the diagram in `overview.md:23`.
  3. Fix I2 wording together with ARCH-01/02.
  4. Leave 0003 unchanged; the 0004 docstring already records the correction.
- **Effort:** S

#### ARCH-10 — Contract fields outside SI contradict I4 as written
- **Severity:** Low
- **Category:** Documentation
- **Confidence:** High
- **Location:** `shared/proto/cogniboiler.proto:103` (`nox_ppmv`), `:104` (`co2_intensity_kg_per_mwh`), `:117,120,122` (`*_hours`), `:123` (`overall_health_pct`); mirrored in `apps/api-gateway/src/api_gateway/clients.py:289-296` and the InfluxDB fields
- **About the file:** the gRPC and telemetry contract.
- **Problem:** I4 says every contract carries SI values; these fields use hours, percent, ppmv and MWh. The unit is in each name, so the mismatch risk I4 guards against is small, but the registry and the contract disagree, and `co2_intensity_kg_per_mwh` coexists with the SI `co2` per joule in `PerformanceMsg`.
- **Recommendation:**
  1. Do not rename (breaking contract change). Propose to the owner an explicit exception list in I4 (dimensionless percent/ppmv, equipment hours) or deprecating `co2_intensity_kg_per_mwh` in favour of the per-joule field.
  2. Add a small proto lint test that every numeric field name ends in an allowed unit suffix, with the exceptions listed.
- **Effort:** S

#### ARCH-11 — The gateway owns KPI formulas that the boundary table forbids it
- **Severity:** Low
- **Category:** Modularity
- **Confidence:** High
- **Location:** `apps/api-gateway/src/api_gateway/routers/kpi.py:94-97` vs `apps/physics-engine/src/physics_engine/proto_mapping.py:151-167`; `docs/architecture/service-boundaries.md:13,15`; `docs/architecture/overview.md:142`
- **About the file:** `kpi.py` answers `GET /api/v1/kpi` from InfluxDB means.
- **Problem:** `service-boundaries.md` lists "KPI formulas" under what the gateway must not own and says physics owns the heat balance, yet the gateway re-implements net efficiency, boiler efficiency and both heat rates as ratios of range means. The formulas currently agree with physics, but they are now defined twice.
- **Recommendation:**
  1. Either record in `service-boundaries.md` that range aggregation of physics-computed quantities is the gateway's (owner decision), or
  2. have the historian store the physics `PerformanceMsg` energy terms only and keep the ratio code in one shared function used by both — refactor, no new logic.
- **Effort:** S

#### ARCH-12 — Unused runtime dependencies blur service boundaries
- **Severity:** Low
- **Category:** Modularity
- **Confidence:** Medium (no import found in `src/`; confirm with a dependency checker)
- **Location:** `apps/alert-manager/pyproject.toml` (`alembic>=1.13.0`), `apps/plc-controller/pyproject.toml` (`pyyaml>=6.0.0`), `apps/api-gateway/pyproject.toml` (`bcrypt>=4.1.0`; passwords are Argon2id, `auth/password.py:4`)
- **About the file:** each service's dependency declaration.
- **Problem:** alert-manager declares Alembic although the chain belongs to the gateway (decision Q2); the PLC declares PyYAML although no configuration file is read (a YAML loader in the safety service invites a future "configurable thresholds" path that I3 forbids); the gateway ships bcrypt it does not use. Each widens the image and the audit surface.
- **Recommendation:**
  1. Remove the three dependencies, run `uv lock`, and run the full gate.
- **Effort:** S

#### ARCH-13 — Interlock thresholds are mutable module objects, not constants
- **Severity:** Low
- **Category:** Safety
- **Confidence:** High
- **Location:** `apps/plc-controller/src/plc_controller/safety_limits.py:57-58` (`@dataclass` without `frozen=True`), `:184,192` (module-level instances); `apps/plc-controller/src/plc_controller/safety.py:278-284` (injectable limits)
- **About the file:** the protection limits of the PLC.
- **Problem:** I3 says thresholds are "code constants". `ParameterLimits` instances can be mutated at runtime (`PRESSURE_LIMITS.trip_high = float("inf")`) by any code in the process, and `ParameterLimits` explicitly allows infinite limits ("an infinite limit leaves that side unprotected"). No external path does this today.
- **Recommendation:**
  1. Make `ParameterLimits` `@dataclass(frozen=True)`.
  2. Add a test that each production limit set has finite `trip_low`/`trip_high` where the docstring table says the side is protected.
- **Effort:** S

#### ARCH-14 — I5 has no automated guard
- **Severity:** Info
- **Category:** Security
- **Confidence:** High
- **Location:** `.pre-commit-config.yaml:19-30`; `.github/workflows/ci.yml`; `docs/architecture/invariants.md:65-66`
- **About the file:** the commit hook and CI pipeline.
- **Problem:** The registry honestly says "`.gitignore` and review". The hook already uses `pre-commit-hooks` but not its `detect-private-key` check, and CI runs no secret scan; `.gitignore:138` ignores only `.env`, not `.env.*` variants.
- **Recommendation:**
  1. Add `- id: detect-private-key` to the existing `pre-commit-hooks` entry.
  2. Add `.env.*` with `!.env.example` to `.gitignore`.
- **Effort:** S

#### ARCH-15 — `audit_log.user_id` is declared `ON DELETE SET NULL`, which the append-only triggers forbid
- **Severity:** Info
- **Category:** Reliability
- **Confidence:** Medium (PostgreSQL implements `SET NULL` as an UPDATE on the referencing row, which fires the BEFORE UPDATE trigger)
- **Location:** `apps/api-gateway/migrations/versions/0001_initial_schema.py:187`; `apps/api-gateway/src/api_gateway/models/user.py:238-240`
- **About the file:** schema of the audit log.
- **Problem:** The FK asks the database to rewrite audit rows when a user is deleted; the triggers refuse it, so deleting any user with audit history fails with an "append-only" error. No route deletes users today (accounts are blocked), so this is latent, but the declared intent contradicts I10.
- **Recommendation:**
  1. In a new migration (never edit 0001), change the FK to `ON DELETE NO ACTION`/`RESTRICT` on PostgreSQL, and mirror it in the model.
- **Effort:** S

## 8. Accepted risks re-checked

These items were accepted earlier by owner decisions. The auditors re-checked them and recorded only new angles.

**API gateway — authentication, sessions, RBAC, audit log, migrations**

- **Д16 (throttle counters kept in process memory):** still accurate. The new angle is GW-AUTH-02: even inside one worker the counter is not atomic across the Argon2 await. That is a separate, unaccepted defect.
- **Refresh token in the response body (decision Q1):** unchanged. It raises the value of GW-AUTH-11 (`no-store`), because the body carries a 7-day credential.
- **Demo users seeded from `.env`:** accepted. GW-AUTH-04 is a defect in how seeding treats *existing* accounts, not in the seeding itself.
- **X-Forwarded-For only from private addresses (2026-09-18):** the implementation matches the decision. nginx overwrites the header, uvicorn trusts only the three private ranges, and repeated headers cannot be smuggled past nginx. The residual risks are in GW-AUTH-10.
- The docstring of migration `0003` (lines 13-15) still says the application connects as the table owner. That was fixed by 0004; applied migrations must not be edited, so this is noted only.

**API gateway — operational API, WebSocket, realtime hub, upstream clients**

- Д16 (login throttle per process): not affected by this area. Note, however, that WebSocket `auth` renewal frames and the 30 s revalidation query the database with no rate limit per connection. They carry tokens, not passwords, so this is not a throttle bypass, only extra DB load (see GW-API-06).
- Q1 (refresh token also in the body): no new angle here. `/ws` takes the access token only in the first frame, and the refresh cookie is scoped to `/auth`, so a cross-site page cannot authenticate a socket (see positives).

**physics-engine — plant model, runtime, PhysicsService, MQTT telemetry**

- **Д17 (superheater without metal heat capacity, restart instability):** not reported again. PHY-01 interacts with it: the documented recovery path reloads a scenario on a hot plant, and a `LoadScenario` cancelled at the gateway's 30 s deadline would race the run loop on exactly that transition.
- The unauthenticated `PhysicsService` is treated as defence in depth (PHY-05). The Compose setup does not publish port 50052, so this is not critical, but the host-run default of binding every interface is a new angle.

**plc-controller — interlocks, E-Stop latch, commands, PLCService**

- **Д17 (restart instability):** not re-reported. PLC-01 is a separate defect. It concerns the *latched* trip command being lost on a reload, not how a restart behaves. The demo guidance "reset before reloading a scenario" (`14-command-reference.md`) happens to avoid it, but a reload while latched is still reachable through the engineer route.
- **A debt note in the assistant rules is stale** (`.claude/agents/cogniboiler-physics-control.md:33` and its mirror `.codex/agents/physics-control.toml:32`): there is no `xfail(strict=True)` left in `apps/plc-controller/tests/test_plc_server.py`. `test_auto_control_holds_nominal_state_for_ten_minutes_simulated` (`test_plc_server.py:288-304`) is now a plain test, with a ±20 bar tolerance that reaches the 160 bar warning band. PID and safety code no longer live in `physics_engine` (no `physics_engine` import in `apps/plc-controller/src`).

**historian and alert-manager**

- **Alarm-change publisher 5 s delay after a broker restart (QoS 1 queue, nothing lost):** still true while the process lives. New angle: the queue is lost on container stop, because SIGTERM is not handled (ALM-05). "Nothing lost" does not hold across a restart of the alert manager itself.
- **InfluxDB restart leaves a gap; the historian counts and drops unwritten batches:** still accurate. New angles:
  - the drops are logged once per batch, not once per outage (HIST-04);
  - `stats["stored"]` counts dropped batches as stored (HIST-09);
  - a slow InfluxDB, as opposed to one that is down, is different: ingestion stalls and aiomqtt's unbounded queue grows (HIST-07).
- **Д14, anonymous OPC UA read:** raises the impact of ALM-02, because `opcua-server` is the host-exposed container that can reach the unauthenticated `AlarmService` write RPCs.

**opcua-server**

- **Д14 (no client certificate trust list):** The acceptance still holds, because certificates protect the channel and users are authenticated at the gateway. New angles: session tokens are guessable and can be rebound across channels (OPC-06), and RSA-1_5 password decryption is accepted (OPC-16). Both can be fixed without a trust list.
- **`None` endpoint with anonymous read and encrypted-only passwords:** Verified. asyncua advertises the Basic256Sha256 token policy on the `None` endpoint, and `_UserAwareSession.activate_session` refuses unencrypted passwords. Sign-only is not offered. Anonymous tokens get `UserRole.User` (needed because `SimpleRoleRuleset` gives `Anonymous` no services). asyncua still refuses AddNodes, DeleteNodes, AddReferences, DeleteReferences and non-Value writes for non-Admin users, and every project variable lacks `CurrentWrite`. No writable path around the gateway was found. A `UserNameIdentityToken` with a null `UserName` is treated as anonymous (`apps/opcua-server/src/opcua_server/identity.py:57-58`). It stays read-only, so this is harmless.
- **Д16 (throttle in gateway memory):** New angle in OPC-04: every OPC UA user shares one per-client bucket.
- **Contract units:** `kg/MWh`, `h` and `%` in the Emissions and Health folders come from non-SI proto field names (`co2_intensity_kg_per_mwh`, `turbine_hours`, `overall_health_pct`) that physics-engine owns. The OPC UA server labels them faithfully. I record this as an observation for the proto owner, not as an opcua-server defect.

**Web console (apps/web)**

- **Refresh token in the response body (Q1):** the console does not store it. `accept()` (`apps/web/src/session/session.ts:178-191`) reads only `access_token`, `username`, `role`, `session_expires_at_ms` and `access_expires_at_ms`. A grep over `src`, `e2e` and `readme` finds no `sessionStorage`/`indexedDB`/`document.cookie`, and `localStorage` is used only for the theme choice (`theme.ts:25`, `:50`). The acceptance holds.
- **SameSite=Strict refresh cookie:** every request that relies on it (`/auth/refresh`, `/auth/logout`) is same-origin with `credentials: "same-origin"` (`http.ts:99`), through the Vite proxy or nginx. No cross-site flow depends on it. New angle: see WEB-04, where logout depends on the cookie alone.
- **Server-side authorization behind the UI:** every mutating endpoint the console calls has a role dependency matching `session/roles.ts:23-33`. Commands load/mode/valve are `OperatorUser` and setpoint/reset are `EngineerUser` (`routers/commands.py:54,77,91,107,128`). Simulation pause/resume/speed/step/scenario/faults are `EngineerUser` (`routers/simulation.py:132-228`). Alarm ack and ack-all are `OperatorUser` (`routers/alarms.py:175,214`). Users are `AdminUser` (`routers/users.py:27-90`) and audit is `AdminUser` (`routers/audit.py:20`). No gap found.
- **Insights panel (AI deferred):** `RecommendationsPanel` returns `null` unless `VITE_INSIGHTS === "true"`, makes no request and has no dependency. It cannot leak or break anything.
- **Demo users from `.env`:** passwords reach the browser only through Playwright's `fill()`. They are not printed; see WEB-20 for traces.

**Platform — Docker, Compose, nginx, Mosquitto, Grafana, CI, supply chain**

- **Gateway not published; nginx on 8080/8443 with a self-signed certificate.** Still holds: `docker-compose.yml:260-307` has no `ports:`, and nginx proxies only `/api`, `/auth`, `/health`, `/ready`, `/ws`, `/docs*` (`site.inc:11-34`). New angle: the `/docs` exposure (PLAT-03) is what makes the same-origin refresh cookie reachable by third-party script.
- **X-Forwarded-For trusted only from private ranges.** Still sound: nginx overwrites the header with `$remote_addr` (`proxy.inc:5`), so a client cannot inject a chain. Caveat for deployers: `--forwarded-allow-ips 172.16.0.0/12,10.0.0.0/8,192.168.0.0/16` (`docker-compose.yml:267`) also trusts every other container on the Compose network, not only nginx. That is acceptable given the network holds only stack services.
- **"A new base-image CVE can block publishing" (2026-09-18).** Still intentional, but it only works if the scanned image is the published one. See PLAT-05.

**Shared packages, developer scripts, cross-service duplication**

- **Scripts share code only through `scripts/_toolkit`** (by design): confirmed, no script imports another. The duplication in SCR-04 and SCR-07 is exactly what `_toolkit` exists for, so the recommendations stay within the design.
- **Demo users and passwords seeded from `.env`**: no new angle. The scripts never print passwords; `processes.run` echoes argv, but no script passes a secret on argv. SCR-01 (file mode) is a separate, new point.

## 9. What is done well

**API gateway — authentication, sessions, RBAC, audit log, migrations**

- The role, the active flag and whether the session is open are resolved from the database on every request (`apps/api-gateway/src/api_gateway/auth/identity.py:112-174`). A role change, a block, a sign-out or an admin's `revoke-sessions` stops an access token on its next use, over HTTP and on the WebSocket's periodic revalidation.
- Refresh rotation is solid:
  - families keep an absolute expiry (rotation never extends a session);
  - `SELECT ... FOR UPDATE` serializes concurrent refreshes of one token;
  - a reuse after the 5 s grace revokes the whole family, including access tokens;
  - the cookie is httpOnly, `SameSite=Strict`, `Secure` by default and scoped to `/auth`.
- Unknown users, wrong passwords and blocked accounts get an identical 401. A dummy Argon2 verification equalises timing, and the throttle key is case- and whitespace-folded. All Argon2 work runs in `asyncio.to_thread`.
- User administration refuses self-demotion and self-blocking and protects the last admin (apart from the race in GW-AUTH-08). Every role or password change closes the user's sessions.
- The audit trail is enforced in the database, not just in code: append-only triggers (0003), and a `cogniboiler_gateway` role with only `SELECT, INSERT` on `audit_log` that owns no table (0004). It therefore cannot `TRUNCATE`, `ALTER` or disable the triggers.
- Problem Details never echo submitted values or driver or gRPC messages, the engine uses `hide_parameters=True`, and the audit endpoint filter escapes `LIKE` wildcards (`apps/api-gateway/src/api_gateway/routers/audit.py:51-52`) with a test for it.

**API gateway — operational API, WebSocket, realtime hub, upstream clients**

- Role discipline is complete and consistent. Every mutating route declares `OperatorUser`/`EngineerUser` and every read declares `ViewerUser`; the operator/engineer split matches `auth/rbac.py` (setpoints, E-Stop reset and simulation control are engineer-only). The audit middleware covers all POST/DELETE routes, and PLC and AlarmService refusals are recorded as outcomes, not hidden behind HTTP 200.
- The control boundary holds. `PhysicsGatewayClient` has no command method at all, valve commands go only through `PLCGatewayClient.send_command`, and the E-Stop reset records the authenticated user and ignores a body-supplied `operator_id`.
- Input validation is tight. Every request float has bounds, which also rejects NaN and ±Infinity (verified). Enums are Literals, scenario names are pattern-checked, and history is limited to 90 days and 2000 points, with a window choice that bounds the result regardless of simulation speed.
- The WebSocket token travels in the first frame, never in the URL, so it does not reach nginx access logs. Refusals are audited, sessions are revalidated every 30 s, token expiry closes with 4401, and renewal is restricted to the same user.
- Upstream errors never leak gRPC or driver text to clients (`problems.upstream_unavailable`, tested). Every unary gRPC call has a deadline, the correlation id propagates through both unary and streaming interceptors (tested), and the MQTT source reuses the shared `MqttSession` with logging once per outage.
- The Flux builder quotes every name, including the environment-supplied bucket, and has regression tests for quote, backslash and newline injection. KPI inputs are filtered to finite numbers.

**physics-engine — plant model, runtime, PhysicsService, MQTT telemetry**

- Valve commands are validated again in the physics engine (`plant.py:265-274`), and the comparison-based checks reject NaN and inf as well as out-of-range values. Confirmed over gRPC: `fuel_valve=nan` and `spray_valve=inf` are refused with a clear reason.
- Operator requests are bounded: speed [0.1, 50] and NaN-safe, steps [1, 3600], fault ramp [0, 3600] s with NaN refused, and scenario names resolved through an enum (`scenarios.py:227-234`), so there is no file lookup and no path traversal.
- The plant is deterministic and needs no wall clock. Tests drive a paused runtime with `runtime.step(n)`, and physics steps and scenario loads run in `asyncio.to_thread` as the rules require.
- The MQTT mirror cannot build a backlog: `wait_for_update` always returns the latest snapshot. Reconnects go through the shared `MqttSession` (warn once, fixed delay), and publish errors are counted per topic.
- `load_scenario` computes the new initial conditions before mutating anything, so a failed load leaves the plant untouched. Scheduled faults that collide with active ones are skipped with a warning rather than crashing a step.
- The model code documents its physical reasoning (calibration notes in `constants.py`, `boiler.py`, `heat_exchanger.py`, `emissions.py`), and the IF97 tables plus the cached turbine expansion keep a steady step near 0.5 ms.

**plc-controller — interlocks, E-Stop latch, commands, PLCService**

- The E-Stop check and the scan share one `asyncio.Lock` (`service.py:299,335,359,456`), so "check E-Stop, then apply the command" is atomic with respect to trips. The latch is set *before* the trip command is sent (`service.py:338-343`).
- Callers cannot pass for the interlock: PID and SAFETY sources are refused (`service.py:63-67,294-297`), and a test covers it (`test_plc_behaviour.py:108-120`).
- Operator input validation is pure and thorough: valves, setpoints and load demand all reject NaN and inf through the bound comparisons, with dedicated tests (`test_plc_commands_and_status.py:103-160`). A reset is refused while any critical condition or a warning on the trip cause stands (`alarms.py:324-339`).
- The trip response depends on its cause: fuel and spray shut, the turbine valve vents only on high pressure, and feedwater keeps the drum wet unless the drum is high (`safety_limits.py:245-252`). Trips are logged at `warning` per the owner decision, and a test checks that (`test_plc_behaviour.py:173-191`).
- The MQTT publisher is bounded (1000 messages, oldest dropped, logged), delivers at least once in order, sends a reconcile snapshot every 10 s, and reuses the shared `MqttSession` (reconnect delay, one warning per outage).
- The lockstep test harness (`plc_harness.py`) steps simulated time deterministically, so no test depends on machine speed.

**historian and alert-manager**

- Both services use the shared `MqttSession` (log once, reconnect with delay, `connected` flag for liveness). The historian flushes its buffer before reconnecting (`on_failure=self._flush`), and a test covers it.
- The lifecycle is split into small pure functions (`lifecycle.py`) with exhaustive parametrized tests. A partial unique index enforces one open alarm per key in the database, and a test covers that too.
- The snapshot reconciliation is well designed: a lost "cleared" message or a pending clear lost on restart is repaired within about 10 s. The `raised_at_ms <= snapshot.timestamp_ms` guard protects against old snapshots, and the 3 s clear hold stops flapping alarms from piling up.
- Bounded queries: `list_alarms` clamps `limit` to 1..1000 and `offset` to ≥ 0. All SQL goes through SQLAlchemy expressions (no string SQL). `hide_parameters=True` keeps values out of error logs. Text fields are truncated to column sizes before insert.
- The historian drops only the NaN/inf field, not the whole point, and counts it. It keeps text out of aggregated fields, applies retention and downsampling idempotently with retry, and runs all blocking InfluxDB calls in `asyncio.to_thread`.
- Secrets hygiene: the InfluxDB token and MQTT password come only from the environment and are never logged, and alert-manager refuses to start until the migrations have run instead of creating schema.

**opcua-server**

- I11 holds structurally. `client.py` has only `GetControlStatus` and `ListAlarms`. Every method goes through `MethodHandlers._forward_as_user`, which refuses anything that is not a `GatewayUser` with a session, before any HTTP call. Both allow and deny are tested at unit and end-to-end level (`test_an_anonymous_session_cannot_call_a_method`, `test_a_signed_in_user_commands_through_the_gateway`).
- Passwords never reach logs or exceptions: `login` logs the username with `%r` and the gateway's `code`, and `test_a_refused_sign_in_is_none_and_logged` asserts that the password is absent. The private key stays in memory.
- The event loop is never blocked: HTTP runs in `asyncio.to_thread` with a 10 s timeout, and gRPC calls carry 3 s deadlines. Upstream outages are logged once and again on recovery, and values are marked `UncertainLastUsableValue` instead of silently going stale.
- The gateway's errors map to OPC UA codes without leaking internals: 401/403 give `BadUserAccessDenied` (with one refresh-and-retry on 401), 4xx gives `BadInvalidArgument`, and 5xx or unreachable gives `BadCommunicationError`. A PLC refusal comes back as `Accepted=False` plus `Reason`, not as an error.
- The address space is a clean catalogue: SI values with UNECE `EUInformation`, instrument quality carried as status codes, and source timestamps taken from `timestamp_ms` (UTC). The probe confirmed that every catalogued node is fed and every fed id is catalogued.
- The burst thinning in the bridge keeps the latest value per topic without writing every physics step, and the MQTT loop reuses the shared `MqttSession` (log once, reconnect with a delay).

**Web console (apps/web)**

- The token architecture is sound and documented where it matters: access token only in `SessionManager` memory, the WebSocket token in the first frame and never in the URL (`apps/web/src/api/realtime.ts:130`), single-flight renewal plus a proactive renew 60 s ahead, StrictMode-safe single restore, and handling of `auth.refresh_superseded` across tabs.
- XSS surface is effectively nil: all server strings (alarm messages, user names, audit details, PLC event details) render as React text; no `innerHTML`, no URL built from server data, no open-redirect parameter (after sign-in the router starts at the current path, with no `next`/`from` query).
- One HTTP layer and one query-key registry: components never call `fetch`, mutations invalidate by `queryKeys` names, 4xx responses are not retried (`apps/web/src/main.tsx:24-25`), and errors are decoded from Problem Details into operator sentences.
- Strict TypeScript with `noUncheckedIndexedAccess`, no `any`, and lint and typecheck both clean. The generated `schema.gen.ts` feeds almost every REST type.
- Dangerous actions (trip, E-Stop reset, valve commands, scenario load, fault injection, block, password reset, sign-out-everywhere) all go through a confirmation that states the effect in display units, and the dialog is re-render-safe through `useEffectEvent`.
- Effects clean up properly: `RealtimeClient.stop()` detaches handlers before closing, `TrendChart` disconnects its `ResizeObserver` and destroys uPlot, the theme watcher unsubscribes, and the horn interval stops on unmount. The only leak found is WEB-16.

**Platform — Docker, Compose, nginx, Mosquitto, Grafana, CI, supply chain**

- Every published port binds `127.0.0.1` (Mosquitto 1883, InfluxDB 8086, PostgreSQL 5432, Grafana 3000, OPC UA 4840, nginx 8080/8443, Prometheus 9090). The physics, PLC and alert-manager gRPC ports and the gateway's 8000 are never published.
- Secrets use `${VAR:?run dev-secrets}` throughout, so no service can start with a known default password. `.env.example` holds only empty secrets and non-sensitive names. `.env` and `certs` are excluded from the Docker context and from git.
- Python runtime images are multi-stage, `--no-dev --no-editable`, run as uid 10001, remove pip, and apply OS security updates. The console runs on `nginx-unprivileged` as uid 101. Services start with `python -m`, not `uv run`.
- The broker refuses anonymous clients, has one account per service written from `.env` at start into a `0600` file (`start-broker.sh:5,29-32`), and has a working per-account ACL with no cross-service write rights. The WebSocket listener is gone.
- CI runs `quality-gate` verbatim with `CI=true`. The workflow defaults to `contents: read`, widens only per job (`packages: write`, `contents: write`), uses `persist-credentials: false`, and passes `github.ref_name` through `env:` rather than inline `${{ }}` in `run:` (no script injection). The gate starts with `uv sync --locked`, so lock drift fails by name.
- The WebSocket authenticates with a first frame, not a query string, so nginx access logs carry no tokens. `backup` and `restore` keep credentials inside the containers, so none reach a host command line. nginx sets `server_tokens off`, a 64 KiB body limit, a strict CSP for the console and `frame-ancestors 'none'`.

**Shared packages, developer scripts, cross-service duplication**

- Correlation ids from outside are validated against `^[A-Za-z0-9._:-]{1,128}$` (`correlation.py:24`) at both edges, HTTP and gRPC. A 5000-character header was replaced with a fresh 32-character id in my probe, so header injection and log flooding are closed.
- The gRPC server interceptor labels only registered methods (`grpc_observability.py:105-108`), handles unary and server-streaming calls, keeps status codes intact, and counts client disconnects as `CANCELLED`. It is well tested.
- The proto contract has never renumbered or reused a field. I compared field numbers across all six historical revisions of `cogniboiler.proto`, so no `reserved` is needed. `timestamp_ms` is used consistently.
- The script runner is stdlib-only, never uses `shell=True`, restricts launchable modules by pattern and by directory (`config_loader.py:190-209`), and turns a bad catalog into one clear error line with exit code 2.
- `clean-caches` is careful: it dry-runs unless `--apply`, never follows symlinks or junctions, protects `.git`, `.venv`, `node_modules` and `.env`, and resolves each extra path inside the root. `confirm()` treats "no terminal" and EOF as "no".
- `MqttSession`, `LivenessFile` and `now_ms` already removed seven hand-written loops and two byte-identical liveness copies. The findings above mostly extend that consolidation to the pieces it has not reached yet.

**Test coverage and test quality**

- The PLC and physics integration tests run in lockstep: one plant step, then exactly that PLC scan (`apps/plc-controller/tests/plc_harness.py:29-40`, `test_plc_server.py:289-300`). Every wait carries a deadline assertion, so these long control tests are deterministic, not timed (debt Д2 is genuinely closed).
- No unit test needs a broker, a database or the internet. MQTT is replaced by `FakeBroker` or `Broker` classes. PostgreSQL is replaced by SQLite (`apps/alert-manager/tests/conftest.py` explains why it uses a file rather than `:memory:`). gRPC runs in process on port 0, and InfluxDB sits behind fakes.
- The gateway suite signs tokens with an ephemeral RSA pair generated per run (`apps/api-gateway/tests/conftest.py:43-49`), so no developer `.env` secret reaches the tests. RBAC (`auth/rbac.py`), the audit middleware (`audit.py`, 0 missed lines) and the login throttle are fully covered.
- Security-relevant escaping is tested with hostile input: Flux string escaping (`test_history_query.py:77`) and database role provisioning (`test_db_roles.py:133`).
- No skip or xfail markers exist in any service suite. All 1 085 Python, 134 script and 212 console tests passed, and no test file exceeds 574 lines.
- The console tests cover the session and renewal client, the realtime socket, units and every screen except Control, using shared fixtures (`apps/web/src/test/fixtures.ts`). The Playwright specs in `apps/web/e2e/` cover the role matrix and the demo against the real stack in CI.

**Architecture conformance — invariants, boundaries, contracts**

- No service imports another service's package; the only cross-service dependency is plc-controller's dev-group use of physics-engine, exactly as `service-boundaries.md:44-45` records, and `ai-predictor` is kept out of the workspace.
- PostgreSQL ownership matches the documents: the gateway maps only its six tables, alert-manager only `alarm_events`/`alarm_transitions`, the gateway reads alarms exclusively through `AlarmService` (`routers/alarms.py:4`), and the migration job is the only owner connection.
- The MQTT ACL (`infrastructure/docker/mosquitto/acl`) matches the topics table publisher-for-publisher, anonymous clients are refused, and there is no WebSocket listener.
- I3 is pinned on both sides: tests show engineers can and operators cannot reset, reserved PID/SAFETY sources are refused, and a reset with an active cause is refused with blockers.
- I11 is implemented cleanly: every OPC UA method goes through the gateway with the session's own token, refused sessions map to `BadUserAccessDenied`, and the end-to-end tests prove both the anonymous refusal and the gateway path.
- Valve bounds are validated twice (PLC `commands.py:137-154` and plant `plant.py:271-274`), including NaN, as the domain rules require; every mutating REST route carries a typed role dependency.

## 10. Remediation status

Fixed on 2026-09-27 … 2026-10-02 by parallel agents in four waves, each merged into the local
`main` after a green full gate and a check on the live stack (decision log, 2026-09-27).
"Commits" are the remediation commits whose message names the finding (up to four shown).
"Owner decision" and the decision-log entries Q9–Q16 name what is left for the owner.

**Totals:** fixed 188 · partial 26 · owner decision 10 · not done 0 (of 224).

| ID | Status | Commits | Note |
|---|---|---|---|
| GW-AUTH-01 | partial | `87f6dce` | steps 1 and 3 done (no digest for password requests); keyed HMAC and rotating the demo passwords are owner decisions (Q14, Q16) |
| GW-AUTH-02 | fixed | `1d21ab5` |  |
| GW-AUTH-03 | partial | `be28b48` | the error no longer carries the statement; sending a SCRAM verifier instead of the password was not done |
| GW-AUTH-04 | fixed | `1e51924` |  |
| GW-AUTH-05 | fixed | `34a3519` |  |
| GW-AUTH-06 | fixed | `59eeeec`, `5e36dc5` |  |
| GW-AUTH-07 | fixed | `dbefb94` |  |
| GW-AUTH-08 | fixed | `40342a3` |  |
| GW-AUTH-09 | fixed | `34a3519` |  |
| GW-AUTH-10 | partial | `9433bd8`, `a1cb509` | IPv6 counted per /64, OPC UA forwards its client address; narrowing --forwarded-allow-ips is an owner decision (Q14) |
| GW-AUTH-11 | fixed | `c7d25e0` |  |
| GW-AUTH-12 | fixed | `5e6737c` |  |
| GW-AUTH-13 | fixed | `02a9574`, `0d1ced6` |  |
| GW-AUTH-14 | fixed | `34a3519`, `5488aa2` |  |
| GW-AUTH-15 | fixed | `89c9eb4` |  |
| GW-AUTH-16 | fixed | `1d21ab5`, `1e51924`, `34a3519`, `5488aa2` … |  |
| GW-API-01 | fixed | `ad72860` |  |
| GW-API-02 | fixed | `e7ba39f` |  |
| GW-API-03 | fixed | `ed7b737` |  |
| GW-API-04 | fixed | `e018402` |  |
| GW-API-05 | fixed | `68bcb44` |  |
| GW-API-06 | partial | `68bcb44`, `ed7b737` | uvicorn frame and queue limits and a connection cap; nginx limit_conn is an owner decision (Q14) |
| GW-API-07 | fixed | `c706d8e` |  |
| GW-API-08 | owner decision |  | same as PLAT-02 (Q11) |
| GW-API-09 | fixed | `a8a2ee2` |  |
| GW-API-10 | fixed | `ec51fbf` |  |
| GW-API-11 | fixed | `3002855`, `33a5019` |  |
| GW-API-12 | fixed | `e018402` |  |
| GW-API-13 | fixed | `ef98cbb` |  |
| GW-API-14 | fixed | `563ef12` |  |
| GW-API-15 | fixed | `e018402`, `ec51fbf` |  |
| GW-API-16 | fixed | `ed7b737` |  |
| GW-API-17 | fixed | `fdb1f1b` |  |
| GW-API-18 | fixed | `8120a17` |  |
| GW-API-19 | fixed | `1eb93c3`, `ade6cfc` |  |
| GW-API-20 | fixed | `238c460`, `563ef12`, `a8a2ee2`, `ad72860` … |  |
| PHY-01 | fixed | `d02a401` |  |
| PHY-02 | partial | `220eba8`, `5a44d17`, `88e1f9a` | metrics, refusal when degraded and the healthcheck done; ending the process on a dead loop is an owner decision (Q14) |
| PHY-03 | fixed | `086a9fd` |  |
| PHY-04 | fixed | `bde9e01` |  |
| PHY-05 | partial | `220eba8`, `d385173`, `ecab511` | binds 127.0.0.1 on a host run; caller authentication is an owner decision (Q10) |
| PHY-06 | fixed | `ecab511` |  |
| PHY-07 | partial | `7adac8a` | finite-state guard and config validation done; an upper bound on step_s is an owner decision (Q14) |
| PHY-08 | fixed | `ecab511` |  |
| PHY-09 | fixed | `ecab511` |  |
| PHY-10 | fixed | `60d3749` |  |
| PHY-11 | fixed | `e052ef8` |  |
| PHY-12 | fixed | `2f25735`, `70e08ac` |  |
| PHY-13 | fixed | `5e19840`, `8120a17`, `bde9e01` |  |
| PHY-14 | fixed | `70e08ac` |  |
| PHY-15 | partial | `b5123d4` | internal rename done; the contract change is an owner decision (Q12) |
| PHY-16 | fixed | `7cce9be`, `e052ef8`, `ecab511` |  |
| PHY-17 | fixed | `5a44d17`, `88e1f9a` |  |
| PLC-01 | fixed | `7304e61` |  |
| PLC-02 | fixed | `19006b0` |  |
| PLC-03 | partial | `806f013` | start-up warning and a test pinning today's behaviour; the fix is an owner decision (Q9) |
| PLC-04 | fixed | `dd0aae0` |  |
| PLC-05 | fixed | `dd0aae0` |  |
| PLC-06 | partial | `77d2e89`, `8a5fe08` | binds 127.0.0.1 on a host run; caller authentication is an owner decision (Q10) |
| PLC-07 | fixed | `aadd278` |  |
| PLC-08 | fixed | `093cf49`, `ecab511` |  |
| PLC-09 | fixed | `6db4781` |  |
| PLC-10 | fixed | `1a39b67` |  |
| PLC-11 | fixed | `92061f3` |  |
| PLC-12 | partial | `a6f0f0e` | conversions, modes and constants unified; sensor ids in the proto are a contract change (Q12) |
| PLC-13 | fixed | `e19347b` |  |
| PLC-14 | fixed | `52e1c09` |  |
| PLC-15 | partial | `c4836bc` | non-finite interval refused, bounds and a stream cap; removing the RPC is an owner decision (Q12) |
| PLC-16 | fixed | `ac205ab` |  |
| PLC-17 | partial | `c0c7e71` | policy written down, an unreported trip sensor warns; tripping on a failed non-trip sensor is an owner decision (Q14) |
| PLC-18 | fixed | `36452a0` |  |
| HIST-01 | fixed | `3002855`, `bd3c790` |  |
| HIST-02 | owner decision |  | same as PLAT-02 (Q11) |
| HIST-03 | fixed | `1a73672`, `858e9b2` |  |
| HIST-04 | fixed | `aea835f` |  |
| HIST-05 | fixed | `edd9947` |  |
| HIST-06 | fixed | `4b8f08e` |  |
| HIST-07 | fixed | `791e083`, `aea835f` |  |
| HIST-08 | fixed | `31d7172`, `3b95a42`, `55a7fd2`, `75fb0dd` … |  |
| HIST-09 | fixed | `db12c28` |  |
| HIST-10 | fixed | `085ccce`, `4b8f08e`, `791e083`, `80f3a22` … |  |
| ALM-01 | partial | `899ccba` | transient-error retry, bounded queue and the unmatched-key counter; raising a lost alarm from the snapshot is an owner decision (Q14) |
| ALM-02 | partial | `3f223c5` | empty operator refused and the caller logged; caller authentication is an owner decision (Q10) |
| ALM-03 | fixed | `2d8804a` |  |
| ALM-04 | fixed | `3002855`, `8c14663` |  |
| ALM-05 | fixed | `4089de1`, `55a7fd2` |  |
| ALM-06 | fixed | `51f3ee2`, `55a7fd2` |  |
| ALM-07 | fixed | `3f223c5`, `ec51fbf` |  |
| ALM-08 | fixed | `5e6737c` |  |
| ALM-09 | fixed | `024af05`, `8c14663` |  |
| ALM-10 | fixed | `085ccce`, `0ee38f2`, `3f223c5`, `4089de1` … |  |
| ALM-11 | partial | `a6c9bfb` | reads split into queries.py; the legacy payload format stays |
| OPC-01 | fixed | `d71377b` |  |
| OPC-02 | fixed | `d71377b` |  |
| OPC-03 | fixed | `2bd5bbd` |  |
| OPC-04 | fixed | `a1cb509` |  |
| OPC-05 | fixed | `933cc21` |  |
| OPC-06 | fixed | `933cc21` |  |
| OPC-07 | fixed | `7426ae1` |  |
| OPC-08 | fixed | `8f04f68` |  |
| OPC-09 | fixed | `3cc4805` |  |
| OPC-10 | fixed | `8f04f68` |  |
| OPC-11 | fixed | `7426ae1` |  |
| OPC-12 | fixed | `f150862` |  |
| OPC-13 | fixed | `7ec98f7` |  |
| OPC-14 | fixed | `f150862` |  |
| OPC-15 | fixed | `a1cb509`, `a3a04a5` |  |
| OPC-16 | fixed | `933cc21` |  |
| OPC-17 | fixed | `144c0a8`, `2bd5bbd`, `8f04f68`, `933cc21` … |  |
| OPC-18 | partial | `10c63e1`, `8f04f68` | two helpers unified; the certificate builder copy in scripts stays (scripts import no package) |
| OPC-19 | fixed | `10c63e1` |  |
| OPC-20 | fixed | `144c0a8` |  |
| WEB-01 | fixed | `37670fe` |  |
| WEB-02 | fixed | `e42b640` |  |
| WEB-03 | fixed | `7bd5e75` |  |
| WEB-04 | fixed | `e42b640` |  |
| WEB-05 | fixed | `3fbabcf` |  |
| WEB-06 | fixed | `72a3b9b` |  |
| WEB-07 | fixed | `e42b640` |  |
| WEB-08 | fixed | `82c3190` |  |
| WEB-09 | fixed | `155f865` |  |
| WEB-10 | fixed | `7bd5e75` |  |
| WEB-11 | fixed | `7bd5e75` |  |
| WEB-12 | fixed | `50d90c2` |  |
| WEB-13 | fixed | `37670fe`, `39cdda0` |  |
| WEB-14 | fixed | `e235c1b` |  |
| WEB-15 | fixed | `82c3190` |  |
| WEB-16 | fixed | `3a4493e` |  |
| WEB-17 | fixed | `7bd5e75`, `82c3190`, `e42b640` |  |
| WEB-18 | partial | `e9cc63f` | three display bugs fixed; behaviour while the live channel is down is an owner decision (Q14) |
| WEB-19 | fixed | `0576789` |  |
| WEB-20 | fixed | `d5f86d6` |  |
| PLAT-01 | fixed | `420afa7` |  |
| PLAT-02 | owner decision |  | a read-only InfluxDB token needs a new way to hand out a secret (Q11) |
| PLAT-03 | fixed | `0576789` |  |
| PLAT-04 | fixed | `8f275d1` |  |
| PLAT-05 | fixed | `6514ca3` |  |
| PLAT-06 | partial | `420afa7`, `6514ca3`, `8f275d1` | actions by SHA and images by digest; no Dependabot (it works through pull requests) |
| PLAT-07 | fixed | `8f275d1` |  |
| PLAT-08 | fixed | `4f18869`, `b353f52` |  |
| PLAT-09 | fixed | `310d188` |  |
| PLAT-10 | fixed | `8f275d1`, `fc4ad0b` |  |
| PLAT-11 | fixed | `0576789` |  |
| PLAT-12 | fixed | `3e01695` |  |
| PLAT-13 | fixed | `68a03fa`, `f8c2059` |  |
| PLAT-14 | fixed | `6514ca3` |  |
| PLAT-15 | partial | `8f275d1` | healthcheck and infra anchors; per-service LOG_DIR and the image CMDs stay |
| PLAT-16 | owner decision |  | Docker secrets and *_FILE settings (Q15) |
| PLAT-17 | fixed | `3543b78`, `9667a73` |  |
| PLAT-18 | fixed | `9667a73` |  |
| PLAT-19 | fixed | `8f275d1`, `bae6ba6`, `ea1ed3e` |  |
| SHR-01 | fixed | `a498082` |  |
| SHR-02 | fixed | `5e36dc5` |  |
| SHR-03 | fixed | `a498082` |  |
| SHR-04 | fixed | `0f7051a` |  |
| SHR-05 | fixed | `a498082` |  |
| SHR-06 | fixed | `f44cd0e` |  |
| SHR-07 | owner decision |  | contract change: enum zero values (Q12) |
| SHR-08 | fixed | `4f25508` |  |
| SHR-09 | fixed | `0f7051a`, `a498082`, `f44cd0e` |  |
| SCR-01 | fixed | `4f18869` |  |
| SCR-02 | fixed | `6dd5eb1` |  |
| SCR-03 | fixed | `ef1c06a` |  |
| SCR-04 | fixed | `eb7bb71` |  |
| SCR-05 | fixed | `7feaa76` |  |
| SCR-06 | fixed | `b353f52` |  |
| SCR-07 | fixed | `40f2c1d` |  |
| SCR-08 | fixed | `df1112c` |  |
| SCR-09 | fixed | `e3d7957` |  |
| SCR-10 | fixed | `2ac88ce` |  |
| SCR-11 | fixed | `26d0ca1` |  |
| SCR-12 | fixed | `8221662` |  |
| DUP-01 | fixed | `10c63e1`, `14d72c1`, `55a7fd2`, `7723e71` … |  |
| DUP-02 | fixed | `14d72c1`, `51f3ee2`, `55a7fd2`, `87895b0` |  |
| DUP-03 | partial | `10c63e1`, `55a7fd2`, `7723e71`, `858e9b2` … | run_service everywhere; the sys.path insert stays until shared/generated is a package (Q15) |
| DUP-04 | fixed | `68bcb44` |  |
| DUP-05 | partial | `3b95a42`, `75fb0dd`, `858e9b2`, `c132f3d` | the consume loop and reconnect delay are shared; client factories stay per service |
| DUP-06 | partial | `14d72c1`, `888a8d4`, `ad72860`, `f150862` | OutageLog used where messages match; four richer outage logs stay |
| DUP-07 | fixed | `2b523b4`, `31d7172`, `75fb0dd`, `7723e71` … |  |
| DUP-08 | fixed | `7426ae1`, `858e9b2`, `9ac74b6`, `b66e607` … |  |
| DUP-09 | owner decision | `f150862` | shared enum labels would change the gateway's answer to an unknown value (Q12) |
| DUP-10 | fixed | `14d72c1`, `3002855`, `55a7fd2`, `7426ae1` … |  |
| TST-01 | fixed | `52e1c09` |  |
| TST-02 | fixed | `52e1c09` |  |
| TST-03 | fixed | `52e1c09` |  |
| TST-04 | fixed | `34a3519` |  |
| TST-05 | fixed | `dbefb94` |  |
| TST-06 | fixed | `933cc21` |  |
| TST-07 | fixed | `52e1c09` |  |
| TST-08 | fixed | `52e1c09` |  |
| TST-09 | fixed | `ecab511` |  |
| TST-10 | fixed | `085ccce`, `33e4963`, `7cce9be`, `ad72860` |  |
| TST-11 | fixed | `093cf49`, `52e1c09`, `aadd278` |  |
| TST-12 | fixed | `238c460`, `ad72860`, `e018402` |  |
| TST-13 | fixed | `ed7b737` |  |
| TST-14 | fixed | `ec51fbf` |  |
| TST-15 | fixed | `919d595` |  |
| TST-16 | fixed | `549ba42`, `ef1c06a` |  |
| TST-17 | fixed | `7cce9be`, `ecab511`, `ef1c06a` |  |
| TST-18 | fixed | `3fbabcf` |  |
| TST-19 | fixed | `bde9e01` |  |
| TST-20 | fixed | `296fcf4`, `cb9ead9` |  |
| TST-21 | owner decision | `cb9ead9` | a shared test package needs dev dependencies in every package (Q15) |
| TST-22 | fixed | `0f4d09e`, `cb9ead9`, `fee83d6` |  |
| TST-23 | fixed | `085ccce`, `4f60250`, `7cce9be`, `933cc21` |  |
| TST-24 | fixed | `82c3190` |  |
| TST-25 | fixed | `68bcb44`, `8a5fe08` |  |
| TST-26 | fixed | `7cce9be` |  |
| TST-27 | fixed | `1b8ab1f`, `6dd5eb1` |  |
| ARCH-01 | partial | `68a03fa`, `8f275d1` | seven networks keep other containers off the plant network; caller authentication is an owner decision (Q10) |
| ARCH-02 | fixed | `238c460`, `439ddaf`, `f150862` |  |
| ARCH-03 | fixed | `295b617`, `5e6737c` |  |
| ARCH-04 | partial | `0ae582d`, `119bac6`, `3837235`, `e7debbc` | one typed model per payload, a round-trip test and gateway validation; the historian keeps its lenient parser, and WebSocket frame models in OpenAPI were not added |
| ARCH-05 | fixed | `19006b0` |  |
| ARCH-06 | owner decision |  | same as PLAT-02 (Q11) |
| ARCH-07 | fixed | `f150862` |  |
| ARCH-08 | fixed | `31d7172`, `3b95a42`, `51f3ee2`, `55a7fd2` … |  |
| ARCH-09 | partial |  | the overview drift is corrected; the invariants registry text is an owner decision (Q13) |
| ARCH-10 | owner decision |  | contract change: units outside SI (Q12) |
| ARCH-11 | owner decision |  | the gateway aggregates KPI ratios over a range; moving it changes ownership (Q14) |
| ARCH-12 | fixed | `3e01695` |  |
| ARCH-13 | fixed | `c0c7e71` |  |
| ARCH-14 | fixed | `9667a73` |  |
| ARCH-15 | fixed | `af1542f` |  |
