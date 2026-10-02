# Task: fix the project by the audit — the whole plan, in waves, by parallel agents

Owner instruction of 2026-09-27: fix everything `docs/__arch__/AUDIT.md` found (224 findings),
with no new features and no new business logic, by several agents working in parallel, without
commit trouble. Findings marked "owner decision" in the audit get only their safe part
(decision log, 2026-09-27). Each wave is merged into the local `main` fast-forward after a green
full gate; `main` is not pushed.

Synchronisation: one git worktree and one `fix/audit-…` branch per work package, disjoint file
zones, shared files changed only by the lead; the lead rebases a finished branch onto
`chore/audit-remediation`, runs the full gate in its worktree and merges it fast-forward, one
at a time.

Acceptance: every finding is fixed, or marked with an honest reason (owner decision, out of
scope, not reproducible); each fix that changes behaviour has a test that failed before it;
no test weakened; the full gate green after every wave; state documents updated.

Marks: `[ ]` open, `+` done, `-` not done or partially done (with a note).

## 1. Preparation

+ Owner decision recorded in the decision log; the Д17 checklist closure carried over
+ Work packages, file zones and briefs defined for every wave

## 2. Wave A — foundations and the High findings

+ A1 shared: MQTT session, shutdown on SIGTERM, queued publisher, outage log, JSON and
    finite-number helpers, clock, logging (SHR, DUP-01/02/06/08/10 as shared helpers)
+ A2 plc-controller: fail-safe inputs, trip re-send, command failures, safety tests
    (PLC, TST-01/02/03/07/08/11, ARCH-05/13)
+ A3 api-gateway auth: credentials in the audit, throttle race, keys, seeding, grants
    migration, metrics, route inventory (GW-AUTH, TST-04/05, SHR-02, ALM-08 migration)
+ A4 alert-manager and historian: alarm intake through a DB outage, payload hardening,
    timeouts (ALM, HIST)
+ Wave A merged; full gate green; merged into local `main` — live stack: migrations 0005
  and 0006 applied on PostgreSQL, grants checked, `smoke` 13/13, `demo` 15/15 with clean logs
  (a first `demo` run failed only because the PC slept 20:10:29–20:15:14 -03:00)

## 3. Wave B — the remaining services and the console

+ B1 api-gateway operational API and realtime (GW-API, TST-12/13/14, DUP-04)
+ B2 physics-engine (PHY, TST-09/17/19/26)
+ B3 opcua-server (OPC, TST-06, ARCH-07)
+ B4 web console (WEB, TST-18/24)
+ Wave B merged; full gate green; merged into local `main` — live stack: migration 0007
  applied, `smoke` 13/13, `demo` 15/15 with clean logs, `console-e2e` 22/22. Two integration
  defects caught at merge: a flaky observability test (fixed, e34c472) and two test modules
  both named `test_boundaries.py` (renamed, fa975af)

## 4. Wave C — platform, scripts, shared helpers adopted

+ C1 platform: Compose network segmentation, InfluxDB tokens, container hardening,
    nginx, Mosquitto, Grafana, CI and supply chain (PLAT, ARCH-01 safe part, ARCH-06/12)
+ C2 developer scripts, coverage floor, audit-table check in smoke (SCR, TST-15/16/25/27,
    ARCH-03)
+ C3 plc-controller, alert-manager, historian adopt the shared helpers (DUP-02/03/05/09,
    ARCH-08, HIST-08, ALM-06)
+ Wave C merged; full gate green; merged into local `main` — live stack on seven networks,
  read-only roots, Grafana 13: every container healthy, `smoke` 19/19 (six new append-only
  audit checks), `demo` 15/15 clean, `console-e2e` 22/22, Prometheus 7/7 targets up, no ACL
  denials, no `too_many_pings`, every service logs "Stopping on SIGTERM"; coverage 98.89 %
  of lines against the new 95 % floor (1915 tests). The InfluxDB read-only token (T3) is an
  owner decision: it needs a new way to hand out a secret

## 5. Wave D — tests of the invariants and test hygiene

+ D1 invariant tests and contract models (ARCH-02/04/11), test hygiene (TST-10/20-23) —
  ARCH-04 partial (the historian keeps its lenient parser), ARCH-11 an owner decision
+ Wave D merged; full gate green; merged into local `main` — live stack with the exact-topic
  ACL: no broker denials, no alarm processed twice, `smoke` 19/19, `demo` clean,
  `console-e2e` 22/22; 1994 Python tests at 98.90 % line coverage, 287 console tests

## 6. Completion

+ State documents: overview, invariants proposals for the owner, roadmap progress note,
    rule modules corrected where stale — the invariants registry itself is unchanged (Q13)
+ Live stack: `stack up`, `smoke`, `demo` green with clean logs
+ Every finding accounted for in a closing table; checklist filled; report in Russian —
  section 10 of `docs/__arch__/AUDIT.md`: fixed 188, partial 26, owner decision 10
+ Worktrees and merged `fix/audit-…` branches removed; `main` not pushed
