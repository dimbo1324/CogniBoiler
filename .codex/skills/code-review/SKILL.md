---
name: code-review
description: Use to review the current diff before finalizing a task — scope, service boundaries, contracts, safety, security, invariants, test honesty, and documentation drift.
---

# Reviewing Changes

Run this **before** merging and before the final report.

## What to look at

```powershell
git status --short --branch
git diff
```

For an independent pass, delegate to the `cogniboiler-quality-reviewer` subagent.

## Mandatory checklist

1. **Scope.** No changes the task did not require.
2. **Service boundaries.** No new cross-service imports; behavior lives in its owning
   service (`docs/architecture/service-boundaries.md`); modules stay under ~700 lines.
3. **Contracts.** Proto fields only added; MQTT, REST and WebSocket shapes compatible or
   every consumer updated; schema changes carry a migration; stubs regenerated.
4. **Safety.** Actuator commands only through `PLCService`; interlocks and E-Stop cannot
   be disabled; valve limits enforced.
5. **Security.** No secrets anywhere tracked; every route has a role; mutations audited;
   allow and deny both tested.
6. **Units and time.** SI in contracts and storage; UTC epoch milliseconds.
7. **Tests.** Not deleted, skipped or loosened; no timing-dependent assertions; no new
   `xfail` without an owner decision.
8. **Documents.** Shape changed → `docs/architecture/overview.md`; stage completed →
   `**Status.**` line; rule module changed → `.ai/CHANGELOG.md` and `sync-agents`; run
   instructions changed → `README.md`.
9. **Leftovers.** Debug output, temp files, commented-out code.

## Handling findings

Serious findings — safety, security, data, contracts — are fixed in the same task.
Unrelated problems are recorded separately rather than mixed into the current diff. If
there are no findings, say so plainly instead of inventing remarks.
