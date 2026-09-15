# Quality: Tests, Errors, Data, Contracts, Performance

Purpose: changes are verified, honest about failure, and safe to release.

## Tests

- Cover new or changed code where reasonable: the happy path, validation errors, edge
  cases, access rights, service and storage behavior, and regressions for fixed bugs.
- NEVER delete, disable, or weaken tests just to make a build green.
- If tests were not added, say why in the final report.
- Prefer the project's designated check runner over ad-hoc command sequences, and keep
  that runner updated when a task adds an important new part of the system.

## Errors and logging

- Handle errors explicitly. Forbidden: silently swallowed errors, empty catch blocks,
  debug logs left after the task, secrets or personal data in logs, print-style
  debugging instead of real handling.
- Logs must say where it broke, which component, what context matters, and how critical
  it is. Useful, not noisy.

## Database migrations

- Schema changes happen ONLY through migrations.
- Never edit or rename an already-applied migration; never delete migrations without a
  separate decision; never make irreversible changes without risk analysis.
- Verify each migration on a clean database, on an existing database where applicable,
  and together with the code that uses the new structure.

## Contracts

- When a task changes an API or an artifact format, update the contract and generated
  types.
- Never silently change field names, data types, response structure, error codes,
  parameter requiredness, or endpoint behavior.
- Any breaking change must be named explicitly in the final report.

## Performance and releases

- No premature optimization, but no obviously wasteful patterns: N+1 queries, heavy
  per-request computation, render loops, unpaginated large reads, redundant calls,
  blocking operations in responsive paths.
- If a change may affect performance, say so in the final report.
- Every change should be revertible; for risky changes plan the rollback before merging.
- Large new features ship behind a feature flag where the project supports them;
  unfinished functionality must not be reachable by accident; stale flags get removed
  after stabilization.
