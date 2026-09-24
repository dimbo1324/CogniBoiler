# Task: console polish and platform sanitation (before 1.0)

Owner instruction of 2026-09-20: before the project is wrapped up, improve the console
(Lucide icons and other open-source, cleanly-integrated building blocks; a dark default
theme), make the whole project more modular, readable and maintainable (no duplicated
business logic, reusable things extracted into their own entity), make it safer (extra
checks where a case is doubtful), remove magic numbers and magic strings, extend the
tests — especially the backend ones, and especially with awkward and adversarial cases —
and delete code and text files the project no longer needs. Then run the full gate, fix
everything it finds, merge into `main`, delete every branch and push.

Order of work is mine to choose; the owner listed the items as they came to mind.

Marks: `[ ]` open, `+` done, `-` not done or partially done (with a note).

## 1. Console: appearance and structure

+ `lucide-react` added (ISC, tree-shaken: +14 KiB of the bundle) — the only new frontend
  dependency; no UI kit, because the console stays minimal by the project's own rule
+ Icons carry meaning next to their labels, never instead of them: navigation, connection
  and PLC state, alarm severity, command results, KPIs, empty states; every glyph is
  `aria-hidden`, so what a screen reader and a test read is unchanged
+ Dark theme is the default; light and follow-the-system stay a click away
+ The palette is declared once per theme (the dark block had been written twice) and only
  the resolved theme reaches the page, as `data-theme` on `<html>`
+ Design tokens for spacing, radius, shadow, control height and icon size
+ Repeated markup extracted: `Panel`, `Badge`, `Icon`, and the error, empty and framed
  notes — fifteen screens stopped re-declaring the same JSX
+ The icon vocabulary is chosen once in `components/ui/icons.ts`: one meaning, one glyph
+ Vitest covers the new primitives and the changed theme behaviour: 22 → 23 files,
  196 → 212 tests, all green; the Playwright theme check follows the new default

## 2. Backend: modularity and duplication

+ `shared/runtime` (`cogniboiler-runtime`) holds the reconnecting MQTT session, the
  liveness file and the clock; both shared packages now ship `py.typed`, so mypy checks
  calls into them instead of treating them as `Any`
+ Seven copies of the session loop became one, in physics-engine, plc-controller,
  historian, alert-manager, opcua-server and api-gateway; the two byte-identical liveness
  modules became one and the compose healthchecks follow
+ `plc_controller/service.py` (893 lines) and `safety.py` (750) split by meaning into
  `commands.py`, `status.py`, `safety_limits.py` and the alarm wiring in `alarms.py`;
  both files are now under the project's 700-line rule (683 and 532)
+ Other duplication removed: `now_ms` in seven modules, the milliseconds of a day in four,
  the severity comparison and the SI unit of a parameter name in the console
+ Magic values named: the PLC's ramp rates, the gateway's step and fault bounds, the
  QoS of the realtime topics, the overview's list lengths, bar and megawatt conversions

## 3. Security

+ A pass over authentication, authorization, audit, input validation and every place a
  value from outside reaches a query, a command line or a file path
+ Fixed: every name reaching a Flux query is escaped where the query is built (the routes
  validated, the builder relied on it); the liveness file no longer takes a directory left
  by a mount for a heartbeat; a NaN no longer costs a whole batch of telemetry; a flood of
  invented names no longer evicts a locked account from the login throttle
+ Checked and left alone, with the reason in the report: the audit trail stores a digest
  and never a body, the WebSocket takes its token in the first frame and not in the URL,
  an unexpected error answers with Problem Details and no internals, role provisioning
  builds its statement inside PostgreSQL, and the scripts never use a shell

## 4. Tests

+ New backend tests for the awkward cases: NaN and infinity through a float field of the
  contract, a value exactly on a bound, a pressure rate of zero, a reading from a failed
  instrument, a broker that refuses every connection, a session that ends politely, a
  sliding throttle window, an eviction flood, and a generated password carrying a quote,
  a backslash, a colon and a percent sign
+ Every new module has its own tests: `shared/runtime` 27, the PLC's pure layers 35, the
  console's primitives 13, the login throttle 10, role provisioning 6
+ Python tests 1001 → 1082; no test weakened or deleted to get green. The three that
  changed did so because the behaviour they asserted changed on purpose: the theme's
  default and cycle, and two log lines that now say "MQTT error" once per outage instead
  of once per retry

## 5. Cleanup

+ Removed: `shared/models` (imported by nothing), the empty `list.todo`, and
  `docs/__arch__/archive/plan-2026-04-helper.md` — a plan from April 2026 that VISION and
  ROADMAP replaced, in the folder agents are told to read
+ Kept, with the reason: `apps/ai-predictor` and `ml/preprocessing` belong to the deferred
  S15 and are named in the roadmap; deleting them would throw away a planned stage
+ Dead constants removed where found (`SUBSCRIBE_TOPIC` in the historian, two re-export
  aliases in physics-engine)

## 6. Verification and completion

+ Full gate green: format, lint, strict types, 1082 pytest, generated artifacts, scripts'
  own tests, frontend lint, types, 212 Vitest and the build
+ The live stack rebuilt from this branch and checked: every container healthy (including
  the two healthchecks that now run `python -m cogniboiler_runtime.liveness`), `smoke`
  green, `demo` green with no `error` line in `logs/`, Playwright console checks green
+ State documents updated: the architecture overview (the shared runtime package and the
  console's primitives), the ROADMAP progress note, the README (dark by default), the
  project map, the rule changelog and the regenerated `AGENTS.md`
+ Checklist filled honestly; report in Russian
+ Merged into `main` fast-forward, branch deleted locally and on `origin`, `main` pushed —
  the owner asked for the publish in this task
