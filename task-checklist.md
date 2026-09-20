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

[ ] `lucide-react` added (MIT, tree-shaken) — the only new frontend dependency; no UI kit,
    because the console stays minimal by the project's own rule
[ ] Icons carry meaning, not decoration: navigation, connection and PLC state, alarm
    severity, command results, empty states; every icon `aria-hidden` next to real text
[ ] Dark theme is the default; the operator can still choose light or follow the system
[ ] The palette is declared once per theme (the dark block was written twice) and the
    resolved theme is one attribute on `<html>`
[ ] Design tokens for spacing, radius, shadow and control height; no stray pixel values
[ ] Repeated markup extracted: panel, section header, empty state, status badge,
    icon button — screens stop re-declaring the same JSX
[ ] Vitest covers the new primitives and the changed theme behaviour; the whole suite green

## 2. Backend: modularity and duplication

[ ] One shared runtime package for what every service repeats: the reconnecting MQTT
    session and the liveness file (two byte-identical copies today)
[ ] historian, alert-manager, opcua-server and physics-engine use it; the duplicates are
    deleted and the compose healthchecks follow
[ ] `plc_controller/service.py` (893 lines) and `safety.py` (750) split by meaning —
    the project's own limit is about 700
[ ] Other duplicated logic found during the pass is extracted or removed
[ ] Magic numbers and magic strings replaced by named constants where they carry meaning

## 3. Security

[ ] A pass over authentication, authorization, audit, input validation and the places
    where a value from outside reaches a query, a command line or a file path
[ ] Every doubtful case either gets a check or an explanation of why it is safe
[ ] Findings and fixes named in the report; anything left open recorded as debt

## 4. Tests

[ ] New backend tests for the awkward cases: boundaries, races, malformed input,
    unauthorized access, partial failures — not only the happy path
[ ] Every new or changed module has its own tests; no test weakened or deleted to get green

## 5. Cleanup

[ ] Dead code and dead files removed (unused modules, empty files, superseded documents)
[ ] Nothing removed that a later stage needs; each removal justified in the report

## 6. Verification and completion

[ ] Full gate green: format, lint, strict types, pytest, generated artifacts, scripts,
    frontend lint, types, tests and build
[ ] The live stack rebuilt and checked: `smoke`, `demo` and the Playwright console checks
[ ] State documents updated (overview, ROADMAP progress, README if a user sees it, rule
    modules and their changelog)
[ ] Checklist filled honestly; report in Russian
[ ] Merged into `main` fast-forward, branch deleted locally and remotely, `main` pushed —
    the owner asked for the publish in this task
