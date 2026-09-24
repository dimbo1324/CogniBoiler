<!--
GENERATED FILE - DO NOT EDIT.
Source of truth: .ai/universal/*.md and .ai/project/*.md.
Edit a module, then run: python dev_tools_scripts_runner.py sync-agents
-->

# CogniBoiler - working notes for Codex

This file is the Codex entry point. It is assembled from the shared rule modules in
`.ai/` so Codex and Claude Code always follow identical rules. Later sections override
earlier ones; an explicit owner instruction in the current conversation overrides
everything.

---

<!-- module: .ai/universal/01-workflow.md -->

# Workflow: Git, Branches, Commits

Purpose: every task follows one predictable git cycle. No exceptions without an
explicit owner instruction in the current conversation.

## Branch discipline

- NEVER develop directly on `main`.
- Start every task from up-to-date `main`:
  `git checkout main` → `git pull --ff-only origin main` → `git checkout -b <branch>`.
- Branch name format: `type/short-task-description`
  (types: `feat`, `fix`, `refactor`, `test`, `chore`, `docs`, `ci`, `perf`, `security`).
- Uninformative branch names (`test`, `fix`, `work`, `final`, `new`) are forbidden.
- Merge into `main` only fast-forward (`git merge --ff-only`) and only after the
  project's full quality gate is green.
- Push to `origin/main` only when the owner explicitly asked for a publish within the
  current task. Work-in-progress branch pushes are allowed (for example, to publish
  the task checklist before starting work).
- Delete the task branch after it is merged.

## Commits

- Message format: `type: short description of what and why`.
- One commit = one logically complete unit. Do not mix a bug fix, a refactor,
  formatting, and new features in a single commit unless they are one inseparable task.
- Forbidden messages: `fix`, `update`, `wip`, `changes`, `final`, `123`.
- Keep commits attributable: include the assistant identity trailer the project already
  uses (check a recent `git log` for the convention).

## Quality gate before merge

- Run the project's checks (see the project commands module) before merging.
- If mandatory checks fail, merging into `main` is forbidden until fixed or the owner
  explicitly decides otherwise.
- Before merging, self-review the diff: changes match the task, no stray files, no
  debug leftovers, no secrets, no accidental unrelated edits.

When unsure whether an action counts as "explicitly requested": ask, or stop after the
branch commit and report instead of pushing.

---

<!-- module: .ai/universal/02-task-checklist.md -->

# Task Checklist and Definition of Done

Purpose: every task is planned before it starts and honestly accounted for after it
ends, in a file any reviewer can read without the conversation.

## task-checklist.md protocol

A file named `task-checklist.md` lives in the repository root and is always tracked by
git (never in `.gitignore`).

Before starting a task:

1. Clear or recreate `task-checklist.md` for the new task.
2. Write the main stages, checks, and expected outcomes as `[ ]` items, grouped into
   short sections (preparation / implementation / verification / completion).
   Moderate detail — stages, not keystrokes.
3. Commit the checklist BEFORE doing the main work.

After finishing the task:

4. Mark every item: `+` done, `-` not done or partially done.
5. Commit the filled checklist together with the completed work.
6. Never hide unfinished items — a `-` with an honest note is correct; a silently
   ignored item is a violation.

## Definition of Done

A task is complete only when ALL of the following hold:

- code written and matching the task, without unrequested scope;
- code formatted; lint and type checks pass;
- tests added or updated where reasonable; existing tests pass;
- project builds; no obvious errors in logs;
- no secrets, no temp files, no accidental changes in unrelated files;
- architecture and state docs updated if the task changed the system's shape;
- task checklist filled with `+`/`-`;
- final report written.

## Final report

The final report is ALWAYS the last step of a task. It states: what was done; which
files and areas changed; which checks ran and their results; dependency, API, database,
or config changes; security, performance, and compatibility risks; and — explicitly and
honestly — anything that was not done or failed.

---

<!-- module: .ai/universal/03-scope-and-code-style.md -->

# Scope Control and Code Style

Purpose: change only what the task requires, and keep code readable without decoration.

## Minimal changes

- Touch only what the current task needs. Forbidden without necessity:
  mass-reformatting other files, renaming things outside the task, changing
  architecture "while at it", rewriting working code without cause, changing UI when
  the task is not about the interface, deleting existing functionality without a
  direct requirement.
- If you discover an unrelated problem, record it separately (report it, or file it per
  the project's process) — do not mix it into the current diff.
- Do not create new documentation (README, `.md`, `.txt`) unless the task requires it
  directly. Exception: the project's designated architecture and progress documents,
  which must be kept current.

## Comments

- No comments by default. Code must be clear through structure and naming.
- A comment is allowed only when the task demands it, or when important logic stays
  non-obvious even with good naming. It explains the non-obvious "why", never restates
  the code.
- Stale, false, or misleading comments are forbidden. No doc comments that merely
  repeat a function name.

## File size

- Regular code files: keep under roughly 1000 lines; split by meaning when approaching
  the limit. Projects may set a stricter limit — the stricter limit wins.
- Application entry points (`main`-type files): under roughly 100 lines; extract
  configuration, startup, and service initialization into modules.
- Exemptions: test files, developer-tool scripts, and any files the project explicitly
  exempts.

## Frontend restraint

- No visual polish (styling, animations, decorative elements, redesign) unless the task
  is explicitly about appearance.
- When a task needs an interface, build the minimum that exercises the business logic
  correctly.
- Never change visual style, interface structure, component behavior, or user flows
  without a direct requirement.

---

<!-- module: .ai/universal/04-architecture-boundaries.md -->

# Architecture Boundaries, Workarounds, Tech Debt

Purpose: respect the project's layering; make every shortcut visible.

## Boundaries

- Follow the project's existing architecture. Forbidden: business logic in UI or
  controllers when a service layer exists; direct storage access in handlers when a
  repository layer exists; bypassing existing services and abstractions without cause;
  circular module dependencies; dumping unrelated logic into catch-all files.
- If the architecture genuinely blocks the task, do not hack around it — propose a
  proper structural change and reflect it in the architecture docs once approved.
- A doc-comment-only boundary is not enforced. Use a type only the check can construct,
  or a gate no caller can skip — and test that it accepts, not only that it rejects.

## Temporary solutions

- Workarounds are allowed only exceptionally, and every one must be recorded
  explicitly: why it exists, where it lives, its limits and risks, and when it must be
  replaced.
- Hidden workarounds are forbidden. Do not scatter uncontrolled `TODO`/`FIXME` marks;
  important debt gets a task or an entry in the project's tracking process.

## Tech debt

- Debt found during a task that cannot be fixed now is recorded explicitly (in the
  final report at minimum), never disguised as a normal solution.
- Debt touching security, data integrity, performance, or stability is priority debt —
  call it out loudly.

---

<!-- module: .ai/universal/05-security-and-secrets.md -->

# Security, Secrets, Dependencies, Portability

Purpose: nothing sensitive in the repo; every change safe within its area; the project
runs on any machine.

## Secrets — absolute ban

- NEVER put in code, git, tests, docs, or examples: passwords, API keys, tokens,
  private keys, real credentials, cookies, production `.env`, or personal user data.
- Secrets live only in `.env` (untracked), environment variables, secret managers, or
  CI/CD secrets. The repo may contain only a safe `.env.example`.
- A secret that ever reached git is compromised: rotate it; deleting the line in a new
  commit is not enough.

## Security in every task

Check within the area you touch: authorization and access rights, input validation,
injection, XSS and CSRF where applicable, unsafe redirects, file uploads, personal data
handling, public endpoints, token storage and transport, and access to admin functions.
Security is part of every task, not a future task.

## Dependencies

- No new dependency without justification: what for, can it be done without, is it
  maintained, known vulnerabilities, stack compatibility, does it duplicate an existing
  dependency, is it too heavy for the need.
- Never add a heavy library for one small function.
- After changing dependencies: update the lock file and verify the build.
- A new production dependency must be named in the final report.

## Portability

- No machine-specific values in code: local absolute paths, usernames, or anything
  environment-dependent. Such values go to configuration.
- The project must remain runnable by someone else using the project's documented tools.

---

<!-- module: .ai/universal/06-quality-and-testing.md -->

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

---

<!-- module: .ai/universal/07-multi-assistant.md -->

# Multi-Assistant Collaboration

Purpose: several AI assistants and humans work in this repository across separate
sessions; git is the coordination surface and the rule modules are shared.

## Coordination through git

- Before non-trivial work, check
  `git log -10 --date=iso-strict --pretty=format:"%h %cd %s"` and
  `git status --short --branch`: recent commits may be another assistant's finished
  work — not yours to redo or second-guess.
- Never rewrite history on another assistant's in-flight branch. Build on top of it or
  ask first.
- Keep commits attributable: include the assistant identity trailer the project already
  uses (see recent `git log` for the convention).

## Shared rule modules

- All assistants obey the same rules from `.ai/universal/` and `.ai/project/`. There is
  exactly one source of truth.
- `CLAUDE.md` imports the modules natively; `AGENTS.md` is GENERATED from them. Never
  hand-edit `AGENTS.md`; edit the module and run the project's sync command (see the
  project commands module).
- When a task changes shared behavior (workflow, gates, style, guardrails), change the
  module once — every assistant picks it up. Mirror-maintained per-assistant files
  (`.claude/` and `.codex/`: agents and skills) still need the same edit on both sides
  in the same task.

## Session hygiene for any model

- Re-read plans and rule files from disk instead of trusting memory of a previous
  session — files change between sessions.
- When context is shaky or the task is large, restate the task's acceptance criteria in
  the checklist before coding, and verify against files, not recollection.
- If two rules appear to conflict: the project module wins over the universal module;
  an explicit owner instruction in the current conversation wins over both. Say out
  loud which rule you chose and why.

---

<!-- module: .ai/project/10-project-map.md -->

# Project: CogniBoiler

A digital twin of a 300 MW gas-fired steam unit (boiler, turbine, virtual PLC) built as
Python microservices speaking MQTT, gRPC and OPC UA, with a historian, an authenticated API
and a web console. It is a portfolio project: it must run and demo end to end on one
machine with Docker Compose.

**AI is deferred** (decision 2026-09-14): the models planned for `apps/ai-predictor` and
`ml/` are out of scope until the owner reopens that stage; the platform must be complete
and demonstrable without them.

## Repository map

uv workspace on Python 3.14; every service is a package with `src/` and `tests/`:

- `apps/physics-engine` — boiler/turbine model, live runtime, `PhysicsService` gRPC API,
  MQTT telemetry; the single owner of process state.
- `apps/plc-controller` — virtual PLC: command validation, setpoints, coordinated control,
  AUTO/MANUAL/ESTOP, interlocks and the E-Stop latch, alarm conditions and events on MQTT,
  `PLCService` gRPC API.
- `apps/api-gateway` — FastAPI edge and user authority: JWT sessions with rotated refresh
  tokens, RBAC, users, append-only audit, REST and WebSocket, simulation control, history
  and KPIs; the Alembic chain in `apps/api-gateway/migrations`.
- `apps/historian` — telemetry, KPIs, labels, alarm changes and PLC events into InfluxDB;
  owns retention and downsampling.
- `apps/alert-manager` — alarm lifecycle, alarm tables in PostgreSQL, `AlarmService` gRPC
  API, alarm changes on MQTT.
- `apps/opcua-server` — OPC UA (IEC 62541) projection of plant, PLC and alarm state; its
  methods run through the gateway as the signed-in user.
- `apps/web` — the operator console (React + TypeScript + Vite, pnpm).
- `apps/ai-predictor` — deferred placeholder, **not** a workspace member.

Shared and supporting areas:

- `shared/proto/cogniboiler.proto` — the gRPC and telemetry contract, stubs in
  `shared/generated/`; `shared/openapi/` — the gateway's REST contract;
  `shared/observability` — logging, correlation ids and metrics for every service;
  `shared/runtime` — the MQTT session and liveness.
- `docker-compose.yml`, `Dockerfile`, `.env.example` — the local stack.
- `infrastructure/` — Mosquitto, Grafana and Prometheus provisioning.
- `ml/` — deferred AI material.
- `dev_tools_scripts_runner.py` + `scripts/` — the developer-tools orchestrator
  (`scripts/runner`); `scripts/_toolkit` is the only shared code, one directory per script
  with its own `config/*.json`; scripts never import each other.
- `.ai/`, `.claude/`, `.codex/` — assistant rules and workspaces.

## Internal vs external documents

A document serves exactly one audience.

**Internal — everything in `docs/__arch__/`, written in Russian.** For the builders:
`VISION.txt` (the product vision without AI), `ROADMAP.md` (stages and what is done),
`open-questions.md` (owner decisions). Nothing a user reads links to them.

**External — written in English.** For newcomers: `README.md` (the hub; every external
document is reachable from it), `docs/architecture/overview.md`,
`docs/architecture/invariants.md`, `docs/architecture/service-boundaries.md`.

## Language policy

- **English**: `.ai/`, `.claude/`, `.codex/`, `CLAUDE.md`, `AGENTS.md`,
  `task-checklist.md`, code, comments, commit messages, test names, UI copy, and every
  external document.
- **Russian**: the internal documents in `docs/__arch__/`, and reports to the owner.
- Do not mix languages inside one file.

## Documentation policy

- `docs/__arch__/VISION.txt` changes only when the product intent changes, by owner
  agreement. `docs/__arch__/ROADMAP.md` is the plan and progress record.
- The no-new-docs exception (`03-scope-and-code-style.md`) covers `docs/architecture/`,
  `README.md` and `docs/__arch__/ROADMAP.md`.

## Product guardrails

- `physics-engine` is the only writer of process state; every other service observes.
- Every actuator command reaches the plant through `plc-controller`; nothing bypasses
  validation, the interlocks, or an active E-Stop.
- The whole platform starts with `stack up` on one machine and demos without internet.
- AI stays optional: no service may need `ai-predictor` to start or to pass its tests.
- **Stage order in `docs/__arch__/ROADMAP.md` is binding.** Skipping ahead needs an owner
  decision recorded in `docs/__arch__/open-questions.md`.

---

<!-- module: .ai/project/11-commands.md -->

# Project Commands and Quality Gates

All commands run from the repository root: PowerShell or Git Bash on Windows, any POSIX
shell elsewhere.

## The script orchestrator — start here

```powershell
python dev_tools_scripts_runner.py          # interactive menu
python dev_tools_scripts_runner.py list     # the catalog — use this, not the menu
python dev_tools_scripts_runner.py <name>   # run one; `help` prints the manuals
```

The orchestrator is stdlib-only on any Python 3.14; a script needing a project package
hands itself to `uv run`. With no arguments and no terminal it runs `quality-gate`, so an
agent or CI can call it safely.

| Script | Purpose |
|---|---|
| `quality-gate` (`--quick`) | the one verification path; CI runs it |
| `format-code` (`--check`) | ruff for Python, Prettier for the frontend |
| `audit-deps` | known vulnerabilities in the lock files |
| `sync-agents` (`--check`) | regenerate `AGENTS.md` from `.ai/` |
| `stack up` / `status` / `logs` / `down` | the Docker Compose stack |
| `dev-secrets` | `.env` with generated secrets |
| `smoke` | end-to-end check of a running stack; CI runs it |
| `demo` | plays the VISION §7 scenario; fails on an `error` in `logs/` |
| `backup` / `restore` | the stack's databases; `restore` asks before overwriting |
| `console-e2e` (`--url`) | Playwright checks of the console |
| `generate-proto`, `generate-openapi` (`--check`) | gRPC stubs; OpenAPI and console types |
| `doctor` | read-only toolchain check |
| `install-hooks` | the pre-commit hook, once per clone |
| `clean-caches` (`--apply`) | **deletes files**; a dry run unless `--apply` |
| `selftest` | the scripts' own tests |

**Standing duty — keep the scripts true.** A task that changes how the project is built,
checked, formatted, run or cleaned updates the matching script in the same task, and runs
`selftest` after touching `scripts/`. A new routine job is a new directory under
`scripts/` plus one entry in `scripts/runner/config/scripts.json` — never new Python in
`scripts/runner/`.

## Direct commands, one layer at a time

```powershell
uv sync --all-packages                    # the environment exactly as uv.lock says
uv run pytest apps/plc-controller/tests   # one service's tests
uv run ruff check . ; uv run mypy         # lint, strict types
pnpm --dir apps/web install               # frontend dependencies, once
```

Run commands per service, ports, topics and platform notes: `14-command-reference.md`.

## Publishing

- No pull requests (owner decision 2026-09-15). A task branch is merged into `main`
  locally with `git merge --ff-only` after a green full gate, then published with
  `git push origin main`. Agents push only when the owner asked for it in the current task.
- `main` on GitHub must not require pull requests; force pushes and non-fast-forward
  updates stay blocked. If a rule declines a push to `main`, stop and tell the owner —
  do not route around it with a side branch.

## Gate policy

- The full gate is green before any merge to `main`; `--quick` is the minimum before a
  push. Docs- and config-only changes still run it.
- The gate runs every section even after a failure and ends with one summary.
- The `--check` modes of `sync-agents`, `generate-proto` and `generate-openapi` are gate
  sections on purpose: a source edited without regenerating its artifact breaks the build.
- Frontend sections are skipped without `apps/web/node_modules`, and fail when `CI` is set.
- The pre-commit hook only formats and checks file hygiene. Lint, strict typing and tests
  are never commit-time checks, and never skipped at merge time.
- `xfail(strict=True)` marks a recorded known defect, allowed only by an owner decision in
  `docs/__arch__/open-questions.md`; never a way to get green.

---

<!-- module: .ai/project/12-domain-rules.md -->

# Domain Rules and House Style (CogniBoiler)

These sharpen the universal rules for this codebase. Stricter wins.

## Python code

- Python 3.14 with full type hints; `mypy --strict` over `apps/` and `shared/` is a gate
  section. A `# type: ignore` carries its error code.
- A module past roughly 700 lines is split by meaning. `__main__.py` entry points hold
  argument parsing and wiring only.
- A service never imports another service's internals — talk over gRPC, MQTT or the
  database contract instead. Tests may run another service in-process as a fixture
  (`plc-controller` tests use the physics runtime), through a development dependency only.
- Configuration comes from environment variables (pydantic-settings, or argparse defaults
  for local runs), never from constants edited per machine.

## Units, time and contracts

- SI units in every contract, message and column: Pa, K, kg/s, W, m, s. Convert to bar,
  °C or MW only at a presentation edge, and name the unit in the field (`pressure_pa`).
- Timestamps are UTC epoch milliseconds (`timestamp_ms`) in contracts and storage.
- `shared/proto/cogniboiler.proto` is a contract: add fields, never renumber or reuse a
  field number; regenerate the stubs with `generate-proto` in the same commit.
- MQTT topics and payloads are a contract listed in `docs/architecture/overview.md`;
  changing one updates every publisher and subscriber in the same task.
- Database schema changes go through Alembic migrations only.

## Control and safety

- The PLC is the only path from an operator to an actuator. The API gateway calls
  `PLCService`; it never calls `PhysicsService.ApplyControlCommand`.
- Interlocks and the E-Stop latch cannot be disabled by configuration, an API parameter,
  or a flag. Resetting a trip is an explicit, role-checked, audited action.
- Valve commands are validated to [0, 1] at the PLC and again in the physics runtime.
- A change to plant parameters, PID tuning or interlock thresholds states its physical
  reasoning in the commit body and is covered by a test of the behavior it changes.

## Async services

- No blocking work on the event loop: physics steps run in `asyncio.to_thread`, and
  blocking client libraries are wrapped the same way.
- Network clients reconnect with a delay and log once per failure; they never spin.
- On Windows, entry points pass `loop_factory=asyncio.SelectorEventLoop`: aiomqtt needs
  `add_reader()`.
- Entry points call `configure_logging(<service>)` (`cogniboiler_observability`) and serve
  `/metrics`; gRPC channels and servers use its interceptors, so logs are JSON lines with
  the correlation id.

## Security

- Keys, passwords and tokens come from `.env` or the environment; `dev-secrets` generates
  local ones. `.env` and `certs/*.pem` are never committed.
- Every mutating API route requires a role (`viewer` < `operator` < `engineer` < `admin`)
  and writes an audit entry. Default users are seeded only for local development.
- Passwords are Argon2id hashes; a failed login never reveals whether the user exists.
- Each service has its own PostgreSQL role and MQTT account: a new table grants its rights
  in its migration, a new topic gets its broker ACL entry.

## Tests

- Unit tests need no broker, database or internet: use fakes, SQLite in memory, or
  in-process gRPC servers on port 0.
- A test whose outcome depends on machine speed is a defect to fix, not a test to loosen.
- Never delete, skip or weaken a test to make the gate green.

## Frontend (`apps/web`)

- React + TypeScript in strict mode + Vite. The browser talks only to the API gateway
  (REST and WebSocket) — never to MQTT, gRPC or a database.
- Unit conversion and formatting live in one presentation module; business rules stay in
  the backend. The console is minimal and functional unless a task is about appearance.

## Assistant workspaces

- `.claude/agents|skills` and `.codex/agents|skills` are name-for-name mirrors.
- `.claude/settings.json` allowlists routine read and verification commands and denies
  destructive git and Docker volume operations. Extend the allowlist rather than routing
  around it; never remove a deny entry without explicit owner approval.

---

<!-- module: .ai/project/13-progress-tracking.md -->

# Project Progress Tracking

Purpose: any assistant, in any session, on any model, can locate exactly where the
project stands and where it is going — **from files, not from memory**. This is the
primary recovery mechanism after a lost conversation.

## Where the truth lives

| Question | File |
|---|---|
| What the product should become (without AI) | `docs/__arch__/VISION.txt` |
| What is planned, in what order, what is done | `docs/__arch__/ROADMAP.md` |
| What is actually built right now | `docs/architecture/overview.md` |
| What must never break | `docs/architecture/invariants.md` |
| Which service owns what | `docs/architecture/service-boundaries.md` |
| Owner decisions and open questions | `docs/__arch__/open-questions.md` |
| What the current or last task was | `task-checklist.md` |
| What actually happened recently | `git log -15 --date=iso-strict --pretty=format:"%h %cd %s"` |
| How the rules themselves changed | `.ai/CHANGELOG.md` |

## Orientation ritual — at the start of EVERY task

In order, without skipping:

1. `git status --short --branch` and
   `git log -15 --date=iso-strict --pretty=format:"%h %cd %s"`.
2. `docs/__arch__/ROADMAP.md` §1 and each stage's `**Status.**` line: a stage with one is
   done; **the first stage without one is next**.
3. `docs/architecture/overview.md` — what exists in the code right now.
4. `task-checklist.md` — what the previous task was and whether it finished cleanly.
5. `docs/__arch__/open-questions.md` — whether a decision changes the plan.
6. Only then plan the new task.

## Update duties when finishing work

- Completed a stage or a significant slice → add or refresh the `**Status.**` line under
  that stage in `docs/__arch__/ROADMAP.md` (what shipped: services, endpoints, screens,
  tests) and the status column of §1. Russian, to match the file.
- Changed the system's shape (a new service, endpoint group, topic, table, screen, or
  operational job) → update `docs/architecture/overview.md`.
- Made or received an owner decision that constrains the future → record it in
  `docs/__arch__/open-questions.md`, not only in the chat.
- Introduced an invariant → record it in `docs/architecture/invariants.md`.
- Changed a rule module → record it in `.ai/CHANGELOG.md` and run `sync-agents`.
- Changed what a user can install, run or see → update `README.md`, in English.
- Wrote a new document → decide its audience before choosing where it lives (the
  internal/external split in the project map).

## Drift guard

If the plan, the state documents and the code disagree: **the code is the fact, the plan
is the intent**. Reconcile them in the same task or report the mismatch explicitly.
Stale documentation is worse than no documentation.

## Unfinished-task rule

If `task-checklist.md` still holds open `[ ]` items from a previous session, resolve them
first: finish them, or mark them `-` with an honest note. Starting a new task on top of a
silently abandoned one is a violation.

---

<!-- module index: extended -->

# Modules loaded on demand

These rules bind exactly like the inlined ones; only their full text lives
outside this file, to stay within the instruction budget. Read the file itself
when a task touches it — that is an obligation, not a suggestion.

## Rules Evolution: Keeping the Instructions Current

File: `.ai/universal/08-rules-evolution.md`

Never weaken a rule to make a task easier — propose changes instead; autonomous edits may only clarify or correct, never loosen; every change needs a changelog entry and a regenerated entry point.

## Time and Timestamps: Never Take a Date Without Its Moment

File: `.ai/universal/09-time-and-timestamps.md`

Read and report every date at full precision — hours, minutes, seconds, zone. For git use `--date=iso-strict` (`git log --pretty=format:"%h %cd %s"`); `--oneline`, `--date=short` and "2 days ago" never answer *when*.

## Command Reference: Services, Stack, Contracts and Platform Notes

File: `.ai/project/14-command-reference.md`

Lookup material — running each service from the host, the Compose stack with its ports and credentials flow, the MQTT topics, protobuf regeneration, the frontend commands, and the Windows notes (selector event loop, a `.venv` held by VS Code, `AppData` redirection under the Claude desktop app). The always-apply policies live in `11-commands.md`.
