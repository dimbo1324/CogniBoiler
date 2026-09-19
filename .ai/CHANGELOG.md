# Rule Changes Changelog

History of changes to the AI assistant rule system (`.ai/`, `CLAUDE.md`, `AGENTS.md`,
`.claude/`, `.codex/`). Every rule change gets an entry here — see
`universal/08-rules-evolution.md` for the protocol.

Format: date, what changed, why, who decided. Newest first.

## 2026-09-19 — A `.venv` without `pyvenv.cfg` in the platform notes

**What changed.** The note on a `.venv` held by VS Code in
`project/14-command-reference.md` now also describes the silent variant: a rebuild that
races VS Code leaves `.venv` without `pyvenv.cfg`, the interpreter runs as the base Python,
and the commit hook fails with `No module named pre_commit` while `uv sync` reports nothing
to do. The module's essence is unchanged, so `AGENTS.md` is not affected.

**Why.** It cost time in the 1.0 readiness check: the existing note named only the loud
failure. A clarification of an existing rule, which `universal/08-rules-evolution.md`
allows without approval.

**Decided by.** Agent, during the owner-requested readiness check of 2026-09-19.

## 2026-09-19 — Service log files in the command reference

**What changed.** `project/14-command-reference.md` describes `LOG_DIR` (a JSON file per
service, rotated by `LOG_FILE_MAX_BYTES` and `LOG_FILE_BACKUPS`), the repository's `logs/`
directory that the Compose stack mounts and `stack up` creates, and `LOG_DIR=logs` for a
service run from the host. The module's essence is unchanged, so `AGENTS.md` is not
affected.

**Why.** The owner asked for service logs to be kept in a project folder; the reference
must say where they are and how a host-run service joins them. A factual addition, which
`universal/08-rules-evolution.md` allows without approval.

**Decided by.** Agent, as part of the owner-requested logging work of 2026-09-19.

## 2026-09-18 — Delivery in the command reference (S12)

**What changed.** `project/14-command-reference.md` describes the Compose profiles
(`infra`, `core`, `observability`, `full`), `stack up --profile`, the per-service image
names `${COGNIBOILER_REGISTRY:-cogniboiler}/<service>:${COGNIBOILER_TAG:-dev}`, running a
published release with `stack up --no-build`, and that CI publishes images on `main` and
`v*` tags. Prometheus is listed under the profiles `observability` and `full`. The module's
essence is unchanged, so `AGENTS.md` is not affected.

**Why.** S12 changed how the stack is started and where its images come from; the old line
"(and Prometheus)" and the single-image assumption were no longer true. Factual
corrections, which `universal/08-rules-evolution.md` allows without approval.

**Decided by.** Agent, as part of the owner-requested S12 work of 2026-09-18.

## 2026-09-18 — Hardening in the project rules (S11)

**What changed.** `project/12-domain-rules.md` gains the rule that each service has its own
PostgreSQL role and MQTT account, so a new table grants its rights in its migration and a
new topic gets its broker ACL entry. `project/11-commands.md` lists `audit-deps`.
`project/14-command-reference.md` describes the authenticated broker without a WebSocket
listener, the database roles, the OPC UA endpoints, the nginx entry on 8080/8443, the
gateway without a host port, and how host-run services pass `MQTT_USERNAME`,
`MQTT_PASSWORD` and `GATEWAY_URL`. To stay within the 30 KiB budget of `AGENTS.md`, the
descriptive lines of `project/10-project-map.md` (the summary, the AI note and the service
list) were condensed without changing a rule; the budget now has almost no room left, which
is raised with the owner.

**Why.** S11 changed how services authenticate and how the stack is reached; without these
lines the next table or topic would silently lack its grant or ACL entry. Additions and
factual corrections, which `universal/08-rules-evolution.md` allows without approval.

**Decided by.** Agent, as part of the owner-requested S11 work of 2026-09-18.

## 2026-09-18 — Observability in the project rules (S9)

**What changed.** `project/12-domain-rules.md` gains the rule that entry points call
`configure_logging` and serve `/metrics`, and that gRPC channels and servers carry the
shared interceptors. `project/10-project-map.md` lists `shared/openapi` and the new workspace
package `shared/observability`, and Prometheus under `infrastructure/`.
`project/14-command-reference.md` lists Prometheus, the gateway's `/metrics`, the metrics
ports, `LOG_FORMAT` and `LOG_LEVEL`, the correlation header and `python -m api_gateway`.
To keep `AGENTS.md` within its 30 KiB budget, service descriptions in the project map and
the MQTT contract rule were shortened: the topic list lives in the architecture overview,
which the rule still names; no rule changed what it permits.

**Why.** S9 changed how every service starts and logs; a new service written without these
lines would log plain text and stay invisible to Prometheus. Additions and clarifications,
which `universal/08-rules-evolution.md` allows without approval.

**Decided by.** Agent, as part of the owner-requested S9 work of 2026-09-18.

## 2026-09-18 — Commands follow S6: the OpenAPI contract and the console checks

**What changed.** `project/11-commands.md` lists the new scripts `console-e2e` and
`generate-openapi` and names `generate-openapi --check` among the gate sections that fail
when an artifact is not regenerated. `project/14-command-reference.md` says how to regenerate
the gateway contract and the console types, how the Playwright checks run and read the demo
passwords, that their demo check trips and resets the running unit, that the Vite dev server binds 127.0.0.1, and that a session of the Claude desktop
app reaches only IPv4 loopback and keeps Playwright's browsers outside AppData. The frontend
agent on both sides (`.claude/agents`, `.codex/agents`) points at the client modules, the
generated types, the token storage decision and `console-e2e`, and says that only Playwright
types a demo password into a page.

**Why.** S6 added a contract check to the gate and two scripts; without these lines the next
session would hand-write gateway types or not know how to verify the console. The IPv4 note
records friction met in this task. Factual additions and clarifications, which
`universal/08-rules-evolution.md` allows without approval; no rule was loosened. To keep
`AGENTS.md` within its 30 KiB budget, a few table cells and the gateway line of
`project/10-project-map.md` were shortened without changing what they say.

**Decided by.** Agent, as part of the owner-requested S6 work of 2026-09-18.

## 2026-09-16 — Project modules follow S5, S8 and S10

**What changed.** `project/10-project-map.md` describes the gateway as the user authority
(sessions, users, append-only audit, WebSocket channels, simulation control, KPIs), the
historian's retention and downsampling, and OPC UA methods running through the gateway.
`project/14-command-reference.md` replaces `/ws/realtime` with `/ws`, lists `/ready`, the
aggregate bucket, the Grafana dashboards, the REST simulation routes and the new MQTT
subscribers, and notes that Docker Desktop must be started by the owner when an agent runs
inside the Claude desktop app. The frontend agent on both sides (`.claude/agents`,
`.codex/agents`) names `/ws`.

**Why.** The S5, S8 and S10 implementation made those statements false, and starting Docker
Desktop from a session crashed its backend. These are factual corrections, which
`universal/08-rules-evolution.md` allows without approval; no rule was loosened.

**Decided by.** Agent, as factual corrections to the owner-requested work of 2026-09-16.

## 2026-09-15 — Project modules follow S2–S4

**What changed.** `project/10-project-map.md` describes the PLC's coordinated control and
alert-manager's alarm lifecycle and `AlarmService`. `project/12-domain-rules.md` no longer
lists `plc-controller` → `physics_engine` and `api-gateway` → `alert_manager.models` as
existing import exceptions — both are gone — and says a test may run another service
in-process through a development dependency; it names `plc/events` and `alarms/changes`
among the MQTT contracts. `project/14-command-reference.md` gains the `AlarmService` port,
the new topics, the physics run flags and how simulation control is reached.

**Why.** The S2–S4 implementation made those statements false. These are factual
corrections, which `universal/08-rules-evolution.md` allows without approval; no rule was
loosened.

**Decided by.** Agent, as factual corrections to the owner-requested S2–S4 work of
2026-09-15.

## 2026-09-15 — Publishing without pull requests

**What changed.** `project/11-commands.md` gains a Publishing section: a task branch is
merged into `main` locally with `git merge --ff-only` after a green gate and pushed; no pull
requests; a push declined by a GitHub rule is reported to the owner, not worked around with
a side branch.

**Why.** Publishing this project's first task hit a GitHub ruleset requiring pull requests
for `main`, and the commits went to a side branch as a workaround. The owner decided to work
with pushes and merges only.

**Decided by.** Owner, 2026-09-15 (recorded in `docs/__arch__/open-questions.md`).

## 2026-09-14 — The rule system arrives, adapted from codepack

**What changed.** CogniBoiler gains the assistant rule system the owner already runs in
`codepack`:

- `.ai/universal/01–09` copied from codepack. Two wording fixes keep them portable: the
  sync command in `07-multi-assistant.md` is now "the project's sync command" instead of
  `cargo xtask sync-agents`, and `08-rules-evolution.md` says "package" where it said
  "crate".
- `.ai/project/10–14` written for this project: the repository map with the deferred-AI
  guardrail, the orchestrator commands and gate policy, domain rules (SI units, epoch-ms
  UTC timestamps, contract discipline, PLC-only actuator path, safety interlocks, async
  and Windows event-loop rules), progress tracking, and a command reference marked
  `tier: extended`.
- `CLAUDE.md` imports every module; `AGENTS.md` is generated by
  `python dev_tools_scripts_runner.py sync-agents`, a Python port of codepack's
  `cargo xtask sync-agents` with the same 30 KiB budget and tier rules.
- `.claude/` and `.codex/` carry mirrored project agents and skills.

**Why.** The owner asked for the project's evolution to be visible to AI agents as well
as to people, by analogy with codepack: rules, checklists, and state documents that
survive a lost conversation.

**Adapted, not copied.** Codepack's gate lives in a Rust `xtask`; here the gate is the
`quality-gate` script itself, so the orchestrator is the single implementation rather
than a door to one. Codepack's legacy-reference module has no counterpart — there is no
previous implementation to reproduce — and its place is taken by the command reference.

**Decided by.** Owner, 2026-09-14 (recorded in `docs/__arch__/open-questions.md`).
