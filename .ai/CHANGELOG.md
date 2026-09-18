# Rule Changes Changelog

History of changes to the AI assistant rule system (`.ai/`, `CLAUDE.md`, `AGENTS.md`,
`.claude/`, `.codex/`). Every rule change gets an entry here — see
`universal/08-rules-evolution.md` for the protocol.

Format: date, what changed, why, who decided. Newest first.

## 2026-09-18 — Commands follow S6: the OpenAPI contract and the console checks

**What changed.** `project/11-commands.md` lists the new scripts `console-e2e` and
`generate-openapi` and names `generate-openapi --check` among the gate sections that fail
when an artifact is not regenerated. `project/14-command-reference.md` says how to regenerate
the gateway contract and the console types, how the Playwright checks run and read the demo
passwords, that the Vite dev server binds 127.0.0.1, and that a session of the Claude desktop
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
