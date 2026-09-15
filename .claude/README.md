# Claude Code Project Configuration

Project-scoped Claude Code configuration for CogniBoiler.

Durable rules live in the shared modules under `.ai/` (universal + project). `CLAUDE.md`
imports them natively via `@` syntax. Codex consumes the same modules through the
generated `AGENTS.md` (regenerate with `python dev_tools_scripts_runner.py sync-agents`;
never edit it by hand). Subagents should read `AGENTS.md` — it is the compiled
single-file ruleset.

## Files

- `settings.json` — permission allowlist for routine read and verification commands, and
  an explicit denylist for destructive git and Docker volume operations.
- `agents/` — project-scoped subagents for focused delegation; name-for-name mirror of
  `.codex/agents/`.
- `skills/` — reusable project workflows; mirror of `.codex/skills/`.
- `settings.local.json` — personal overrides, ignored by git.

## Recommended delegation

Do task-owning work on the main thread; spawn a subagent only for independent work that
does not need the main thread's full context:

- `cogniboiler-stage-planner` — scope a `docs/__arch__/ROADMAP.md` stage before code:
  boundaries, contracts touched, risks, acceptance criteria.
- `cogniboiler-physics-control` — `physics-engine` and `plc-controller`: plant model,
  runtime, PID, interlocks, E-Stop.
- `cogniboiler-backend` — `api-gateway`, `historian`, `alert-manager`, `opcua-server`,
  PostgreSQL, InfluxDB, migrations.
- `cogniboiler-frontend` — the operator console in `apps/web`.
- `cogniboiler-platform` — Dockerfile, Compose, Grafana, CI, Kubernetes/Helm.
- `cogniboiler-security` — auth, RBAC, audit, secrets, transport security.
- `cogniboiler-quality-reviewer` — review a diff before finalizing it.
- `cogniboiler-ci-triage` — debug a failing local or CI check.
- `cogniboiler-repo-maintainer` — formatting, docs upkeep, rule sync, explicit publishing.

## Quality shortcuts

```powershell
python dev_tools_scripts_runner.py quality-gate --quick
python dev_tools_scripts_runner.py format-code
python dev_tools_scripts_runner.py doctor
python dev_tools_scripts_runner.py sync-agents --check
```

Push to `main` only when the owner explicitly asked for it in the current task.

## Evolving the rules

The rule set is expected to change as the project learns. The protocol — mandatory
triggers, what may change autonomously, what needs owner approval — is in
`.ai/universal/08-rules-evolution.md`. Every rule change is recorded in
`.ai/CHANGELOG.md`. Never weaken a rule to make the current task pass.
