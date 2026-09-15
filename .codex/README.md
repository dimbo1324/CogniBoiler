# Codex Project Configuration

Project-scoped Codex configuration for CogniBoiler.

Rules live in the shared modules under `.ai/`. Codex reads them through the generated
`AGENTS.md` in the repository root — the compiled single-file ruleset. Claude Code gets the
same rules through `CLAUDE.md` with native `@` imports.

`AGENTS.md` is **generated** and never hand-edited. After changing any `.ai/` module:

```powershell
python dev_tools_scripts_runner.py sync-agents
```

## Files

- `config.toml` — project-scoped model and agent defaults. No secrets belong here;
  personal overrides go to the ignored `config.local.toml`.
- `agents/` — project agents; name-for-name mirror of `.claude/agents/`.
- `skills/` — reusable workflows; mirror of `.claude/skills/`.

## Agents

- `cogniboiler-stage-planner` — scope a `docs/__arch__/ROADMAP.md` stage before code.
- `cogniboiler-physics-control` — physics-engine and plc-controller.
- `cogniboiler-backend` — api-gateway, historian, alert-manager, opcua-server, databases.
- `cogniboiler-frontend` — the operator console in `apps/web`.
- `cogniboiler-platform` — Dockerfile, Compose, provisioning, CI, deployment.
- `cogniboiler-security` — auth, RBAC, audit, secrets.
- `cogniboiler-quality-reviewer` — review a diff before finalizing.
- `cogniboiler-ci-triage` — debug a failing check.
- `cogniboiler-repo-maintainer` — formatting, state documents, rule sync, publishing.

## Skills

`stage-episode`, `contract-change`, `code-review`, `ci-fix`, `project-maintenance`,
`rules-evolution`.

## Mirroring

`.codex/agents|skills` and `.claude/agents|skills` are name-for-name mirrors. Changing one
side requires the equivalent change on the other in the same task.

## Evolving the rules

The protocol lives in `.ai/universal/08-rules-evolution.md`; every change is recorded in
`.ai/CHANGELOG.md`. Never weaken a rule to make the current task pass.
