---
name: project-maintenance
description: Use for routine CogniBoiler upkeep — formatting, quality gates, rule sync, mirror consistency, script catalog accuracy, state-document updates, and explicitly requested publishing.
---

# Repository Maintenance

Use the project's automation instead of hand-rolled command sequences.

## Fast paths

```powershell
python dev_tools_scripts_runner.py list          # the catalog, machine-readable
python dev_tools_scripts_runner.py doctor        # what this host can and cannot do
python dev_tools_scripts_runner.py format-code   # ruff + Prettier
python dev_tools_scripts_runner.py quality-gate  # --quick before a push
python dev_tools_scripts_runner.py sync-agents   # regenerate AGENTS.md from .ai/
python dev_tools_scripts_runner.py selftest      # after touching anything under scripts/
```

## Rules

- `AGENTS.md` is **generated** and never hand-edited. Edit the module in `.ai/`, then run
  `sync-agents`. The budget is 30 KiB: tighten a module, or mark a situational one
  `<!-- tier: extended -->` with an `> **Essence.**` line.
- Rule changes follow the `rules-evolution` skill, including the `.ai/CHANGELOG.md` entry.
- `.claude/agents|skills` and `.codex/agents|skills` are name-for-name mirrors; change both
  in the same task.
- `.claude/settings.json`: extending the allowlist is fine; removing a deny entry needs
  explicit owner approval.
- **Scripts are infrastructure.** A task that changes how the project is built, checked,
  formatted, run or cleaned updates the matching script in that same task. New routine work
  gets a new directory under `scripts/` plus one catalog entry — never new Python in
  `scripts/runner/`. Scripts stay cross-platform: resolve tools through
  `scripts/_toolkit/processes.py`.

## State documents

- Stage completed → `**Status.**` line under that stage in `docs/__arch__/ROADMAP.md`
  (Russian) plus the §1 table.
- System shape changed → `docs/architecture/overview.md`; user-visible → `README.md`.
- Owner decision → `docs/__arch__/open-questions.md`.
- New invariant → `docs/architecture/invariants.md`.
- Task closed → `task-checklist.md` with honest `+`/`-` marks.

## Publishing

Push to `main` only when the owner explicitly asked in the current task, and only with a
green gate.
