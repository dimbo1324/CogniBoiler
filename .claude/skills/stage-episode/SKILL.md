---
name: stage-episode
description: Use to plan and execute one ROADMAP stage end to end — orientation, scoping, implementation, verification against the running stack, and the status update.
---

# Executing a Roadmap Stage

One `docs/__arch__/ROADMAP.md` stage is one task. Do not merge two stages into one task, and
do not skip the stage order without an owner decision.

## 1. Orientation (mandatory, no skipping)

```powershell
git status --short --branch
git log -15 --date=iso-strict --pretty=format:"%h %cd %s"
```

Then read, in order: `docs/__arch__/ROADMAP.md` §1 and the `**Status.**` lines (the first
stage without one is yours), `docs/architecture/overview.md`, `task-checklist.md`,
`docs/__arch__/open-questions.md`. Product intent: `docs/__arch__/VISION.txt`.

If `task-checklist.md` still has open `[ ]` items from a previous session, resolve them
first.

## 2. Planning

For a large or unfamiliar stage, delegate to the `cogniboiler-stage-planner` subagent.

Define: boundaries, the owning service for each piece of behavior, every contract touched
(proto, MQTT, REST, WebSocket, tables), risks to invariants, and acceptance criteria.

Fill `task-checklist.md` with `[ ]` items grouped into preparation / implementation /
verification / completion, and **commit it before writing code**.

## 3. Branch

```powershell
git checkout main
git pull --ff-only origin main
git checkout -b feat/<stage>-short-description
```

## 4. Implementation

Delegate to the specialists where it helps: `cogniboiler-physics-control`,
`cogniboiler-backend`, `cogniboiler-frontend`, `cogniboiler-platform`,
`cogniboiler-security`. Contract changes follow the `contract-change` skill.

## 5. Verification

```powershell
python dev_tools_scripts_runner.py quality-gate
python dev_tools_scripts_runner.py stack up
```

Exercise the stage's acceptance criteria against the running stack, not only in unit
tests. If the stage changed how the project is built, checked, run or cleaned, update the
matching script in this same task and run `selftest`.

Run the `code-review` skill or the `cogniboiler-quality-reviewer` subagent before
finalizing.

## 6. Completion

- Mark checklist items `+`/`-` honestly.
- Add the `**Status.**` line under the stage in `docs/__arch__/ROADMAP.md` (Russian) and
  update the §1 table.
- Update `docs/architecture/overview.md`, and `README.md` if a user can see the change.
- If the stage exposed stale or missing rules, apply the `rules-evolution` skill.
- Fast-forward merge into `main` only with a green gate.
- Write the final report: what was done, what was verified, what was not done.
