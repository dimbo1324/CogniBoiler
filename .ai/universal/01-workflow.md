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
