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
