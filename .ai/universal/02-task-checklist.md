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
