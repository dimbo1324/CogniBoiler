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
