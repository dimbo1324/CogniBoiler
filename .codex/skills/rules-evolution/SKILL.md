---
name: rules-evolution
description: Use when the agent rules need to change — a rule is stale, wrong, missing, or the owner gave a new standing instruction. Covers what may change autonomously, what needs approval, and the sync procedure.
---

# Evolving the Agent Rules

The rule set is a living system. Keeping it accurate is part of the work, not overhead.
The full protocol is `.ai/universal/08-rules-evolution.md`; this skill is the operational
path.

## The prime safeguard

**Never weaken a rule to make your current task easier.** If a rule blocks you and you
believe it is wrong, stop, report the conflict, and propose the change. Do not edit and
proceed.

## 1. Decide whether a change is required

A change is **mandatory** when: a rule is factually false (a stale command, path, service
or file name); the same friction cost you time twice; reviews keep finding the same class
of defect; the owner gave an instruction that constrains future work; the workflow
changed; or following a rule produced a worse outcome.

## 2. Decide whether you may do it alone

**Autonomous** — making rules more accurate or clearer: fixing a stale path or command,
resolving a contradiction against the code, clarifying misleading wording, adding an
example, adding a constraint the owner just stated.

**Needs owner approval** — anything that loosens the system: removing a rule, relaxing a
constraint, gate or invariant, changing the definition of done, the branch or publish
policy, the language policy, or the documentation policy.

When in doubt, ask.

## 3. Apply the change

```powershell
# 1. Edit exactly one concern in .ai/universal/ or .ai/project/
# 2. Add an entry to .ai/CHANGELOG.md: date, what, why, who decided
python dev_tools_scripts_runner.py sync-agents          # 3. regenerate AGENTS.md
python dev_tools_scripts_runner.py sync-agents --check  # 4. verify sync and budget
```

Universal modules stay portable — no project names, paths, or stack specifics. If the
change touches per-assistant workspaces, mirror it in `.claude/` and `.codex/` in the same
task.

Commit the module, the changelog entry, and the regenerated `AGENTS.md` **together**.

## 4. If the size budget blocks you

The compiled `AGENTS.md` must stay under 30 KiB. Tighten the module, or mark a situational
one `<!-- tier: extended -->` and give it an `> **Essence.**` line right after its title.
Extended rules remain binding; only their full text moves out of the compiled file.

## 5. Retiring a rule

Obsolete rules are deleted, not commented out — git history is the archive. Deletion needs
owner approval, a changelog entry explaining why, and a check that no other module still
references it.

## Review cadence

At the end of every task, fix anything the task proved stale. At the end of every stage,
re-read the project modules against the code — commands, ports, topics and gates drift
most. Rule problems you cannot fix now go into the final report and, if significant, into
`docs/__arch__/open-questions.md`.
