# Rules Evolution: Keeping the Instructions Current

<!-- tier: extended -->

> **Essence.** Never weaken a rule to make a task easier — propose changes instead; autonomous edits may only clarify or correct, never loosen; every change needs a changelog entry and a regenerated entry point.

Purpose: the rule set is a living system, not a monument. It must absorb what agents
learn while working, stay factually true, and never silently rot. This module governs
how the rules change themselves.

## The prime safeguard

**Never weaken a rule to make your current task easier.** Editing the rules so that a
failing check passes, a forbidden action becomes allowed, or an inconvenient constraint
disappears is the single most damaging thing an agent can do here — it converts one bad
task into a permanent defect in every future task.

If a rule blocks you and you believe it is wrong, you do not edit it and proceed. You
stop, report the conflict, and propose the change for the owner. Finish the task under
the existing rule or leave it unfinished with an honest note.

## When you MUST propose a rule change

Proposing is mandatory, not optional, when any of these happen:

- **A rule is factually false.** A command, path, package, or file name in a module no
  longer matches reality.
- **You hit the same friction twice.** The same ambiguity or missing guidance cost you
  time in two separate tasks — that is a documentation defect, not bad luck.
- **A review keeps finding the same class of defect.** Repeated findings mean the rules
  failed to prevent them.
- **The owner gives an instruction that constrains future work.** Chat-only decisions
  are invisible to the next session; they must land in a module or in the decisions log.
- **The workflow changed.** A new stage, tool, gate, or command changed how work is
  actually done.
- **A rule proved harmful.** Following it produced a worse outcome than ignoring it
  would have.

## What you may change autonomously

Without asking, you may make the rules **more accurate** or **more clear**:

- correcting a stale path, command, package name, or file reference;
- fixing a contradiction between two modules by making them agree with the code;
- clarifying wording that misled you, without changing what is permitted;
- adding a concrete example of an existing rule;
- adding a genuinely new constraint the owner stated in the current conversation.

## What requires explicit owner approval

Ask, and wait, before you:

- remove a rule or make it optional;
- loosen a constraint, threshold, gate, or invariant;
- change the definition of done, the branch policy, or the publish policy;
- change anything in the project's invariants registry;
- change the language policy or the documentation policy;
- move a module between tiers in a way that drops content from the compiled entry point.

## How to change a module

1. Edit exactly one concern at a time in `.ai/universal/` or `.ai/project/`.
   Universal modules stay portable — no project names, paths, or stack specifics.
2. Add an entry to `.ai/CHANGELOG.md`: date, what changed, why, and who decided.
3. Regenerate the compiled entry point with the project's sync command.
4. If the change affects per-assistant workspaces, apply the mirrored edit on every
   side in the same task.
5. Commit the module, the changelog entry, and the regenerated entry point **together**.
   A module change without its regenerated entry point is a broken commit.

Never hand-edit the generated entry point. If it drifts, regenerate — do not patch.

## Review cadence

- **End of every task:** if the task surfaced friction, a stale fact, or a new
  constraint, fix the module now. Deferring is how rules rot.
- **End of every stage or milestone:** re-read the project modules against the code.
  Anything that no longer matches reality gets corrected or deleted.
- **On stage transition:** commands, layout, and gates usually change; the commands and
  project map modules almost always need an update.

## Retiring a rule

A rule that no longer applies is deleted, not left to confuse people. Deletion requires
owner approval, a changelog entry explaining why it is obsolete, and a check that no
other module still references it.

Superseded rules are removed, not commented out. Git history is the archive.

## Conflict resolution

When two rules disagree: the project module wins over the universal module; an explicit
owner instruction in the current conversation wins over both. Say out loud which rule
you followed and why — then, if the conflict is structural rather than one-off, fix it
in the modules so the next agent does not face the same ambiguity.

## Rule debt

Rule problems you notice but cannot fix in the current task are recorded like any other
debt: named in the final report, and, if significant, added to the decisions log as an
open question. Silently ignoring a known-wrong rule is a violation.
