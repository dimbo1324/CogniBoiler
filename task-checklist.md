# Task: remove the `.gitkeep` placeholders and publish `main`

Owner instruction of 2026-09-19: delete every `.gitkeep` file, commit, and push to the
remote `main`. The push also publishes the 15 local commits of S6, S7, S9, S11 and S12,
and CI on `main` runs the `publish` job (images to GHCR).

Marks: `[ ]` open, `+` done, `-` not done or partially done (with a note).

## Preparation

[ ] Previous checklist (S6–S12) closed: every item marked
[ ] List the tracked `.gitkeep` files; find the directories that become empty and anything
    that relies on them

## Implementation

[ ] Remove every tracked `.gitkeep` on branch `chore/remove-gitkeep`

## Verification

[ ] Full gate green
[ ] Fast-forward merge into `main`, push `main` to `origin`
[ ] CI on `main` checked: gate, audit, stack, publish

## Completion

[ ] Checklist filled honestly
[ ] Report in Russian: what is done, what remains
