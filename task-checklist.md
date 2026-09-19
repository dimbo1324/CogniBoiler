# Task: remove the `.gitkeep` placeholders and publish `main`

Owner instruction of 2026-09-19: delete every `.gitkeep` file, commit, and push to the
remote `main`. The push also publishes the 15 local commits of S6, S7, S9, S11 and S12,
and CI on `main` runs the `publish` job (images to GHCR).

Marks: `[ ]` open, `+` done, `-` not done or partially done (with a note).

## Preparation

+ Previous checklist (S6–S12) closed: every item marked
+ 28 tracked `.gitkeep` files; 11 directories become empty and leave the tree (`certs`,
  `tests`, `shared/crypto`, `infrastructure/ci-cd`, `helm`, `influxdb`, `k8s`,
  `ml/datasets`, `ml/inference`, `ml/models`, `ml/training`). Nothing relies on them:
  `pytest` collects from `apps` and `shared/observability/tests`, `dev-secrets` writes keys
  into `.env`, and the dataset generator creates `ml/datasets/raw` itself

## Implementation

+ Removed every tracked `.gitkeep` on branch `chore/remove-gitkeep`

## Verification

+ Full gate green (14 sections); `docker compose --profile full config` valid
[ ] Fast-forward merge into `main`, push `main` to `origin`
[ ] CI on `main` checked: gate, audit, stack, publish

## Completion

[ ] Checklist filled honestly
[ ] Report in Russian: what is done, what remains
