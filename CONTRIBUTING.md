# Contributing to CogniBoiler

## Branches

Start every change from an up-to-date `main` on a branch named `type/short-description`,
where the type is one of `feat`, `fix`, `refactor`, `test`, `chore`, `docs`, `ci`, `perf`
or `security`. Do not commit to `main` directly.

## Commits

One logical change per commit, with a message `type: what and why`, for example
`fix: hold the valves when a reading is not finite`.

## Checks

Run the quality gate before you ask for a merge:

```bash
python dev_tools_scripts_runner.py quality-gate
```

It runs formatting, lint, strict typing, every test suite with a coverage floor, the
contract checks and the console checks. `python dev_tools_scripts_runner.py list` shows the
other routine jobs.

## Merging

The project takes no pull requests. A branch with a green gate is merged into `main`
locally with `git merge --ff-only` and then pushed; force pushes are blocked.
