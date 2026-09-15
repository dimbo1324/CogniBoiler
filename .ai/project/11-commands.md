# Project Commands and Quality Gates

All commands run from the repository root: PowerShell or Git Bash on Windows, any POSIX
shell elsewhere.

## The script orchestrator — start here

```powershell
python dev_tools_scripts_runner.py          # interactive menu
python dev_tools_scripts_runner.py list     # machine-readable catalog — use this, not the menu
python dev_tools_scripts_runner.py <name>   # run one directly; `help` prints the manuals
```

The orchestrator is stdlib-only and runs on any Python 3.14; a script that needs a project
package hands itself to `uv run`. With no arguments and no terminal it runs
`quality-gate`, which is what makes it safe for an agent or CI to call.

| Script | Purpose |
|---|---|
| `quality-gate` (`--quick`) | the one verification path; CI runs exactly this |
| `format-code` (`--check`) | ruff for Python, Prettier for the frontend |
| `sync-agents` (`--check`) | regenerate `AGENTS.md` from `.ai/` |
| `stack up` / `status` / `logs <svc>` / `down` | the Docker Compose stack |
| `dev-secrets` | create or complete `.env` with generated development secrets |
| `smoke` | end-to-end check of a running stack through the gateway; CI runs it |
| `generate-proto` (`--check`) | regenerate `shared/generated` after a `.proto` change |
| `doctor` | read-only toolchain check |
| `install-hooks` | the pre-commit hook, once per clone |
| `clean-caches` (`--apply`) | **deletes files**; a dry run unless `--apply` |
| `selftest` | the scripts' own tests |

**Standing duty — keep the scripts true.** A task that changes how the project is built,
checked, formatted, run or cleaned updates the matching script in that same task, and
runs `selftest` after touching `scripts/`. A new routine job is a new directory under
`scripts/` plus one entry in `scripts/runner/config/scripts.json` — never new Python in
`scripts/runner/`.

## Direct commands, one layer at a time

```powershell
uv sync --all-packages                        # the environment exactly as uv.lock says
uv run pytest apps/plc-controller/tests       # one service's tests
uv run ruff check . ; uv run mypy             # lint, strict types
pnpm --dir apps/web install                   # frontend dependencies, once
```

Service-by-service run commands, ports, topics and platform notes are in
`14-command-reference.md`.

## Gate policy

- The full gate is green before any merge to `main`; `--quick` is the minimum before a
  push. Documentation- and configuration-only changes still run it.
- The gate runs every section even after one fails and ends with a single summary.
- `sync-agents --check` and `generate-proto --check` are gate sections on purpose: a
  source edited without regenerating its artifact breaks the build.
- Frontend sections are skipped locally without `apps/web/node_modules` and fail when the
  `CI` variable is set.
- The pre-commit hook only formats and checks file hygiene. Lint, strict typing and tests
  are never commit-time checks and are never skipped at merge time.
- `xfail(strict=True)` marks a recorded known defect and is allowed only by an owner
  decision in `docs/__arch__/open-questions.md`; it is never a way to get green.
