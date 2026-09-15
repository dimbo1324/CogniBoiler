---
name: ci-fix
description: Use when a local check or CI job is failing — reproduce, find the root cause, apply the minimal correct fix, and re-verify.
---

# Fixing a Failing Check

## Procedure

1. **Reproduce** with a minimal command. No diagnosis without reproduction.

   ```powershell
   python dev_tools_scripts_runner.py quality-gate --quick
   ```

   Or one section: `uv sync --all-packages --locked`, `uv run ruff format --check .`,
   `uv run ruff check .`, `uv run mypy`, `uv run pytest apps/<service>/tests`,
   `python dev_tools_scripts_runner.py sync-agents --check`,
   `python dev_tools_scripts_runner.py generate-proto --check`,
   `pnpm --dir apps/web run lint|typecheck|test|build`.

2. **Localize the layer:** formatting, lint, types, a test, the lock file, a generated
   artifact, the frontend, or a platform difference.
3. **Find the cause, not the symptom.** A failing test usually means a code defect.
4. **Apply the minimal fix** without expanding scope.
5. **Re-verify** with the same command, then with the full gate.

## Forbidden

- Deleting, skipping, or loosening tests; adding `xfail` without an owner decision.
- A bare `# type: ignore` or `# noqa` instead of a fix, unless justified in place.
- Suppressing output so a check appears to pass.
- Hand-editing `AGENTS.md` or `shared/generated/` — regenerate them.

## Common local-versus-CI differences

- Line endings: the repository normalizes text to LF.
- `uv.lock` behind `pyproject.toml`: `uv sync --locked` fails by name — run `uv lock` and
  commit the lock with the dependency change.
- A `.venv` built for another Python: `uv sync --all-packages` recreates it (close VS Code
  first on Windows).
- Windows asyncio: MQTT code needs the selector event loop.
- Frontend sections skip locally without `apps/web/node_modules` but fail under `CI`.
- Timing-dependent tests pass on a fast machine and fail on a runner.

## Report

The reproduction command, the root cause, the fix, and the re-run result. If the cause
was not found, say so and list what was checked and ruled out.
