# Security, Secrets, Dependencies, Portability

Purpose: nothing sensitive in the repo; every change safe within its area; the project
runs on any machine.

## Secrets — absolute ban

- NEVER put in code, git, tests, docs, or examples: passwords, API keys, tokens,
  private keys, real credentials, cookies, production `.env`, or personal user data.
- Secrets live only in `.env` (untracked), environment variables, secret managers, or
  CI/CD secrets. The repo may contain only a safe `.env.example`.
- A secret that ever reached git is compromised: rotate it; deleting the line in a new
  commit is not enough.

## Security in every task

Check within the area you touch: authorization and access rights, input validation,
injection, XSS and CSRF where applicable, unsafe redirects, file uploads, personal data
handling, public endpoints, token storage and transport, and access to admin functions.
Security is part of every task, not a future task.

## Dependencies

- No new dependency without justification: what for, can it be done without, is it
  maintained, known vulnerabilities, stack compatibility, does it duplicate an existing
  dependency, is it too heavy for the need.
- Never add a heavy library for one small function.
- After changing dependencies: update the lock file and verify the build.
- A new production dependency must be named in the final report.

## Portability

- No machine-specific values in code: local absolute paths, usernames, or anything
  environment-dependent. Such values go to configuration.
- The project must remain runnable by someone else using the project's documented tools.
