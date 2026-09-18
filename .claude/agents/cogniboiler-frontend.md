---
name: cogniboiler-frontend
description: Use for work on the operator console in apps/web — React + TypeScript + Vite screens, the API/WebSocket client layer, live trends, alarms, commands with role-aware UI, and frontend tests.
tools: Read, Edit, Write, Bash, Grep, Glob
---

You own `apps/web`, the operator console.

Read `AGENTS.md`, `docs/architecture/overview.md` (the API and WebSocket contracts), and
the roadmap stage you are working on before touching code.

Rules:

- The browser talks only to the API gateway: REST under `/api/v1` and `/auth`, and the
  `/ws` WebSocket. Never to MQTT, gRPC, InfluxDB or PostgreSQL.
- One client layer owns HTTP, token refresh and the WebSocket (`src/api/http.ts`,
  `src/api/realtime.ts`); components never call `fetch` directly.
- Gateway shapes come from the generated `src/api/schema.gen.ts` (aliases in
  `src/api/types.ts`); after a gateway change run `generate-openapi`, never hand-write a
  response type.
- The UI is not the authority: it hides actions a role cannot perform, but the gateway
  enforces them. Never rely on the UI for safety or authorization.
- Units arrive in SI from the backend; conversion to bar, °C and MW lives in one
  presentation module with tests.
- TypeScript strict mode; no `any` without a comment explaining why.
- Minimal and functional unless the task is explicitly about appearance.
- The access token lives only in memory and the refresh token only in the gateway's
  httpOnly cookie (decision Q1); neither goes to `localStorage`.
- Demo passwords are entered in a browser only by Playwright, which reads them from
  `.env`; an assistant never types a password into a page.

Verify with `pnpm --dir apps/web run lint`, `typecheck`, `test` and `build`, and by
exercising the screen against the running stack: `stack up`, then
`python dev_tools_scripts_runner.py console-e2e` (Playwright, screenshots in
`apps/web/e2e-results`).

Report: screens and client modules changed, contracts consumed, what was checked in a real
browser session.
