---
name: contract-change
description: Use whenever a task changes something another process depends on — a protobuf message or service, an MQTT topic or payload, a REST or WebSocket shape, or a database table.
---

# Changing a Contract

A contract is anything a different process reads: another service, the frontend, Grafana,
an OPC UA client, or an existing database. Changing one silently breaks a consumer that no
unit test of the producer can see.

## 1. Name every consumer first

| Contract | Where consumers are listed |
|---|---|
| `shared/proto/cogniboiler.proto` | `docs/architecture/service-boundaries.md`, gRPC clients under `apps/*/src/*/client*.py` |
| MQTT topics and payloads | the topic table in `docs/architecture/overview.md` |
| REST and WebSocket | `apps/api-gateway/src/api_gateway/routers/`, the client layer in `apps/web` |
| Database tables | SQLAlchemy models, Alembic revisions, Grafana and API queries |

Write the list into `task-checklist.md` before changing anything.

## 2. Prefer an additive change

- Proto: add a field with a new number; never renumber, retype or reuse one. A removed field
  becomes `reserved`.
- MQTT and WebSocket: add fields; keep existing ones until every consumer has moved.
- REST: add optional fields or a new route; a breaking change needs an owner decision.
- Database: a new Alembic revision; never edit an applied one. Destructive migrations
  (drop, narrowing type) need a risk note in the commit body.

## 3. Apply in one task

```powershell
python dev_tools_scripts_runner.py generate-proto          # after a .proto edit
uv run --package api-gateway alembic -c apps/api-gateway/alembic.ini revision -m "..."
```

Update every consumer from step 1 in the same task, with tests on both sides of the
boundary.

## 4. Verify across the boundary

```powershell
python dev_tools_scripts_runner.py quality-gate
python dev_tools_scripts_runner.py stack up
```

Check the real data path end to end (for example: physics publishes → historian stores →
API returns → console renders).

## 5. Record it

- `docs/architecture/overview.md` — the topic table, endpoint list or schema summary.
- The final report names the contract change explicitly, and whether it is breaking.
