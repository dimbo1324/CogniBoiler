# CogniBoiler — Internal Execution Plan / Technical Specification / Progress Tracker

Version: 1.0  
Audience: Internal developers only  
Scope: Branch-level implementation control, delivery tracking, and self-assessment  
Project: CogniBoiler

---

## 1. Purpose of this document

This document is the internal execution tracker for CogniBoiler.  
It is intended to answer four practical questions at any point in time:

1. What exact implementation step are we on right now?
2. What must exist for the step to be considered complete?
3. How much of the step is already done: 0%, 25%, 50%, 75%, or 100%?
4. What is the rough remaining effort in hours / days / weeks?

This file is intentionally stricter than a normal roadmap:
- it is a delivery control document;
- each step has explicit completion criteria;
- partial completion can be scored consistently;
- phase transitions are gated;
- implementation is measured against code, tests, configs, and runnable behavior.

---

## 2. How to use this file

For every step, maintain these fields:

- **Status**: `NOT STARTED` / `IN PROGRESS` / `BLOCKED` / `DONE`
- **Completion**: `0 / 25 / 50 / 75 / 100`
- **Owner**: developer responsible for the step
- **ETA remaining**: rough estimate to 100%
- **Last review date**
- **Notes / blockers**

Recommended discipline:
- update the step after every meaningful work session;
- do not mark 100% unless all exit criteria are met;
- if something is implemented but not tested, it is not 100%;
- if behavior exists but is undocumented or unstable, cap at 75%;
- if the code exists but is not integrated, cap at 50%.

---

## 3. Completion scale

### 0% — Not started
- No meaningful implementation.
- No validated deliverables exist.

### 25% — Skeleton / draft
- Basic structure exists.
- Initial files/modules/classes/configs are created.
- Major behavior is missing or unverified.

### 50% — Core implementation exists
- Main code path is written.
- Partial local testing may exist.
- Integration, robustness, and validation are incomplete.

### 75% — Functionally usable
- The feature works in normal conditions.
- Relevant tests exist and pass in the main path.
- Edge cases, polish, docs, or full validation are still incomplete.

### 100% — Done
- All listed deliverables exist.
- All exit criteria are satisfied.
- Relevant tests/checks pass.
- The step is stable enough to build on top of it.

---

## 4. Global effort model

Reference effort baseline:
- Full project target: ~25 weeks at ~20–25 hours/week
- Total effort range: ~500–625 hours

Phase-level rough effort:
- Phase 1: 3 weeks
- Phase 2: 4 weeks
- Phase 3: 3 weeks
- Phase 4: 4 weeks
- Phase 5: 3 weeks
- Phase 6: 4 weeks
- Phase 7: 4 weeks

Important:
- These are planning estimates, not guarantees.
- Real duration depends on rework, debugging, integration friction, and research depth.

---

## 5. Phase gates

A phase may be considered complete only if:
- all mandatory steps in the phase are at **100%**;
- no critical blocker remains open;
- downstream phases are not built on unstable assumptions;
- relevant checks/tests for the phase pass reliably.

A phase may be considered **functionally passable** if:
- all steps are at least **75%**;
- no unsafe or contradictory implementation is carried forward;
- remaining work is limited to polish, documentation, or non-critical hardening.

---

# PHASE 1 — Foundation and Developer Environment
**Target duration:** 3 weeks  
**Difficulty:** Low to Medium  
**Goal:** establish repository structure, dependency management, code quality baseline, and repeatable development workflow.

---

## Step 1.1 — Repository initialization and project structure

**Status:** NOT STARTED  
**Completion:** 0  
**Owner:**  
**ETA remaining:** 8–12 hours  
**Depends on:** none  
**Last review date:**  
**Notes / blockers:**  

### Objective
Create a clean and professional monorepo foundation with the required root files, top-level structure, branch rules, and base documentation.

### Deliverables
- GitHub repository created and usable
- Correct monorepo directory structure
- `.gitignore`
- License
- `README.md`
- `CONTRIBUTING.md`
- `CHANGELOG.md`
- `CODE_OF_CONDUCT.md`
- branch strategy configured

### Exit criteria for 100%
- Repository can be cloned without issues
- Root structure matches intended architecture
- Core project files exist and are valid
- Branch protection for `main` is enabled
- First meaningful commit is made with clean history

### Partial scoring
- **25%**: repo exists, folders exist, minimal root files added
- **50%**: structure is correct, base docs added, initial commits made
- **75%**: branch policy and repo hygiene are configured, docs render properly
- **100%**: all exit criteria satisfied

### Self-check
- Can a new developer clone the repo and understand the basic structure?
- Are there any missing mandatory root files?
- Is `main` protected from direct pushes?

---

## Step 1.2 — `uv` workspace and dependency management

**Status:** NOT STARTED  
**Completion:** 0  
**Owner:**  
**ETA remaining:** 12–16 hours  
**Depends on:** 1.1  
**Last review date:**  
**Notes / blockers:**  

### Objective
Create a deterministic Python workspace with `uv`, service-level packaging, lockfile, and root automation commands.

### Deliverables
- root workspace `pyproject.toml`
- app-level `pyproject.toml` files
- `uv.lock`
- `Makefile`
- dependency groups for dev/runtime

### Exit criteria for 100%
- `uv sync` works for the workspace
- package imports work from expected environments
- lockfile is committed and reproducible
- `make install` and `make lint` work from a clean machine

### Partial scoring
- **25%**: root workspace exists, some packages registered
- **50%**: workspace resolves, lockfile exists, basic commands added
- **75%**: commands are stable and reproducible, package isolation is sane
- **100%**: all exit criteria satisfied

### Self-check
- Can the workspace be recreated from scratch?
- Are service dependencies separated correctly?
- Are dev tools installed through the intended workflow?

---

## Step 1.3 — Linting, formatting, typing, and pre-commit

**Status:** NOT STARTED  
**Completion:** 0  
**Owner:**  
**ETA remaining:** 10–14 hours  
**Depends on:** 1.2  
**Last review date:**  
**Notes / blockers:**  

### Objective
Establish a strict baseline for code quality and developer hygiene.

### Deliverables
- Ruff config
- mypy config
- `.pre-commit-config.yaml`
- `.editorconfig`
- one typed reference module

### Exit criteria for 100%
- `ruff check .` passes
- `mypy .` passes on intended scope
- `pre-commit run --all-files` passes
- `git commit` runs hooks correctly
- missing type annotations are caught where expected

### Partial scoring
- **25%**: tool configs created
- **50%**: tools run but still produce unresolved issues
- **75%**: hooks and local workflow are stable
- **100%**: all checks pass cleanly

### Self-check
- Does every developer get the same quality gates?
- Are typing rules strict enough for long-term maintainability?
- Are hooks blocking bad commits as intended?

---

## Phase 1 gate
Phase 1 is complete when:
- repo structure is stable;
- workspace is reproducible;
- quality tooling is enforced by default.

---

# PHASE 2 — Physics Engine
**Target duration:** 4 weeks  
**Difficulty:** Medium  
**Goal:** build a physically meaningful simulation core for the boiler and turbine.

---

## Step 2.1 — Boiler thermodynamic ODE model

**Status:** NOT STARTED  
**Completion:** 0  
**Owner:**  
**ETA remaining:** 24–36 hours  
**Depends on:** Phase 1  
**Last review date:**  
**Notes / blockers:**  

### Objective
Implement the boiler model as an ODE-based physical system with meaningful state variables and controlled numerical behavior.

### Deliverables
- `physics_engine/boiler.py`
- `physics_engine/models.py`
- `physics_engine/constants.py`
- unit tests for key equations and behaviors

### Exit criteria for 100%
- Pressure rises monotonically under high fuel input
- Water level falls when feedwater is absent
- Energy balance error is within acceptable tolerance
- Boiler cools plausibly without fuel
- tests pass and the model remains numerically stable

### Partial scoring
- **25%**: class structure and state model exist
- **50%**: equations implemented, behavior partially works
- **75%**: model behaves plausibly in normal scenarios, tests mostly pass
- **100%**: all exit criteria satisfied

### Self-check
- Are all units consistent?
- Are the state transitions physically plausible?
- Are instability cases understood and contained?

---

## Step 2.2 — Steam turbine model

**Status:** NOT STARTED  
**Completion:** 0  
**Owner:**  
**ETA remaining:** 16–24 hours  
**Depends on:** 2.1  
**Last review date:**  
**Notes / blockers:**  

### Objective
Implement the turbine as a linked downstream model driven by boiler steam output.

### Deliverables
- `physics_engine/turbine.py`
- `physics_engine/system.py`
- turbine tests and linked-system tests

### Exit criteria for 100%
- Power output is in a realistic range for target input conditions
- turbine efficiency stays within physically plausible bounds
- power changes smoothly with steam flow changes
- no pathological jumps or negative nonsense values appear
- linked boiler+turbine execution is stable

### Partial scoring
- **25%**: model shell and interfaces exist
- **50%**: turbine math exists and produces outputs
- **75%**: linked simulation behaves plausibly in normal cases
- **100%**: all exit criteria satisfied

### Self-check
- Is the coupling to the boiler explicit and clean?
- Are limits enforced?
- Does the system remain numerically stable under step changes?

---

## Step 2.3 — Async simulator, valves, and operating scenarios

**Status:** NOT STARTED  
**Completion:** 0  
**Owner:**  
**ETA remaining:** 24–32 hours  
**Depends on:** 2.2  
**Last review date:**  
**Notes / blockers:**  

### Objective
Create a usable simulation runtime with valve models, sensor noise, scenario execution, and accelerated data generation.

### Deliverables
- valve models
- async simulator
- scenario library
- tests for simulator cadence and scenario stability

### Exit criteria for 100%
- stable target sampling frequency is achieved
- valve behavior affects plant state plausibly
- cold start and other main scenarios complete without numerical collapse
- sensor noise matches intended magnitude
- accelerated mode is usable for dataset generation

### Partial scoring
- **25%**: simulator shell and scenario files exist
- **50%**: core loop works, scenarios partially run
- **75%**: main scenarios behave plausibly
- **100%**: all exit criteria satisfied

### Self-check
- Is async orchestration clean?
- Are valve dynamics modeled explicitly enough?
- Can the simulator be used both live and for fast dataset generation?

---

## Step 2.4 — Visualization and dataset export

**Status:** NOT STARTED  
**Completion:** 0  
**Owner:**  
**ETA remaining:** 14–22 hours  
**Depends on:** 2.3  
**Last review date:**  
**Notes / blockers:**  

### Objective
Provide visual confirmation of physical behavior and enable export of training data.

### Deliverables
- visualization script
- dataset generation script
- CSV and Parquet export path
- reference plots/screenshots

### Exit criteria for 100%
- plots reflect physically plausible trajectories
- CSV and Parquet exports are valid
- dataset generation completes successfully
- generated data can be consumed downstream

### Partial scoring
- **25%**: scripts exist
- **50%**: exports run but are unstable or incomplete
- **75%**: visualization and exports work for common cases
- **100%**: all exit criteria satisfied

### Self-check
- Are charts good enough to catch physics mistakes?
- Is the export schema stable?
- Can ML preprocessing consume the generated files?

---

## Phase 2 gate
Phase 2 is complete when:
- boiler and turbine models behave plausibly;
- simulator scenarios are stable;
- exports are usable for downstream ML and integration work.

---

# PHASE 3 — Virtual PLC and Control Logic
**Target duration:** 3 weeks  
**Difficulty:** Medium  
**Goal:** implement control loops and safety logic around the physical process.

---

## Step 3.1 — Base PID controller

**Status:** NOT STARTED  
**Completion:** 0  
**Owner:**  
**ETA remaining:** 12–18 hours  
**Depends on:** Phase 2  
**Last review date:**  
**Notes / blockers:**  

### Objective
Implement a reusable PID controller with anti-windup, derivative filtering, limits, and reset support.

### Deliverables
- `plc_controller/pid.py`
- response tests
- anti-windup tests
- derivative behavior tests

### Exit criteria for 100%
- step response is acceptable for baseline tuning
- anti-windup works under saturation
- integral term removes steady-state error when expected
- derivative path is not destroyed by noise
- tests pass quickly and reliably

### Partial scoring
- **25%**: class shell exists
- **50%**: controller runs but lacks robustness
- **75%**: core behavior is correct in standard tests
- **100%**: all exit criteria satisfied

### Self-check
- Is controller state resettable and reusable?
- Is derivative filtering explicit and testable?
- Are output limits enforced correctly?

---

## Step 3.2 — Cascade control system

**Status:** NOT STARTED  
**Completion:** 0  
**Owner:**  
**ETA remaining:** 24–32 hours  
**Depends on:** 3.1, Phase 2  
**Last review date:**  
**Notes / blockers:**  

### Objective
Implement the multi-loop cascade controller for plant operation.

### Deliverables
- `plc_controller/cascade.py`
- PID configuration file
- integration tests with the physics engine

### Exit criteria for 100%
- target power is reached in a realistic time
- pressure overshoot stays within target limits
- drum water level is held within acceptable bounds
- steam temperature is regulated adequately
- loops work together without unstable coupling

### Partial scoring
- **25%**: controller layout exists
- **50%**: loops exist individually, integration incomplete
- **75%**: full loop system works in nominal scenarios
- **100%**: all exit criteria satisfied

### Self-check
- Are loop responsibilities clearly separated?
- Is tuning externalized and configurable?
- Is plant behavior stable after setpoint changes?

---

## Step 3.3 — Safety interlocks and emergency logic

**Status:** NOT STARTED  
**Completion:** 0  
**Owner:**  
**ETA remaining:** 18–26 hours  
**Depends on:** 3.2  
**Last review date:**  
**Notes / blockers:**  

### Objective
Implement safety boundaries, emergency stops, permissives, and structured safety event logging.

### Deliverables
- safety classes
- safety limits config
- emergency and rate-of-change protection
- tests for critical fault scenarios

### Exit criteria for 100%
- unsafe pressure/temperature conditions cause immediate protective action
- permissive logic blocks invalid control actions
- rate-of-change checks work
- safety events are logged with enough detail
- emergency-stop scenarios are covered by tests

### Partial scoring
- **25%**: safety module shell exists
- **50%**: some checks implemented, response incomplete
- **75%**: critical scenarios handled properly
- **100%**: all exit criteria satisfied

### Self-check
- Is the system safer with this code than without it?
- Are hard shutdown conditions truly enforced?
- Can the plant be restarted only under explicit safe conditions?

---

## Phase 3 gate
Phase 3 is complete when:
- basic control is operational;
- cascade loops behave coherently;
- safety logic prevents obviously unsafe operation.

---

# PHASE 4 — Communication Infrastructure
**Target duration:** 4 weeks  
**Difficulty:** Medium to High  
**Goal:** connect services through protobuf, gRPC, MQTT, OPC UA, and InfluxDB.

---

## Step 4.1 — Protocol Buffers contracts

**Status:** NOT STARTED  
**Completion:** 0  
**Owner:**  
**ETA remaining:** 12–18 hours  
**Depends on:** Phase 1  
**Last review date:**  
**Notes / blockers:**  

### Objective
Define stable data contracts for plant state, commands, and AI results.

### Deliverables
- `.proto` files for plant state, commands, AI events
- generated Python stubs
- generation command integrated into dev workflow

### Exit criteria for 100%
- contracts compile cleanly
- generated modules import correctly
- round-trip serialization works
- protobuf payloads are materially more efficient than JSON
- linting/validation for contracts passes

### Partial scoring
- **25%**: `.proto` skeletons exist
- **50%**: contracts compile but still churn
- **75%**: main messages are stable and usable
- **100%**: all exit criteria satisfied

### Self-check
- Are contracts clear and minimal?
- Are message names stable enough to build on?
- Are generated artifacts reproducible?

---

## Step 4.2 — gRPC service layer

**Status:** NOT STARTED  
**Completion:** 0  
**Owner:**  
**ETA remaining:** 18–28 hours  
**Depends on:** 4.1  
**Last review date:**  
**Notes / blockers:**  

### Objective
Implement inter-service RPC and streaming between core services.

### Deliverables
- gRPC server implementation
- gRPC client implementation
- streaming telemetry path
- health checks
- retry/reconnect handling
- integration tests

### Exit criteria for 100%
- registered methods are callable
- streaming is stable for extended runs
- health checks report serving state
- repeated local RPC calls succeed reliably
- reconnect logic works under temporary disconnection

### Partial scoring
- **25%**: service definitions and server shell exist
- **50%**: unary RPC works
- **75%**: streaming and reconnect behavior mostly work
- **100%**: all exit criteria satisfied

### Self-check
- Does the service contract match the `.proto` contract exactly?
- Are timeouts/retries reasonable?
- Can downstream services depend on the RPC layer safely?

---

## Step 4.3 — MQTT broker and telemetry pipeline

**Status:** NOT STARTED  
**Completion:** 0  
**Owner:**  
**ETA remaining:** 20–28 hours  
**Depends on:** 4.1, Phase 2  
**Last review date:**  
**Notes / blockers:**  

### Objective
Establish the live telemetry bus for sensor and command topics.

### Deliverables
- broker config
- ACL rules
- physics publisher
- historian subscriber
- LWT handling
- message delivery/load testing

### Exit criteria for 100%
- broker runs reliably
- topic permissions are enforced
- telemetry is published and consumed correctly
- high-volume messaging works without unacceptable loss
- disconnects produce intended LWT behavior

### Partial scoring
- **25%**: broker config and topic plan exist
- **50%**: messages flow in normal cases
- **75%**: ACL and reliability behavior mostly work
- **100%**: all exit criteria satisfied

### Self-check
- Are topic boundaries explicit and secure?
- Does the historian actually receive the intended stream?
- Are QoS expectations verified, not assumed?

---

## Step 4.4 — OPC UA server

**Status:** NOT STARTED  
**Completion:** 0  
**Owner:**  
**ETA remaining:** 18–26 hours  
**Depends on:** 4.3  
**Last review date:**  
**Notes / blockers:**  

### Objective
Expose plant state and control hooks through a structured OPC UA namespace.

### Deliverables
- OPC UA server
- address space / namespace model
- MQTT-to-OPC update bridge
- callable methods
- manual validation with an OPC UA client

### Exit criteria for 100%
- clients can connect and browse the namespace
- node values update in real time
- method calls route correctly into the system
- emergency/control paths behave as designed

### Partial scoring
- **25%**: server shell and namespace skeleton exist
- **50%**: namespace is exposed but not fully live
- **75%**: updates and methods mostly work
- **100%**: all exit criteria satisfied

### Self-check
- Is the namespace clean and understandable?
- Are node types and metadata consistent?
- Does the server reflect real plant state instead of a dead model?

---

## Step 4.5 — Historian and InfluxDB integration

**Status:** NOT STARTED  
**Completion:** 0  
**Owner:**  
**ETA remaining:** 18–28 hours  
**Depends on:** 4.3  
**Last review date:**  
**Notes / blockers:**  

### Objective
Persist live telemetry in time-series storage with sensible measurements, retention, and dashboards.

### Deliverables
- historian service
- InfluxDB setup
- retention/downsampling design
- Grafana dashboard for live views

### Exit criteria for 100%
- telemetry is stored reliably
- batch writes are stable
- retention/downsampling behavior exists
- Grafana displays live plant data
- common queries perform acceptably

### Partial scoring
- **25%**: storage stack exists
- **50%**: writes work, schema incomplete or fragile
- **75%**: historian is operational in normal conditions
- **100%**: all exit criteria satisfied

### Self-check
- Is measurement design query-friendly?
- Can the dashboard serve as a real operator/developer signal?
- Is long-term retention strategy defined instead of postponed?

---

## Phase 4 gate
Phase 4 is complete when:
- services speak through stable contracts;
- live telemetry moves through MQTT/gRPC;
- OPC UA and historian paths are operational.

---

# PHASE 5 — API Gateway and Security
**Target duration:** 3 weeks  
**Difficulty:** High  
**Goal:** provide a secure, typed, auditable entry point into the system.

---

## Step 5.1 — FastAPI gateway

**Status:** NOT STARTED  
**Completion:** 0  
**Owner:**  
**ETA remaining:** 20–28 hours  
**Depends on:** Phases 2 and 4  
**Last review date:**  
**Notes / blockers:**  

### Objective
Implement the main HTTP/WebSocket gateway for status, commands, history, and alarms.

### Deliverables
- FastAPI app
- routers
- schemas
- websocket endpoint
- integration tests

### Exit criteria for 100%
- OpenAPI docs are usable
- key endpoints exist and validate inputs correctly
- WebSocket streaming works in the browser
- core endpoints are tested for success and failure paths
- status and response semantics are consistent

### Partial scoring
- **25%**: app shell and routers exist
- **50%**: basic endpoints work
- **75%**: websocket and tests are mostly operational
- **100%**: all exit criteria satisfied

### Self-check
- Are request/response schemas explicit and typed?
- Do docs reflect actual behavior?
- Is API versioning stable?

---

## Step 5.2 — JWT auth and RBAC

**Status:** NOT STARTED  
**Completion:** 0  
**Owner:**  
**ETA remaining:** 20–30 hours  
**Depends on:** 5.1  
**Last review date:**  
**Notes / blockers:**  

### Objective
Implement user authentication, token lifecycle, and role-based access control.

### Deliverables
- JWT handling
- login / refresh / logout flow
- RBAC checks
- password hashing
- role/user/token persistence
- auth tests

### Exit criteria for 100%
- access and refresh flows work correctly
- expired or invalid tokens are rejected clearly
- roles are enforced correctly
- password storage is hashed only
- refresh rotation / blacklist logic works as designed

### Partial scoring
- **25%**: token/auth scaffolding exists
- **50%**: login works, lifecycle incomplete
- **75%**: RBAC and token lifecycle mostly work
- **100%**: all exit criteria satisfied

### Self-check
- Is the auth model secure enough for the intended system?
- Are 401 vs 403 cases separated properly?
- Is refresh token behavior explicit and test-covered?

---

## Step 5.3 — TLS / mTLS and secret protection

**Status:** NOT STARTED  
**Completion:** 0  
**Owner:**  
**ETA remaining:** 16–24 hours  
**Depends on:** 5.2, Phase 4  
**Last review date:**  
**Notes / blockers:**  

### Objective
Protect transport channels and sensitive configuration/state.

### Deliverables
- dev CA
- service certificates
- HTTPS for gateway
- mTLS for MQTT/gRPC where intended
- encryption utilities for sensitive fields
- cert generation automation

### Exit criteria for 100%
- HTTPS works
- services requiring mTLS reject unauthenticated clients
- gRPC and broker transport security are verifiable
- sensitive data encryption/decryption works correctly
- cert generation workflow is repeatable

### Partial scoring
- **25%**: certificate tooling exists
- **50%**: HTTPS works locally
- **75%**: mTLS paths and encryption mostly work
- **100%**: all exit criteria satisfied

### Self-check
- Is transport security really enforced, not just configured?
- Are certificates easy to regenerate cleanly?
- Are secrets protected both in transit and at rest where required?

---

## Step 5.4 — Immutable audit log and PostgreSQL schema

**Status:** NOT STARTED  
**Completion:** 0  
**Owner:**  
**ETA remaining:** 20–30 hours  
**Depends on:** 5.2  
**Last review date:**  
**Notes / blockers:**  

### Objective
Create the persistent schema for security, alarms, and operational metadata with real audit guarantees.

### Deliverables
- full PostgreSQL schema
- Alembic migrations
- audit middleware
- alert manager persistence path
- admin audit retrieval path

### Exit criteria for 100%
- migrations run cleanly on a fresh database
- audit events are recorded automatically
- immutability is enforced in practice
- alarm events land in the database through the intended path
- upgrade/downgrade flow is controlled

### Partial scoring
- **25%**: schema draft and migration skeletons exist
- **50%**: tables and migrations exist, behavior incomplete
- **75%**: audit/alarm persistence mostly works
- **100%**: all exit criteria satisfied

### Self-check
- Is the audit table actually immutable in practice?
- Are migrations reversible and understandable?
- Are alarms and operational records queryable and useful?

---

## Phase 5 gate
Phase 5 is complete when:
- gateway routes exist and are typed;
- auth and RBAC are enforced;
- transport and persistence security are not placeholders.

---

# PHASE 6 — AI / ML Subsystem
**Target duration:** 4 weeks  
**Difficulty:** High  
**Goal:** generate training data, train three model families, and integrate them into inference services.

---

## Step 6.1 — Synthetic dataset generation and feature engineering

**Status:** NOT STARTED  
**Completion:** 0  
**Owner:**  
**ETA remaining:** 24–36 hours  
**Depends on:** Phases 2–4  
**Last review date:**  
**Notes / blockers:**  

### Objective
Generate a large labeled synthetic dataset and transform it into a stable ML-ready feature pipeline.

### Deliverables
- raw and split datasets
- feature engineering pipeline
- scalers/normalization artifacts
- dataset metadata

### Exit criteria for 100%
- dataset volume is sufficient
- train/val/test do not leak across time
- features are computed reproducibly
- no broken values remain
- metadata explains class balance and generation conditions

### Partial scoring
- **25%**: generation scripts and schema exist
- **50%**: datasets are produced but not validated
- **75%**: features and splits are mostly stable
- **100%**: all exit criteria satisfied

### Self-check
- Is the split time-safe?
- Are labels trustworthy?
- Are features meaningful instead of decorative?

---

## Step 6.2 — Anomaly detector

**Status:** NOT STARTED  
**Completion:** 0  
**Owner:**  
**ETA remaining:** 24–36 hours  
**Depends on:** 6.1  
**Last review date:**  
**Notes / blockers:**  

### Objective
Train and package an anomaly detection model on normal operating behavior.

### Deliverables
- training code
- inference code
- trained model artifacts
- experiment tracking

### Exit criteria for 100%
- model trains stably
- thresholding is defined and justified
- test metrics are acceptable
- CPU inference latency is acceptable
- saved artifacts are reusable

### Partial scoring
- **25%**: model skeleton exists
- **50%**: training works, metrics are weak or unstable
- **75%**: inference is usable and metrics are close to target
- **100%**: all exit criteria satisfied

### Self-check
- Is the anomaly threshold based on evidence?
- Are false positives controlled?
- Is model serialization verified, not assumed?

---

## Step 6.3 — Efficiency advisor

**Status:** NOT STARTED  
**Completion:** 0  
**Owner:**  
**ETA remaining:** 24–36 hours  
**Depends on:** 6.1  
**Last review date:**  
**Notes / blockers:**  

### Objective
Train a recommendation model for short-term operating setpoints under physical constraints.

### Deliverables
- training script
- inference artifact
- recommendation API path
- what-if analysis path

### Exit criteria for 100%
- predictions are numerically reasonable
- outputs remain within physical bounds
- recommendation quality is measured
- API responds acceptably
- simulated comparison shows tangible gain

### Partial scoring
- **25%**: model scaffold exists
- **50%**: model predicts but lacks reliability
- **75%**: constrained recommendations mostly work
- **100%**: all exit criteria satisfied

### Self-check
- Are recommendations physically legal?
- Is there a measurable benefit over baseline?
- Is the model explaining or merely guessing?

---

## Step 6.4 — Predictive maintenance model

**Status:** NOT STARTED  
**Completion:** 0  
**Owner:**  
**ETA remaining:** 24–34 hours  
**Depends on:** 6.1  
**Last review date:**  
**Notes / blockers:**  

### Objective
Estimate component health and failure risk for maintenance prioritization.

### Deliverables
- maintenance model
- training code
- health scoring logic
- survival/risk estimation
- maintenance schedule output

### Exit criteria for 100%
- health score degrades plausibly under degradation scenarios
- failure probability estimates are calibrated enough to be useful
- ranking of components reflects actual risk order
- schedule output is actionable and interpretable

### Partial scoring
- **25%**: architecture and labels are defined
- **50%**: model outputs scores, interpretation incomplete
- **75%**: schedule/ranking is mostly usable
- **100%**: all exit criteria satisfied

### Self-check
- Does the health score track degradation meaningfully?
- Is risk ranking believable?
- Is the output something an engineer could act on?

---

## Step 6.5 — AI Predictor service integration

**Status:** NOT STARTED  
**Completion:** 0  
**Owner:**  
**ETA remaining:** 20–30 hours  
**Depends on:** 6.2, 6.3, 6.4, Phase 4, Phase 5  
**Last review date:**  
**Notes / blockers:**  

### Objective
Load trained models into a service that performs scheduled inference and emits operational outputs.

### Deliverables
- AI predictor service
- model loading
- inference scheduler
- data fetch / preprocess / infer / publish pipeline
- integration with alert path
- AI status endpoint

### Exit criteria for 100%
- all models load correctly
- scheduled inference runs reliably
- anomalies and predictions are emitted through intended paths
- resource usage stays within reasonable bounds
- graceful shutdown works correctly

### Partial scoring
- **25%**: service shell and loading logic exist
- **50%**: one or more models infer successfully
- **75%**: scheduled pipeline mostly works
- **100%**: all exit criteria satisfied

### Self-check
- Can the service survive routine operation?
- Is the scheduler deterministic enough?
- Are outputs actually integrated into the rest of the platform?

---

## Phase 6 gate
Phase 6 is complete when:
- data generation is reproducible;
- model training has measurable quality;
- scheduled inference is integrated into the platform.

---

# PHASE 7 — Containers, Kubernetes, and CI/CD
**Target duration:** 4 weeks  
**Difficulty:** High  
**Goal:** package, deploy, observe, and automate the full platform.

---

## Step 7.1 — Dockerization of all services

**Status:** NOT STARTED  
**Completion:** 0  
**Owner:**  
**ETA remaining:** 20–30 hours  
**Depends on:** core services existing  
**Last review date:**  
**Notes / blockers:**  

### Objective
Package each service into production-style images and enable full local orchestration.

### Deliverables
- Dockerfiles
- `.dockerignore`
- `docker-compose.yml`
- dev override configuration
- image scanning workflow
- published images

### Exit criteria for 100%
- full stack boots through Compose
- images are reasonably optimized
- critical vulnerabilities are absent
- images are taggable and publishable
- cold start is acceptable

### Partial scoring
- **25%**: some service images exist
- **50%**: most services containerize, orchestration incomplete
- **75%**: stack boots locally with manageable issues
- **100%**: all exit criteria satisfied

### Self-check
- Are images reproducible?
- Is the local stack actually usable for development?
- Are build layers optimized enough to avoid painful rebuild times?

---

## Step 7.2 — Kubernetes manifests and Helm chart

**Status:** NOT STARTED  
**Completion:** 0  
**Owner:**  
**ETA remaining:** 24–36 hours  
**Depends on:** 7.1  
**Last review date:**  
**Notes / blockers:**  

### Objective
Define deployment topology for the platform under Kubernetes and package it through Helm.

### Deliverables
- raw manifests
- Deployments, Services, Ingress
- Namespaces
- NetworkPolicies
- PVCs
- HPA where intended
- Helm chart and values files

### Exit criteria for 100%
- Helm install/upgrade works
- pods become healthy
- network isolation is meaningful
- storage survives restarts where required
- autoscaling and environment-specific values are valid

### Partial scoring
- **25%**: base chart and manifests exist
- **50%**: workload deploys partially
- **75%**: cluster deployment mostly works
- **100%**: all exit criteria satisfied

### Self-check
- Is the deployment model understandable and maintainable?
- Are environment differences parameterized?
- Are security boundaries encoded, not left informal?

---

## Step 7.3 — Observability stack

**Status:** NOT STARTED  
**Completion:** 0  
**Owner:**  
**ETA remaining:** 24–34 hours  
**Depends on:** 7.1, 7.2, Phase 4, Phase 5  
**Last review date:**  
**Notes / blockers:**  

### Objective
Introduce metrics, tracing, logs, dashboards, and alerts across the platform.

### Deliverables
- OpenTelemetry instrumentation
- collector config
- Prometheus config
- Loki/Promtail config
- Grafana dashboards
- alert rules

### Exit criteria for 100%
- live metrics are visible
- traces show cross-service request paths
- logs are centralized and correlated
- operational dashboards are meaningful
- alerts trigger under simulated fault conditions

### Partial scoring
- **25%**: stack components defined
- **50%**: basic telemetry visible
- **75%**: dashboards and alerts mostly work
- **100%**: all exit criteria satisfied

### Self-check
- Can a developer/debugger trace a request end-to-end?
- Are logs useful, not just numerous?
- Would an operator notice a real problem through this stack?

---

## Step 7.4 — CI/CD pipelines

**Status:** NOT STARTED  
**Completion:** 0  
**Owner:**  
**ETA remaining:** 24–36 hours  
**Depends on:** 7.1, 7.2  
**Last review date:**  
**Notes / blockers:**  

### Objective
Automate quality checks, tests, security scans, model validation, image publishing, and deployment.

### Deliverables
- lint/typecheck workflow
- test workflow
- security scan workflow
- ML validation workflow
- build/publish workflow
- deployment workflow
- environment controls
- badges if desired

### Exit criteria for 100%
- workflows execute reliably
- failing checks block unsafe merges/deploys
- image publishing is automated
- deployment path is controlled
- overall pipeline duration is acceptable

### Partial scoring
- **25%**: workflow files exist
- **50%**: some workflows execute
- **75%**: full CI mostly works, deployment path needs polish
- **100%**: all exit criteria satisfied

### Self-check
- Is every critical quality gate automated?
- Can broken code reach deployment?
- Are security and ML validation real gates, not decorative steps?

---

## Step 7.5 — Documentation and final polish

**Status:** NOT STARTED  
**Completion:** 0  
**Owner:**  
**ETA remaining:** 18–28 hours  
**Depends on:** most previous phases  
**Last review date:**  
**Notes / blockers:**  

### Objective
Finish the project as a professional, understandable, reproducible system.

### Deliverables
- English README
- architecture document
- deployment guide
- optional Spanish README
- demo script
- future roadmap file

### Exit criteria for 100%
- a new developer can follow docs and run the system
- architecture is explained clearly
- demo path works
- no stale placeholder docs remain
- the project presentation is coherent and professional

### Partial scoring
- **25%**: document shells exist
- **50%**: docs are present but incomplete or outdated
- **75%**: main docs are usable
- **100%**: all exit criteria satisfied

### Self-check
- Can an external engineer understand and run the system?
- Do docs match the actual repo?
- Is the demo robust enough for interviews and demonstrations?

---

## Phase 7 gate
Phase 7 is complete when:
- the stack is containerized, deployable, observable, and automated;
- documentation makes the platform operable by someone other than the original author.

---

# 6. Executive progress summary

Use this section as the current one-page control board.

| Phase | Name | Target duration | Current completion | Status | Blockers |
|---|---|---:|---:|---|---|
| 1 | Foundation | 3 weeks | 0% | NOT STARTED | |
| 2 | Physics Engine | 4 weeks | 0% | NOT STARTED | |
| 3 | PLC / Control | 3 weeks | 0% | NOT STARTED | |
| 4 | Communication | 4 weeks | 0% | NOT STARTED | |
| 5 | API / Security | 3 weeks | 0% | NOT STARTED | |
| 6 | AI / ML | 4 weeks | 0% | NOT STARTED | |
| 7 | Infra / CI/CD | 4 weeks | 0% | NOT STARTED | |

---

# 7. Current step tracker

| Step | Title | Completion | Status | Owner | ETA remaining | Notes |
|---|---|---:|---|---|---|---|
| 1.1 | Repository initialization and structure | 0% | NOT STARTED |  |  |  |
| 1.2 | `uv` workspace and dependency management | 0% | NOT STARTED |  |  |  |
| 1.3 | Linting, formatting, typing, pre-commit | 0% | NOT STARTED |  |  |  |
| 2.1 | Boiler thermodynamic ODE model | 0% | NOT STARTED |  |  |  |
| 2.2 | Steam turbine model | 0% | NOT STARTED |  |  |  |
| 2.3 | Async simulator, valves, and scenarios | 0% | NOT STARTED |  |  |  |
| 2.4 | Visualization and dataset export | 0% | NOT STARTED |  |  |  |
| 3.1 | Base PID controller | 0% | NOT STARTED |  |  |  |
| 3.2 | Cascade control system | 0% | NOT STARTED |  |  |  |
| 3.3 | Safety interlocks and emergency logic | 0% | NOT STARTED |  |  |  |
| 4.1 | Protocol Buffers contracts | 0% | NOT STARTED |  |  |  |
| 4.2 | gRPC service layer | 0% | NOT STARTED |  |  |  |
| 4.3 | MQTT broker and telemetry pipeline | 0% | NOT STARTED |  |  |  |
| 4.4 | OPC UA server | 0% | NOT STARTED |  |  |  |
| 4.5 | Historian and InfluxDB integration | 0% | NOT STARTED |  |  |  |
| 5.1 | FastAPI gateway | 0% | NOT STARTED |  |  |  |
| 5.2 | JWT auth and RBAC | 0% | NOT STARTED |  |  |  |
| 5.3 | TLS / mTLS and secret protection | 0% | NOT STARTED |  |  |  |
| 5.4 | Immutable audit log and PostgreSQL schema | 0% | NOT STARTED |  |  |  |
| 6.1 | Synthetic dataset generation and features | 0% | NOT STARTED |  |  |  |
| 6.2 | Anomaly detector | 0% | NOT STARTED |  |  |  |
| 6.3 | Efficiency advisor | 0% | NOT STARTED |  |  |  |
| 6.4 | Predictive maintenance model | 0% | NOT STARTED |  |  |  |
| 6.5 | AI Predictor service integration | 0% | NOT STARTED |  |  |  |
| 7.1 | Dockerization | 0% | NOT STARTED |  |  |  |
| 7.2 | Kubernetes and Helm | 0% | NOT STARTED |  |  |  |
| 7.3 | Observability stack | 0% | NOT STARTED |  |  |  |
| 7.4 | CI/CD pipelines | 0% | NOT STARTED |  |  |  |
| 7.5 | Documentation and final polish | 0% | NOT STARTED |  |  |  |

---

# 8. Rules for honest scoring

Use these rules to avoid inflating progress:

- Code without tests or verification is not 100%.
- A local happy-path demo without error handling is usually 50–75%, not 100%.
- A config file added but never executed is not “done”.
- A service that boots but is not integrated with its dependencies is not “done”.
- A model that trains once but is not reproducible is not “done”.
- A dashboard that renders but has no operational value is not “done”.
- A deployment that works once manually but has no repeatable path is not “done”.

---

# 9. Definition of project completion

CogniBoiler may be considered fully implemented only when:

- all 29 steps are at 100%;
- phase gates are satisfied;
- core services operate together as one platform;
- security paths are implemented, not mocked;
- observability is usable in practice;
- ML outputs are integrated into operations;
- a new developer can clone, run, test, and understand the system from repo materials.

---

# 10. Final note

This file is not a marketing roadmap.  
It is an execution control document.

The correct use of this file is to:
- mark reality honestly,
- expose blockers early,
- prevent fake progress,
- and keep implementation aligned with the actual target system.
