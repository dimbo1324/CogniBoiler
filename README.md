<div align="center">

```
   ██████╗ ██████╗  ██████╗ ███╗   ██╗██╗██████╗  ██████╗ ██╗██╗     ███████╗██████╗
  ██╔════╝██╔═══██╗██╔════╝ ████╗  ██║██║██╔══██╗██╔═══██╗██║██║     ██╔════╝██╔══██╗
  ██║     ██║   ██║██║  ███╗██╔██╗ ██║██║██████╔╝██║   ██║██║██║     █████╗  ██████╔╝
  ██║     ██║   ██║██║   ██║██║╚██╗██║██║██╔══██╗██║   ██║██║██║     ██╔══╝  ██╔══██╗
  ╚██████╗╚██████╔╝╚██████╔╝██║ ╚████║██║██████╔╝╚██████╔╝██║███████╗███████╗██║  ██║
   ╚═════╝ ╚═════╝  ╚═════╝ ╚═╝  ╚═══╝╚═╝╚═════╝  ╚═════╝ ╚═╝╚══════╝╚══════╝╚═╝  ╚═╝
```

### AI-Driven Digital Twin Platform for Industrial Steam Power Generation

_A production-grade simulation and intelligence platform for steam boilers,_
_turbines, and virtual PLC control — built for real-world deployment._

---

[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org)
[![uv](https://img.shields.io/badge/uv-workspace-DE5FE9?style=flat-square)](https://docs.astral.sh/uv)
[![License](https://img.shields.io/badge/License-Apache_2.0-green?style=flat-square)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-F7B500?style=flat-square)](https://docs.astral.sh/ruff)
[![Typed: mypy](https://img.shields.io/badge/typed-mypy%20strict-2A6DB5?style=flat-square)](https://mypy-lang.org)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-FAB040?style=flat-square&logo=pre-commit)](https://pre-commit.com)
[![Status](https://img.shields.io/badge/status-in%20development-orange?style=flat-square)]()

</div>

---

## What is CogniBoiler?

CogniBoiler is a **digital twin platform** that simulates the complete thermodynamic cycle of a modern 300 MW gas-fired steam power plant — boiler, turbine, and virtual PLC — enriched with an AI layer for anomaly detection, efficiency optimization, and predictive maintenance.

The project is designed from the ground up to reflect real industrial software: it follows IEC standards, uses the same protocols found on actual plant floors (OPC UA, MQTT), and is built to be deployable on real infrastructure via Kubernetes.

This is not a toy simulation. The physics model is derived from real thermodynamic equations. The AI models train on that physics — not random noise. The security model follows NERC CIP and IEC 62443 conventions. Every architectural decision maps to something you would find in production at a utility company.

> **Target markets:** United States · Uruguay · Argentina

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        EXTERNAL CLIENTS                         │
│              Browser  ·  REST API  ·  OPC UA Clients            │
└────────────────────────────┬────────────────────────────────────┘
                             │ HTTPS / WebSocket
                    ┌────────▼────────┐
                    │   API GATEWAY   │  FastAPI · JWT · RBAC
                    │  (auth & route) │  TLS 1.3 · Rate Limiting
                    └──┬──────┬───┬──┘
               gRPC    │      │   │    gRPC
        ┌──────────────┘      │   └──────────────┐
        │                     │                  │
┌───────▼──────┐    ┌─────────▼──────┐   ┌──────▼───────┐
│   PHYSICS    │    │ PLC CONTROLLER │   │ AI PREDICTOR │
│   ENGINE     │    │                │   │              │
│  ODE Solver  │    │ Cascade PID    │   │ LSTM Autoenc.│
│  Boiler+     │    │ Safety Layer   │   │ Transformer  │
│  Turbine     │    │ Interlocks     │   │ GRU + Weibull│
└──────┬───────┘    └────────────────┘   └──────┬───────┘
       │                                         │
       └──────────────┬──────────────────────────┘
                      │
         ┌────────────▼────────────┐
         │      MESSAGE BUS        │
         │  MQTT (Mosquitto 2.x)   │
         │  Protocol Buffers       │
         └──┬──────────────────┬───┘
            │                  │
   ┌────────▼──────┐  ┌────────▼──────┐
   │  OPC UA       │  │   HISTORIAN   │
   │  SERVER       │  │               │
   │  IEC 62541    │  │  InfluxDB     │
   │  Field level  │  │  PostgreSQL   │
   └───────────────┘  └───────┬───────┘
                               │
                      ┌────────▼───────┐
                      │    GRAFANA     │
                      │  Dashboards    │
                      │  Prometheus    │
                      │  Loki · OTel   │
                      └────────────────┘
```

**Data flows:**

- **Sensor stream** (1–10 Hz): Physics Engine -> MQTT -> Historian -> InfluxDB -> Grafana
- **Control commands**: Operator -> API Gateway -> PLC -> MQTT -> Physics Engine
- **AI analysis** (scheduled): InfluxDB -> AI Predictor -> Alert Manager -> PostgreSQL
- **Observability** (continuous): All services -> OpenTelemetry -> Prometheus + Loki -> Grafana

---

## Technology Stack

| Layer                  | Technologies                                       |
| ---------------------- | -------------------------------------------------- |
| **Language**           | Python 3.12+                                       |
| **Package management** | uv (workspace monorepo)                            |
| **Web framework**      | FastAPI, uvicorn                                   |
| **Physics simulation** | NumPy, SciPy (ODE solver RK45)                     |
| **AI / ML**            | PyTorch, scikit-learn, Pandas, MLflow              |
| **Protocols**          | MQTT 5.0 (Mosquitto), OPC UA (asyncua), gRPC       |
| **Serialization**      | Protocol Buffers (protobuf)                        |
| **Time-series DB**     | InfluxDB 2.x (Flux)                                |
| **Relational DB**      | PostgreSQL 16 (SQLAlchemy async, asyncpg)          |
| **Migrations**         | Alembic                                            |
| **Security**           | PyJWT (RS256), cryptography (AES-256, RSA), bcrypt |
| **Visualization**      | Grafana, Matplotlib                                |
| **Observability**      | OpenTelemetry, Prometheus, Loki, Promtail          |
| **Containers**         | Docker (multi-stage builds)                        |
| **Orchestration**      | Kubernetes, Helm 3                                 |
| **CI/CD**              | GitHub Actions                                     |
| **Code quality**       | ruff, mypy (strict), pre-commit                    |

---

## Services

The platform is structured as a microservice workspace. Each service is an independent Python package with its own dependencies, tests, and Dockerfile.

```
apps/
├── physics-engine/     Thermodynamic ODE model of boiler + turbine
├── plc-controller/     Virtual PLC with cascade PID and safety interlocks
├── api-gateway/        FastAPI gateway — auth, routing, WebSocket
├── historian/          MQTT subscriber -> InfluxDB time-series writer
├── alert-manager/      Alarm management and immutable audit log
├── opcua-server/       OPC UA field-level server (IEC 62541)
└── ai-predictor/       PyTorch inference: anomaly · efficiency · maintenance
```

---

## AI Capabilities

Three independent neural network models run on a continuous inference schedule:

**Anomaly Detector** — LSTM Autoencoder trained exclusively on normal operating data. Detects deviations via reconstruction error. Targets: burner fouling, steam leaks, turbine blade wear.

**Efficiency Advisor** — Temporal Transformer that takes 30 minutes of plant history and recommends optimal setpoints for the next 15 minutes. Includes a physical constraint layer to ensure all recommendations are thermodynamically feasible.

**Predictive Maintenance** — GRU network computing a health score (0–100) for each component, combined with Weibull Survival Analysis to estimate time-to-failure probability. Output: prioritized maintenance schedule with confidence intervals.

All three models train on **synthetic data generated by the physics engine itself** — making the project fully self-contained and reproducible without access to real plant data.

---

## Physical Model

The boiler is described by a system of ODEs solved numerically (Runge-Kutta 4/5):

```
Thermal balance:    dU/dt  = Q_fuel - Q_steam - Q_loss
Steam pressure:     dP/dt  = f(T, m_water, V_drum)
Drum water level:   dh/dt  = (m_feed - m_steam) / (ρ · A)
Flue gas temp:      dT_g/dt = f(Q_fuel, m_air, η_combustion)
Turbine power:      W_elec = η_turbine · m_steam · (h_in - h_out)
```

**Nominal parameters** (300 MW class gas boiler):

| Parameter         | Value          |
| ----------------- | -------------- |
| Steam pressure    | 100 – 180 bar  |
| Steam temperature | 540 – 565 °C   |
| Steam flow        | 500 – 1000 t/h |
| Electrical output | 100 – 300 MW   |
| Boiler efficiency | 88 – 93 %      |

Control is implemented as a **cascade PID system** — the industrial standard for boiler control:

- `PID_1` Master: power setpoint -> pressure setpoint
- `PID_2` Slave: pressure setpoint -> fuel valve position
- `PID_3` Independent: water level -> feedwater valve
- `PID_4` Independent: steam temperature -> desuperheater spray

---

## Security Model

Security is implemented at every layer independently — compromise of one layer does not grant access to the next.

```
Transport:   TLS 1.3 (all HTTP)  ·  mTLS (MQTT, OPC UA, gRPC)
Identity:    JWT RS256 · 15-min access tokens · 7-day refresh rotation
Passwords:   bcrypt (cost 12) — plaintext never stored
Data at rest: AES-256-GCM for sensitive fields
Audit log:   INSERT-only PostgreSQL table — physically immutable
Network:     Kubernetes NetworkPolicy — zero-trust between namespaces
Scanning:    Trivy (images) · Bandit (Python source)
```

RBAC roles: `viewer` · `operator` · `engineer` · `admin`

---

## Project Status

This project is under active development following a structured 7-phase roadmap.

| Phase | Description                                            | Status      |
| ----- | ------------------------------------------------------ | ----------- |
| 1     | Project foundation — uv workspace, tooling, structure  | ✅ Complete |
| 2     | Physics Engine — ODE boiler and turbine model          | 🔄 Up next  |
| 3     | Virtual PLC — cascade PID, safety interlocks           | ⏳ Planned  |
| 4     | Communication — protobuf, gRPC, MQTT, OPC UA, InfluxDB | ⏳ Planned  |
| 5     | API Gateway — FastAPI, JWT, TLS, PostgreSQL            | ⏳ Planned  |
| 6     | AI/ML — PyTorch models, MLflow, inference service      | ⏳ Planned  |
| 7     | Infrastructure — Docker, Kubernetes, Helm, CI/CD       | ⏳ Planned  |

---

## Getting Started

> The project is in early development. Full quick-start instructions will be added when Phase 2 is complete. For now, you can explore the workspace structure and tooling setup.

**Prerequisites:** Python 3.12+, [uv](https://docs.astral.sh/uv/getting-started/installation/), Git

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/cogniboiler.git
cd cogniboiler

# Install all workspace dependencies
uv sync --all-packages

# Verify the environment
uv run python -c "import numpy; print('NumPy', numpy.__version__)"
uv run python -c "import fastapi; print('FastAPI', fastapi.__version__)"

# Run code quality checks
make check
```

---

## Development

```bash
make install        # Install all dependencies
make lint           # Run ruff linter
make format         # Format code with ruff
make typecheck      # Run mypy strict type checking
make check          # All quality checks at once
make test           # Run test suite
make test-cov       # Run tests with coverage report
make clean          # Remove caches and build artifacts
```

Pre-commit hooks run automatically on every `git commit`:
ruff lint · ruff format · mypy · trailing whitespace · YAML/TOML validation · debug statement detection

---

## Repository Structure

```
cogniboiler/
├── apps/                   Microservices (7 services)
├── shared/                 Cross-service code (models, proto, crypto)
├── ml/                     Training scripts, datasets, saved models
├── infrastructure/         Docker, Kubernetes, Helm, Grafana dashboards
├── scripts/                Development and operational scripts
├── tests/                  Top-level integration tests
├── docs/                   Architecture and deployment documentation
├── certs/                  TLS certificates (dev environment only)
├── pyproject.toml          uv workspace root
├── uv.lock                 Deterministic dependency lock file
└── Makefile                Development automation
```

---

## Roadmap Highlights

Features planned for future phases:

- **Reinforcement Learning controller** — compare RL agent vs classic PID in real time
- **3D plant visualization** — WebGL dashboard showing live sensor state on 3D boiler model
- **Real OPC UA integration** — connect to an actual DCS or SCADA system
- **NERC CIP compliance module** — automated compliance reporting for US utilities
- **Multi-unit simulation** — scale to simulate an entire power plant with multiple boilers

---

## License

Licensed under the [Apache License 2.0](LICENSE).

---

<div align="center">

_Built with precision. Designed for industry._

**CogniBoiler** · AI-Driven Digital Twin Platform

</div>
