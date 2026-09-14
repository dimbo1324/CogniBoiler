# CogniBoiler

**An AI-driven digital twin platform for industrial steam boiler and turbine systems.**

CogniBoiler is a software project that brings together **physics simulation**, **industrial control logic**, **real-time telemetry**, **historical data storage**, **web APIs**, and **machine learning** into one coherent platform.

In simple terms, it is a digital environment that can **simulate how a large industrial steam power unit behaves**, **control it like a real plant**, **stream and store operational data**, and eventually **analyse that data with AI** to detect abnormal behaviour, improve efficiency, and support predictive maintenance.

This project is designed to feel much closer to a real industrial system than to a classroom demo. It uses the same types of concepts and technologies that appear in modern automation and energy systems: microservices, MQTT, gRPC, OPC UA, time-series databases, web gateways, role-based security, and ML-based operational intelligence.

---

## Table of Contents

- [What is CogniBoiler?](#what-is-cogniboiler)
- [The Big Idea](#the-big-idea)
- [What Problem the Project Solves](#what-problem-the-project-solves)
- [How to Think About the Platform](#how-to-think-about-the-platform)
- [Core Capabilities](#core-capabilities)
- [System Architecture](#system-architecture)
- [Main Components](#main-components)
- [AI Layer](#ai-layer)
- [Industrial Communication and Data Flow](#industrial-communication-and-data-flow)
- [Security and Reliability Philosophy](#security-and-reliability-philosophy)
- [Who This Project Is For](#who-this-project-is-for)
- [Typical Use Cases](#typical-use-cases)
- [Technology Overview](#technology-overview)
- [Project Status](#project-status)
- [Design Principles](#design-principles)
- [Long-Term Direction](#long-term-direction)
- [Final Note](#final-note)

---

## What is CogniBoiler?

CogniBoiler is an **industrial digital twin platform** focused on a steam boiler–turbine system.

A **digital twin** is a software representation of a real physical system. In this case, the goal is to represent the behaviour of equipment commonly found in steam power generation and process industries:

- the **boiler**, where fuel is converted into thermal energy;
- the **steam system**, where pressure, temperature, and flow must be controlled;
- the **turbine**, where steam energy is converted into useful mechanical and electrical output;
- the **control layer**, which behaves like a virtual industrial PLC;
- the **data layer**, which records system history and makes it observable;
- the **AI layer**, which interprets behaviour and supports decision-making.

This means CogniBoiler is not just “a simulator” and not just “an AI project”. It is a **complete software architecture** that tries to represent how a modern industrial system can be modeled, controlled, observed, and analysed.

---

## The Big Idea

The central idea behind CogniBoiler is straightforward:

> Build one platform that can simulate an industrial steam system, operate like a real control environment, and eventually provide intelligent insight about efficiency, risk, and maintenance.

That idea turns into several layers working together:

1. A **physics layer** produces realistic process behaviour.
2. A **control layer** acts like an industrial controller.
3. A **communication layer** moves data between services.
4. A **storage and observability layer** records and visualises everything.
5. A **security layer** protects access and actions.
6. An **AI layer** learns from system history and helps interpret what is happening.

---

## What Problem the Project Solves

Industrial systems are complex.

A large steam unit is not something you can understand from one sensor or one graph. Pressure, temperature, water level, steam flow, valve positions, fuel input, and turbine output all influence each other. Real systems also require:

- safe control logic;
- reliable communication between subsystems;
- structured historical storage;
- monitoring and alarms;
- clear access control;
- explainable operational insight.

CogniBoiler addresses this by creating a software platform where all of those concerns can live together.

At a high level, the project helps answer questions like these:

- What does a realistic industrial steam system look like in software?
- How can a digital twin be structured as a modern microservice platform?
- How can control, telemetry, storage, APIs, and AI coexist in one architecture?
- How can simulation data be turned into useful signals for anomaly detection and maintenance planning?

---

## How to Think About the Platform

The easiest way to understand CogniBoiler is to imagine a **virtual industrial plant** built out of software services.

Each service has a specific job.

Some services are responsible for **simulating the physical process**.
Some are responsible for **control and commands**.
Some move data between layers using **industrial protocols**.
Some store the system’s history.
Some expose information through a web API.
Some will analyse the data with machine learning.

Together, they form a system that behaves like a miniature industrial ecosystem.

You can think of CogniBoiler as a combination of:

- a **physics simulator**;
- a **virtual automation stack**;
- a **data platform**;
- an **AI-assisted monitoring system**.

---

## Core Capabilities

CogniBoiler is being designed to provide the following broad capabilities.

### 1. Physics-based process simulation

The platform models boiler and turbine behaviour using engineering and thermodynamic logic rather than arbitrary fake numbers. The purpose is to generate system states that behave like a real process and respond to control actions in a realistic way.

### 2. Virtual industrial control

The project includes a virtual control layer intended to resemble PLC-style behaviour. This includes actuator logic, operating constraints, and safety-oriented decision paths.

### 3. Real-time telemetry streaming

The system is built to move process data in near real time between components, allowing live monitoring, state distribution, and downstream processing.

### 4. Historical recording

Operational data can be recorded as time series so the platform has memory. This is essential for dashboards, diagnostics, model training, and later analysis.

### 5. External access through APIs

A web gateway provides a structured way for external clients to read system state, retrieve history, and submit controlled actions.

### 6. Industrial protocol integration

The architecture includes technologies commonly used in industrial and plant environments, which makes the project relevant beyond pure software experimentation.

### 7. AI-assisted operational intelligence

The platform is designed to support anomaly detection, efficiency guidance, and predictive maintenance based on simulated operational history.

---

## System Architecture

CogniBoiler follows a **microservice-oriented architecture**.

That means the platform is split into multiple focused services rather than one giant application. Each service owns a specific responsibility and communicates with others through explicit contracts.

At a conceptual level, the architecture looks like this:

- **Process simulation services** generate the operational state.
- **Control services** interpret commands and enforce operating logic.
- **Messaging services** move data through the system.
- **Historical storage services** record process behaviour over time.
- **Integration services** expose data using industrial protocols.
- **API services** provide external access for dashboards and clients.
- **AI services** consume historical and real-time data to produce higher-level insight.

This kind of structure helps keep the platform modular, understandable, and extensible.

---

## Main Components

### Physics Engine

The Physics Engine is the heart of the digital twin.

Its job is to simulate how the boiler and turbine behave over time. Instead of using random placeholder numbers, it is intended to model relationships between physical quantities such as:

- pressure;
- temperature;
- water level;
- energy balance;
- steam flow;
- turbine output.

This makes it possible to generate realistic operating scenarios and use them as the foundation for monitoring, control, and AI training.

### PLC Controller

The PLC Controller is the virtual control layer.

In real industrial environments, PLCs and related control systems are responsible for turning high-level goals into low-level actions. CogniBoiler mirrors that idea in software.

The controller layer is meant to:

- receive commands or target values;
- manage actuators such as valves;
- apply operating logic;
- respect safety constraints;
- keep the simulated process within acceptable limits.

### API Gateway

The API Gateway acts as the main external entry point.

It is responsible for exposing the platform to clients through a controlled and structured HTTP/WebSocket interface. This is where status queries, command submission, authentication, and future web integrations naturally belong.

### Historian

The Historian stores process history.

Industrial systems are heavily dependent on historical trends. You usually do not understand a boiler from a single value; you understand it from how values change over time.

The Historian is designed to:

- consume live telemetry;
- store it as time-series data;
- support trend visualisation;
- enable retrospective analysis;
- feed future AI models.

### OPC UA Server

The OPC UA Server is the industrial interoperability layer.

OPC UA is widely used in automation and industrial environments to expose machine state and structured variables. In CogniBoiler, this layer allows the digital twin to behave more like an industrial-grade system and less like a simple software-only experiment.

### Alert Manager

The Alert Manager is intended to handle abnormal events and structured alarm workflows.

Its role is to centralise operational signals such as warnings, alarms, and event records so they can be stored, reviewed, and acted upon in a disciplined way.

### AI Predictor

The AI Predictor is the intelligence layer of the platform.

It is designed to transform raw plant history into higher-level interpretation. Rather than only answering “what is the current pressure?”, it aims to answer questions like:

- Is the current behaviour normal?
- Is the system drifting away from an efficient operating point?
- Does this pattern suggest future maintenance risk?

---

## AI Layer

One of the defining ideas of CogniBoiler is that machine learning should not be an isolated add-on. It should be part of the broader operational architecture.

The planned AI layer is centered around three major classes of intelligence.

### Anomaly Detection

This part of the platform is intended to identify behaviour that deviates from learned normal operating patterns.

In practical terms, anomaly detection helps answer:

- Is the plant behaving in a way that looks unusual?
- Does this pattern resemble drift, instability, or degradation?
- Should this be surfaced as an operational concern?

### Efficiency Advisory

This part of the platform is intended to estimate or recommend operating targets that improve efficiency while respecting physical and operational constraints.

The purpose is not to replace the control system directly, but to support better decisions by highlighting more effective operating regions.

### Predictive Maintenance

This part of the platform is intended to infer component health and estimate future maintenance risk from observed behaviour over time.

Instead of only reacting when something fails, the goal is to recognise gradual degradation early enough to act before failure becomes critical.

### Why the AI Layer Matters

The AI features are especially interesting because they are designed to be trained from the digital twin itself.

That means the physics and data layers are not just for visualisation. They also create the foundation for model development, experimentation, and intelligent diagnostics.

---

## Industrial Communication and Data Flow

CogniBoiler is not just about calculations inside one Python process. It is designed as a connected system where information flows between services.

At a broad level, the expected data movement looks like this:

### Process data flow

1. The simulation layer generates system state.
2. That state is published into the internal communication backbone.
3. Storage and integration services consume the data.
4. Dashboards and APIs expose current and historical state.

### Command flow

1. A user or client issues a command through the gateway.
2. The command is validated and routed to the relevant control layer.
3. The control logic updates the process or the virtual actuators.
4. The resulting process state is propagated back through telemetry.

### AI flow

1. Historical and/or live data is collected.
2. The AI layer evaluates the behaviour.
3. The result is converted into events, recommendations, or maintenance-oriented signals.
4. Those results can be surfaced through APIs, dashboards, or alerting paths.

This architecture reflects an important engineering idea: **information is a first-class part of the system**, not an afterthought.

---

## Security and Reliability Philosophy

CogniBoiler is designed with the assumption that industrial-style systems should not be open by default.

Even though the project is still evolving, its direction clearly includes structured security principles such as:

- authentication before sensitive operations;
- role-based access control;
- separation of concerns between services;
- auditability of actions;
- secure communication between components;
- operational logging and observability.

This matters because a realistic industrial platform is not only about simulating physics. It is also about representing how real systems are governed, monitored, and protected.

Reliability is treated as part of the design as well. That includes ideas such as:

- clear service boundaries;
- explicit contracts between components;
- structured telemetry;
- historical storage;
- validation and tests;
- support for monitoring and operational visibility.

---

## Who This Project Is For

CogniBoiler can be interesting to several different audiences.

### Software engineers

Developers can use it as an example of how to build a serious multi-service system with APIs, messaging, storage, observability, and security.

### Industrial and control engineers

Automation-oriented readers can use it as a software interpretation of process control, telemetry flow, and industrial integration concepts.

### Data and ML practitioners

People working in machine learning can view it as a self-contained environment for building models on top of synthetic operational data.

### Students and learners

Anyone trying to understand what a modern industrial software system looks like can use the project as a structured learning artifact.

### Employers and technical reviewers

As a portfolio project, CogniBoiler demonstrates not only coding ability, but also architectural thinking, system design, and domain modeling.

---

## Typical Use Cases

Although the platform is not yet positioned as a finished commercial product, its architecture naturally supports several meaningful use cases.

### Educational demonstration

The project can help explain how a boiler-turbine system, control layer, communication stack, and AI layer fit together.

### Portfolio-grade systems engineering

CogniBoiler shows the ability to design and implement a complex technical system that crosses several disciplines at once.

### Simulation-driven experimentation

It can be used as a sandbox for testing control logic, telemetry designs, storage models, and AI methods without requiring a real industrial plant.

### Synthetic data generation

The project can generate realistic operational data for downstream analytics and machine learning experiments.

### Industrial software prototyping

The architecture can serve as a prototype for ideas related to digital twins, remote monitoring, predictive analytics, or process optimisation.

---

## Technology Overview

CogniBoiler combines several technology domains into one platform.

### Application and service layer

Python is used as the main implementation language, with a monorepo layout managed through `uv`.

### API and backend layer

FastAPI and related backend technologies support structured external access and service orchestration.

### Simulation and numerical computation

Scientific Python tooling supports modeling, numerical processing, and dataset generation.

### Messaging and contracts

Protobuf, gRPC, and MQTT provide typed contracts and service communication.

### Industrial integration

OPC UA brings the project closer to real automation and plant-style interoperability.

### Data storage and visualisation

Time-series and relational storage, combined with dashboarding tools, support both operational visibility and historical analysis.

### AI and model lifecycle

Machine learning frameworks and model-management tooling support future development of intelligent diagnostics and advisory systems.

The significance of this stack is not just that it is “modern”. It is that each part supports a real responsibility in the platform.

---

## Project Status

CogniBoiler is **under active development**.

It should be understood as a growing platform rather than a finished end-user product.

At this stage, the project is best described as:

- architecturally ambitious;
- technically serious;
- already structured around real service boundaries;
- still evolving toward a more complete end-to-end system.

Some parts of the platform already represent concrete implementation work, while other parts are still being expanded or refined.

For that reason, this README focuses on **what the project is**, **what it is meant to become**, and **how the architecture is organized**, rather than giving rigid production-style operating instructions.

---

## Design Principles

Several principles define the spirit of the project.

### 1. Realism over toy simplification

The platform aims to reflect how industrial software is actually structured.

### 2. Clear boundaries between responsibilities

Each service should have an understandable purpose and a controlled interface.

### 3. Data should be useful, not decorative

Telemetry, historical storage, and model outputs should all support meaningful analysis.

### 4. AI should be grounded in system behaviour

The AI layer is intended to learn from the behaviour of the simulated system rather than from disconnected synthetic randomness.

### 5. Engineering clarity matters

The system is meant to be understandable, inspectable, and explainable to humans.

### 6. Security and observability are part of the architecture

They are not “extra polish”; they are part of what makes a system realistic and trustworthy.

---

## Long-Term Direction

CogniBoiler is designed with room to grow.

The broader vision includes a platform that can eventually support:

- richer physical simulation;
- more advanced control strategies;
- stronger industrial interoperability;
- deeper observability and diagnostics;
- more capable AI-driven analysis;
- clearer operational and educational interfaces.

In other words, the long-term goal is not just to model one machine. It is to build a convincing software environment around how such a machine would be simulated, controlled, observed, and analysed in a modern engineering context.

---

## Final Note

CogniBoiler is a project about **systems thinking**.

It connects physics, software architecture, control logic, industrial communication, historical data, security, and AI into one design.

Even in its current evolving state, the project already represents a clear idea:

> a digital twin should not only imitate a machine — it should also imitate the environment around that machine: its control layer, data layer, integration layer, and decision-support layer.

That is what CogniBoiler is trying to build.
