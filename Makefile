# Shortcuts only. Every target calls the script orchestrator, which is the real,
# cross-platform implementation: `python dev_tools_scripts_runner.py list` shows it all.

PYTHON ?= python
RUN := $(PYTHON) dev_tools_scripts_runner.py

.PHONY: help install secrets gate gate-quick format format-check agents proto up up-infra status down smoke doctor hooks clean selftest web-install web-dev

.DEFAULT_GOAL := help

help:
	@$(RUN) list

install:
	uv sync --all-packages

web-install:
	pnpm --dir apps/web install

web-dev:
	pnpm --dir apps/web dev

secrets:
	$(RUN) dev-secrets

gate:
	$(RUN) quality-gate

gate-quick:
	$(RUN) quality-gate --quick

format:
	$(RUN) format-code

format-check:
	$(RUN) format-code --check

agents:
	$(RUN) sync-agents

proto:
	$(RUN) generate-proto

up:
	$(RUN) stack up

up-infra:
	$(RUN) stack up --infra-only

status:
	$(RUN) stack status

down:
	$(RUN) stack down

smoke:
	$(RUN) smoke

doctor:
	$(RUN) doctor

hooks:
	$(RUN) install-hooks

clean:
	$(RUN) clean-caches

selftest:
	$(RUN) selftest
