# One image per Python service, each a target of this file:
#   physics-engine, plc-controller, api-gateway (also runs the migrations), historian,
#   alert-manager, opcua-server.
# Each environment is built by uv from uv.lock alone, with the workspace packages installed
# as wheels; the runtime is plain Python with that environment, the protobuf stubs and, for
# the gateway, the migrations — no uv, no sources, no development tools, not root.
# The console has its own image: apps/web/Dockerfile.

ARG UV_IMAGE=ghcr.io/astral-sh/uv:python3.14-bookworm-slim
ARG PYTHON_IMAGE=python:3.14-slim-bookworm

FROM ${UV_IMAGE} AS build
ENV UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_COMPILE_BYTECODE=1
WORKDIR /app
COPY . .

FROM build AS env-physics-engine
RUN uv sync --frozen --no-dev --no-editable --package physics-engine

FROM build AS env-plc-controller
RUN uv sync --frozen --no-dev --no-editable --package plc-controller

FROM build AS env-api-gateway
RUN uv sync --frozen --no-dev --no-editable --package api-gateway

FROM build AS env-historian
RUN uv sync --frozen --no-dev --no-editable --package historian

FROM build AS env-alert-manager
RUN uv sync --frozen --no-dev --no-editable --package alert-manager

FROM build AS env-opcua-server
RUN uv sync --frozen --no-dev --no-editable --package opcua-server

FROM ${PYTHON_IMAGE} AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:${PATH}" \
    PYTHONPATH="/app/shared/generated"
# Security updates the base image does not have yet; pip is removed because nothing
# installs packages at runtime and its vendored libraries are scanner findings.
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get upgrade --yes --no-install-recommends \
    && rm -rf /var/lib/apt/lists/* \
    && python -m pip uninstall --yes --root-user-action=ignore pip \
    && useradd --system --uid 10001 --home-dir /app --shell /usr/sbin/nologin app
WORKDIR /app
COPY --from=build /app/shared/generated /app/shared/generated
LABEL org.opencontainers.image.source="https://github.com/dimbo1324/CogniBoiler" \
      org.opencontainers.image.licenses="Apache-2.0"

FROM runtime AS physics-engine
COPY --from=env-physics-engine /app/.venv /app/.venv
USER app
EXPOSE 50052 9100
CMD ["python", "-m", "physics_engine", "--grpc-port", "50052", "--metrics-port", "9100", "--metrics-host", "0.0.0.0"]

FROM runtime AS plc-controller
COPY --from=env-plc-controller /app/.venv /app/.venv
USER app
EXPOSE 50051 9100
CMD ["python", "-m", "plc_controller", "--port", "50051", "--metrics-port", "9100", "--metrics-host", "0.0.0.0"]

FROM runtime AS api-gateway
COPY --from=env-api-gateway /app/.venv /app/.venv
COPY apps/api-gateway/alembic.ini /app/apps/api-gateway/alembic.ini
COPY apps/api-gateway/migrations /app/apps/api-gateway/migrations
USER app
EXPOSE 8000
CMD ["python", "-m", "api_gateway", "--host", "0.0.0.0", "--port", "8000"]

FROM runtime AS historian
COPY --from=env-historian /app/.venv /app/.venv
USER app
EXPOSE 9100
CMD ["python", "-m", "historian", "--metrics-port", "9100", "--metrics-host", "0.0.0.0"]

FROM runtime AS alert-manager
COPY --from=env-alert-manager /app/.venv /app/.venv
USER app
EXPOSE 50053 9100
CMD ["python", "-m", "alert_manager", "--grpc-port", "50053", "--metrics-port", "9100", "--metrics-host", "0.0.0.0"]

FROM runtime AS opcua-server
COPY --from=env-opcua-server /app/.venv /app/.venv
USER app
EXPOSE 4840 9100
CMD ["python", "-m", "opcua_server", "--opc-port", "4840", "--metrics-port", "9100", "--metrics-host", "0.0.0.0"]
