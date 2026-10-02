# One image per Python service, each a target of this file:
#   physics-engine, plc-controller, api-gateway (also runs the migrations), historian,
#   alert-manager, opcua-server.
# Each environment is built by uv from uv.lock alone, with the workspace packages installed
# as wheels; the runtime is plain Python with that environment, the protobuf stubs and, for
# the gateway, the migrations — no uv, no sources, no development tools, not root.
# The console has its own image: apps/web/Dockerfile.

# Base images are pinned by digest; the uv image is the last uv release built on bookworm,
# the same Debian as the runtime. Rebuild on a new digest when the image scan asks for it.
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.9.30-python3.14-bookworm-slim@sha256:7cf77f594be8042dab6daa9fe326f90962252268b4f120a7f5dccce4d947e6c1
ARG PYTHON_IMAGE=python:3.14-slim-bookworm@sha256:82bc3c539b8813ada9d68c63b40158fa002f7f33de9bf3312a3dfdc0620dff56

# The manifests and the lock alone: third-party dependencies are one layer per service that
# stays cached until uv.lock or a pyproject.toml changes, and uv keeps downloaded wheels in
# a build cache, so a rebuild after a source change needs no network.
FROM ${UV_IMAGE} AS build
ENV UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_COMPILE_BYTECODE=1
WORKDIR /app
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY apps/physics-engine/pyproject.toml apps/physics-engine/pyproject.toml
COPY apps/plc-controller/pyproject.toml apps/plc-controller/pyproject.toml
COPY apps/api-gateway/pyproject.toml apps/api-gateway/pyproject.toml
COPY apps/historian/pyproject.toml apps/historian/pyproject.toml
COPY apps/alert-manager/pyproject.toml apps/alert-manager/pyproject.toml
COPY apps/opcua-server/pyproject.toml apps/opcua-server/pyproject.toml
COPY shared/observability/pyproject.toml shared/observability/pyproject.toml
COPY shared/runtime/pyproject.toml shared/runtime/pyproject.toml

FROM build AS env-physics-engine
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-workspace --package physics-engine
COPY shared/observability shared/observability
COPY shared/runtime shared/runtime
COPY apps/physics-engine apps/physics-engine
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable --package physics-engine

FROM build AS env-plc-controller
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-workspace --package plc-controller
COPY shared/observability shared/observability
COPY shared/runtime shared/runtime
COPY apps/plc-controller apps/plc-controller
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable --package plc-controller

FROM build AS env-api-gateway
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-workspace --package api-gateway
COPY shared/observability shared/observability
COPY shared/runtime shared/runtime
COPY apps/api-gateway apps/api-gateway
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable --package api-gateway

FROM build AS env-historian
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-workspace --package historian
COPY shared/observability shared/observability
COPY shared/runtime shared/runtime
COPY apps/historian apps/historian
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable --package historian

FROM build AS env-alert-manager
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-workspace --package alert-manager
COPY shared/observability shared/observability
COPY shared/runtime shared/runtime
COPY apps/alert-manager apps/alert-manager
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable --package alert-manager

FROM build AS env-opcua-server
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-workspace --package opcua-server
COPY shared/observability shared/observability
COPY shared/runtime shared/runtime
COPY apps/opcua-server apps/opcua-server
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable --package opcua-server

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
COPY shared/generated /app/shared/generated
LABEL org.opencontainers.image.source="https://github.com/dimbo1324/CogniBoiler" \
      org.opencontainers.image.licenses="Apache-2.0"

FROM runtime AS physics-engine
COPY --from=env-physics-engine /app/.venv /app/.venv
USER app
EXPOSE 50052 9100
CMD ["python", "-m", "physics_engine", "--grpc-host", "0.0.0.0", "--grpc-port", "50052", "--metrics-port", "9100", "--metrics-host", "0.0.0.0"]

FROM runtime AS plc-controller
COPY --from=env-plc-controller /app/.venv /app/.venv
USER app
EXPOSE 50051 9100
CMD ["python", "-m", "plc_controller", "--host", "0.0.0.0", "--port", "50051", "--metrics-port", "9100", "--metrics-host", "0.0.0.0"]

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
