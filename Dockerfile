# One image for every Python service; each Compose service picks its module with
# `python -m`. uv is used only at build time: the runtime never resolves or syncs.
FROM ghcr.io/astral-sh/uv:python3.14-bookworm-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PATH="/app/.venv/bin:${PATH}" \
    PYTHONPATH="/app/shared/generated"

WORKDIR /app

COPY . .

RUN uv sync --all-packages --frozen --no-dev \
    && useradd --system --uid 10001 --home-dir /app --shell /usr/sbin/nologin app \
    && chown -R app:app /app

USER app
