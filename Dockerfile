# syntax=docker/dockerfile:1
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# Install dependencies in a separate layer for build caching
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-dev

# Copy application source tree and install project
COPY . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# ---------------------------------------------------------------------------
# Runner Stage
# ---------------------------------------------------------------------------
FROM python:3.11-slim-bookworm AS runner

# System libraries required by opencv-contrib-python and curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Run as dedicated non-root user
RUN groupadd -g 10001 cryoguard && \
    useradd -u 10001 -g cryoguard -s /bin/bash -m cryoguard

WORKDIR /app

# Copy application and virtualenv from builder stage
COPY --from=builder --chown=cryoguard:cryoguard /app /app

# Ensure artifacts directory is writeable by cryoguard user
RUN mkdir -p /app/artifacts/incidents/dashboard && \
    chown -R cryoguard:cryoguard /app/artifacts

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src" \
    PYTHONUNBUFFERED=1

USER cryoguard

EXPOSE 8080

HEALTHCHECK --interval=15s --timeout=5s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/api/manifest')" || exit 1

ENTRYPOINT ["python", "-m"]
CMD ["cryoguard.server", "--host", "0.0.0.0", "--port", "8080", "--scenario", "leak"]
