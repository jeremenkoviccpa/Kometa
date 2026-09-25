# Kometa server image (Railway or any container host). Runs `at demo run` on 0.0.0.0:$PORT.
# State (journal, risk state, registry, audit) lives under /data: mount a persistent volume there.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

# dependencies first (cached layer), then the workspace itself
COPY pyproject.toml uv.lock ./
COPY packages ./packages
RUN uv sync --locked --no-dev

COPY . .
RUN uv sync --locked --no-dev
ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8080
CMD ["sh", "scripts/serve.sh"]
