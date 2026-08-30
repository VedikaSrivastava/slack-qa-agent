# syntax=docker/dockerfile:1.7
FROM ghcr.io/astral-sh/uv:0.12.0 AS uv

FROM python:3.12-slim-bookworm AS builder
COPY --from=uv /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable

FROM builder AS validation
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --all-groups --no-editable
COPY tests ./tests
COPY alembic ./alembic
COPY scripts ./scripts
# Keep the assignment fixture out of the runtime image, but include it here so the supplied-schema
# integration test cannot silently skip during image validation.
COPY data/synthetic_startup.sqlite ./data/synthetic_startup.sqlite
RUN uv run ruff check . \
    && uv run ruff format --check . \
    && uv run mypy src tests \
    && uv run pytest -m "not integration" \
    && uv run pytest -m integration

FROM python:3.12-slim-bookworm AS runtime
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --home-dir /app app
WORKDIR /app
COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --chown=app:app alembic ./alembic
COPY --chown=app:app alembic.ini pyproject.toml README.md ./
RUN mkdir -p /app/data && chown app:app /app/data
USER app
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=10s --retries=5 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2)" || exit 1
CMD ["uvicorn", "knowledge_assistant.main:app", "--host", "0.0.0.0", "--port", "8000"]
