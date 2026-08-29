.PHONY: install up down reset logs test lint format format-check typecheck check inspect-db migrate \
	eval-smoke eval-full eval-follow-up-workflow \
	eval-local eval-derived-local eval-multiturn-local eval-retrieval-matrix eval-matrix \
	eval-followup-variants

PROFILE ?= balanced-gpt-4.1-mini
ROUTING_PROMPT_VARIANT ?= current
# Local (Postgres-free) evaluation. Needs only OPENAI_API_KEY in ENV_FILE plus the bundled
# SQLite database; the graph runs against an in-process LangGraph checkpointer. Reports land in
# the tracked evals/reports/ tree so completed take-home experiments remain reviewable.
ENV_FILE ?= .env.local
REPEATS ?= 3
SUITE ?= full

install:
	uv sync --frozen --all-groups

up:
	docker compose --env-file .env.local up --build

down:
	docker compose --env-file .env.local down

reset:
	docker compose --env-file .env.local down --volumes

logs:
	docker compose --env-file .env.local logs --follow app slack-ingress migrate inngest postgres

test:
	uv run pytest --cov=knowledge_assistant --cov-report=term-missing

lint:
	uv run ruff check .

format:
	uv run ruff format . --exclude DESIGN.md

format-check:
	uv run ruff format --check .

typecheck:
	uv run mypy src tests

check: lint format-check typecheck test

inspect-db:
	uv run python scripts/inspect_database.py

migrate:
	docker compose --env-file .env.local run --rm migrate

eval-smoke:
	docker compose --env-file .env.local run --rm eval python -m knowledge_assistant.evals run --suite smoke --profile $(PROFILE) --output /app/evals/reports/smoke-$(PROFILE).json

eval-full:
	docker compose --env-file .env.local run --rm eval python -m knowledge_assistant.evals run --suite full --profile $(PROFILE) --output /app/evals/reports/full-$(PROFILE).json

eval-follow-up-workflow:
	docker compose --env-file .env.local run --rm eval python -m knowledge_assistant.evals follow-up-workflow --profile $(PROFILE) --prompt-variant $(ROUTING_PROMPT_VARIANT) --output /app/evals/reports/follow-up-workflow-$(ROUTING_PROMPT_VARIANT)-$(PROFILE).json

# --- Local, Postgres-free evaluation (no Docker) --------------------------------------------
eval-local:
	uv run python -m knowledge_assistant.evals run --suite $(SUITE) --profile $(PROFILE) --env-file $(ENV_FILE) --output evals/reports/$(SUITE)-$(PROFILE).json

eval-derived-local:
	uv run python -m knowledge_assistant.evals run --suite derived --profile $(PROFILE) --env-file $(ENV_FILE) --output evals/reports/derived-$(PROFILE).json

eval-multiturn-local:
	uv run python -m knowledge_assistant.evals run --suite multiturn --profile $(PROFILE) --env-file $(ENV_FILE) --output evals/reports/multiturn-$(PROFILE).json

# Focused five-profile model matrix on the gold suite, REPEATS runs each, one rollup report.
eval-matrix:
	uv run python -m knowledge_assistant.evals matrix --label model-matrix-$(SUITE) --suite $(SUITE) --repeats $(REPEATS) --model-matrix --env-file $(ENV_FILE)

# Fixed-model retrieval screening: global BM25 control versus first-pass diversification.
eval-retrieval-matrix:
	uv run python -m knowledge_assistant.evals matrix --label retrieval-matrix-$(SUITE) --suite $(SUITE) --repeats $(REPEATS) --retrieval-matrix --env-file $(ENV_FILE)

# Follow-up routing prompt variants across a short profile list.
eval-followup-variants:
	uv run python -m knowledge_assistant.evals matrix --label followup-variants --suite smoke --repeats $(REPEATS) --profiles $(PROFILE) --follow-up-variants current,latest_agent_context --env-file $(ENV_FILE)
