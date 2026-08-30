.PHONY: install up down reset logs test lint format format-check typecheck check inspect-db migrate \
	eval-local eval-derived-local eval-multiturn-local eval-retrieval-matrix eval-matrix \
	eval-followup-variants eval-judge

PROFILE ?= balanced-gpt-4.1-mini
# Local (Postgres-free) evaluation. Needs only OPENAI_API_KEY in ENV_FILE plus the bundled
# SQLite database; the graph runs against an in-process LangGraph checkpointer. Reports land in
# the tracked evals/reports/ tree so completed take-home experiments remain reviewable.
ENV_FILE ?= .env.local
REPEATS ?= 3
SUITE ?= full
# Override this with a unique value for every preserved experiment. The CLIs refuse to overwrite
# an existing report or non-empty label directory.
EVAL_LABEL ?= take-home-evaluation
CONFIRM_DATA_TRANSFER ?=

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

# --- Local, Postgres-free evaluation (no Docker) --------------------------------------------
eval-local:
	uv run python -m knowledge_assistant.evals run --suite $(SUITE) --profile $(PROFILE) --env-file $(ENV_FILE) --output evals/reports/$(EVAL_LABEL).json

eval-derived-local:
	uv run python -m knowledge_assistant.evals run --suite derived --profile $(PROFILE) --env-file $(ENV_FILE) --output evals/reports/$(EVAL_LABEL).json

eval-multiturn-local:
	uv run python -m knowledge_assistant.evals run --suite multiturn --profile $(PROFILE) --env-file $(ENV_FILE) --output evals/reports/$(EVAL_LABEL).json

# Semantic take-home score against the assignment answers. This sends questions, retrieved
# evidence, generated answers, and references to the configured providers. Require an explicit
# per-command acknowledgement instead of treating ordinary evaluation access as authorization.
eval-judge:
	$(if $(filter YES,$(CONFIRM_DATA_TRANSFER)),,$(error Re-run with CONFIRM_DATA_TRANSFER=YES only after the expanded transfer is authorized))
	uv run python -m knowledge_assistant.evals judge --label $(EVAL_LABEL) --suite $(SUITE) --profiles $(PROFILE) --env-file $(ENV_FILE) --confirm-data-transfer

# Focused deterministic model screen on the gold inputs; semantic selection uses eval-judge.
eval-matrix:
	uv run python -m knowledge_assistant.evals matrix --label $(EVAL_LABEL) --suite $(SUITE) --repeats $(REPEATS) --model-matrix --env-file $(ENV_FILE)

# Fixed-model retrieval screening: global BM25 control versus first-pass diversification.
eval-retrieval-matrix:
	uv run python -m knowledge_assistant.evals matrix --label $(EVAL_LABEL) --suite $(SUITE) --repeats $(REPEATS) --retrieval-matrix --env-file $(ENV_FILE)

# Follow-up routing prompt variants across a short profile list.
eval-followup-variants:
	uv run python -m knowledge_assistant.evals matrix --label $(EVAL_LABEL) --suite smoke --repeats $(REPEATS) --profiles $(PROFILE) --follow-up-variants current,latest_agent_context --env-file $(ENV_FILE)
