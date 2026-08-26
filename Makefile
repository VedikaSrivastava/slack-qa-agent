.PHONY: install run up down reset logs test lint format format-check typecheck check inspect-db migrate eval-smoke eval-full eval-sync eval-experiment eval-augment ask

PROFILE ?= balanced-gpt-4.1-mini
PROTOCOL ?= screening

install:
	uv sync --all-groups

run:
	uv run uvicorn knowledge_assistant.main:app --reload --port 8000

up:
	docker compose --env-file .env.local up --build

down:
	docker compose --env-file .env.local down

reset:
	docker compose --env-file .env.local down --volumes

logs:
	docker compose --env-file .env.local logs --follow app inngest postgres

test:
	uv run pytest --cov=knowledge_assistant --cov-report=term-missing

lint:
	uv run ruff check .

format:
	uv run ruff format .

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
	docker compose --env-file .env.local run --rm app python -m knowledge_assistant.evals run --suite smoke --profile $(PROFILE) --output /app/evals/results/smoke-$(PROFILE).json

eval-full:
	docker compose --env-file .env.local run --rm app python -m knowledge_assistant.evals run --suite full --profile $(PROFILE) --output /app/evals/results/full-$(PROFILE).json

eval-sync:
	docker compose --env-file .env.local run --rm app python -m knowledge_assistant.evals sync

eval-experiment:
	docker compose --env-file .env.local run --rm app python -m knowledge_assistant.evals experiment --profile $(PROFILE) --protocol $(PROTOCOL) --output /app/evals/results/langsmith-$(PROFILE)-$(PROTOCOL).json

eval-augment:
	docker compose --env-file .env.local run --rm app python -m knowledge_assistant.evals augment --per-case 2

ask:
	docker compose --env-file .env.local run --rm app python -m knowledge_assistant.cli ask "$(Q)"
