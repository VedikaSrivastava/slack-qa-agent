.PHONY: install run test lint format typecheck check inspect-db

install:
	uv sync --all-groups

run:
	uv run uvicorn slack_qa_agent.main:app --reload --port 8000

test:
	uv run pytest --cov=slack_qa_agent --cov-report=term-missing

lint:
	uv run ruff check .

format:
	uv run ruff format .

typecheck:
	uv run mypy src tests

check: lint typecheck test

inspect-db:
	uv run python scripts/inspect_database.py
