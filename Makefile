.PHONY: agent test lint format typecheck install

install:
	uv sync

agent:
	uv run python -m agent.main

test:
	uv run pytest tests/unit

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run pyright

format:
	uv run ruff format .
	uv run ruff check --fix .

typecheck:
	uv run pyright
