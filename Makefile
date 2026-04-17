.PHONY: agent mock-payer test lint format typecheck install

install:
	uv sync

agent:
	uv run python -m agent.main

mock-payer:
	uv run uvicorn mock_payer.main:app --host 0.0.0.0 --port 8080

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
