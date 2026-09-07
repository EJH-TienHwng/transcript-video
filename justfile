set shell := ["powershell", "-NoLogo", "-NoProfile", "-Command"]

install:
    python -m uv sync

format:
    python -m uv run ruff format .
    python -m uv run ruff check --fix .

lint:
    python -m uv run ruff format --check .
    python -m uv run ruff check .

test:
    python -m uv run pytest

test-fast:
    python -m uv run pytest -m "not slow and not gpu and not integration"

check: lint test

doctor:
    python -m uv run transcript-video doctor

tui:
    python -m uv run transcript-video course tui

wizard:
    python -m uv run transcript-video course create

test-tui:
    python -m uv run pytest tests/test_tui.py

coverage:
    python -m uv run pytest -m "not slow and not gpu" --cov=transcript_video --cov-report=term-missing
