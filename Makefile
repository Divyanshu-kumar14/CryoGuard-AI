PY := .venv/bin/python
RUFF := .venv/bin/ruff
MYPY := .venv/bin/mypy

.PHONY: setup lint format typecheck test check export

## Create the pinned environment from uv.lock. uv owns the venv; .python-version pins 3.11.
setup:
	uv sync --all-extras --frozen

## Match CI exactly: no autofix, no reformat.
lint:
	$(RUFF) check .
	$(RUFF) format --check .

format:
	$(RUFF) format .
	$(RUFF) check --fix .

typecheck:
	$(MYPY)

test:
	$(PY) -m pytest -q

check: lint typecheck test

## Judges get a plain requirements file; uv.lock remains the source of truth.
export:
	uv export --format requirements-txt --no-hashes --no-emit-project -o requirements.txt
