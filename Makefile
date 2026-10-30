PY := .venv/bin/python
RUFF := .venv/bin/ruff
MYPY := .venv/bin/mypy

SCENARIO ?= leak
SPLIT ?= all

.PHONY: setup lint format typecheck test check export fixtures probe serve demo sims bench bench-check seed-sam checkpoints

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

## Regenerate the SIMULATED fixture clips and their manifest.
fixtures:
	PYTHONPATH=src $(PY) scripts/make_sim_fixtures.py

## Measure the tracker on one clip, e.g. make probe CLIP=fixtures/videos/simulated/sim_leak_qd4.mp4
probe:
	PYTHONPATH=src $(PY) scripts/probe_clip.py $(CLIP)

## End-to-end vertical slice: clip -> incident JSON.
demo:
	PYTHONPATH=src $(PY) -m cryoguard.app --scenario $(SCENARIO) --headless

## Interactive dashboard on http://127.0.0.1:8080. SCENARIO=leak auto-runs on startup.
serve:
	PYTHONPATH=src $(PY) -m cryoguard.server --scenario $(SCENARIO)

## Start the SIMULATED PLC (Modbus-TCP 5020) and BMC (Redfish 8443).
sims:
	PYTHONPATH=src $(PY) -m cryoguard.sims.run

## Deterministic offline benchmark over every manifest clip (P8).
bench:
	PYTHONPATH=src $(PY) -m cryoguard.bench --split $(SPLIT)

## Re-run the benchmark and fail if the committed runHash or rows drifted.
bench-check:
	PYTHONPATH=src $(PY) -m cryoguard.bench --split $(SPLIT) --check

## Seed the deterministic SAM cache (fixture provenance, never a hosted call).
seed-sam:
	for s in leak condensation glint hidden; do \
		PYTHONPATH=src $(PY) scripts/seed_sam_cache.py --scenario $$s; \
	done

## Emit deterministic training checkpoints for the checkpoint gate.
checkpoints:
	PYTHONPATH=src $(PY) -m cryoguard.sims.train_job --duration 30
