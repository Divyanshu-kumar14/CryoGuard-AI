# CryoGuard AI — Phase-Wise Implementation Plan

**Derived from:** `specs/CryoGuard AI — Product Requirements Document (Build Spec for Coding Agent).md` (v1.0, 2026-10-30)
**Target:** Meta Global AI Developer Hackathon, Indie track — solo build
**Hard deadline:** Wed **2026-11-04, 21:00 ET** — submit by **19:00 ET** (2 h buffer)
**Plan author:** orchestrator · **Status:** active

---

## 0. How to use this plan

1. Phases run in order. **Do not start phase N+1 until phase N's exit criteria are green.** The only exception is explicitly marked parallel work.
2. Every task has an ID (`P3-T04`), target files, and a verify step. A task is done only when its verify command has been run and its output pasted into the commit message or PR note.
3. **No P2 work while any P0/P1 item is red** (PRD §0).
4. `main` must always have a working `make demo` and `make bench` (PRD §0.7). If a phase would break them, land the phase behind a config flag until green.
5. Commit after each task group. Conventional commits, small diffs, no WIP on `main`.

### 0.1 Input gap — closed

The PRD's §7–§14 arrived as `CryoGuard AI — PRD v1.1 (complete).md` and are now canonical. `prd-delta-v1.2.md` records the eleven conflicts inside that document (C-01…C-11), the red-team of its fusion weights (R-1…R-3), and the parameters its `FR-FUS-01` still leaves undefined.

- **Binding source of truth:** complete PRD §7–§14 → `prd-delta-v1.2.md` for the conflicts it resolves → this plan. The archived addendum v1.1 (`specs/archive/`) is superseded — its invented §7–§14 must not be used.
- **Acceptance criteria:** PRD §14 is the floor; delta §7 adds the six tests the pitch depends on (`T-LLM`, `T-DET`, `T-DEC-09`, `T-FSM-01`, `T-BEN-05`, `T-DOC-01`). §15 of this plan holds the working list.
- **Remaining Day-1 unknown:** whether the organisers' Muse Spark Open endpoint matches the community-cited Baseten model id (P0-T01/P0-T03). This is external and cannot be closed on paper.

---

## 1. Non-negotiables: engineering rules and stack

### 1.1 Rules

These are gates, not guidelines. Each maps to a test or a CI check.

| Rule | Enforcement |
| :-- | :-- |
| No fabricated numbers | All UI/README metrics read from `reports/benchmark.json` or live runtime state. Grep gate in CI for hard-coded probabilities in `ui/` and `README.md`. |
| Label simulation honestly | `SIMULATED` badge component required on every simulator-backed panel; README real-vs-simulated table is a P0 test fixture (`tests/test_docs_contract.py` asserts the table rows exist). |
| LLM never in safety loop | Architecture test: no import of `cryoguard.report.*` from `fusion/`, `orchestrator/`, `adapters/`. `deterministicActionsTaken` assembled before the report task is scheduled. |
| Everything tunable in config | No numeric literal in decision/FSM modules except `0`, `1`, bounds. Lint rule + review. Config hash logged at startup and written into every incident. |
| Determinism | SAM cache key = `sha256(image_bytes + prompt + model)`; bench writes a `run_hash` over its JSON output; CI asserts two runs produce identical bytes. |
| Vertical slice first | Phase 2 hard gate: one fixture reaches incident JSON by end of Day 2, crude is acceptable. |
| Small commits, always-green `main` | Pre-push hook runs `ruff` + `pytest -m "not slow"`. |
| Secrets via env | `.env.example` committed; `.env` ignored; startup fails loudly if a required key is missing **only** when the live path is enabled. |

### 1.2 Tech stack — final decisions

The PRD fixes the stack (§5). That stack is right for this build: one asyncio process, a single HTML page, real protocols, no training. The changes below are corrections, not preferences — each one removes a known failure mode.

**Runtime and tooling**

| Layer | Decision | Why | Rejected |
| :-- | :-- | :-- | :-- |
| Python | **3.11, pinned** (`.python-version`) | Every dep has a tested wheel; 3.12 buys ~nothing here because the bottlenecks are API latency and `cv2`, not the interpreter. Do not spend Day 1 chasing wheels. | 3.12/3.13 (unpaid risk) |
| Installer | **`uv` + committed `uv.lock`**, `uv sync --frozen` in CI | Reproducible judge/CI installs in seconds; `pip` has no lockfile and slow cold installs. | pip + requirements only (keep `requirements.txt` as an export for judges without `uv`) |
| Lint/format | **ruff** (lint + format), one tool | Replaces black/isort/flake8; single config, fast. | black + isort + flake8 |
| Types | **mypy**, scoped strict: `fusion/`, `orchestrator/`, `adapters/`, `report/schema.py` | These four carry the safety argument; type errors there are incidents. Rest of the tree stays loose to keep velocity. | mypy whole repo (Day-1 sink), Pyright (needs node) |
| Tests | pytest + **pytest-asyncio** + **pytest-timeout** + **pytest-cov** + **hypothesis** | `hypothesis` gives the rule-totality property in ~15 lines: *every* `(P, zone, ticks, health, gates)` tuple matches exactly one rule. Catches boundary holes a hand table misses. | unittest, nose |
| CI | GitHub Actions + `uv` cache: `ruff` → `mypy` → `pytest -m "not slow"` → `make bench --check` | Matches the PRD "always green `main`" rule. | pre-commit only (no gate on the remote) |

**Application**

| Layer | Decision | Why | Rejected |
| :-- | :-- | :-- | :-- |
| Config | pydantic v2 + **pydantic-settings** + PyYAML, `yaml.safe_load` | Env + YAML + validation with one model per section; `config_hash` falls out of `model_dump_json()`. | hand-rolled dicts, dynaconf |
| HTTP/API | FastAPI + **`uvicorn[standard]`** (uvloop, websockets) | Native async, WS, static files. Nothing else needed. | Flask (sync), Litestar (no gain) |
| Vision | **`opencv-contrib-python` only** — never alongside `opencv-python` | Both ship the `cv2` namespace and silently clobber each other. `contrib` is a superset and includes ArUco (`cv2.aruco`). | `opencv-python` + `opencv-contrib-python` (conflict), scikit-image (slower, no ArUco) |
| Numerics | numpy + scipy (`scipy.stats.beta.ppf` for Clopper–Pearson) | One function, exact intervals. | statsmodels (heavy for one call), hand-rolled bisection (fine but no gain) |
| GPU | **`nvidia-ml-py`** (imports as `pynvml`), optional import | PyPI `pynvml` is deprecated; this is the official NVML binding. `Real` backend is P1 only. | `pynvml` (deprecated), `py3nvml` (stale) |
| Modbus | `pymodbus` 3.x, **exact pin** (`==3.11.*` at install time) | Server API moved between 3.x minors; an unpinned minor can break the PLC sim on demo day. | `pymodbus` unpinned |
| LLM clients | `openai` SDK v1, `base_url` overridden per endpoint | One client for both Meta `…/v1/responses` and the Baseten-compatible Muse endpoint. | raw httpx (loses typed errors/retries), LangChain (huge, no value) |
| Concurrency | Single process, asyncio, `asyncio.Queue` bus | PRD §5 process model. 4 GPUs, one rack. | Celery/Redis/RabbitMQ (deploy surface, zero benefit), threads |
| Persistence | JSON files (`artifacts/incidents/`) + SAM cache PNG/JSON | Diffable, judge-readable, no schema migration, no server. | SQLite/Postgres (no query need), Redis (a cache we already have on disk) |
| Time | `datetime` UTC only, ISO-8601 `Z`; `Clock` protocol for sim time | Determinism in bench; one formatting path. | `arrow`/`pendulum` (dep for `strftime`), naive datetimes |

**Frontend**

Single `ui/index.html` + ES modules + `ui/app.js` with `// @ts-check` and JSDoc types + Canvas 2D + native `WebSocket` + a **vendored** compiled Tailwind CSS (`ui/tailwind.css`). **No framework, no bundler, no node_modules, no CDN** — `FR-DSH-01` allows pre-compiled CSS, and a CDN request during the recorded demo is a needless failure mode (delta C-11).

This is the deliberate "worst-sounding, best" call: judges clone and run, and one screen with one canvas does not justify React, Vite, and a Node toolchain that the Python-only PRD never asks for. `// @ts-check` recovers most editor type safety for free.

**Explicitly not installed:** pandas, torch, ultralytics, `kubernetes`, Docker, React/Next/Vite, Redis, SQLite, LangChain, structlog, `tenacity`. Each adds surface without moving a P0.

**Install smoke test (P0-T04 must prove, not assume):** after `uv sync --frozen`, one command imports `cv2` (with `cv2.aruco`), `pymodbus`, `fastapi`, `pydantic`, `scipy`, `numpy`, `openai`, `yaml` and asserts the pinned versions, and imports `pynvml` only if present. Any wheel failure ⇒ drop to Python 3.11.x latest, never to a different library.

---

## 2. Calendar and milestones

Assumes solo, ~11 h/day, local clock ≈ ET (adjust the hour blocks, not the order, if not).

| Day | Date | Phases | Milestone |
| :-- | :-- | :-- | :-- |
| 1 | Fri 2026-10-30 | P0, P1, start P2 | **M0** repo boots, SAM + Muse spikes answered, one clip rectified |
| 2 | Sat 2026-10-31 | P2 → P3 | **M1 (hard gate)** `make demo SCENARIO=leak` writes a schema-valid incident JSON from video |
| 3 | Sun 2026-11-01 | P4, P5 | **M2** sims live; fusion + full decision table under test |
| 4 | Mon 2026-11-02 | P6, P7 | **M3** closed loop: clamp→valve→checkpoint→requeue via real protocols; report via Muse |
| 5 | Tue 2026-11-03 | P8, P9, P10 | **M4** `make bench` offline-deterministic; dashboard live; 3 scenarios replay |
| 6 | Wed 2026-11-04 | P10 finish, P11 | **M5** video ≤3:00, README, submission filed by 19:00 ET |

**Time budget (P0 only):** 3 + 4 + 8 + 6 + 6 + 6 + 6 + 3 + 4 + 4 ≈ 50 h, plus demo/README/submission 6 h. Buffer is deliberately thin: any slip in P1 or P2 must be recovered by cutting P1 scope, never by cutting the P0 chain.

### 2.1 Scope ladder (what gets cut, in order)

1. Cut first: Kubernetes stub, multi-rack, SWIR documentation, webcam live mode (file replay only is enough for the demo).
2. Cut second: real NVML backend (Fake backend is P0; Real is P1 and only if an NVIDIA GPU is present).
3. Cut third: UI polish, canvas trails, report streaming animation.
4. **Never cut:** clamp-first ordering, LLM-out-of-loop test, SAM cache determinism, honest labeling, benchmark CIs, three scenarios.

---

## 3. Phase 0 — Preflight and guardrails (Day 1 AM · ~3 h)

**Goal:** eliminate unknowns that can invalidate the architecture, and stand up the skeleton so every later task has a home.

| ID | Task | Files | Verify |
| :-- | :-- | :-- | :-- |
| P0-T01 | Confirm hackathon facts against the official guide: SAM 3.1 endpoint/model id, text-only prompt limit (32 tokens), Meta credit balance, Muse Spark Open partner model id (`meta-models/muse-spark-open-1026` on Baseten), open-weights verification procedure. Record findings + URLs in `specs/preflight.md`. | `specs/preflight.md` | Each item has a source URL or an explicit `UNCONFIRMED` marker |
| P0-T02 | **SAM spike:** one real image, one ≤32-token text prompt, decode response into a boolean mask. Time the round trip. Save raw JSON. | `/tmp/opencode/sam_spike.py` | Script prints `mask_shape`, `area_px`, `latency_ms` |
| P0-T03 | **Muse spike:** one prompt against the partner endpoint; confirm the model returns JSON and what it costs/limits. Save request+response for `artifacts/open_weights_proof.jsonl`. | `/tmp/opencode/muse_spike.py` | Valid JSON back; log line written |
| P0-T04 | Lock the stack exactly as §1.2: pinned Python 3.11 (`.python-version`), `uv` with committed `uv.lock`, `pyproject.toml` (ruff/mypy/pytest config), `requirements.txt` export for judges, `Makefile` targets (`setup`, `lint`, `format`, `typecheck`, `test`, `acceptance`, `demo`, `bench`, `sims`, `ci`), `.env.example`, `.gitignore`. Run the **install smoke test** from §1.2: import `cv2` + `cv2.aruco`, `pymodbus`, `fastapi`, `pydantic`, `scipy`, `numpy`, `openai`, `yaml`; assert pinned versions; import `pynvml` only if present. Never install `opencv-python` alongside `opencv-contrib-python`. | root files | `make setup && make lint && make typecheck && make test` exit 0; smoke script prints each version and `cv2.aruco` availability |
| P0-T05 | Skeleton packages exactly as PRD §5 layout with `__init__.py`, plus `configs/default.yaml`, `configs/topology.yaml`, `configs/scenarios/{leak,condensation,hidden}.yaml`. | `src/cryoguard/**`, `configs/**` | `python -c "import cryoguard"` |
| P0-T06 | Canonical IDs as a single module (`models.py` enums + `configs/topology.yaml`): `RACK-08B`, `BRANCH-SEC-02`, `QD-01..08`, `gpu-node-08b`, `gpu-node-09a`, `train-llm-0427`, `INC-RACK-08B-YYYYMMDD-NN`. **No ID string may be typed twice.** (D-17) | `src/cryoguard/models.py`, `configs/topology.yaml` | `tests/test_identifiers.py` asserts every ID comes from config/enum |
| P0-T07 | Core plumbing: async `EventBus` (topics + typed events), `Clock` abstraction (sim time for bench, wall time for demo), `Config` loader with `config_hash`. | `bus.py`, `clock.py`, `config.py` | `tests/test_bus.py`, `tests/test_config_hash.py` |
| P0-T08 | CI (GitHub Actions, `uv` cache): `ruff check` + `ruff format --check` → `mypy` (scoped) → `pytest -m "not slow"` → `make demo --headless --dry-run`. | `.github/workflows/ci.yml` | Workflow green on the first push |
| P0-T09 | **Reconcile the complete PRD.** Walk delta v1.2 §2 (C-01…C-11), accept or reject each, and pin every value in delta §4 into `configs/default.yaml` + `configs/topology.yaml`. Write the resolved values into `specs/preflight.md` with the config hash. | `configs/*.yaml`, `specs/preflight.md` | `python -m cryoguard.config --check` prints a hash and rejects unknown keys; no symbol from `FR-FUS-01` is left undefined |

**Exit criteria:** both spikes answered in writing; `make setup lint test` green; config hash printed at startup; CI running.

**Kill-switch:** if SAM access fails, Phase 3 falls back to the HSV path only and G2 is lost — escalate to the user immediately, do not fake it.

---

## 4. Phase 1 — Fixtures and rectification (Day 1 PM · ~4 h)

**Goal:** the only real signal in the project exists on disk, calibrated, with ground truth. Everything else is code.

> This is the long pole. Camera work cannot be compressed later and cannot be simulated without destroying the product claim.

| ID | Task | Files | Verify |
| :-- | :-- | :-- | :-- |
| P1-T01 | Capture rig: dark enclosure, 365–395 nm excitation, long-pass filter on the camera, fluorescent tracer mixed into a water reservoir. Shoot **same framing** for all three scenarios. | `fixtures/videos/raw/` | 3 clips ≥20 s each, visible fluorescence in `leak` |
| P1-T02 | Print and mount ArUco board coplanar with the chassis face; shoot a calibration clip per camera setup. | `fixtures/videos/raw/calib_*.mp4` | Detection rate ≥95% of frames |
| P1-T03 | `rectify.py`: ArUco detect → homography → chassis-plane mm frame; emit the matrix + reprojection error. **Planar assumption and its limits documented.** (D-19) | `src/cryoguard/capture/rectify.py` | `tests/test_rectify.py`: synthetic board → known mm points within 1 mm |
| P1-T04 | `source.py`: file-replay source with deterministic frame timing (no wall-clock drift), `--fps` override, frame timestamps in the incident. | `capture/source.py` | Two replays produce identical timestamps |
| P1-T05 | Zone polygons in `topology.yaml` in **chassis-plane mm**, Zone 2 includes the 20 mm margin baked in. Zone 0 = QD manifold/hose, Zone 1 = tray/gutter, Zone 2 = busbar/VRM/PCIe envelope. | `configs/topology.yaml` | `tests/test_zones.py`: ≥5%-area rule, `max` zone index wins (PRD §6.1) |
| P1-T06 | `fixtures/manifest.yaml` + labels: per clip `{scenario, camera, fps, calibration, ground_truth_label, qd_origin, expected_zone, notes}`. Labels are hand-authored, timestamped. | `fixtures/manifest.yaml`, `fixtures/labels/` | Manifest loads; every clip has a label; test asserts no clip is unlabeled |
| P1-T07 | Freeze dev/test split **now**: ~60/40 per class, no augmented copy counted as independent (D-12). Write the split into `fixtures/manifest.yaml`. | manifest | Split file committed; test asserts disjoint sets |

**Exit criteria:** three labeled clips + calibration; rectified overlay visually correct; zone polygons load and classify synthetic points.

---

## 5. Phase 2 — Vertical slice (Day 1 PM → Day 2 · ~8 h) · **M1 hard gate**

**Goal:** one leak clip flows video → incident JSON. Crude is fine; the *chain* must be complete and honest. This de-risks integration, which is what actually kills hackathon builds.

| ID | Task | Files | Verify |
| :-- | :-- | :-- | :-- |
| P2-T01 | Pydantic models: `Frame`, `Candidate`, `FeatureVector`, `FusionResult`, `Decision`, `ActionRecord`, `Incident`, `Report`. Enums from PRD §6.2. UTC ISO-8601 `Z` serialization. | `models.py` | Round-trip tests; timestamp format test |
| P2-T02 | Minimal tracker: HSV/fluorescence threshold on the rectified frame + morphology + connected components → candidates with area, centroid, mean intensity. | `perception/fast_tracker.py` | On the leak clip: ≥1 candidate in the QD-04 ROI in >50% of frames; unit test on synthetic blobs |
| P2-T03 | Minimal fusion: canonical sigmoid `P = σ(w_v φv + w_h φh + w_t·min(φt,0) + w_W W + b)` with the §9 weights; `φv/φh/φt` per delta §4. **Hand-set priors in config, marked as priors — never "calibrated"** (D-03, delta C-01). | `fusion/features.py`, `fusion/engine.py` | `P ∈ [0,1]`; `∂P/∂φt = 0` for `φt > 0`; config-hash logged |
| P2-T04 | Minimal decision: R1–R7 first-match order + hysteresis counters, strict half-open intervals. | `fusion/decision.py` | `tests/test_decision_table.py` covers each rule + boundaries |
| P2-T05 | Minimal FSM with **clamp first** and an in-memory adapter stub: `CLAMPING` requires verified readback, then `ISOLATING`. Checkpoint verification runs concurrently. (D-06) | `orchestrator/incident_fsm.py` | FSM unit test: no transition to `ISOLATING` without readback OK |
| P2-T06 | Incident writer: `artifacts/incidents/INC-RACK-08B-YYYYMMDD-NN.json`, deterministic numbering per run, includes `deterministicActionsTaken`, config hash, clip id, frame range. | `orchestrator/actions.py` | File exists, validates against `Incident`, ID matches regex |
| P2-T07 | Deterministic fallback report (template, no LLM) so the chain completes with the network down. | `report/fallback.py` | Incident JSON contains `report.source = "deterministic_fallback"` |
| P2-T08 | `make demo SCENARIO=leak --headless` wiring: app → source → tracker → fusion → decision → FSM → incident. | `app.py`, `Makefile` | Command exits 0 and prints incident path |

**Exit criteria (M1):** `make demo SCENARIO=leak --headless` produces a schema-valid incident on `main`, with a report (fallback acceptable at this point). No hard-coded probabilities anywhere.

---

## 6. Phase 3 — Perception hardening (Day 2 · ~6 h)

**Goal:** real SAM 3.1 in the loop, asynchronously, with a deterministic cache and a tracker that survives between calls.

| ID | Task | Files | Verify |
| :-- | :-- | :-- | :-- |
| P3-T01 | `sam_client.py`: OpenAI SDK → `https://api.meta.ai/v1/responses`, `model="sam-3.1"`, **text prompt on a crop** (D-01), prompts from `sam.prompts` in config, ≤32 tokens validated client-side, `timeout_sec: 4.0`, `async_rate_hz: 1.0`, retries, structured errors. | `perception/sam_client.py` | Test against a recorded fixture response; prompt-token guard test rejects a 40-token prompt |
| P3-T02 | Disk cache: key `sha256(image_bytes + prompt + model)`; writes atomic; stores raw response + decoded mask PNG; `--offline` mode fails loudly on miss (never silently falls back). | `perception/sam_client.py` | Second call = 0 network requests; `tests/test_sam_cache.py` asserts byte-identical mask |
| P3-T03 | Async worker: ≤2 Hz, in-flight cap 1, ROI crop per hydraulic-enclosure region, results re-anchor the tracker (IoU match, else new candidate). (D-02) | `perception/sam_worker.py` | Latency-independent: feed a 3 s delayed mock, pipeline keeps ticking |
| P3-T04 | Candidate manager: lifecycle, ID assignment, association, kinematics (`dA/dt`, centroid velocity, `C_fluor`, variance), `W` continuity per D-09 (8-neighborhood path to originating QD ROI for ≥3 ticks). | `perception/candidates.py` | Unit tests for merge/split; `W` true only on connected path, false on a disconnected blob |
| P3-T05 | Adaptive sampling: idle 2 Hz discovery → 5 Hz tracking → 10 Hz on SUSPECT. (D-18) | `app.py` loop policy | Log asserts tick rate changes on state change |
| P3-T06 | Camera-health signal: mean brightness, focus (Laplacian variance), saturation; drives R4's camera-healthy gate and suppresses actuation when unhealthy. | `perception/fast_tracker.py` | Test: black frame / saturated frame ⇒ `camera_healthy=false` |
| P3-T07 | Mask quality gates (min area, max noise, exposure clipping) exposed in config and required by R1/R2. | `configs/default.yaml` | Decision test proves a failed gate blocks R1 |

**Exit criteria:** SAM masks visibly drive fusion on the leak clip; cache makes a second run fully offline; ~0 SAM calls during a 60 s replay after warm cache.

---

## 7. Phase 4 — Simulators and telemetry (Day 3 AM · ~6 h)

**Goal:** every simulated channel is a real protocol with a real state machine, and every panel in the UI is honestly badge-able.

| ID | Task | Files | Verify |
| :-- | :-- | :-- | :-- |
| P4-T01 | `psychrometrics.py`: dew point (Magnus), absolute humidity, wet-bulb. Reference values in tests. | `telemetry/psychrometrics.py` | `tests/test_psychrometrics.py` vs published table within 0.2 °C |
| P4-T02 | `telemetry/sim.py`: scenario-scripted channels (pressure, flow, ΔT, humidity, ambient) sampled at fixed dt, **time-aligned to video timestamps**, not wall clock. | `telemetry/sim.py` | Replay twice ⇒ identical series hash |
| P4-T03 | `sims/thermal_model.py`: lumped `ΔT·C_th/P` model with `C_th = 350 J/K` labelled **assumption** in config; watchdog at 90 °C. (D-07) | `sims/thermal_model.py` | 40 K / 200 W ⇒ 70 s ±1%; 1000 W ⇒ 14 s ±1% |
| P4-T04 | `sims/plc_server.py`: `pymodbus` TCP server, **port 5020** default, unit id `1`, FSM `VALVE_OPEN→CLOSING_COMMANDED→CLOSED/FAULT`. Per PRD §10.1: **Coil 0001** = valve solenoid (0 = spring-close), **Coil 0002** = vacuum drain, **Discrete Input 1001** = limit switch closed, **1002** = line depressurized, **HR 4001** = pressure int16 psi×10, **HR 4002** = flow uint16 L/min×100, **Coil 0003** = `VALVE_FAULT` alarm. Travel 800–1200 ms nominal (D-15 bands: ≤1200 `VERIFIED`, 1200–1500 `VERIFIED_SLOW`, >1500 `FAULT`). | `sims/plc_server.py` | `tests/test_plc_modbus.py` drives coils and reads discrete inputs over the wire; asserts scaling + both timing bands |
| P4-T05 | `sims/bmc_server.py`: Redfish-shaped mock (`/redfish/v1/Systems/gpu-node-08b`, `Actions/ComputerSystem.Reset`), ForceOff honoured only when powered. | `sims/bmc_server.py` | httpx test: ForceOff returns 202, subsequent GET shows `PowerState=Off` |
| P4-T06 | `sims/slurm_mock.py`: job `train-llm-0427` on 4 GPUs; DRAIN, REQUEUE on standby `gpu-node-09a`; Kubernetes as an interface stub only (D-14). | `sims/slurm_mock.py` | State machine test: drain → requeue → running on 09a |
| P4-T07 | `sims/checkpoint_store.py`: write/verify with SHA-256; a corrupted checkpoint must fail verification, not pass. | `sims/checkpoint_store.py` | Corruption test: flip one byte ⇒ `verified=false` |
| P4-T08 | `sims/train_job.py`: emits checkpoints every N s so the incident has something real to verify. | `sims/train_job.py` | Checkpoint files appear with manifests |

**Exit criteria:** `make sims` starts PLC (5020) + BMC (8443); all simulator tests green; thermal numbers match the PRD's table within tolerance.

---

## 8. Phase 5 — Fusion and decision (Day 3 PM · ~6 h) · **M2**

**Goal:** the intellectual core — the thing the judges score under "technical depth". Must be defensible line by line.

| ID | Task | Files | Verify |
| :-- | :-- | :-- | :-- |
| P5-T01 | Finalize the fusion function and **write down the math** in `src/cryoguard/fusion/README.md`: inputs, bounds, weight semantics, why thermal only modulates (D-10). No "calibrated" wording (D-03). | `fusion/README.md`, `engine.py` | Doc reviewed against §6.3; no forbidden words (`calibrated`, `trained`) |
| P5-T02 | `φv` (growth + fluorescence contrast − jitter, clamped), `φh` (pressure drop, slope, flow imbalance), `φt` (dew-point margin). Condensation veto path (D-08): fluorescence discriminates, dew point vetoes. `W` computed on **fluorescence-positive** pixels only (delta R-2). | `fusion/features.py` | Unit vectors: glint ⇒ low φv; dry air + pressure drop ⇒ high φh; beads + dew point ⇒ φt negative |
| P5-T03 | Decision engine: canonical table **plus the three holes closed** — R2b, R2c, R8 `CAMERA_DEGRADED` (delta §2/E-02/E-03) — first match wins, strict half-open intervals, `maintenance_mode` suppresses Zone 0/1 but never Zone 2. **R1/R2 require `dA/dt > growth_epsilon` OR `φh ≥ 0.2`** (delta R-1: without it a static reflection at a QD scores 0.61 and trips Zone 2). | `fusion/decision.py` | Table-driven test for every rule and boundary; Hypothesis totality test; static-reflection case must stay below R1 |
| P5-T04 | Hysteresis per §6.4: escalation counts, de-escalation at `P < thr − 0.05` for 5 ticks, latch forever after `CONFIRMED_LEAK`/`EMERGENCY_TRIP`. | `fusion/decision.py` | 12-tick scripted sequences per rule |
| P5-T05 | Zone assignment: `max` zone index among mask pixels covering ≥5% of area; polygons already include the 20 mm margin. | `fusion/decision.py` | Synthetic mask spanning Zone 1+2 ⇒ Zone 2 |
| P5-T06 | Priors tuned **on the dev split only**, then frozen: write `strict_freeze: true` + weights + `config_hash` into `reports/fusion-freeze.json` before touching test clips. | `configs/default.yaml`, `reports/fusion-freeze.json` | Freeze file committed with timestamp **earlier** than the first test-split run |
| P5-T07 | Architecture test: `report/` is unreachable from `fusion/`, `orchestrator/`, `adapters/` (AST import scan). | `tests/test_architecture.py` | Fails if anyone adds the import |

**Exit criteria (M2):** all `T-DEC-*` tests green; freeze record exists; the condensation clip yields a weak visual candidate that hydraulic/thermal reject.

---

## 9. Phase 6 — Incident orchestrator and adapters (Day 4 AM · ~6 h) · **M3**

**Goal:** the safety argument. Ordering, readback, fault escalation, and the proof that no LLM is in the path.

| ID | Task | Files | Verify |
| :-- | :-- | :-- | :-- |
| P6-T01 | FSM using the canonical §6.2 state names with the §11 transitions (delta §5 alias map). `CLAMPING` → `ISOLATING` → `ISOLATED` → `NODE_POWERED_DOWN` → `REQUEUED` → `RESOLVED_PENDING_REPAIR`, with `CHECKPOINT_VERIFY` as a **parallel branch from clamp time**; exits `HARD_TRIP`, `FAILED_SAFE`. `ISOLATED` is reached on limit switch alone — a pending checkpoint must not delay it (delta C-04). | `orchestrator/incident_fsm.py` | Exhaustive transition test: every illegal transition raises; `T-FSM-02` proves the valve-closed timestamp is independent of the checkpoint timestamp |
| P6-T02 | **Ordering guarantee** (D-06): power clamp issued and **readback-verified** before valve close; checkpoint verification starts in parallel at clamp time. Test asserts timestamps, not intentions. | FSM + `actions.py` | `tests/test_safety_ordering.py`: `t_valve_cmd > t_clamp_verified`; checkpoint starts before valve CLOSED |
| P6-T03 | `adapters/nvml.py`: `Fake` and `Real` backends behind one interface. **Never hard-code watts** — query `min_limit`; if `min_limit` is unknown, fail safe (do not pretend). (D-07) | `adapters/nvml.py` | Fake backend returns configured `min_limit=200 W`; Real backend skipped when `pynvml` absent |
| P6-T04 | `adapters/plc_client.py`: Modbus-TCP client, scale decode, valve command + status poll, `VERIFIED` / `VERIFIED_SLOW` / `FAULT` per D-15, FAULT ⇒ escalate to node power cut. | `adapters/plc_client.py` | Wire test against the sim: 1300 ms ⇒ `VERIFIED_SLOW`, 1600 ms ⇒ `FAULT` ⇒ power cut path |
| P6-T05 | `adapters/redfish_client.py`: ForceOff, power state polling, idempotent retries. | `adapters/redfish_client.py` | httpx test against mock; retry test with 503 then 202 |
| P6-T06 | `adapters/scheduler.py`: DRAIN, REQUEUE to `gpu-node-09a`, resume from last verified checkpoint `N−1`; log which checkpoint was used and why. | `adapters/scheduler.py` | Test: corrupted checkpoint N ⇒ resumes N−1 and records the reason |
| P6-T07 | Watchdog: thermal ≥90 °C, clamp readback failure, PLC FAULT ⇒ `EMERGENCY_TRIP` / `HARD_TRIP` per R1b. Runs on its own task, independent of fusion loop. | `orchestrator/actions.py` | Test: kill the perception loop, watchdog still trips |
| P6-T08 | `deterministicActionsTaken`: append-only, ordered, timestamped, immutable after incident close. Report service receives a **read-only snapshot**. | `orchestrator/actions.py` | Test: attempting to mutate a closed incident raises |

**Exit criteria (M3):** `make demo SCENARIO=leak` drives the real Modbus + Redfish sims to a closed incident, with `RESOLVED_PENDING_REPAIR` and a requeued job. Ordering test green.

---

## 10. Phase 7 — Reporting with Muse Spark Open (Day 4 · ~3 h)

**Goal:** G3 and the $10K open-weights bonus, done honestly.

| ID | Task | Files | Verify |
| :-- | :-- | :-- | :-- |
| P7-T01 | `report/schema.py`: strict report schema (summary, timeline, evidence per channel, actions, uncertainty, what-was-not-observed). LLM output validated with `pydantic`; one repair retry with the validation error, then fallback. | `report/schema.py` | Test: malformed JSON ⇒ one retry ⇒ fallback, never a partial report |
| P7-T02 | `report/muse_client.py`: partner endpoint via OpenAI-compatible SDK, prompt assembled **only** from incident facts (no free-text invention opportunity), timeout, retries, request/response logging to `artifacts/open_weights_proof.jsonl` with model id + latency + token counts. | `report/muse_client.py` | Log line per request; offline test uses recorded fixture |
| P7-T03 | Constrained prompts per system label: R4 report must state "no visual evidence"; `BENIGN_CONDENSATION` must state the rejection reasons; never propose actuation. | `configs/prompts.yaml` | Prompt snapshot tests |
| P7-T04 | Async scheduling: report requested **after** `deterministicActionsTaken` is sealed; failure/timeout cannot block, delay, or alter the safety path. | `app.py` | Test with endpoint returning 500 and with a 60 s hang: incident completes, report falls back |
| P7-T05 | Report render into the incident JSON + UI pane, with `source: muse_spark_open | deterministic_fallback`. | `models.py`, `ui/app.js` | Schema test; UI shows the source badge |

**Exit criteria:** schema-valid Muse report on the leak scenario with a saved proof log; full incident completes with the LLM endpoint down.

---

## 11. Phase 8 — Benchmarks and CI (Day 5 AM · ~4 h) · **M4**

**Goal:** G4. Re-runnable, offline, byte-identical, and statistically honest.

| ID | Task | Files | Verify |
| :-- | :-- | :-- | :-- |
| P8-T01 | `bench/runner.py`: runs every manifest clip through the **same** pipeline, records per-clip decision trace, fired rule, P, and latency breakdown. | `bench/runner.py` | Run completes offline (`--offline`) with zero network calls (assert via a socket guard) |
| P8-T02 | `bench/metrics.py`: confusion counts, recall/precision **with Clopper–Pearson 95% CIs** (`scipy.stats.beta`), plus per-class counts so the reader sees `n`. No augmented clip counted twice (D-12). | `bench/metrics.py` | Known fixture: 10/10 ⇒ CI ≈ [0.69, 1.0]; test asserts bounds |
| P8-T03 | `bench/report.py`: emits the §12 format — `reports/benchmark.md` + `reports/benchmark.json`; JSON contains config hash, freeze hash, `run_hash`, git SHA, per-clip rows, CI bounds. The table in PRD §12 is a **format template with placeholder numbers only** (delta C-08): every shipped value must exist in the JSON. | `bench/report.py` | `make bench` twice ⇒ byte-identical files; `T-BEN-05` fails on any Markdown number absent from JSON |
| P8-T04 | Latency measurement: perception tick, SAM call (cached vs uncached, clearly separated), fusion, decision→action. Medians + p95 over measured runs, **measured not assumed**. | `reports/benchmark.json` | Values present and non-constant across rows |
| P8-T05 | CI job: `make bench` on the test split, diff against committed baseline, fail on nondeterminism. | `.github/workflows/ci.yml` | Deliberate cache edit breaks CI as expected |

**Exit criteria (M4):** `make bench` green offline; report regenerated; CIs present; second run byte-identical.

---

## 12. Phase 9 — Dashboard (Day 5 AM · ~4 h)

**Goal:** the demo surface. One page, no build step, every simulated element visibly badged.

| ID | Task | Files | Verify |
| :-- | :-- | :-- | :-- |
| P9-T01 | FastAPI app: static `ui/`, `GET /api/state`, `GET /api/incidents/{id}`, `WS /ws` broadcasting typed bus events. | `api/routes.py`, `api/ws.py` | `curl` returns state; WS test client receives ≥1 event |
| P9-T02 | Canvas: rectified video frame, zone polygon overlay (colour by risk), tracker mask, SAM mask, candidate centroid trail. Fixed 30 fps render loop decoupled from data rate. | `ui/app.js` | Visual check on all three clips |
| P9-T03 | Panels: optical, hydraulic, thermal, dew point; **each simulator panel carries a `SIMULATED` badge**; camera panel carries `REAL`. | `ui/index.html` | DOM test asserts badges present on the five simulated panels |
| P9-T04 | Decision strip: current `P`, φv/φh/φt, `W`, zone, candidate label, fired rule ID, hysteresis counter. Values from WS only — no literals. | `ui/app.js` | Grep gate: no numeric probability literals in `ui/` |
| P9-T05 | Action timeline: ordered `deterministicActionsTaken` with timings; badge `VERIFIED` / `VERIFIED_SLOW` / `FAULT`; checkpoint N−1 note. | `ui/app.js` | Matches incident JSON exactly |
| P9-T06 | Report pane: streamed text, `source` badge, `SIMULATED` footer for environment. | `ui/app.js` | Renders fallback and Muse cases |
| P9-T07 | `--headless` and UI modes share one pipeline; no duplicated logic in the UI. | `app.py` | Trace shows identical event sequence in both modes |

**Exit criteria:** dashboard shows a full leak incident live, with honest badges and no fabricated numbers.

---

## 13. Phase 10 — Scenario hardening and acceptance (Day 5 PM · ~4 h)

**Goal:** three scenarios, all green, with the acceptance suite from §6 (stand-in for the missing §14).

| ID | Task | Files | Verify |
| :-- | :-- | :-- | :-- |
| P10-T01 | `condensation`: real beads under the same excitation; optical produces a weak candidate; hydraulic flat; dew point vetoes ⇒ `BENIGN_CONDENSATION`, no actuation. (D-08) | fixture + `configs/scenarios/condensation.yaml` | `make demo SCENARIO=condensation` ⇒ no actuator command in the action log |
| P10-T02 | `hidden`: no visual candidate, pressure drop ≥3 s ⇒ `HIDDEN_HYDRAULIC_EVENT`, scheduler DRAIN only, **no valve actuation**. (D-11) | fixture + scenario yaml | Action log contains zero PLC writes |
| P10-T03 | `leak`: goes to `CONFIRMED_LEAK` or `EMERGENCY_TRIP` per zone, full safe-state sequence, requeue. | scenario yaml | Action log matches FSM order |
| P10-T04 | Negative controls: static reflection clip, glint clip, camera-blackout clip ⇒ no trip. | fixtures | Each stays ≤ `LOG_ONLY`/`SUSPECT_LEAK` |
| P10-T05 | Acceptance suite `T-*` from §14 of this plan, marked `pytest -m acceptance`. | `tests/acceptance/` | All green |
| P10-T06 | Docs contract test: README real-vs-simulated table, limitations (planar homography, hand-set priors, small n, assumption `C_th`), and setup in ≤10 commands. | `tests/test_docs_contract.py` | Green |

**Exit criteria:** all three scenarios replay headless and with UI; acceptance suite green.

---

## 14. Phase 11 — Ship (Day 6 · ~5 h)

| ID | Task | Verify |
| :-- | :-- | :-- |
| P11-T01 | Record the demo from the live UI, **real footage on screen**, `SIMULATED` badges legible at 1080p. Beat sheet below. | File ≤ 3:00, audio intelligible |
| P11-T02 | README: pitch, architecture diagram, real-vs-simulated table, setup (≤10 commands), honest limitations, benchmark numbers **copied from `reports/`**, sourced cost figures or "illustrative" label. (D-16) | A fresh clone reaches `make demo` by following only the README |
| P11-T03 | Repo public check: no `.env`, no keys, `git ls-files | grep -i env`, license, description, topics. | Clean |
| P11-T04 | Submission: video URL, repo URL, model usage proof, short write-up. | Filed **by 19:00 ET** |
| P11-T05 | Post-submit smoke: clone into `/tmp/opencode/verify`, `make setup demo bench`. | All three pass |

**Demo beat sheet (2:50 max):**

| Time | Beat |
| :-- | :-- |
| 0:00–0:20 | Problem: a drip on a 48 V busbar; why optical alone cries wolf and hydraulic alone is blind |
| 0:20–0:50 | Real fluorescent footage, tracker + SAM masks, zone overlay, `REAL` vs `SIMULATED` labels |
| 0:50–1:20 | Condensation scenario: optical candidate exists, dew point + pressure veto it. "It refuses to cry wolf" |
| 1:20–2:10 | Leak: fusion → rule R2 → clamp readback → valve close → checkpoint N−1 → requeue. Action timeline on screen |
| 2:10–2:35 | Hidden hydraulic event: no visual, DRAIN only, explicitly no valve, human in the loop |
| 2:35–2:50 | Honesty card: what is simulated, measured benchmark counts with CIs, what this is not |

---

## 15. Acceptance suite (PRD §14 + the delta's additions)

Grouped by what a skeptical judge would attack. All run in CI. PRD §14 is the floor; the delta adds `T-LLM`, `T-DET`, `T-DEC-09`, `T-FSM-01`, `T-BEN-05`, `T-DOC-01`.

### Decision logic (`T-DEC`)
- `T-DEC-01` R1 needs **all** of: zone 2, `P ≥ 0.50`, ≥3 consecutive ticks, camera healthy, mask gates pass.
- `T-DEC-02` A single-tick zone-2 glint does **not** trip (D-04).
- `T-DEC-03` Escalation latches; no auto-de-escalation after `CONFIRMED_LEAK`/`EMERGENCY_TRIP`.
- `T-DEC-04` Boundary values 0.40/0.50/0.70/0.85 land in exactly one rule (strict half-open intervals, R3 before R5).
- `T-DEC-05` `maintenance_mode` suppresses Zone 0 actuation but still logs (D-05).
- `T-DEC-06` R4 requires camera healthy and φv ≤ 0.1; a dirty lens cannot manufacture a hidden event.
- `T-DEC-07` CONFIRMED requires φv ≥ 0.2 **and** φh ≥ 0.2; thermal alone never confirms (D-10).

### Safety path (`T-FSM`)
- `T-FSM-01` Valve command timestamp strictly after clamp readback verification (D-06).
- `T-FSM-02` Checkpoint verification starts before valve reaches CLOSED.
- `T-FSM-03` 1200/1500 ms valve bands ⇒ `VERIFIED` / `VERIFIED_SLOW` / `FAULT` (D-15).
- `T-FSM-04` FAULT ⇒ node power cut, incident `HARD_TRIP`, `FAILED_SAFE` if power cut also fails.
- `T-FSM-05` Thermal 90 °C watchdog fires with the perception loop dead.
- `T-FSM-06` Clamp readback failure escalates; never proceeds to isolation on an unverified clamp.

### LLM isolation (`T-LLM`)
- `T-LLM-01` Full incident completes with the report endpoint returning 500 / hanging 60 s.
- `T-LLM-02` `deterministicActionsTaken` is identical with and without the LLM (byte compare).
- `T-LLM-03` No module in `fusion/`, `orchestrator/`, `adapters/` imports `report/` (AST scan).
- `T-LLM-04` Malformed report ⇒ validated fallback, never partial output.

### Determinism (`T-DET`)
- `T-DET-01` Same clip + same cache ⇒ byte-identical incident (excluding timestamps).
- `T-DET-02` `make bench` twice ⇒ identical `benchmark.json`.
- `T-DET-03` SAM cache key changes when prompt or model changes, and only then.

### Protocols (`T-PROTO`)
- `T-PROTO-01` Modbus registers are int16/uint16 scaled; no floats (D-13).
- `T-PROTO-02` PLC on port 5020, unit id 1; document the 502 root requirement.
- `T-PROTO-03` Redfish ForceOff idempotent; 202 then power-off state.
- `T-PROTO-04` Corrupted checkpoint fails hash verification.

### Data and semantics (`T-DATA`)
- `T-DATA-01` Zone = max zone index over pixels covering ≥5% of mask area.
- `T-DATA-02` `W = 1` only on a connected 8-neighborhood path to the QD ROI for ≥3 ticks (D-09).
- `T-DATA-03` Psychrometrics match published values within 0.2 °C.
- `T-DATA-04` Thermal soak matches `ΔT·C_th/P_min` for the configured `C_th`, labelled an assumption.

---

## 16. Traceability: PRD decision → where it lands

| ID | Phase / task | Guarded by |
| :-- | :-- | :-- |
| D-01 text-only SAM + HSV fallback | P3-T01, P2-T02 | `T-DET-03`, SAM prompt-token test |
| D-02 async ≤2 Hz + tracker bridge | P3-T03 | delayed-mock test |
| D-03 hand-set priors, frozen | P2-T03, P5-T06 | `reports/fusion-freeze.json` |
| D-04 zone-2 gate | P5-T03 | `T-DEC-01/02` |
| D-05 canonical table + `maintenance_mode` | P5-T03 | `T-DEC-04/05` |
| D-06 clamp first, parallel checkpoint | P2-T05, P6-T02 | `T-FSM-01/02` |
| D-07 query `min_limit` | P4-T03, P6-T03 | `T-DATA-04` |
| D-08 condensation veto framing | P5-T02, P10-T01 | condensation scenario test |
| D-09 `W` definition | P3-T04 | `T-DATA-02` |
| D-10 thermal modulates only | P5-T01/02 | `T-DEC-07` |
| D-11 `HIDDEN_HYDRAULIC_EVENT` | P5-T03, P10-T02 | hidden scenario test (zero PLC writes) |
| D-12 raw counts + CIs | P8-T02 | `T-DET-02`, CI-bound test |
| D-13 scaled integer registers | P4-T04 | `T-PROTO-01` |
| D-14 Slurm only | P4-T06 | adapter stub test |
| D-15 valve timing bands | P4-T04, P6-T04 | `T-FSM-03` |
| D-16 sourced/illustrative figures | P11-T02 | docs contract test |
| D-17 canonical IDs | P0-T06 | `test_identifiers.py` |
| D-18 sampling rates | P3-T05 | tick-rate log test |
| D-19 ArUco homography | P1-T03 | `test_rectify.py` |

---

## 17. Risk register

| # | Risk | Likelihood | Impact | Mitigation | Trigger |
| :-- | :-- | :-- | :-- | :-- | :-- |
| R1 | Fluorescent footage doesn't read well (weak signal, motion blur, reflections) | Medium | **Fatal** — kills the "real" claim | Phase 1 first; test rig same day; increase tracer concentration, longer exposure, tripod, dark enclosure | If leak clip shows no ≥50 px signal by Day 1 18:00, re-shoot Day 2 AM before any other work |
| R2 | SAM API shape/latency/credits differ from spec | Medium | High | P0-T02 spike; cache; HSV fallback keeps the pipeline alive | Spike fails ⇒ escalate, drop G2 claim from README |
| R3 | Muse Spark Open access or partner verification unavailable | Medium | Medium (loses bonus) | P0-T03 spike; deterministic fallback always works | No access by Day 1 ⇒ report section reworded, never faked |
| R4 | PRD §7–§14 missing from v1.0 | Closed | — | Complete PRD supplied; conflicts catalogued in `prd-delta-v1.2.md` | If the organisers publish a newer revision, diff against delta v1.2 before changing code |
| R5 | Solo time overrun | High | High | Strict cut ladder (§2.1); P2 is a hard gate | Any phase > 1.5× estimate ⇒ cut P1 scope that day |
| R6 | Modbus port conflicts / pymodbus 3.x API drift | Medium | Low | 5020 default, integration test at start of P4 | Test red > 30 min ⇒ pin version, isolate behind adapter |
| R7 | Nondeterministic bench (threading, timestamps) | Medium | Medium | Sim clock, fixed seeds, sorted JSON keys, cache required | `T-DET-02` fails ⇒ fix before any new feature |
| R8 | Scope creep from "we could also…" | High | High | P2 list is closed (§19); PRD §0 rule 7 | Any unplanned file or unplanned dependency ⇒ revert |

---

## 18. Definition of Done (project)

- [ ] `make demo SCENARIO=leak|condensation|hidden` works headless **and** with UI.
- [ ] `make bench` regenerates `reports/benchmark.md` + `reports/benchmark.json` offline from cache.
- [ ] Every `T-*` test green in CI on `main`.
- [ ] `deterministicActionsTaken` proven identical with the LLM down (`T-LLM-02`).
- [ ] Config hash + fusion freeze recorded in every incident and in the bench report.
- [ ] README: setup ≤10 commands, real-vs-simulated table, limitations section, all numbers from `reports/`.
- [ ] UI: `SIMULATED` badge on PLC, BMC, NVML, Slurm, telemetry; `REAL` on camera.
- [ ] Video ≤3:00, uploaded, link in submission.
- [ ] Repo public, no secrets, submission filed by **19:00 ET on 2026-11-04**.

---

## 19. Explicitly out of scope (do not build)

Kubernetes execution (interface stub only) · multi-rack · 3D fluid trajectory · SWIR 1450 nm · real PLC firmware · live webcam mode (file replay is sufficient for the demo) · training any model · "calibrated" language anywhere · claims of production readiness or safety certification.
