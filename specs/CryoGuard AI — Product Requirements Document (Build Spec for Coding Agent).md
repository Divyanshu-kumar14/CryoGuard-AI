# CryoGuard AI — Product Requirements Document

**Build spec for the coding agent · Meta Global AI Developer Hackathon (Indie track)** Version 1.0 · Written 2026-10-30 · Submission deadline **Wed 2026-11-04, 9:00 PM ET** (Cerebral Valley portal) · Solo builder

---

## 0. How the coding agent must use this document

Read sections 0–6 fully before writing code. Sections 7–14 are the requirements. Every requirement has an ID (e.g. `FR-FUS-03`) and a priority:

- **P0**: demo and submission fail without it.
- **P1**: strongly improves score; build after all P0 items work end to end.
- **P2**: only if time remains. Never start a P2 while a P0/P1 is red.

**Non-negotiable agent rules**

1. **No fabricated numbers.** Any metric shown in the UI, README or video must be computed by code from fixtures or logs. Never hard-code a probability, latency, recall or FPR. Scenario outputs like "P(leak) = 0.94" in the research dossier are *illustrations of expected behavior*, not targets to hard-code.
2. **Label simulation honestly.** Only the optical channel is real (recorded footage). The PLC, BMC, NVML, Slurm, telemetry and storage are simulators. Every simulated element carries a visible `SIMULATED` badge in the UI and is called a simulator in the README.
3. **The LLM is never in the safety loop.** No LLM output may influence any actuation, zone decision, or the `deterministicActionsTaken` field. The system must complete a full incident with the LLM endpoint down.
4. **Everything tunable lives in config** (`configs/*.yaml`), not in code. Thresholds, weights, timings, topology, prompts.
5. **Determinism for benchmarks.** Cache every SAM API response on disk keyed by `sha256(image bytes + prompt + model)`. Benchmarks must be re-runnable offline and byte-identical.
6. **Build vertical slice first.** By end of Day 2 one fixture must flow from video → incident report, even if crude. Then widen.
7. **Small commits, tests with each module, `make demo` and `make bench` always working on `main`.**
8. **Secrets**: only via env vars (`.env.example` committed, `.env` git-ignored).

---

## 1. Product summary

**CryoGuard AI** detects and localizes direct-to-chip liquid-cooling leaks in a GPU rack using fluorescence video, then verifies them against hydraulic and thermodynamic telemetry before triggering a *deterministic* safe-state sequence (power clamp, valve isolation, checkpoint verification, workload resumption). An open-weights LLM (Muse Spark Open) writes the incident report after the fact.

**One-line pitch:** *A leak detector that proves it is a leak (optical growth + pressure drop + thermodynamics), refuses to cry wolf on condensation, refuses to hallucinate when it cannot see, and never lets an LLM touch a valve.*

### 1.1 What is real vs simulated (must be reflected in README and UI)

| Element | Status | Implementation |
| :-- | :-- | :-- |
| Fluorescent leak/condensation/glint footage | **Real** | Self-recorded fixtures (Section 13) |
| SAM 3.1 segmentation | **Real** | Meta Model API (text-prompted) |
| Evidence fusion, zone logic, tracker | **Real** | Our code |
| Pressure, flow, temperature, humidity | **Simulated** | Scenario-scripted telemetry synchronized to video |
| Rack Safety PLC | **Simulated, real protocol** | `pymodbus` Modbus-TCP server with FSM |
| Server BMC | **Simulated, real protocol shape** | Redfish-style REST mock |
| GPU power clamp | **Simulated** (real backend if NVIDIA GPU present) | NVML adapter with `Fake` and `Real` backends |
| Slurm / checkpoint storage | **Simulated** | Mock scheduler + real file I/O checkpoint with hash verification |
| Incident report | **Real** | Muse Spark Open via partner inference |

---

## 2. Hackathon constraints and scoring

| Rubric axis | Weight | What wins it here |
| :-- | :-- | :-- |
| Impact | 30% | Sourced cost figures, clear who-pays story, honest scope |
| Use of Meta AI tooling | 25% | SAM 3.1 doing real work in the pipeline; Muse Spark Open doing real work in reporting |
| Demo | 25% | Real footage, three scenarios, closed loop visible on screen, under 3:00 |
| Technical depth and execution | 20% | Fusion math, PLC FSM, checkpoint verification, reproducible benchmarks |

**Verified facts (from public event pages and community repos, Oct 2026)**

- Deadline: Nov 4, 9:00 PM ET via Cerebral Valley portal. Judging Nov 4–8; finals Nov 9.
- Participants receive about $150 in Meta Model API credits.
- **SAM 3.1 via Meta Model API**: called through the OpenAI SDK Responses API at `https://api.meta.ai/v1/responses` with `model="sam-3.1"`. Returns a bounding box and a pixel mask per match. **The hosted API accepts text prompts only**; box/point prompts exist only in the open-weight release. Reported pricing: about $2.5 per 1,000 images and $0.2 per 1,000 video frames. Text prompt limit: 32 tokens.
- **Open-weights bonus ($10K)** counts **only** Muse Glimmer or Muse Spark Open served on your own hardware or partner inference (Baseten, Fireworks, Modal), verified by the infrastructure partner. **SAM 3.1 through Meta's API does not count.** Community repos cite hackathon model id `meta-models/muse-spark-open-1026` on Baseten; confirm in the official hackathon guide on Day 1.
- Muse Spark Open appears to be a very large model (public repo names suggest \~816B total / \~42B active); do not plan to self-host it. Use partner inference.

**Unverified (do not state publicly without confirming):** first-place prize amounts, the "OCP specifies COOLWAVE DC 25" claim, all dollar-loss figures in the dossier.

---

## 3. Decisions log: corrections to the research dossier

The dossier (files 01–07) is a good concept brief but contains defects that would break a build or hurt credibility with a skeptical judge. **These decisions override the dossier wherever they conflict.** Class: **Fatal** (breaks the idea), **Fixable**, **Blind Spot**.

| ID | Class | Dossier says | Problem | Decision |
| :-- | :-- | :-- | :-- | :-- |
| D-01 | Fixable | SAM prompted with a spatial grid and specular peaks | Hosted SAM 3.1 is text-only; grid/point prompts are impossible | Crop each hydraulic-enclosure ROI and send a **text prompt** on the crop. Add a classical HSV/fluorescence threshold path as fallback and as the high-rate tracker |
| D-02 | Fixable | SAM runs every 200 ms (5 Hz) | API round-trip latency is unknown and likely 0.5–3 s | Run SAM asynchronously at ≤2 Hz with in-flight cap; classical mask tracker bridges frames at 5–10 Hz; SAM masks re-anchor the tracker |
| D-03 | Fixable | Fusion weights are "calibrated logistic regression" | There is no training data; "calibrated" would be false | Weights are **hand-set priors**, tuned on the dev split only, frozen (config hash recorded) before the test split. Never claim calibration unless ECE/Brier are computed |
| D-04 | Fixable | "Any P(leak) in Zone 2 → hard trip" | Contradicts the entire false-positive thesis; one glint near a busbar trips the rack | Zone 2 trip requires `P ≥ 0.50` AND ≥3 consecutive ticks AND healthy camera AND mask quality gates |
| D-05 | Fixable | Zone 0 at P ≥ 0.85 is undefined; threshold tables disagree (0.50 vs 0.40/0.70) | Ambiguous behavior | Single canonical table in Section 6.3. Zone 0 and Zone 1 both CONFIRMED at P ≥ 0.85. Add per-branch `maintenance_mode` to suppress Zone 0 actions during servicing |
| D-06 | Fatal-if-unfixed | Order: cordon → checkpoint → NVML → valve | Valve closure with 1 kW load gives \~14 s to 105°C. Waiting up to 30 s for a checkpoint before clamping/isolating is unsafe or leaks longer | **Clamp power first (verified by readback), then close valve, and run checkpoint verification in parallel during the thermal soak.** Thermal watchdog forces power-off at 90°C |
| D-07 | Fixable | Clamp to 60–80 W gives "over 5 minutes" of soak | 40°C ÷ 0.22°C/s ≈ 182 s (\~3 min), not >5. Data-center GPUs' NVML minimum power limits are typically well above 80 W | Never hard-code a wattage. Query `min_limit`. Soak time = `ΔT × C_th / P_min`. For 40 K, C\_th = 350 J/K (assumed): 1000 W → 14 s; 200 W → 70 s; 100 W → 140 s; 80 W → 175 s. Default fake-NVML `min_limit` = 200 W (so soak = 70 s, still > 30 s checkpoint budget). `C_th` is labeled an assumption |
| D-08 | Blind Spot | Thermal/dew-point factor is the star of Scenario 2 | The coolant tracer fluoresces; plain condensation (water) does **not**. A judge will ask why dew point matters | Frame correctly: fluorescence is the primary discriminator; dew-point/pressure are **independent vetoes** for filter leakage, specular water-bead glint under excitation light, dried tracer residue wetting, and RGB fallback mode. Scenario 2 footage must show real beads under the same excitation light so the optical channel *does* produce a weak candidate that the other channels reject |
| D-09 | Blind Spot | Wetness continuity signal `W` appears in the equation, never defined | Unimplementable | `W = 1` iff the candidate mask is spatially connected (≥1 px 8-neighborhood path) to the originating QD ROI for ≥3 consecutive ticks, else 0 |
| D-10 | Fixable | Thermal factor counted as leak evidence | "Condensation impossible" is permissive, not positive evidence of a leak | CONFIRMED requires **visual AND hydraulic** positive (each φ ≥ 0.2). Thermal only modulates |
| D-11 | Fixable | Scenario 3 labeled `UNCERTAIN`, `HIDDEN_HYDRAULIC_EVENT`, `UNVERIFIED_HYDRAULIC_EVENT` | Three names | Canonical label: `HIDDEN_HYDRAULIC_EVENT`. No autonomous valve actuation on it (human-in-loop) |
| D-12 | Fixable | Targets: recall ≥95%, FPR ≤2% | With \~10–15 clips per class these are statistically meaningless (10/10 recall has a 95% lower bound of \~69%) | Report raw counts + Clopper-Pearson 95% CIs. Targets are stated as goals, never as achieved claims. Never count augmented copies as independent clips |
| D-13 | Fixable | Modbus "16-bit float" registers | Modbus holding registers are 16-bit integers | Scaled integers: pressure int16 = psi×10; flow uint16 = L/min×100 |
| D-14 | Fixable | Kubernetes **and** Slurm | Scope creep for a solo 5-day build | **Slurm-only mock (P0).** Kubernetes adapter is an interface stub with documented `cordon`/`drain` mapping (P2) |
| D-15 | Fixable | Valve FSM leaves 1200–1500 ms undefined; "limit switch within 1200 ms" | Gap | ≤1200 ms = VERIFIED; 1200–1500 ms = VERIFIED\_SLOW (warning); >1500 ms = FAULT → escalate to node power cut |
| D-16 | Fixable | Pitch quotes: $300k tray, $40k GPU, $100k/hr, "zero hardware damage" | Dossier files disagree ($40–64k/hr vs $100k/hr); unsourced; "zero damage" is unearned in a simulation | Cite sources or label "illustrative". Remove "zero physical hardware damage". Use "in simulation" wording |
| D-17 | Fixable | Naming drift: `compute-042`, `compute-08b`, `gpu-node-08b` | Breaks log/UI coherence | Canonical IDs in Section 6.5 |
| D-18 | Blind Spot | Sampling "1 Hz → 10 Hz on suspect" but temporal window needs 10 frames @200 ms | 1 Hz idle cannot compute `dA/dt` | Idle scan 2 Hz (candidate discovery), tracking window 5 Hz once a candidate exists, 10 Hz on SUSPECT |
| D-19 | Blind Spot | Millimeter claims ("within 20 mm of conductors") from a 2D camera | Meaningless without calibration | ArUco-fiducial homography to a planar chassis frame; zone polygons already include the 20 mm margin, expressed in chassis-plane mm. State planar-assumption limits in README |

---

## 4. Goals, non-goals, success criteria

**Goals (P0)**

- G1. End-to-end closed loop on 3 canonical scenarios, replayable from `make demo`.
- G2. SAM 3.1 produces the masks that drive fusion in the shipped demo (not a decorative call).
- G3. Muse Spark Open (partner-served) writes a schema-valid incident report; request logs saved for partner verification.
- G4. Reproducible benchmark report generated by `make bench` with CIs.
- G5. 2:50 demo video showing real footage and visible simulation labels.

**Non-goals**

- Real hardware control, real GPUs required, real PLC firmware certification.
- 3D fluid trajectory prediction. SWIR 1450 nm channel (documented as rejected alternative only).
- Kubernetes execution (stub only). Multi-rack scale-out.
- Any claim of production readiness or safety certification.

**Definition of Done for the project**

1. `make demo SCENARIO=leak|condensation|hidden` runs headless and with UI.
2. `make bench` regenerates `reports/benchmark.md` and `reports/benchmark.json` from fixtures with no network (SAM cache).
3. All P0 acceptance tests in Section 14 pass in CI (`pytest`).
4. README documents setup in ≤10 commands, real-vs-simulated table, honest limitations.
5. Video ≤ 3:00, repo public, submission filed ≥2 hours before deadline.

---

## 5. System architecture

```
[Video source: file replay | webcam] ──► Capture ──► Rectifier (ArUco homography)
                                                         │
                          ┌──────────────────────────────┴───────────────┐
                          ▼                                              ▼
                 Fast tracker (HSV/fluor mask, 5–10 Hz)       SAM 3.1 worker (async ≤2 Hz, ROI crops)
                          └──────────────► Candidate manager ◄───────────┘
                                                  │ per-candidate kinematics (dA/dt, centroid, C_fluor, var)
[Telemetry source: scenario script | live sim] ──┤
                                                  ▼
                                      Evidence Fusion  →  P(leak), φv/φh/φt, W
                                                  ▼
                                      Decision Engine (zone + rules R1–R6)
                                                  ▼
                                      Incident Orchestrator (FSM)
        ┌───────────────┬──────────────┬──────────────┬────────────────┐
        ▼               ▼              ▼              ▼                ▼
  NVML adapter    PLC client     Redfish client   Slurm mock +    Report service
  (Fake/Real)     (Modbus-TCP)   (BMC mock)       checkpoint FS   (Muse Spark Open, async)
                          │
                  Event bus (asyncio) ──► FastAPI + WebSocket ──► Dashboard
```

**Process model.** One Python process (`cryoguard.app`) running asyncio tasks, plus two in-process-launchable simulator servers (Modbus PLC on port 5020 by default; Redfish BMC mock on 8443/HTTP for demo). Port 502 requires root; default to **5020** and document it.

**Tech stack (fixed unless blocked)**

- Python 3.11, `pydantic` v2, `fastapi` + `uvicorn`, `numpy`, `opencv-python` (with `opencv-contrib` for ArUco), `httpx`, `openai` SDK (for Meta and Baseten OpenAI-compatible endpoints), `pymodbus` 3.x, `pynvml` (optional import), `PyYAML`, `pytest`, `scipy` (Clopper-Pearson via `scipy.stats.beta`), `ruff`.
- Dashboard: single-page static HTML + vanilla JS + WebSocket + `<canvas>`. No build step. Served by FastAPI.
- Task runner: `Makefile`. Package manager: `uv` or `pip` with pinned `requirements.txt`.

**Repository layout**

```
cryoguard/
  README.md  Makefile  pyproject.toml  .env.example
  configs/  default.yaml  topology.yaml  scenarios/*.yaml
  src/cryoguard/
    app.py  bus.py  models.py  config.py  clock.py
    capture/   source.py  rectify.py
    perception/ fast_tracker.py  sam_client.py  sam_worker.py  candidates.py
    telemetry/ psychrometrics.py  sim.py  replay.py
    fusion/    features.py  engine.py  decision.py
    orchestrator/ incident_fsm.py  actions.py
    adapters/  nvml.py  plc_client.py  redfish_client.py  scheduler.py
    sims/      plc_server.py  bmc_server.py  slurm_mock.py  checkpoint_store.py  thermal_model.py  train_job.py
    report/    muse_client.py  schema.py  fallback.py
    api/       routes.py  ws.py
    bench/     runner.py  metrics.py  report.py
  ui/  index.html  app.js  style.css
  fixtures/  manifest.yaml  videos/  telemetry/  labels/
  artifacts/  sam_cache/  incidents/  open_weights_proof.jsonl
  reports/  tests/
```

---

## 6. Canonical definitions (single source of truth)

### 6.1 Zones (chassis-plane polygons, defined in `topology.yaml`)

- **Zone 0, Hydraulic origin:** QD fittings, manifold, hose routing. Droplets possible during servicing.
- **Zone 1, Containment buffer:** drip tray and gutter channel. Non-conductive, passive capture.
- **Zone 2, Electrical hazard plane:** 48 V busbars, VRMs, PCIe sockets **including a 20 mm margin** baked into the polygon.

A candidate's zone = highest-risk zone containing its **mask's lowest-risk-to-highest-risk extent**: use the max zone index among mask pixels covering ≥5% of mask area.

### 6.2 Labels and states

Candidate labels: `STATIC_REFLECTION`, `GLINT_NOISE`, `SUSPECT_LEAK`, `ACTIVE_LEAK`, `BENIGN_CONDENSATION`. System labels: `NOMINAL`, `LOG_ONLY`, `SUSPECT_LEAK`, `UNVERIFIED_ANOMALY`, `CONFIRMED_LEAK`, `HIDDEN_HYDRAULIC_EVENT`, `EMERGENCY_TRIP`. Incident FSM: `OPEN → CLAMPING → ISOLATING → ISOLATED → CHECKPOINT_VERIFY → NODE_POWERED_DOWN → REQUEUED → RESOLVED_PENDING_REPAIR`; failure exits: `HARD_TRIP`, `FAILED_SAFE`.

### 6.3 Canonical decision table (first match wins, evaluated each fusion tick)

| Rule | Condition | System label | Action |
| :-- | :-- | :-- | :-- |
| R1 | Candidate zone = 2 AND P ≥ 0.50 AND ≥3 consecutive ticks AND camera healthy AND mask gates pass | `EMERGENCY_TRIP` | Parallel: PLC close valve + BMC ForceOff + scheduler drain. No checkpoint wait. Resume from last verified checkpoint (N−1) |
| R1b | PLC valve FAULT, or thermal watchdog ≥ 90°C, or clamp readback failure | `EMERGENCY_TRIP` | Same as R1 |
| R2 | P ≥ 0.85 AND zone ∈ {0,1} AND φv ≥ 0.2 AND φh ≥ 0.2 AND not `maintenance_mode` | `CONFIRMED_LEAK` | Incident sequence (Section 7.9) |
| R3 | 0.70 ≤ P < 0.85 | `UNVERIFIED_ANOMALY` | Query CDU telemetry (sim), alert technician, async report. No actuation |
| R4 | No visual candidate (φv ≤ 0.1) AND φh ≥ 0.6 for ≥3 s AND camera healthy | `HIDDEN_HYDRAULIC_EVENT` | Soft cordon (scheduler DRAIN only), alert technician, async report constrained to "no visual evidence". **No valve actuation** |
| R5 | 0.50 ≤ P < 0.70 with candidate | `SUSPECT_LEAK` (or `BENIGN_CONDENSATION` if φt < 0 and φh ≤ 0.1) | Raise sampling to 10 Hz; log |
| R6 | 0.40 ≤ P < 0.50 | `LOG_ONLY` | Log |
| R7 | P < 0.40 | `NOMINAL` | Telemetry logging |

Note: R3 and R5 ranges overlap at 0.70 boundaries by design only through ordering (R3 evaluated before R5). Implement strict half-open intervals.

### 6.4 Hysteresis

A state escalates only after its condition holds for `M` consecutive ticks (default M=3 for R2/R3, 3 for R1, 3 s for R4). De-escalation requires P below threshold − 0.05 for 5 ticks. Incidents never auto-de-escalate once `CONFIRMED_LEAK` or `EMERGENCY_TRIP` fires; they require operator reset.

### 6.5 Canonical identifiers

Rack `RACK-08B` · Branch `BRANCH-SEC-02` · QD fittings `QD-01`…`QD-08` (leak demo uses `QD-04`) · Failing node `gpu-node-08b` (4 GPUs: `GPU-0..3`) · Standby node `gpu-node-09a` · Job `train-llm-0427` · Incident ID `INC-RACK-08B-YYYYMMDD-NN` · PLC unit id `1` · Timestamps UTC ISO-8601 with `Z`.
