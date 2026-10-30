# CryoGuard AI — PRD Addendum v1.1

**Supplies:** §7 Requirements · §8 Contracts · §9 Fusion specification · §10 Decision semantics · §11 Simulator & protocol spec · §12 Benchmark protocol · §13 Fixture protocol · §14 Acceptance tests · §15 Errata · §16 Impact narrative policy · §17 Open-weights proof · §18 Submission checklist
**Base document:** `CryoGuard AI — Product Requirements Document (Build Spec for Coding Agent).md` v1.0
**Status:** proposed — this is the missing half of the PRD, reconstructed from the product intent in §0–§6.

---

## 0. Precedence and conventions

1. **This addendum overrides v1.0 where they conflict.** Where it is silent, v1.0 stands.
2. Every item marked **⟐** is **new or changed in v1.1** and is a proposal: accept or reject it explicitly. Unmarked items restate v1.0.
3. **Every numeric default here is a design prior or a physical assumption — never a measured result.** Priors live in `configs/*.yaml`, are frozen with a hash before the test split, and are never quoted as performance.
4. Requirement IDs follow v1.0 §0: `P0` = demo/submission fails without it, `P1` = score improver, `P2` = only if time remains.
5. Requirement → test traceability is in §14; requirement → build phase is in §19.

---

## §7. Requirements

### 7.1 Capture (`FR-CAP`)

| ID | Requirement | P | Verified by |
| :-- | :-- | :-- | :-- |
| FR-CAP-01 | File-replay video source with deterministic frame timestamps derived from fps, not wall clock. | P0 | `T-DET-01` |
| FR-CAP-02 | ArUco homography to the planar chassis frame; reprojection error recorded per clip; **hard failure** if the board is not detected in the first N frames. | P0 | `T-CAP-01` |
| FR-CAP-03 | Zone polygons and candidate geometry expressed in chassis-plane **millimetres**, not pixels. | P0 | `T-CAP-02` |
| FR-CAP-04 | ⟐ Webcam source behind the same interface as file replay, disabled by default. | P2 | manual |
| FR-CAP-05 | ⟐ Recorded clip metadata (fps, calibration, exposure) is read from `fixtures/manifest.yaml`; no clip-specific constants in code. | P0 | `T-DOC-02` |

### 7.2 Perception (`FR-PER`)

| ID | Requirement | P | Verified by |
| :-- | :-- | :-- | :-- |
| FR-PER-01 | Classical HSV/fluorescence tracker produces masks at the idle scan rate (2 Hz) with no network dependency. | P0 | `T-PER-01` |
| FR-PER-02 | SAM 3.1 is called through the OpenAI SDK Responses API, `model="sam-3.1"`, **text prompt on an ROI crop**, prompt ≤32 tokens enforced client-side. | P0 | `T-PER-02` |
| FR-PER-03 | SAM runs asynchronously at ≤2 Hz with in-flight cap 1; a slow call must never stall the tracker or decision loop. | P0 | `T-PER-03` |
| FR-PER-04 | Every SAM response is cached on disk keyed by `sha256(image_bytes + prompt + model)`; `--offline` fails loudly on a miss. | P0 | `T-DET-03` |
| FR-PER-05 | Candidate manager assigns stable IDs, associates masks across frames, and computes `dA/dt`, centroid velocity, `C_fluor`, mask variance, and `W`. | P0 | `T-PER-04`, `T-DATA-02` |
| FR-PER-06 | ⟐ Adaptive sampling: 2 Hz idle discovery → 5 Hz tracking while a candidate exists → 10 Hz on `SUSPECT_LEAK`. | P0 | `T-PER-05` |
| FR-PER-07 | ⟐ Camera-health signal with a defined contract (§10.6): health is false if brightness, focus, or saturation are outside configured bounds. | P0 | `T-PER-06` |
| FR-PER-08 | ⟐ Mask quality gates (`q`) with a defined contract (§10.6); R1 and R2 must fail closed when gates fail. | P0 | `T-DEC-01` |
| FR-PER-09 | ⟐ SAM/tracker disagreement policy: tracker is authoritative for tracking continuity; SAM re-anchors only when IoU ≥ 0.3. Disagreement is logged, never silently merged. | P1 | `T-PER-07` |
| FR-PER-10 | ⟐ No candidate may exist without an originating QD ROI within `candidate_link_radius_mm`. | P1 | `T-PER-08` |

### 7.3 Telemetry (`FR-TEL`)

| ID | Requirement | P | Verified by |
| :-- | :-- | :-- | :-- |
| FR-TEL-01 | Scenario telemetry replays at fixed `dt`, indexed by the **video** timestamp, not wall clock. | P0 | `T-DET-01` |
| FR-TEL-02 | Dew point via Magnus; relative and absolute humidity; all values in documented units. | P0 | `T-DATA-03` |
| FR-TEL-03 | Baselines (`p0`, `f0`, `ΔT_expected`) computed from a rolling pre-incident window, per run — never hard-coded. | P0 | `T-TEL-01` |
| FR-TEL-04 | ⟐ Every telemetry sample carries `simulated: true` and a channel id, so the UI badge cannot be forgotten. | P0 | `T-UI-02` |
| FR-TEL-05 | ⟐ Live simulator mode (no scenario file) behind the same interface. | P2 | manual |

### 7.4 Fusion (`FR-FUS`)

| ID | Requirement | P | Verified by |
| :-- | :-- | :-- | :-- |
| FR-FUS-01 | Visual evidence `φv` per §9.2, including `W` and quality `q`. | P0 | `T-FUS-01` |
| FR-FUS-02 | Hydraulic evidence `φh` per §9.3. | P0 | `T-FUS-02` |
| FR-FUS-03 | Thermal term `φt` per §9.4 is a **veto modulator only**; it can never raise `P` above the visual/hydraulic base. | P0 | `T-DEC-07` |
| FR-FUS-04 | `P` is bounded `[0,1]`, monotone non-decreasing in `φv` and `φh`, and never constant. | P0 | `T-FUS-03` |
| FR-FUS-05 | All weights, references, and windows come from config; the fusion config hash is recorded in every incident and every benchmark row. | P0 | `T-FUS-04` |
| FR-FUS-06 | ⟐ No temporal smoothing of `P` before decisioning. Evidence persistence is handled by hysteresis, not by a filter that can hide a tick. (`P_ui` may be smoothed for display only, labelled as such.) | P0 | `T-DEC-03` |
| FR-FUS-07 | ⟐ Fusion emits a per-candidate explanation payload (each term, weight, and contribution) for the UI and the report. | P1 | `T-UI-04` |

### 7.5 Decision (`FR-DEC`)

| ID | Requirement | P | Verified by |
| :-- | :-- | :-- | :-- |
| FR-DEC-01 | The rule table in §10.2 is **total**: every reachable input combination matches exactly one rule. | P0 | `T-DEC-09` |
| FR-DEC-02 | First match wins, evaluated top to bottom, in a fixed documented order. | P0 | `T-DEC-04` |
| FR-DEC-03 | Hysteresis per §10.4, using both consecutive-tick **and** minimum wall-time conditions. | P0 | `T-DEC-03` |
| FR-DEC-04 | `CONFIRMED_LEAK` and `EMERGENCY_TRIP` latch forever; only an explicit operator reset clears them. | P0 | `T-DEC-05` |
| FR-DEC-05 | `maintenance_mode` suppresses Zone 0/1 actuation but never suppresses Zone 2 emergency trip nor logging. | P0 | `T-DEC-06` |
| FR-DEC-06 | Zone assignment = max zone index among mask pixels covering ≥5% of mask area (§6.1), computed on the rectified mask. | P0 | `T-DATA-01` |
| FR-DEC-07 | ⟐ A camera-unhealthy system must never report `NOMINAL`; it reports `CAMERA_DEGRADED`. | P0 | `T-DEC-08` |
| FR-DEC-08 | ⟐ Multiple simultaneous candidates are decided independently and aggregated by severity → `P` → zone (§10.5). | P0 | `T-DEC-10` |
| FR-DEC-09 | ⟐ Every decision event carries `rule_id`, so the UI and the report can name the rule that fired. | P0 | `T-UI-05` |

### 7.6 Orchestrator (`FR-ORC`)

| ID | Requirement | P | Verified by |
| :-- | :-- | :-- | :-- |
| FR-ORC-01 | Power clamp is issued **first** and verified by readback before any valve command. | P0 | `T-FSM-01` |
| FR-ORC-02 | Clamp target is the queried `min_limit`, never a literal wattage. | P0 | `T-FSM-07` |
| FR-ORC-03 | Valve closure is classified `VERIFIED` ≤1200 ms, `VERIFIED_SLOW` 1200–1500 ms, `FAULT` >1500 ms. | P0 | `T-FSM-03` |
| FR-ORC-04 | Checkpoint verification runs **in parallel** from clamp time and never blocks isolation. | P0 | `T-FSM-02` |
| FR-ORC-05 | Thermal watchdog at 90 °C runs on its own task and trips even if perception is dead. | P0 | `T-FSM-05` |
| FR-ORC-06 | Valve `FAULT` escalates to node power cut; a failed power cut ends in `FAILED_SAFE`, not silence. | P0 | `T-FSM-04` |
| FR-ORC-07 | Workload resumes from the newest **hash-verified** checkpoint, normally N−1, and records why. | P0 | `T-FSM-06` |
| FR-ORC-08 | `deterministicActionsTaken` is append-only and sealed before the report task is scheduled. | P0 | `T-LLM-02` |
| FR-ORC-09 | ⟐ Operator reset is exposed as `POST /api/incidents/{id}/reset` with an explicit `confirm` body; no implicit reset exists. | P1 | `T-ORC-01` |

### 7.7 Adapters (`FR-ADP`)

| ID | Requirement | P | Verified by |
| :-- | :-- | :-- | :-- |
| FR-ADP-01 | NVML adapter with `Fake` and `Real` backends behind one interface; `Real` is optional-import. | P0 | `T-ADP-01` |
| FR-ADP-02 | PLC client speaks Modbus-TCP, unit id 1, port 5020 by default, scaled integer registers per §11.2. | P0 | `T-PROTO-01` |
| FR-ADP-03 | Redfish client implements `ForceOff` and `GracefulShutdown` idempotently against the BMC mock. | P0 | `T-PROTO-03` |
| FR-ADP-04 | Scheduler adapter implements DRAIN and REQUEUE to `gpu-node-09a` with checkpoint reference. | P0 | `T-ADP-02` |
| FR-ADP-05 | Every adapter call records request, response, latency, and outcome into the action log, success or not. | P0 | `T-ORC-02` |
| FR-ADP-06 | ⟐ Kubernetes is an interface stub only, with `cordon`/`drain` mapping documented and no runtime path. | P2 | code review |

### 7.8 Reporting (`FR-RPT`)

| ID | Requirement | P | Verified by |
| :-- | :-- | :-- | :-- |
| FR-RPT-01 | Muse Spark Open (partner-served) writes the report; the model id and endpoint are config, not code. | P0 | `T-RPT-01` |
| FR-RPT-02 | Report output is validated against the schema in §8.6; one repair retry, then deterministic fallback. | P0 | `T-LLM-04` |
| FR-RPT-03 | The report is scheduled only after `deterministicActionsTaken` is sealed and has no write path into it. | P0 | `T-LLM-02` |
| FR-RPT-04 | The report must not propose, describe as taken, or imply any actuation not present in the sealed action log. | P0 | `T-RPT-02` |
| FR-RPT-05 | Each request/response pair is appended to `artifacts/open_weights_proof.jsonl` per §17. | P0 | `T-RPT-03` |
| FR-RPT-06 | ⟐ Constrained prompts: the R4 report must state "no visual evidence"; the condensation report must state the veto reasons. | P0 | `T-RPT-04` |
| FR-RPT-07 | ⟐ Deterministic fallback produces a complete, schema-valid report with `source="deterministic_fallback"`. | P0 | `T-LLM-01` |

### 7.9 Incident sequence (canonical, replaces v1.0's missing §7.9)

**Clock:** `t=0` is the moment the decision is sealed. All budgets are config keys with these defaults.

| Step | Actor | Action | Success condition | Failure branch |
| :-- | :-- | :-- | :-- | :-- |
| 0 | Decision | Seal `{rule_id, label, P, φv, φh, φt, W, zone, candidate_id, config_hash, tick}` into the incident; state `OPEN` | Incident exists with a sealed trigger block | — |
| 1 | NVML | Read `min_limit`; set power limit to it | Readback equals request within `clamp_tolerance_w` before `clamp_readback_timeout_ms` (2000) | **→ `EMERGENCY_TRIP`** (R1b): parallel valve close + `ForceOff`, no checkpoint wait |
| 2 | PLC | On clamp verified only: command valve close; poll status | `CLOSED` within 1200 ms → `VERIFIED`; 1200–1500 ms → `VERIFIED_SLOW` + warning | >1500 ms or `FAULT` → **`EMERGENCY_TRIP`**, `ForceOff`; if power-off fails → `HARD_TRIP` → `FAILED_SAFE` |
| 3 | Checkpoint store | **Started at t=0, in parallel:** enumerate checkpoints newest→oldest, verify SHA-256, take first verified as `resume_from` | Verified within `checkpoint_budget_ms` (30000) | Budget expires → resume from last verified; if none → `resume_from=null`, requeue from scratch and record it |
| 4 | BMC | After `ISOLATED`: `GracefulShutdown`, then `ForceOff` after `bmc_graceful_timeout_ms` (10000) | `PowerState=Off` observed | Retry once; then `HARD_TRIP` |
| 5 | Scheduler | DRAIN `gpu-node-08b`; REQUEUE `train-llm-0427` to `gpu-node-09a` from `resume_from` | Job running on standby | `FAILED_SAFE` with reason, alert technician |
| 6 | System | State `RESOLVED_PENDING_REPAIR`; seal `deterministicActionsTaken`; schedule report | Sealed action log hash recorded | — |
| 7 | Report | Async, `report_timeout_ms` (60000), no effect on state | Schema-valid report or fallback | Timeout/error → fallback, incident unchanged |
| — | Watchdog | Throughout: junction temp ≥ 90 °C → `ForceOff` immediately | — | Power-off fails → `FAILED_SAFE` |

**EMERGENCY_TRIP (R1 / R1b):** step 1 and step 2 are issued **in parallel**, no checkpoint wait, requeue from `resume_from` if verification completed before requeue, else N−1.

**Timing budget (asserted in tests, not in prose):** decision → clamp verified ≤ 2000 ms; clamp verified → valve closed ≤ 1500 ms; decision → isolated ≤ 3500 ms; checkpoint ≤ 30000 ms in parallel; decision → requeued ≤ 35000 ms.

**Hard invariant:** no timestamp of a valve command may precede the clamp-verified timestamp. Violation is a test failure, not a warning.

### 7.10 Benchmark (`FR-BEN`)

| ID | Requirement | P | Verified by |
| :-- | :-- | :-- | :-- |
| FR-BEN-01 | `make bench` runs every manifest clip through the production pipeline, offline, with no network access. | P0 | `T-DET-02` |
| FR-BEN-02 | Per-clip rows include: clip id, split, expected label, final label, fired rule, peak `P`, detection outcome, latency breakdown. | P0 | `T-BEN-01` |
| FR-BEN-03 | Recall/precision reported with raw counts `k/n` **and** Clopper–Pearson 95% CIs. | P0 | `T-BEN-02` |
| FR-BEN-04 | ⟐ Augmented or duplicated clips are never counted as independent samples; the split is frozen before tuning. | P0 | `T-BEN-03` |
| FR-BEN-05 | Two consecutive runs produce byte-identical `benchmark.json` and `benchmark.md`. | P0 | `T-DET-02` |
| FR-BEN-06 | Latency is measured at four points: perception tick, SAM call (cached/uncached separated), fusion, decision→action. | P0 | `T-BEN-04` |

### 7.11 UI (`FR-UI`)

| ID | Requirement | P | Verified by |
| :-- | :-- | :-- | :-- |
| FR-UI-01 | Single page, no build step, served by FastAPI. | P0 | manual |
| FR-UI-02 | Every simulator-backed panel carries a visible `SIMULATED` badge; the camera panel carries `REAL`. | P0 | `T-UI-02` |
| FR-UI-03 | Canvas overlay: rectified frame, zone polygons, tracker mask, SAM mask, candidate trail. | P0 | manual |
| FR-UI-04 | Decision strip: `P`, `φv`, `φh`, `φt`, `W`, zone, candidate label, fired rule, hysteresis counter — all from the WebSocket, no literals. | P0 | `T-UI-03` |
| FR-UI-05 | Action timeline matches `deterministicActionsTaken` exactly, including `VERIFIED_SLOW` and `FAULT`. | P0 | `T-UI-05` |

### 7.12 Operations (`FR-OPS`)

| ID | Requirement | P | Verified by |
| :-- | :-- | :-- | :-- |
| FR-OPS-01 | Config loads from `configs/*.yaml`; unknown keys are errors; `config_hash` is logged at startup. | P0 | `T-OPS-01` |
| FR-OPS-02 | Secrets only from env; `.env.example` complete; missing keys fail only when the live path is enabled. | P0 | `T-OPS-02` |
| FR-OPS-03 | ⟐ Structured JSON logs with `incident_id`, `rule_id`, `actor`, `latency_ms` where applicable. | P0 | `T-OPS-03` |
| FR-OPS-04 | ⟐ Incident files are self-contained: clip id, config hash, fusion freeze hash, git SHA. | P0 | `T-OPS-04` |
| FR-OPS-05 | ⟐ Schema version in every incident and report (`schema_version`). | P0 | `T-OPS-04` |

---

## §8. Contracts

### 8.1 Event bus topics

| Topic | Payload | Rate |
| :-- | :-- | :-- |
| `frame` | `frame_id, ts, camera_healthy` | source fps |
| `telemetry` | `ts, channels{pressure, flow, coolant_temp, humidity, ambient}, simulated` | 2–10 Hz |
| `candidate` | `candidate_id, zone, area_mm2, centroid_mm, W, q, label` | per tick |
| `fusion` | `candidate_id, phi_v, phi_h, phi_t, W, P, rule_id, label` | per tick |
| `decision` | `candidate_id, rule_id, label, hysteresis{tick_count, hold_ms}` | on change |
| `incident` | `incident_id, state, trigger, resume_from` | on transition |
| `action` | full `ActionRecord` | per action |
| `report` | `incident_id, source, schema_version` | on completion |
| `health` | `camera, sam, plc, bmc, scheduler, llm` each `{ok, detail}` | 1 Hz |

### 8.2 Incident schema

```
incident_id, schema_version, created_at, updated_at
rack_id, branch_id, node_id, job_id
trigger{ rule_id, system_label, P, phi_v, phi_h, phi_t, W, zone,
         candidate_id, tick_count, hold_ms, config_hash, fusion_freeze_hash }
timeline[ ActionRecord ]          # every attempt, in order
deterministic_actions_taken[ ActionRecord ]   # sealed, append-only
checkpoint{ resume_from, expected, verified, reason }
camera{ healthy, clip_id, frame_index, quality }
report{ source, model, schema_version, latency_ms, text, validated }
outcome{ state, requeued_to, failed_reason }
operator{ reset_at, reset_by }
```

`ActionRecord`:

```
seq, ts, actor(NVML|PLC|BMC|SCHEDULER|SYSTEM|OPERATOR), command, target
requested, observed
status(OK|VERIFIED|VERIFIED_SLOW|FAULT|TIMEOUT|SKIPPED|FAILED)
latency_ms, detail, error
```

### 8.3 WebSocket messages

`{"type": "<topic>", "ts": "<ISO-8601Z>", "seq": <int>, "payload": {…}}` — one message per bus event; no batching; dropped frames are logged as `dropped` and never replayed out of order.

### 8.4 Config key inventory

| File | Section | Keys (defaults in parentheses) |
| :-- | :-- | :-- |
| `default.yaml` | `camera` | `width, height, fps, exposure_locked, health{brightness_min, brightness_max, focus_min, saturation_max}` |
| | `perception` | `idle_hz(2), track_hz(5), suspect_hz(10), min_area_mm2, max_area_mm2, link_radius_mm` |
| | `sam` | `model("sam-3.1"), base_url, prompt_max_tokens(32), max_hz(2), in_flight(1), cache_dir, offline(bool)` |
| | `fusion` | `priors{w_area, w_fluor, w_dp, w_slope, w_flow, kappa_t}, refs{growth_per_s, dp_rel, dp_rate_per_s, flow_imbalance, dew_margin_c, thermal_excess_c}, windows{baseline_s, growth_s}` |
| | `decision` | `thresholds{r1(0.50), r2(0.85), r3(0.70), r5(0.50), r6(0.40)}, hold_ms{r1(600), r2(600), r4(3000)}, drop_margin(0.05), drop_ticks(5), drop_ms(1500), zones{r1(2)}` |
| | `orchestrator` | `clamp_readback_timeout_ms(2000), clamp_tolerance_w, valve_verified_ms(1200), valve_fault_ms(1500), checkpoint_budget_ms(30000), bmc_graceful_timeout_ms(10000), thermal_trip_c(90), watchdog_period_ms(100)` |
| | `thermal` | `c_th_j_per_k(350), ambient_c, expected_dt_k, assumption_note` |
| | `report` | `provider, model, base_url, timeout_ms(60000), max_retries(1), fallback_enabled(true)` |
| | `bench` | `split, repeats(3), confidence(0.95), seed` |
| `topology.yaml` | `chassis` | `width_mm, height_mm, aruco{id, dictionary, size_mm}` |
| | `zones` | per zone: `id, polygon_mm[], margin_mm(20 for zone 2)` |
| | `qd` | `QD-01..08: {roi_mm[], zone(0), branch}` |
| | `nodes` | `gpu-node-08b{gpus[GPU-0..3]}, gpu-node-09a{…}` |
| `scenarios/*.yaml` | | `clip_id, t0_offset_s, channels{pressure[], flow[], coolant_temp[], humidity[], ambient[]}, expected{label, zone, qd}, fault_injection{…}` |

### 8.5 Modbus register map (unit id 1, port 5020)

| Offset | Name | Type | Scaling | Access |
| :-- | :-- | :-- | :-- | :-- |
| 0 | `pressure` | int16 | psi × 10 | RO |
| 1 | `flow` | uint16 | L/min × 100 | RO |
| 2 | `coolant_temp` | int16 | °C × 10 | RO |
| 3 | `humidity` | uint16 | %RH × 10 | RO |
| 4 | `valve_state` | uint16 | 0 OPEN, 1 CLOSING, 2 CLOSED, 3 FAULT | RO |
| 5 | `valve_cmd` | uint16 | 0 noop, 1 close, 2 open | WO |
| 6 | `heartbeat` | uint16 | increments every 100 ms | RO |
| 7 | `fault_code` | uint16 | 0 none, 1 stuck, 2 limit-timeout, 3 sensor | RO |

No floats on the wire (D-13). The client must reject unknown `fault_code` values rather than treating them as 0.

### 8.6 Report schema

```
schema_version, incident_id
headline            # ≤ 120 chars
summary             # 3–6 sentences, facts only
evidence{ optical, hydraulic, thermal, continuity }   # each: what was observed + the numbers from the incident
timeline_summary[]  # derived from deterministic_actions_taken
actions_taken[]     # must equal the sealed log verbatim; validator compares
uncertainty[]       # what the system could not determine
not_observed[]      # e.g. "no visual evidence" for HIDDEN_HYDRAULIC_EVENT
recommendation      # human follow-up, never an actuation command
provenance{ model, source, latency_ms, prompt_hash }
```

Validator rejects the report if `actions_taken` differs from the sealed log by even one element.

### 8.7 Checkpoint manifest

```
{ job_id, step, node, path, sha256, bytes, created_at }
```
Layout: `checkpoints/train-llm-0427/step-{n}/manifest.json`. `N−1` is the newest step whose manifest hash matches the payload; a corrupted newest step is skipped and the skip reason is recorded.

---

## §9. Fusion specification

Written so a reviewer can re-derive every number. All references are **design priors** in `configs/default.yaml` and are frozen before the test split.

### 9.1 Notation

Per candidate `c` at tick `t`, on the rectified chassis-plane mask: `A_t` area (mm²), `C_t` mean fluorescence intensity inside the mask, `q_t ∈ [0,1]` quality, `W_t ∈ {0,1}` continuity, `dA/dt` from the window in `fusion.windows.growth_s`.

### 9.2 Visual evidence

```
g   = clamp( ln(max(A_t,1) / max(A_{t-Δ},1)) / Δt / growth_ref , 0, 1 )     # normalized growth rate
cn  = clamp( (C_t - C_bg) / max(C_ref - C_bg, ε) , 0, 1 )                  # normalized fluorescence
φv  = q_t * W_t * ( w_area * g + w_fluor * cn )        with w_area + w_fluor = 1
```

`C_bg`, `C_ref` are per-clip background and 99th-percentile references computed at load time. `W=0` ⇒ `φv=0` (a detached blob is not a leak from a fitting). Defaults: `w_area=0.6`, `w_fluor=0.4`, `growth_ref=0.01 s⁻¹`.

### 9.3 Hydraulic evidence

```
d_rel   = clamp( (p0 - p_t) / max(p0, ε) / dp_ref , 0, 1 )
s_rel   = clamp( (-dp/dt) / dp_rate_ref , 0, 1 )            # window = baseline_s
i_rel   = clamp( (f0 - f_t) / max(f0, ε) / flow_imbalance_ref , 0, 1 )
φh      = clamp( w_dp*d_rel + w_slope*s_rel + w_flow*i_rel , 0, 1 )
```

Defaults: `dp_ref=0.03` (3 % relative drop), `dp_rate_ref=0.01 s⁻¹`, `flow_imbalance_ref=0.10`, `w_dp=0.5, w_slope=0.3, w_flow=0.2`. `p0`, `f0` are rolling medians over `baseline_s` from before the first candidate.

### 9.4 Thermal modulator

`φt ∈ [-1, 1]`, negative = condensation plausible:

```
dew_margin = T_surface - T_dew
v_dew      = clamp( dew_margin / dew_ref , -1, 1 )                   # -1 at/below dew point
excess     = ΔT_coolant_observed - ΔT_expected(f_t, P_gpu)
v_heat     = clamp( excess / thermal_excess_ref , 0, 1 )
φt         = clamp( 0.7*v_dew + 0.3*v_heat , -1, 1 )
```

Defaults: `dew_ref=5 K`, `thermal_excess_ref=5 K`, `T_surface` approximated by coolant inlet temperature (documented assumption).

### 9.5 Leak probability

```
base = 1 - (1 - φv)(1 - φh)
P    = clamp( base * (1 + kappa_t * min(φt, 0)) , 0, 1 )      # kappa_t = 0.6
```

Properties, all asserted in tests:

1. `P ∈ [0,1]` for all inputs.
2. `∂P/∂φv ≥ 0`, `∂P/∂φh ≥ 0`, `∂P/∂φt ≥ 0` but only through `min(φt,0)`, so **positive `φt` has zero effect** (D-10).
3. `φt=-1` scales the base by `1-0.6=0.4`; `φt=-0.5` by 0.7.
4. `φv=φh=0 ⇒ P=0`.
5. `φv=1, φh=1 ⇒ P=1`.
6. The **AND** that makes `CONFIRMED_LEAK` require both channels lives in the rule table (φv≥0.2 and φh≥0.2), not in `P`. `P` is a suspicion score; the rules are the safety argument.

**Explicit non-claims:** `P` is not a calibrated probability. No ECE, Brier, or reliability diagram is computed. The README and report must not call it "calibrated", "confidence", or "probability of a leak" without that qualifier — use "suspicion score".

---

## §10. Decision semantics

### 10.1 Evaluation order

Decisions run every fusion tick. If an incident is latched (`CONFIRMED_LEAK`, `EMERGENCY_TRIP`, `HARD_TRIP`, `FAILED_SAFE`), the **FSM owns the state** and the label table is not evaluated; only escalation is possible.

### 10.2 Total rule table (⟐ closes the two v1.0 holes)

| # | ID | Condition | System label | Action |
| :-- | :-- | :-- | :-- | :-- |
| 1 | R1b | PLC `FAULT`, or junction temp ≥ 90 °C, or clamp readback failed | `EMERGENCY_TRIP` | Valve close + `ForceOff` + DRAIN in parallel; no checkpoint wait |
| 2 | R8 ⟐ | `camera_healthy == false` | `CAMERA_DEGRADED` | No actuation, never `NOMINAL`; alert; hold last non-degraded label in history; escalate sampling |
| 3 | R1 | `zone == 2` AND `P ≥ 0.50` AND ticks ≥ `hold_ticks` AND `hold_ms ≥ hold_ms` AND `q ≥ q_min` | `EMERGENCY_TRIP` | Parallel: valve close + `ForceOff` + DRAIN |
| 4 | R2 | `P ≥ 0.85` AND `zone ∈ {0,1}` AND `φv ≥ 0.2` AND `φh ≥ 0.2` AND `q ≥ q_min` AND NOT `maintenance_mode` | `CONFIRMED_LEAK` | §7.9 |
| 5 | R2b ⟐ | `P ≥ 0.85` AND `zone ∈ {0,1}` AND (any R2 precondition unmet) | `UNVERIFIED_ANOMALY` | Alert + 10 Hz + CDU query; **DRAIN only** when `φv ≤ 0.1` AND `φh ≥ 0.6` (same situation R4 would catch). **No valve** |
| 6 | R2c ⟐ | `P ≥ 0.85` AND `zone == 2` AND R1 preconditions unmet | `UNVERIFIED_ANOMALY` | Alert + 10 Hz + **DRAIN only** (soft cordon). **No valve** |
| 7 | R4 | `φv ≤ 0.1` AND `φh ≥ 0.6` for ≥ 3000 ms AND camera healthy | `HIDDEN_HYDRAULIC_EVENT` | DRAIN only, alert, report constrained to "no visual evidence". **No valve** |
| 8 | R3 | `0.70 ≤ P < 0.85` | `UNVERIFIED_ANOMALY` | Alert + 10 Hz + CDU query. Zone 2 ⇒ also DRAIN only |
| 9 | R5 | `0.50 ≤ P < 0.70` | `BENIGN_CONDENSATION` if `φt < 0` AND `φh ≤ 0.1`, else `SUSPECT_LEAK` | 10 Hz, log, alert |
| 10 | R6 | `0.40 ≤ P < 0.50` | `LOG_ONLY` | Log |
| 11 | R7 | `P < 0.40` | `NOMINAL` | Telemetry logging |

**Why R4 moved above R3** (⟐): with no candidate, `P` can only come from `φh`, so ordering R3 first would split an identical hydraulic situation into "drain" or "no action" purely by the magnitude of the same channel. R4's condition is strictly more informative and should win.

**Totality proof:** after R1b and R8, camera is healthy and no hard fault exists. If `P ≥ 0.85`, exactly one of R2/R2b/R2c applies by zone and preconditions; if `P < 0.85` and R4 does not apply, R3 covers `[0.70,0.85)`, R5 `[0.50,0.70)`, R6 `[0.40,0.50)`, R7 `(−∞,0.40)`. Every interval is half-open and adjacent, so exactly one rule matches. `T-DEC-09` asserts this exhaustively with Hypothesis.

### 10.3 Severity ranking (for aggregation and UI)

`FAILED_SAFE > HARD_TRIP > EMERGENCY_TRIP > CONFIRMED_LEAK > UNVERIFIED_ANOMALY > HIDDEN_HYDRAULIC_EVENT > SUSPECT_LEAK > BENIGN_CONDENSATION > LOG_ONLY > CAMERA_DEGRADED > NOMINAL`

### 10.4 Hysteresis (⟐ both criteria, not either)

- Escalation to a rule requires its condition true for `hold_ticks` consecutive ticks **and** for at least `hold_ms` of wall time between the first and last of those ticks. This stops 3 ticks at 10 Hz (300 ms) from counting the same as 3 ticks at 2 Hz (1500 ms).
- De-escalation requires `P < threshold − 0.05` for 5 consecutive ticks and ≥1500 ms.
- Counters are per `(candidate_id, rule_id)` and reset on candidate loss or an intervening higher rule.
- Defaults: `hold_ticks=3`, `hold_ms{r1:600, r2:600}`, R4 is time-only (`3000 ms`).
- Latched labels (`CONFIRMED_LEAK`, `EMERGENCY_TRIP`) never de-escalate.

### 10.5 Multiple candidates

Each candidate is evaluated independently. The system label is the **highest severity** across candidates; ties break by higher `P`, then higher zone, then lower `candidate_id` for determinism. Only the winning candidate drives the incident. All candidates remain in the timeline.

### 10.6 Two definitions v1.0 left open (⟐)

**Camera healthy** — all must hold, else `false`:
- mean luma within `[health.brightness_min, health.brightness_max]`;
- Laplacian variance ≥ `health.focus_min` (defocus ⇒ blind);
- clipped-pixel fraction ≤ `health.saturation_max` (glare ⇒ blind);
- frame arrives within `2/fps` of its expected timestamp (frozen feed ⇒ blind).

**Mask quality `q`** — product of three clamped terms:
`q = q_area * q_clip * q_compact`, where `q_area = clamp(A/min_area, 0, 1)` (0 below `min_area_mm2`), `q_clip = 1 - clipped_fraction_in_mask`, `q_compact = clamp(4πA/perimeter² / compactness_ref, 0, 1)`. `q_min` default 0.5. R1/R2 fail closed when `q < q_min`.

---

## §11. Simulator and protocol specification

### 11.1 PLC FSM

States `OPEN → CLOSING → CLOSED`, fault state `FAULT`. Travel time is sampled from config around the nominal (default 900 ms) so the 1200/1500 bands are actually exercised. `valve_cmd` is edge-triggered; a command while `CLOSING` is rejected with `fault_code=1`. Heartbeat stops ⇒ client marks PLC unhealthy within 500 ms.

### 11.2 Thermal model

`dT/dt = (P_gpu − P_min) / C_th` while clamping, `C_th = 350 J/K` (assumption, labelled in config and README). Ambient and coolant inlet are scenario inputs. The model is a lumped single-node approximation — state that in the README, do not imply CFD.

### 11.3 BMC mock

`GET /redfish/v1/Systems/gpu-node-08b` → `{PowerState}`, `POST …/Actions/ComputerSystem.Reset {ResetType}` → 202. Plain HTTP on `127.0.0.1:8443` for the demo (⟐ documented deviation: 8443 is conventionally TLS; the mock has no TLS and no auth, and must never be exposed off-host).

### 11.4 Scheduler mock

`drain(node)`, `requeue(job, target, checkpoint)`, `state(job)`. States `PENDING|RUNNING|DRAINING|REQUEUED|COMPLETED|FAILED`. REQUEUE without a checkpoint starts from step 0 and records `resume_from=null`.

### 11.5 Fault injection (test-only, ⟐)

`fault_injection` in a scenario file: `plc_fault_at_ms`, `clamp_readback_fail`, `valve_travel_ms_override`, `thermal_runaway`, `camera_blackout_at_ms`, `sam_timeout`. Used by the acceptance suite; ignored unless `--allow-faults` is passed.

---

## §12. Benchmark protocol

- **Splits:** dev (tuning) and test (reporting), frozen in `fixtures/manifest.yaml` before any tuning. No clip appears in both.
- **Sample size:** report `n` per class. With the expected 10–15 clips, CIs are wide — say so in the README. Never state a target as achieved.
- **Detection definition:** a positive clip is detected if its final system label is `CONFIRMED_LEAK`, `EMERGENCY_TRIP`, or `HIDDEN_HYDRAULIC_EVENT` for its expected class. A negative clip is a false positive under the same labels. `SUSPECT_LEAK` alone is not a detection.
- **Endpoints reported:** counts, recall, precision, Clopper–Pearson 95 % CI, per-clip table, latency medians and p95, SAM cache hit rate, `run_hash`, git SHA, config hash, freeze hash.
- **Repetitions:** 3 runs; if any byte differs, the run is invalid and nondeterminism must be fixed before proceeding.
- **No augmentation counting**, no per-frame metrics presented as per-clip metrics.

---

## §13. Fixture protocol

**Rig (minimum viable):**
- Camera ≥1080p, 30 fps, **exposure and focus locked**, tripod, fixed distance recorded in the manifest.
- Excitation 365–395 nm UV source; **long-pass filter ≥420 nm on the lens**; room lights off.
- Fluorescent tracer in the reservoir (fluorescein-class dye; record concentration in the manifest). Tracer is the reason fluorescence discriminates coolant from condensation — without it the whole thesis fails.
- ArUco board (`DICT_5X5_100`, 100 mm) coplanar with the chassis face, visible in every clip.
- A clap or flash at `t=0` so video and telemetry share an origin; the offset goes in the scenario file.

**Per scenario, minimum 3 takes** (keeps a usable one when one is blurred):

| Clip | Content | Must show |
| :-- | :-- | :-- |
| `leak` | Tracer drip from QD-04, growing pool toward Zone 2 | growth over ≥10 s, signal ≥50 px above 3σ background for ≥10 consecutive frames |
| `condensation` | Plain water beads on cold surfaces under the same excitation | weak/variable candidate, **no growth**, hydraulic flat |
| `hidden` | No visual anomaly; scripted pressure decay | clean optical channel, hydraulic drop ≥3 s |
| negatives | static reflection, moving glint, blackout | no persistent candidate / camera unhealthy |

**Clip acceptance gate:** ArUco detected in ≥95 % of frames, no autofocus hunting, no clipping in the ROI, ground-truth label and zone agreed by two passes (author + review). A clip failing the gate is re-shot, not relabelled.

**Manifest fields:** `clip_id, path, scenario, camera, fps, resolution, calibration{aruco_id, dict, size_mm, homography}, t0_offset_s, ground_truth{label, zone, qd}, split, notes`.

---

## §14. Acceptance tests

Every test is a pytest node; `-m acceptance` runs the P0 set.

### Decision (`T-DEC`)
| ID | Asserts |
| :-- | :-- |
| `T-DEC-01` | R1 requires zone 2 **and** `P ≥ 0.50` **and** hold ticks **and** hold ms **and** `q ≥ q_min` |
| `T-DEC-02` | A single-tick zone-2 glint does not trip |
| `T-DEC-03` | Hysteresis needs ticks **and** wall time; 3 ticks in 300 ms do not escalate |
| `T-DEC-04` | Boundaries 0.40/0.50/0.70/0.85 each land in exactly one rule |
| `T-DEC-05` | `CONFIRMED_LEAK`/`EMERGENCY_TRIP` latch; only operator reset clears |
| `T-DEC-06` | `maintenance_mode` suppresses Zone 0/1 but never Zone 2 trip |
| `T-DEC-07` | `CONFIRMED_LEAK` requires `φv ≥ 0.2` **and** `φh ≥ 0.2`; thermal alone never confirms |
| `T-DEC-08` | Camera unhealthy ⇒ `CAMERA_DEGRADED`, never `NOMINAL` |
| `T-DEC-09` | **Totality**: Hypothesis over the reachable input space — exactly one rule matches, always |
| `T-DEC-10` | Multi-candidate aggregation picks severity → `P` → zone deterministically |
| `T-DEC-11` | Zone = max zone index over pixels ≥5 % of mask area |

### Safety path (`T-FSM`)
| ID | Asserts |
| :-- | :-- |
| `T-FSM-01` | `t_valve_cmd > t_clamp_verified`; no exception path violates it |
| `T-FSM-02` | Checkpoint verification starts before valve `CLOSED` |
| `T-FSM-03` | 1200/1500 ms bands map to `VERIFIED` / `VERIFIED_SLOW` / `FAULT` |
| `T-FSM-04` | `FAULT` ⇒ power cut; power-cut failure ⇒ `HARD_TRIP` ⇒ `FAILED_SAFE` |
| `T-FSM-05` | Thermal 90 °C trips with the perception loop dead |
| `T-FSM-06` | Corrupted checkpoint N ⇒ resume N−1 with recorded reason |
| `T-FSM-07` | Clamp target == queried `min_limit`; no literal wattage anywhere |
| `T-FSM-08` | Decision→isolated ≤ 3500 ms in the simulator |

### LLM isolation (`T-LLM`)
| ID | Asserts |
| :-- | :-- |
| `T-LLM-01` | Full incident completes with the report endpoint at 500 and with a 60 s hang |
| `T-LLM-02` | `deterministicActionsTaken` byte-identical with and without the LLM |
| `T-LLM-03` | AST scan: no `report` import reachable from `fusion/`, `orchestrator/`, `adapters/` |
| `T-LLM-04` | Malformed report ⇒ validated fallback, never partial |

### Determinism (`T-DET`)
| ID | Asserts |
| :-- | :-- |
| `T-DET-01` | Two replays of one clip produce identical timestamps and event sequence |
| `T-DET-02` | `make bench` twice ⇒ byte-identical JSON and Markdown |
| `T-DET-03` | SAM cache key changes iff image, prompt, or model changes |

### Protocols (`T-PROTO`)
| ID | Asserts |
| :-- | :-- |
| `T-PROTO-01` | Modbus registers are scaled integers; no float decoding |
| `T-PROTO-02` | Unit id 1, port 5020, unknown `fault_code` rejected |
| `T-PROTO-03` | ForceOff idempotent: 202 then `PowerState=Off`; repeat is harmless |
| `T-PROTO-04` | Corrupted checkpoint fails hash verification |

### Fusion (`T-FUS`)
| ID | Asserts |
| :-- | :-- |
| `T-FUS-01` | `φv=0` when `W=0` or mask detached; grows with `g` and `cn` |
| `T-FUS-02` | `φh` monotone in pressure drop, slope, flow imbalance; 0 with flat telemetry |
| `T-FUS-03` | `P` bounds and monotonicity; positive `φt` has zero effect |
| `T-FUS-04` | Changing a weight changes the config hash and invalidates a frozen run |

### Data (`T-DATA`)
| ID | Asserts |
| :-- | :-- |
| `T-DATA-01` | Zone rule with a synthetic mask spanning zones |
| `T-DATA-02` | `W=1` only on an 8-neighbour path to the QD ROI for ≥3 ticks |
| `T-DATA-03` | Psychrometrics within 0.2 °C of published values |
| `T-DATA-04` | Thermal soak matches `ΔT·C_th/P_min`; `C_th` labelled an assumption |

### Bench (`T-BEN`)
| ID | Asserts |
| :-- | :-- |
| `T-BEN-01` | Per-clip rows complete; latency measured, not constant |
| `T-BEN-02` | 10/10 ⇒ CI ≈ [0.69, 1.0]; `n` and `k` present |
| `T-BEN-03` | No clip counted twice; split frozen |
| `T-BEN-04` | Cached vs uncached SAM latency reported separately |

### Capture / UI / Ops / Report
| ID | Asserts |
| :-- | :-- |
| `T-CAP-01` | Missing ArUco ⇒ hard failure with the clip id |
| `T-CAP-02` | Zone polygon area equals configured mm² within 1 % |
| `T-PER-01…08` | Tracker on synthetic blobs; SAM token guard; async non-blocking; cache; `W`; sampling rates; health; disagreement |
| `T-TEL-01` | Baselines computed from pre-incident window, not literals |
| `T-UI-02` | Five simulated panels carry `SIMULATED`, camera carries `REAL` |
| `T-UI-03` | No numeric probability literals in `ui/` |
| `T-UI-05` | Timeline equals `deterministicActionsTaken` |
| `T-OPS-01…04` | Config hash, env handling, log fields, incident self-containment |
| `T-ORC-01/02` | Operator reset requires explicit confirm; every adapter call logged |
| `T-ADP-01/02` | NVML backends interchangeable; scheduler drain/requeue |
| `T-RPT-01…04` | Model id from config; schema; no invented actuation; constrained R4 text |
| `T-DOC-01/02` | README real-vs-simulated table; every numeric claim traceable to `reports/` or labelled illustrative |

---

## §15. Errata to v1.0

| ID | v1.0 statement | Correction |
| :-- | :-- | :-- |
| E-01 | §6.3 note: "R3 and R5 overlap at the 0.70 boundary" | They do not: R3 is `[0.70,0.85)`, R5 is `[0.50,0.70)`. Note deleted; half-open intervals stand. |
| E-02 | §6.3 table, rules R1–R7 | Not total. `P ≥ 0.85` with unmet confirmation preconditions matched nothing. Fixed by R2b/R2c. |
| E-03 | §6.2 system labels | No blind state. Added `CAMERA_DEGRADED` (R8). |
| E-04 | §6.2 FSM is linear through `CHECKPOINT_VERIFY` | Contradicts D-06. `CHECKPOINT_VERIFY` is a **parallel branch** started at clamp time, not a serial state. |
| E-05 | §6.4 de-escalation "5 ticks" | Ambiguous at 2/5/10 Hz. Now 5 ticks **and** ≥1500 ms. |
| E-06 | §6.3 rule order R4 after R3 | R4 moved before R3 (§10.2) to avoid label flapping on the same hydraulic signal. |
| E-07 | §6.3 R5 "with candidate" | Left `P ∈ [0.50,0.70)` without a candidate unhandled. Qualifier removed; `BENIGN_CONDENSATION` branch keyed on `φt`/`φh`. |
| E-08 | §5 "BMC mock on 8443/HTTP" | Documented as plain HTTP on 127.0.0.1:8443, no TLS, no auth, host-local only. |
| E-09 | §6.1 "lowest-risk-to-highest-risk extent" | Ambiguous phrasing; the operative rule is the max zone index over pixels covering ≥5 % of mask area, and that is what is tested. |
| E-10 | §0.1 "Sections 7–14 are the requirements" | Those sections were absent from the circulated file; this addendum supplies them. Where both exist, the addendum wins. |

---

## §16. Impact narrative policy (the 30 % axis)

v1.0 has no impact content and explicitly distrusts its own figures (D-16). This section defines *what may be claimed* without inventing anything.

**Argument structure (no numbers in the skeleton):**

1. **Who pays:** the operator running high-density liquid-cooled GPU capacity, and their customer whose job is interrupted.
2. **Failure modes today:** optical-only (glare, reflections, condensation → alert fatigue → muted alarms) and hydraulic-only (blind to which branch, and to slow external seepage).
3. **What changes here:** two independent physical channels must agree before a valve moves; the actuation path is deterministic and LLM-free; the optical channel is real footage, not a render.
4. **What this is not:** not certified, not deployed, not a replacement for a leak-detection standard, simulated except for the camera.

**Claim table — every figure the pitch or README uses must land in one row:**

| Claim | Status | Rule |
| :-- | :-- | :-- |
| Cost of a leaked tray / GPU / hour of downtime | **UNSOURCED** | Cite a public source with a URL, or write "illustrative" in the same sentence. Never a bare number. |
| GPU thermal time constant / soak behaviour | Assumption | Label `C_th = 350 J/K` as assumed; show the formula, not a result. |
| Detection performance | Measured | Only from `reports/benchmark.json`, with `n` and CI, never as a target. |
| Latency | Measured | Only from the bench, cached and uncached separated. |
| "Zero hardware damage" | **Banned** | Not demonstrated in simulation. Replace with "contained in simulation". |
| Market size / TAM | Not computed | Omit unless sourced; a hackathon judge does not need a TAM. |

---

## §17. Open-weights proof format

`artifacts/open_weights_proof.jsonl`, one JSON object per line, append-only:

```
{ ts, incident_id, provider, model, endpoint, request_id,
  prompt_sha256, latency_ms, prompt_tokens, completion_tokens,
  http_status, schema_valid, fallback_used }
```

No prompt text and no key material in the log — hashes only. This is what the infrastructure partner verifies.

---

## §18. Submission checklist

- [ ] Repo public, description and topics set, no `.env`, no keys (`git ls-files | grep -i env` clean).
- [ ] `make demo SCENARIO=leak|condensation|hidden` works headless and with UI from a fresh clone.
- [ ] `make bench` reproduces `reports/benchmark.{md,json}` offline, byte-identical.
- [ ] README: pitch, architecture, real-vs-simulated table, ≤10 setup commands, limitations, every number traceable.
- [ ] `artifacts/open_weights_proof.jsonl` present with ≥1 real Muse Spark Open request.
- [ ] Video ≤3:00, uploaded, `SIMULATED` badges legible.
- [ ] All P0 acceptance tests green on `main`.
- [ ] Submission filed by **19:00 ET, 2026-11-04** (2 h before the 21:00 deadline).

---

## §19. Requirement → build phase map

| Requirements | Phase (implementation plan) |
| :-- | :-- |
| FR-CAP-01…05 | P1 |
| FR-PER-01…10 | P2 (01–05 minimal), P3 (all) |
| FR-TEL-01…05, FR-ADP-01…06 | P4 |
| FR-FUS-01…07, FR-DEC-01…09 | P2 (minimal), P5 (full) |
| FR-ORC-01…09 | P2 (minimal FSM), P6 (full) |
| FR-RPT-01…07 | P2 (fallback), P7 (full) |
| FR-BEN-01…06 | P8 |
| FR-UI-01…05 | P9 |
| FR-OPS-01…05 | P0, enforced by CI |

---

## §20. What this addendum deliberately does not do

- It does not claim any performance number. Every threshold above is a design prior.
- It does not define a 3D fluid model, a training procedure, or a certification path.
- It does not remove the possibility that the fixtures are unusable; if the leak clip fails §13's acceptance gate, the optical claim must be restated honestly rather than worked around in code.
- It does not resolve the open question of whether the organisers' Muse Spark Open endpoint matches the community-cited Baseten model id — that stays a Day-1 verification item (plan P0-T01/P0-T03).
