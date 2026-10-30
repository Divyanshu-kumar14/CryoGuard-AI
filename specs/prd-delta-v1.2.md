# CryoGuard AI — PRD Delta v1.2

**Purpose:** reconcile the complete PRD (`CryoGuard AI — PRD v1.1 (complete).md`, §7–§14 now present) against its own §0–§6, and list what is *still* unspecified.
**Supersedes:** `archive/prd-addendum-v1.1.md` (written against the truncated PRD; its §7–§14 are now obsolete).
**Precedence:** complete PRD §7–§14 → this delta (for the items in §2–§4) → v1.0 §0–§6.

---

## 1. What the complete PRD settles

The following are now canonical and no longer need reconstruction: requirement IDs and priorities (§7), config skeleton (§9), Modbus/Redfish surface (§10), FSM graph (§11), benchmark output format (§12), fixture list (§13), acceptance matrix (§14). The implementation plan is patched to match (§8 below).

---

## 2. Conflicts inside the complete PRD

Each is a real contradiction between two normative parts of the same document. Resolution stated; none requires new product scope.

### C-01 — "Calibrated" logistic contradicts D-03

**Where:** §7.3 heading "Calibrated Prior Logistic Leak Probability", `FR-FUS-02`; vs D-03 "never claim calibration unless ECE/Brier are computed".
**Problem:** there is no training data, no held-out calibration set, no ECE or Brier score. Calling the output "calibrated" is false and is exactly what D-03 forbids.
**Resolution:** keep the logistic form (it is fine and now canonical), keep `fusion.weights` as hand-set priors, and **never** describe `P` as calibrated or as a probability. UI label: `LEAK SUSPICION`. Report/README wording: "prior-weighted score". A code comment in `fusion/engine.py` cites D-03. `T-DOC-01` greps for "calibrated" in `ui/`, `README.md`, and the prompts.

### C-02 — `w_thermal` positive makes dry air count as leak evidence (vs D-10)

**Where:** `FR-FUS-01` φ_thermal = (T_surface − T_dew)/ΔT_scale; `FR-FUS-02` adds `w3·φ_thermal`; vs D-10 "thermal only modulates, never positive evidence".
**Problem:** a dry room (large positive margin) increases `P`, so a leak could be "confirmed" partly because the air is dry.
**Resolution:** evaluate `w_thermal · min(φ_thermal, 0)`. Dry conditions contribute exactly 0; proximity to dew point subtracts. One-line change, no config change, preserves `w_thermal: 1.25`. `T-FUS-03` asserts `∂P/∂φt = 0` for `φt > 0`.

### C-03 — Fixed 5 Hz decision tick vs adaptive 2/5/10 Hz

**Where:** `NFR-LAT-01` "fixed 5 Hz (200 ms) system update tick"; `FR-FUS-04` "every 200 ms tick"; vs D-18 "idle 2 Hz, tracking 5 Hz, 10 Hz on SUSPECT".
**Resolution:** separate the two rates. **Perception** samples adaptively (2/5/10 Hz) as D-18 requires; **fusion + decision evaluate at a fixed 5 Hz** on the latest feature vector, as NFR-LAT-01 requires. The 10 Hz suspect mode therefore buys more evidence per decision tick, not more decisions. Documented in `configs/default.yaml` as `system.sampling_rate_hz: 5` plus `perception.*_hz`.

### C-04 — `ISOLATED_SAFE` requires a committed checkpoint

**Where:** §11 `AWAITING_CHECKPOINT --> ISOLATED_SAFE: Input 1001 verified & Checkpoint committed`; `FR-ORC-03` "shall not power down until the checkpoint commit is verified or the 30 s timeout".
**Check before calling this fatal:** the physical order is still correct — `CLAMPING_POWER → CLOSING_VALVE → AWAITING_CHECKPOINT` issues the valve command *before* waiting. So D-06's "never wait for a checkpoint before clamping/isolating" is **not** violated. The defect is in the state label: the system reports "not isolated" for up to 30 s after the valve is physically closed.
**Resolution:** `ISOLATED` is reached on limit-switch verification alone. Checkpoint verification gates `NODE_POWERED_DOWN` and the requeue, not isolation. This also gives the correct UI story ("valve closed at 1.9 s, checkpoint verified at 12 s"). `T-FSM-02` asserts the valve-closed timestamp is independent of the checkpoint timestamp.

### C-05 — Resume from step `N` or `N−1`

**Where:** `FR-ORC-04` "loading from checkpoint step N"; §11 "`EMERGENCY_TRIP --> WORKLOAD_REQUEUED: … Requeued from Step N−1`"; v1.0 §6.3 R1 "Resume from last verified checkpoint (N−1)".
**Resolution:** resume from the **newest hash-verified** checkpoint. In a clean shutdown that is `N`; in an emergency trip the newest step is often in flight and fails verification, so it is `N−1`. The incident records which step was chosen and why. This satisfies all three statements without special-casing.

### C-06 — Two FSM vocabularies

**Where:** §6.2 `OPEN → CLAMPING → ISOLATING → ISOLATED → CHECKPOINT_VERIFY → NODE_POWERED_DOWN → REQUEUED → RESOLVED_PENDING_REPAIR` + `HARD_TRIP`, `FAILED_SAFE`; §11 `NOMINAL / SUSPECT / CONFIRMED_LEAK / CLAMPING_POWER / CLOSING_VALVE / AWAITING_CHECKPOINT / ISOLATED_SAFE / WORKLOAD_REQUEUED / EMERGENCY_TRIP`.
**Problem:** two state machines, two names for the same states, and §11 doubles as both the *system-label* machine and the *incident* machine.
**Resolution:** §6.2 is canonical for the **incident FSM** (it is labelled "single source of truth"); §11 is canonical for **transitions**. Alias map in §5. The label vocabulary (`NOMINAL`, `SUSPECT_LEAK`, `CONFIRMED_LEAK`, …) stays as §6.2 defines it and is a separate axis.

### C-07 — Identity homography fallback vs the 20 mm claim

**Where:** `FR-PER-02` "if markers are absent, fall back to an identity transform with an explicit uncalibrated warning"; vs D-19 ("millimetre claims are meaningless without calibration").
**Problem:** with an identity transform, zone polygons in chassis mm cannot be mapped, so a Zone-2 trip would be decided from meaningless geometry.
**Resolution:** identity fallback is allowed only with `system.allow_uncalibrated: true` (default **false**) **and** it disables Zone-2 emergency trips and all mm-based claims, showing an `UNCALIBRATED` banner. Simulation tests run calibrated. `T-CAP-01` asserts the default path fails hard.

### C-08 — The benchmark table in §12 contains illustrative numbers

**Where:** §12 sample output: 12/12 recall, 0/15 FPR, 1.84 s, 980 ms.
**Problem:** PRD rule 1 forbids fabricated numbers in the README, UI, or video, and a judge will compare the shipped report against this example.
**Resolution:** the §12 block is a **format template**. `reports/benchmark.md` is generated only by `bench/report.py`; the template lives in `bench/templates/` and every value is a placeholder. CI fails if `reports/benchmark.*` contains a number that is not present in `benchmark.json` (`T-BEN-05`).

### C-09 — Thermal watchdog is P1 in §7 but P0 in the decision table

**Where:** `FR-ORC-05 [P1]`; v1.0 §6.3 R1b makes ≥90 °C a `P0` trip; D-06 requires it.
**Resolution:** P0. A P1 watchdog would be the first thing cut and it is the last line of defence.

### C-10 — Scenario 3 is two different scenarios

**Where:** §13 `scenario_03_glint_noise.mp4` (dry chassis, moving flashlights); §14.1 "Scenario 3 E2E: pressure drop without visual droplet triggers `HIDDEN_HYDRAULIC_EVENT`".
**Problem:** the glint clip has no pressure drop script, and the hidden scenario has no clip.
**Resolution:** shoot four clips. `scenario_03_hidden_hydraulic.mp4` = dry chassis, motionless camera, **no visual anomaly at all**, paired with a pressure-decay script. Keep `scenario_04_glint_noise.mp4` from §13 as the negative control for temporal-variance rejection. Two different claims, two different clips.

### C-11 — Tailwind via CDN breaks the offline demo

**Where:** `FR-DSH-01` "Tailwind (via CDN or pre-compiled CSS)".
**Resolution:** vendor a pre-compiled `ui/tailwind.css` (or plain CSS). A CDN request during a recorded demo is an avoidable failure mode; `make demo` must work with networking off.

---

## 3. Red-team of the canonical fusion weights

Weights from §9: `w_visual 1.85, w_hydraulic 2.10, w_thermal 1.25, w_wetness 2.50, bias −2.80`. Three inputs worth checking before anyone builds against them. Arithmetic, not measurements.

| # | Situation | Terms | Result |
| :-- | :-- | :-- | :-- |
| R-1 | **Static bright reflection at a QD fitting in Zone 2.** Mask connected to the fitting ⇒ `W=1`; no growth ⇒ the growth term is 0; strong contrast ⇒ `φv≈0.3–0.4`; no hydraulic signal ⇒ `φh=0`; quiet room ⇒ `φt≈0`. | `z = 1.85·0.4 + 2.5 − 2.8 = +0.44` | **P = 0.61** → above the R1 threshold of 0.50 → **3 ticks (600 ms) and the rack trips.** |
| R-2 | **Condensation, connected beads** (`φv≈0.1`, `φh=0`, `W=1`, `φt=0`) | `z = 0.185 + 2.5 − 2.8 = −0.115` | **P = 0.47** → `LOG_ONLY`, but §14.1 demands `< 0.40` for Scenario 2. The test only passes because `φt` goes negative: with `φt=−1`, `P=0.20`. |
| R-3 | **Ideal leak**, all terms saturated | `z = 1.85+2.10+2.5−2.8 = 3.65` | P = 0.975. Fine. |

**R-1 is a real defect, not a tuning preference.** `w_wetness` alone (2.50) against `bias` (−2.80) puts any persistent, connected, fluorescent-looking blob at 0.43 before a single unit of growth or hydraulic evidence. Near a busbar that is a false trip — the exact failure D-04 exists to prevent. GLINT_NOISE does not save it: a static reflection is *persistent*, not a 1-tick glint.

**Fix (R-1):** add a growth-or-hydraulic gate to R1 and R2: require `dA/dt > growth_epsilon` **OR** `φh ≥ 0.2`. A real leak grows; a hydraulic-confirmed event near busbars still trips; a static reflection cannot. Keep `w_wetness` and `bias` untouched — the gate is in the rule table, where it is auditable.

**Fix (R-2):** `W` must be computed on **fluorescence-positive** mask pixels, not on any wet-looking region. Otherwise water beads inherit `W=1` and the condensation scenario can never satisfy its own acceptance criterion. Also `ΔT_scale` must exist in config (see §4) or `φt` is dimensionless nonsense.

**Consequence for the label set:** with the canonical weights, `BENIGN_CONDENSATION` (`P ∈ [0.50,0.70)`, `φt<0`, `φh≤0.1`) is nearly unreachable — a negative `φt` strong enough to matter also pushes `P` below 0.50. Treat `BENIGN_CONDENSATION` as a **candidate** label carrying the explanation, and expect the system label to be `NOMINAL`/`LOG_ONLY`. Do not write a test that demands the system label.

---

## 4. Still unspecified — required before Phase 5

`FR-FUS-01` names eight symbols and `configs/default.yaml` defines none of them. Each needs a value and a home.

| Symbol | Meaning | Proposed default | Unit | Where |
| :-- | :-- | :-- | :-- | :-- |
| `ΔT_scale` | normalises thermal margin | 5.0 | K | `fusion.refs` |
| `A_ref` | normalises area growth | 100 | mm²/s | `fusion.refs` |
| `C_max` | normalises fluorescence contrast | per-clip p99, not a constant | intensity | computed at load |
| Var normalisation | `Var(I_centroid)` must be scale-free or it swamps the sum | divide by `255²`, clamp `φv` to `[0,1]` | — | `fusion/engine.py` |
| `α, β` | visual term weights, `α+β=1` | 0.6, 0.4 | — | `fusion.priors` |
| `γ` | jitter penalty | 0.5 | — | `fusion.priors` |
| `ψ_p` | pressure response | `clamp((−dP/dt)/0.5, 0, 1)` PSI/s | — | `fusion.refs` |
| `ψ_q` | flow imbalance response | `clamp((Q_in−Q_out)/1.0, 0, 1)` L/min | — | `fusion.refs` |
| `w_p, w_q` | hydraulic term weights | 0.6, 0.4 | — | `fusion.priors` |
| `growth_epsilon` | R1/R2 growth gate (from R-1) | 0.02 | mm²/s | `decision` |
| `ΔT_expected` | thermal baseline for the heat term | `P_gpu / (ρ·c_p·Q)`, `ρ·c_p = 4.18×10⁶ J·m⁻³·K⁻¹` | K | `telemetry/sim.py` |
| `hysteresis_hold_ms` | §6.4 is tick-only; wall-time floor needed at variable rates | 600 (R1/R2), 3000 (R4) | ms | `decision` |
| `drop_margin` | de-escalation margin | 0.05 | — | `decision` |
| `maintenance_mode` | per-branch suppression flag | `false` | bool | `topology.yaml` |
| camera health | brightness / focus / saturation / frame-age bounds | see delta §6 | — | `camera.health` |
| mask gates `q` | `q_area · q_clip · q_compact`, `q_min` | `q_min = 0.5` | — | `perception` |
| `allow_uncalibrated` | C-07 gate | `false` | bool | `system` |
| NVML fake | `min_limit` | 200 W | W | `nvml` |
| bench split | dev/test partition | 60/40 per class, frozen pre-tuning | — | `fixtures/manifest.yaml` |

**Unit inconsistency to settle:** `FR-PER-04/05` compute area in **pixels** (`area_px`, centroid gating ≤30 px); `FR-FUS-01` normalises growth by `A_ref` and §6.1 maps masks in **millimetres**. Decision: perception tracks in rectified **mm²**; pixel units appear only inside the tracker before rectification. `T-DATA-05` asserts no px value reaches the fusion layer.

**Unused signal:** `FR-PER-05` requires `specular_jitter` in the candidate state, but no rule or formula consumes it. Either feed it into `Var(I_centroid)` (recommended) or delete it — an unreferenced field in a safety-adjacent struct invites silent rot.

**Still absent entirely:** `topology.yaml` contents (zones, QD ROIs, chassis size, ArUco IDs), `configs/scenarios/*.yaml` schema, and the `HIDDEN_HYDRAULIC_EVENT` handling in §11's FSM. The delta's §6 and the plan's Phase 1/4 cover these.

---

## 5. FSM canonicalisation

Canonical incident states are §6.2 names. §11 names map as:

| §11 name | Canonical (§6.2) | Note |
| :-- | :-- | :-- |
| `CLAMPING_POWER` | `CLAMPING` | starts at decision |
| `CLOSING_VALVE` | `ISOLATING` | issued only after clamp readback |
| `AWAITING_CHECKPOINT` | `ISOLATED` + parallel `CHECKPOINT_VERIFY` | checkpoint is a **branch**, not a state (C-04) |
| `ISOLATED_SAFE` | `ISOLATED` | reached on limit switch alone |
| `WORKLOAD_REQUEUED` | `REQUEUED` | |
| `EMERGENCY_TRIP` | `EMERGENCY_TRIP` (label) / `HARD_TRIP` (FSM exit) | |
| — | `NODE_POWERED_DOWN`, `RESOLVED_PENDING_REPAIR`, `FAILED_SAFE` | missing from §11; keep from §6.2 |

The mermaid diagram in §11 is a useful sketch of *transitions*; it is not the authoritative state list.

---

## 6. Definitions §14 still assumes

`T-DEC`, `T-FSM` and `T-UI` cannot be written without these two contracts (carried from the archived addendum):

**Camera healthy** — all must hold, else `false`: mean luma within `[brightness_min, brightness_max]`; Laplacian variance ≥ `focus_min`; clipped-pixel fraction ≤ `saturation_max`; frame age ≤ `2/fps`.
**Mask quality** `q = q_area · q_clip · q_compact` with `q_area = clamp(A/min_area, 0, 1)`, `q_clip = 1 − clipped_fraction`, `q_compact = clamp(4πA/perimeter² / compactness_ref, 0, 1)`; `q_min = 0.5`; R1/R2 fail closed below it.

---

## 7. Tests to add beyond §14

§14's matrix is a good floor but misses the four claims the pitch rests on:

| Test | Why it must exist |
| :-- | :-- |
| `T-LLM-02/03` | `deterministicActionsTaken` byte-identical with the LLM down; AST scan proves `report/` is unreachable from `fusion/`, `orchestrator/`, `adapters/` |
| `T-DET-02/03` | two `make bench` runs byte-identical; SAM cache key changes iff image/prompt/model changes |
| `T-DEC-09` | rule-table totality, property-based — this is what catches the `P ≥ 0.85` holes |
| `T-FSM-01` | valve command timestamp strictly after clamp readback, asserted on timestamps |
| `T-BEN-05` | no number in `benchmark.md` that is absent from `benchmark.json` (C-08) |
| `T-DOC-01` | "calibrated" never appears in shipped prose (C-01) |

---

## 8. Plan patches applied

1. Source of truth is now the complete PRD; the addendum is archived.
2. Phase 0 gains a reconciliation task for C-01…C-11 and the §4 parameter table.
3. Phase 4's PLC map is rewritten to §10.1 coils/discrete inputs (the plan previously used a holding-register-only map).
4. Phase 5 uses the canonical logistic weights, the `min(φt,0)` clamp, and the R-1 growth gate.
5. Phase 6 uses §6.2 state names with the §11 transitions.
6. Phase 8 generates the §12 report format; the §12 sample table is a template, never shipped as data.
7. Frontend vendors compiled CSS instead of a CDN.

---

## 9. Open items that cannot be closed on paper

- Muse Spark Open endpoint/model id for the organisers' verification (`P0-T01/P0-T03`).
- Whether the 5 Hz decision tick plus 10 Hz perception meets `NFR-LAT-01`'s 80 ms budget on the demo machine — must be **measured** in Phase 8, and if it misses, the plan says so rather than quoting the budget as achieved.
- Whether the recorded footage is good enough for `φv` to separate fluorescence from specular glint — Phase 1's acceptance gate decides this, not the PRD.

---

## 10. Parameter pinning record (2026-10-30)

The proposed parameter table was checked against FR-FUS-01 and the §9 weights. Where a
proposal changed the physics or contradicted a rule, it was rejected. Everything pinned
here lives in `configs/default.yaml` and is covered by `tests/test_fusion_math.py`.

| Parameter | Pinned | Rejected | Reason |
| :-- | :-- | :-- | :-- |
| `w_visual, w_hydraulic, w_thermal, w_wetness, b` | 1.85, 2.10, 1.25, 2.50, −2.80 | `w_hydraulic = 2.20` | §9 is the only traceable source. 2.20 moves scenario 1 from 0.776 to 0.780 — immaterial, and it breaks the single-source claim. |
| `α, β, γ` | 0.4, 0.4, 0.2 | 0.6, 0.4, 0.5 | β = 0.4 caps a purely static candidate at φv = 0.40, so P ≤ 0.608 and a sustained glint cannot reach the 0.70 band (needs z = 0.847). |
| Growth reference | `growth_ref_mm2_per_s = 2.0` (a rate) | `A_ref = 50.0 mm²` (an area) | At 50, the growth term is 0.4·min(1.2/50, 1) = 0.0096, i.e. 1.6 % of its range; a real 1.2 mm²/s growth would be invisible in the score. The renaming makes the units match the derivative. |
| `C_max` | 1.0, absolute contrast | per-clip p99 normalisation | p99 makes the same physical leak score differently depending on what else is in the clip, which cannot be benchmarked deterministically. |
| `ψ_p` (pressure-rate ref) | 2.0 PSI/s | 0.5 PSI/s | 0.5 saturates on a 0.29 PSI/s slow leak, turning the pressure channel into a binary and hiding the transient shape. |
| `ψ_q` (flow-imbalance ref) | 0.5 L/min | 1.0 L/min | Scenario 1 has a 0.9 L/min imbalance; at 1.0 the score is φh = 0.447 and P = 0.8494, **below** the 0.85 confirmed threshold. The confirmed-leak fixture would not confirm. |
| `ΔT_scale` | 5.0 °C | — | Scale only. |
| `ΔT_margin` | **removed** | 3.0 °C | FR-FUS-01 has no margin term. A 3 °C offset puts the neutral point at +3 °C instead of the dew point, and makes the penalty zero between 0 and +3 °C, where condensation is physically impossible — anti-conservative in the direction that matters. |
| `ΔT_expected` | **not defined** | fixed 15.0 °C | Not in the PRD, and not the same quantity: `P/(ρ·c_p·Q)` is bulk coolant rise, 1.12 K at 700 W and 9 L/min. The simulator scripts surface temperature directly, so no such constant is needed. If a physical model is added later it must be labelled an assumption. |

Two rule-level proposals were also rejected:

- **R1-HYD (`φh ≥ 0.80` → immediate trip, no visual).** Contradicts PRD R4, which is
  `HIDDEN_HYDRAULIC_EVENT` → scheduler cordon only, and D-11's human-in-the-loop
  requirement. A hydraulic-only event must not move a valve. Implemented as R4 with the
  3 s hold; see `test_hidden_hydraulic_event_drains_and_never_touches_the_valve`.
- **`VALVE_ISOLATED` / `WORKLOAD_DRAINED_SAFE`.** Resolves to the canonical §6.2 names
  `ISOLATED` and `NODE_POWERED_DOWN`; `ISOLATED` is reached on the limit switch alone and
  the checkpoint gates only the power-down (delta C-04, `test_isolation_does_not_wait_for_the_checkpoint`).

One naming fix: the PRD's `DRAIN` is `scontrol update ... State=DRAIN`, a reversible
scheduler cordon, not a coolant valve. Actions now say `SCHEDULER_DRAIN`, and
`Decision.actuates` counts only `CLAMP`, `VALVE_CLOSE`, `FORCE_OFF`. This makes the
"zero actuation" claim checkable: R3, R2b, R2c and R4 never satisfy it, per §6.3's
explicit "No actuation" on R3 and "No valve actuation" on R4.
