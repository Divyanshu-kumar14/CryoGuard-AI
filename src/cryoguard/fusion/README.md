# Evidence fusion and the decision table

Scope: this package is a deterministic function of one tick of evidence. It has
no network client, no model, and no clock. The reporting layer is downstream and
the import boundary is enforced by `tests/test_architecture.py`.

The numbers below are the values in `configs/default.yaml`. The freeze file
`reports/fusion-freeze.json` pins the arithmetic and every gate that changes a
rule outcome; `tests/test_fusion_freeze.py` fails when the two drift apart.

---

## 1. Inputs

| Field | Domain | Meaning |
| :-- | :-- | :-- |
| `phi_v` | `[0, 1]` | Visual evidence, section 2.1 |
| `phi_h` | `[0, 1]` | Hydraulic evidence, section 2.2 |
| `phi_t` | `[-1, 1]` | Thermal margin, section 2.3 |
| `W` | `[0, 1]` | Wetness continuity, section 2.4 |
| `dA_dt` | finite | Candidate area slope, mm²/s |
| `q` | `[0, 1]` | Mask quality, section 2.5 |
| `zone` | `-1` or a zone id | `-1` means no visual candidate |
| `ticks` | integer `>= 0` | Consecutive ticks in the current evidence band |
| `camera_healthy` | bool | Brightness, focus, saturation checks |
| `plc_fault` | bool | Valve fault from the PLC |
| `temp_c` | finite | Surface temperature, °C |
| `clamp_readback_ok` | bool | NVML clamp readback |
| `maintenance` | bool | Suppresses Zone 0/1 confirmation, never Zone 2 |
| `hydraulic_low_ms` | integer `>= 0` | Continuous hydraulic-only evidence |

`decide()` validates the domains first. Non-finite or out-of-range evidence raises
`DecisionError`; it never falls through to `NOMINAL`. There are therefore only two
outcomes: one canonical rule, or a loud failure.

---

## 2. Features

### 2.1 Visual evidence

```
phi_v = clamp(
    alpha * clamp((dA/dt) / growth_ref, 0, 1)
  + beta  * clamp(I / contrast_max, 0, 1)
  - gamma * clamp(Var(I), 0, 1),
    0, 1)
```

| Parameter | Value | Note |
| :-- | :-- | :-- |
| `alpha` | 0.4 | growth weight |
| `beta` | 0.4 | contrast weight |
| `gamma` | 0.2 | jitter penalty |
| `growth_ref_mm2_per_s` | 2.0 | rate reference, not an area |
| `contrast_max` | 1.0 | `I` is mean value / 255 |

`dA/dt` is the least-squares slope of the candidate's area over the last
`growth_window = 5` samples. `Var(I)` is the variance of the same normalised
values, so it is already in `[0, 1]`.

The PRD's `A_ref` is an area; dividing a rate by an area makes the growth term
contribute about 2 % of `phi_v` at 50 mm², which is numerically irrelevant. The
reference is therefore defined as a **growth rate** (delta v1.2 §3). A static
connected reflection still scores 0.61, which is why the growth gate below is
load-bearing rather than cosmetic.

### 2.2 Hydraulic evidence

```
phi_h = clamp(
    w_pressure * clamp((-dP/dt) / pressure_ref, 0, 1)
  + w_flow     * clamp((Q_in - Q_out) / flow_ref, 0, 1),
    0, 1)
```

| Parameter | Value |
| :-- | :-- |
| `w_pressure` | 0.6 |
| `w_flow` | 0.4 |
| `pressure_rate_ref_psi_per_s` | 2.0 |
| `flow_imbalance_ref_lpm` | 0.5 |

`-dP/dt` is positive on a pressure drop. The one-second derivative comes from the
scripted telemetry replay, not from a wall clock.

### 2.3 Thermal evidence

```
phi_t = clamp((T_surface - T_dew) / dt_scale_c, -1, 1)
```

`dt_scale_c = 5.0`. The value is exactly zero at the dew point and negative below
it. No extra margin is added: an offset would move the neutral point away from
the physical onset of condensation.

### 2.4 Wetness continuity

`W = 1` only when the candidate's fluorescent component is 8-connected to a
Quick Disconnect ROI for **3 consecutive ticks** (`wetness_ticks = 3`).
Connectivity is checked on the full fluorescence mask, so a thin wet trail is
enough. A droplet that appears across the chassis without a path back to the
fitting is a reflection or a splash, not coolant that leaked there.

`W` is measured on fluorescence-positive pixels only. A water bead is bright but
unsaturated; it never becomes a fluorescent component, so it cannot inherit the
wetness credit of a real leak.

### 2.5 Mask quality

```
q = q_area * q_clip * q_solid
q_area  = min(1, area / min_area_mm2)
q_clip  = 1 - clipped_fraction
q_solid = min(1, solidity / solidity_ref)
```

| Parameter | Value |
| :-- | :-- |
| `min_area_mm2` | 4.0 |
| `solidity_ref` | 0.6 |
| `exposure_clip_max` | 0.05 |
| `q_min` | 0.5 |

Solidity is area divided by convex-hull area. Compactness (`4πA/P²`) measures
roundness and punishes the elongation every wet trail has; solidity measures
whether the mask is one coherent region. If `clipped_fraction` exceeds
`exposure_clip_max`, `q` is forced to zero: a mostly saturated mask cannot
localise a leak regardless of its shape.

---

## 3. Score

```
P = sigma(w_v * phi_v + w_h * phi_h + w_t * min(phi_t, 0) + w_w * W + bias)

sigma(z) = 1 / (1 + exp(-z))
```

| Weight | Value |
| :-- | :-- |
| `w_visual` | 1.85 |
| `w_hydraulic` | 2.10 |
| `w_thermal` | 1.25 |
| `w_wetness` | 2.50 |
| `bias` | -2.80 |

`min(phi_t, 0)` makes dry air add exactly zero leak evidence; only proximity to
the dew point subtracts. These are hand-set design priors. There is no training
set, no held-out evaluation of the score, and no claim that `P` is a measured
frequency. It is a bounded suspicion score used to order the rule table.

Worked values, computed by `tests/test_fusion_math.py`:

| Case | Inputs | `P` |
| :-- | :-- | :-- |
| dry nominal | `phi_v=0, phi_h=0, phi_t=1, W=0` | 0.057 |
| static connected reflection | `phi_v=0.40, phi_h=0, phi_t=0.50, W=1` | 0.608 |
| leak fixture | `phi_v=0.665, phi_h=0.402, phi_t>0, W=1` | 0.855 |
| condensation, no path | `phi_v=0.35, phi_h=0, phi_t=-0.418, W=0` | 0.065 |
| condensation, path assumed | `phi_v=0.35, phi_h=0, phi_t=-0.418, W=1` | 0.456 |

The third row is above the Zone 2 trip threshold of 0.50. The score alone would
trip the rack on a static reflection; the growth gate is what prevents it. The
last two rows are why `W` is measured only on fluorescence-positive pixels.

---

## 4. Decision table

First match wins. The order is fixed; the table is total (see section 5).

| # | Rule | Condition | Label | Actions | Moves hardware? |
| :-- | :-- | :-- | :-- | :-- | :-- |
| 1 | **R1b** | `plc_fault` or `temp_c >= 90` or clamp readback failed | `EMERGENCY_TRIP` | `VALVE_CLOSE`, `FORCE_OFF`, `SCHEDULER_DRAIN` | yes |
| 2 | **R8** | `camera_healthy` is false | `CAMERA_DEGRADED` | — | no |
| 3 | **R1** | zone 2, `P >= 0.50`, `ticks >= 3`, `q >= 0.5`, growth gate | `EMERGENCY_TRIP` | `VALVE_CLOSE`, `FORCE_OFF`, `SCHEDULER_DRAIN` | yes |
| 4 | **R2** | zone 0/1, `P >= 0.85`, `ticks >= 3`, `phi_v >= 0.2`, `phi_h >= 0.2`, `q >= 0.5`, not maintenance, growth gate | `CONFIRMED_LEAK` | `CLAMP`, `VALVE_CLOSE`, `CHECKPOINT_VERIFY`, `SCHEDULER_DRAIN` | yes |
| 5 | **R2b** | `P >= 0.85`, zone 0/1, R2 not satisfiable | `UNVERIFIED_ANOMALY` | `ALERT`, `SAMPLE_10HZ`, `CDU_QUERY` | no |
| 6 | **R2c** | `P >= 0.85`, zone 2, R1 not satisfiable | `UNVERIFIED_ANOMALY` | `ALERT`, `SAMPLE_10HZ`, `CDU_QUERY` | no |
| 7 | **R4** | `phi_v <= 0.1`, `phi_h >= 0.6`, `hydraulic_low_ms >= 3000` | `HIDDEN_HYDRAULIC_EVENT` | `ALERT`, `SCHEDULER_DRAIN` | no valve |
| 8 | **R3** | `0.70 <= P < 0.85` | `UNVERIFIED_ANOMALY` | `ALERT`, `SAMPLE_10HZ`, `CDU_QUERY` | no |
| 9 | **R5** | `0.50 <= P < 0.70` | `SUSPECT_LEAK`, or `BENIGN_CONDENSATION` when `phi_t < 0` and `phi_h <= 0.10` | `SAMPLE_10HZ`, `LOG` | no |
| 10 | **R6** | `0.40 <= P < 0.50` | `LOG_ONLY` | `LOG` | no |
| 11 | **R7** | otherwise | `NOMINAL` | — | no |

`Decision.actuates` is true only for `CLAMP`, `VALVE_CLOSE`, and `FORCE_OFF`.
`SCHEDULER_DRAIN` is `scontrol ... State=DRAIN`, a reversible scheduling cordon,
not a coolant valve (D-11). `ALERT`, `SAMPLE_10HZ`, `CDU_QUERY`, and `LOG` are
observational.

### 4.1 The growth gate

```
growth_ok = (dA_dt > 0.5 mm²/s) or (phi_h >= 0.20)
```

R1 and R2 require it. A static reflection connected to a fitting has no growth
and no hydraulic corroboration, so it cannot actuate even when `P` is high. The
same evidence with real growth does trip Zone 2; the pair of tests is
`test_static_reflection_scores_above_trip_threshold_but_never_actuates` and
`test_identical_evidence_with_real_growth_trips_zone_two`.

### 4.2 Zones and maintenance

Zone 0 is the hydraulic origin, Zone 1 the containment buffer, Zone 2 the
electrical hazard plane. The map tiles the chassis to a 20 mm border; a candidate
in flight is never in "no zone". A candidate covering at least 5 % of its area in
a zone takes the highest qualifying zone id.

`maintenance_mode` suppresses R2 (Zone 0/1) but never R1 (Zone 2). Under
maintenance a high-scoring Zone 0 candidate still reaches R2b and is logged and
sampled; it does not clamp or isolate.

### 4.3 Closed holes

The PRD's table left three states with no matching row. They are closed here:

- **R2b** — `P >= 0.85` in Zone 0/1 with an unmet confirmation precondition used
  to match nothing. It is now an anomaly with an alert and a 10 Hz sample, never
  silence and never an actuation.
- **R2c** — the same score in Zone 2 when R1's mask gates or tick count fail. It
  is observational, because a high score is not yet an electrical breach.
- **R8** — an unhealthy camera used to fall through to `NOMINAL`. A blind chassis
  is now `CAMERA_DEGRADED`; it never confirms a leak and never actuates.

### 4.4 Escalation and latching

`ticks` counts consecutive ticks in the same **evidence band**. R2, R2b, and R2c
share one `HIGH` band, so a rule refining from R2b to R2 when the quality gate
passes does not restart the count.

`CONFIRMED_LEAK` and `EMERGENCY_TRIP` latch in `HysteresisDecision`. A later tick
with calm evidence still returns the latched incident; only an operator
`reset()` clears it (T-DEC-03).

Non-incident bands follow the instantaneous table. The 5-tick de-escalation hold
in PRD §6.4 is not yet implemented in the wrapper; because those bands never
actuate, this affects logging cadence rather than hardware.

R4 needs 3 s of continuous hydraulic evidence (`hydraulic_low_ms >= 3000`), which
is longer than the 3-tick escalation used by R1/R2 because a hidden event has no
visual corroboration. The timer starts when `phi_h` first reaches 0.20; R4 still
requires `phi_h >= 0.60` at the current tick, so a fading hydraulic signal does
not keep the rule alive by itself.

---

## 5. Totality and determinism

`tests/test_rule_totality.py` runs Hypothesis over `phi_v`, `phi_h`, `phi_t`,
`W`, `dA_dt`, `q`, zone, ticks, camera health, fault flags, temperature,
maintenance, and the hydraulic hold. Every valid vector must produce one of the
eleven rules, `P` in `[0, 1]`, and the same output on a second call. A separate
witness table proves all eleven rules are reachable, so no branch is dead.

`tests/test_fusion_freeze.py` compares the live config with
`reports/fusion-freeze.json`; a weight, threshold, gate, or mask-quality input
that drifts fails the suite until the freeze is regenerated deliberately.

There is no wall clock in `decide()`. The hysteresis wrapper takes `t_s` only to
measure holds; the score and the rules themselves are functions of the evidence
and the config alone.
