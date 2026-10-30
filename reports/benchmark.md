# Deterministic Regression & Safety Verification Suite (4 Synthetic Scenarios)
Generated: 2026-10-30T21:26:15.907Z | Config Hash: f0781ebc586322ce | Git: 5713b4b
Split: all (n=4) | Fusion freeze: 164849b04a9afb87 | Run hash: 98ac032bc7dc8b3d52954ef5d67f1da6e660c972e1aeb2e052ab3640b429d858

> Every value below is computed from `reports/benchmark.json`; the Markdown
> renderer is a pure function of that payload. Wall-clock samples are in
> `reports/latency.json` and are explicitly not deterministic.

## Headline

**Automated Scenario Verification: 4/4 passing (2 positive detection scenarios, 2 negative control scenarios)**

| Metric | Measured Value | 95% Confidence Interval (Clopper-Pearson) | Sample Count |
| :--- | :--- | :--- | :--- |
| **Positive Scenario Detection** | 100.0% (2/2) | [15.8%, 100.0%] | 2 positive fixtures |
| **Negative Control False Positive Rate** | 0.0% (0/2) | [0.0%, 84.2%] | 2 negative fixtures |
| **Mean Detection Latency** | 5.55 s ± 2.90 s | — | 2 confirmed events |
| **Fitting Localization (≤ 25 mm)** | 0.0% (0/1) | [0.0%, 97.5%] | 1 localizable event |
| **Mean Localization Error** | 52.8 mm | — | 1 localizable event |
| **Checkpoint Recovery Success** | 100.0% (1/1) | [2.5%, 100.0%] | 1 checkpointed incident |
| **Valve Isolation Verified** | 100.0% (1/1) | [2.5%, 100.0%] | 1 commanded close |

> **Notice: Sample size n=4 across four synthetic benchmark fixtures. These results are deterministic regression verification of specified failure modes — recall X/Y with exact Clopper-Pearson intervals — not population performance estimates and not evidence of generalization to unmodeled data center environments.**

## Per split

| Split | n | Recall | FPR | Precision |
| :--- | :--- | :--- | :--- | :--- |
| all | 4 | 100.0% (2/2) | 0.0% (0/2) | 100.0% (2/2) |
| dev | 2 | 100.0% (1/1) | 0.0% (0/1) | 100.0% (1/1) |
| test | 2 | 100.0% (1/1) | 0.0% (0/1) | 100.0% (1/1) |

## Source estimation

Back-projected source hypothesis vs the independent manifest `leak_source_mm`: 1 of 4 clips produced a genuine estimate; the rest are unavailable, never zero. The spill centroid above is the observation, not the origin.

| Metric | Measured Value | 95% Confidence Interval (Clopper-Pearson) | Sample Count |
| :--- | :--- | :--- | :--- |
| **Source Within Tolerance (≤ 25 mm)** | 100.0% (1/1) | [2.5%, 100.0%] | 1 estimated event |
| **Mean Source Error** | 0.4 mm | — | 1 estimated event |

## Fault injection

Deterministic fault-injection matrix `tests/test_bench_faults.py`: 11/11 cases passing. Counted separately; never merged into scenario metrics or runHash.

| Case | Category | Result |
| :--- | :--- | :--- |
| camera-degraded | perception | pass |
| sam-timeout | perception | pass |
| sam-malformed | perception | pass |
| plc-disconnect | actuation | pass |
| valve-seal-leak | actuation | pass |
| bmc-poweroff-unverified | actuation | pass |
| stale-telemetry | telemetry | pass |
| frozen-sensor | telemetry | pass |
| checkpoint-corrupt | recovery | pass |
| watchdog-primary-failed | safety | pass |
| watchdog-fallback-failed | safety | pass |

## Per clip

| Clip | Split | Ground truth | Predicted | Label | Rule | P | Detection (s) | Zone | Loc. error (mm) | Within tolerance | Src err (mm) | Src | Actions |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| sim_leak_qd4 | dev | CONFIRMED_LEAK | incident | CONFIRMED_LEAK | R2 | 0.853 | 7.60 | 0 | 52.8 | no | 0.4 | yes | 6 |
| sim_condensation | dev | BENIGN_CONDENSATION | none | NOMINAL | R7 | 0.035 | — | — | — | n/a | — | n/a | 0 |
| sim_glint | test | SUSPECT_LEAK | none | NOMINAL | R7 | 0.099 | — | — | — | n/a | — | n/a | 0 |
| sim_hidden_hydraulic | test | HIDDEN_HYDRAULIC_EVENT | incident | HIDDEN_HYDRAULIC_EVENT | R4 | 0.332 | 3.50 | — | — | n/a | — | n/a | 2 |

## Limits

- Every clip is synthetic (source: synthetic); no captured hardware exists in this repository.
- The manifest split is provisional: two clips per side, so each confidence interval is wide.
- The pipeline is rule-based and not trained on these clips; the split is reported, not used for fitting.
- PLC, BMC, NVML, Slurm, and telemetry are simulators; the in-memory transport was used for this run.
- Localization is geometric: the candidate centroid in rectified chassis millimetres is compared with the configured QD-04 ROI centre. A matching origin string is no longer accepted as localization.
- Localization error is measured at the decision frame; the simulated leak drifts 11 mm/s from QD-04, so the error includes the spill that accumulated before confirmation.
- Source error compares the back-projected source hypothesis (RunResult.source_estimate, method recorded per clip in sourceMethod) with the independent manifest leak_source_mm, which is declared in the scenario file — never derived from the detector's QD ROI. The spill centroid is reported separately and never as the origin. Clips with no genuine estimate report the source fields as unavailable (None), never zero.
- Fault-injection cases (tests/test_bench_faults.py, src/cryoguard/bench/faults.py) use mocked inputs without video fixtures and are counted separately; they never enter scenario recall, localization, source error, or runHash.
- SAM ran in cache mode from a fixture-seeded cache (provenance=fixture); no hosted SAM call was made.
- Detection latency is scenario time from clip start to the decision. Wall-clock stage samples are in latency.json.
- The sequence timing is in-memory transport; the Modbus/Redfish socket path is exercised by tests/test_closed_loop.py.
