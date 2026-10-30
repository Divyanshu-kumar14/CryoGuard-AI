# Deterministic Regression & Safety Verification Suite (4 Physical Scenarios)
Generated: 2026-10-30T19:28:50.482Z | Config Hash: 1ce8a0508d6fddc5 | Git: a777b44
Split: all (n=4) | Fusion freeze: 164849b04a9afb87 | Run hash: 0b2d56ddf07d65669f09070c21237c267457aa81551939cce7fa9b67545a1fa2

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

> **Notice: Sample size n=4 across four synthetic benchmark fixtures. These results establish deterministic reproducibility across specified physical failure modes, not statistical generalization across unmodeled data center environments.**

## Per split

| Split | n | Recall | FPR | Precision |
| :--- | :--- | :--- | :--- | :--- |
| all | 4 | 100.0% (2/2) | 0.0% (0/2) | 100.0% (2/2) |
| dev | 2 | 100.0% (1/1) | 0.0% (0/1) | 100.0% (1/1) |
| test | 2 | 100.0% (1/1) | 0.0% (0/1) | 100.0% (1/1) |

## Per clip

| Clip | Split | Ground truth | Predicted | Label | Rule | P | Detection (s) | Zone | Loc. error (mm) | Within tolerance | Actions |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| sim_leak_qd4 | dev | CONFIRMED_LEAK | incident | CONFIRMED_LEAK | R2 | 0.853 | 7.60 | 0 | 52.8 | no | 6 |
| sim_condensation | dev | BENIGN_CONDENSATION | none | NOMINAL | R7 | 0.035 | — | — | — | n/a | 0 |
| sim_glint | test | SUSPECT_LEAK | none | NOMINAL | R7 | 0.099 | — | — | — | n/a | 0 |
| sim_hidden_hydraulic | test | HIDDEN_HYDRAULIC_EVENT | incident | HIDDEN_HYDRAULIC_EVENT | R4 | 0.332 | 3.50 | — | — | n/a | 2 |

## Limits

- Every clip is synthetic (source: synthetic); no captured hardware exists in this repository.
- The manifest split is provisional: two clips per side, so each confidence interval is wide.
- The pipeline is rule-based and not trained on these clips; the split is reported, not used for fitting.
- PLC, BMC, NVML, Slurm, and telemetry are simulators; the in-memory transport was used for this run.
- Localization is geometric: the candidate centroid in rectified chassis millimetres is compared with the configured QD-04 ROI centre. A matching origin string is no longer accepted as localization.
- Localization error is measured at the decision frame; the simulated leak drifts 11 mm/s from QD-04, so the error includes the spill that accumulated before confirmation.
- SAM ran in cache mode from a fixture-seeded cache (provenance=fixture); no hosted SAM call was made.
- Detection latency is scenario time from clip start to the decision. Wall-clock stage samples are in latency.json.
- The sequence timing is in-memory transport; the Modbus/Redfish socket path is exercised by tests/test_closed_loop.py.
