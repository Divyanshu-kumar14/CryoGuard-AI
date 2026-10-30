# CryoGuard AI — Empirical Benchmark Report
Generated: 2026-10-30T15:41:00.067Z | Config Hash: de5bd0f0d16e7607 | Git: 90ac543
Split: all (n=4) | Fusion freeze: 164849b04a9afb87 | Run hash: 76edbee924a65714060ed962142d09c6e061cf086fca5fba27a66119c00b3aab

> Every value below is computed from `reports/benchmark.json`; the Markdown
> renderer is a pure function of that payload. Wall-clock samples are in
> `reports/latency.json` and are explicitly not deterministic.

## Headline

| Metric | Measured Value | 95% Confidence Interval (Clopper-Pearson) | Sample Count |
| :--- | :--- | :--- | :--- |
| **Leak Detection Recall** | 100.0% (2/2) | [15.8%, 100.0%] | 2 positive clips |
| **False Positive Rate** | 0.0% (0/2) | [0.0%, 84.2%] | 2 negative clips |
| **Mean Detection Latency** | 5.55 s ± 2.90 s | — | 2 confirmed events |
| **Fitting Localization Accuracy** | 100.0% (1/1) | [2.5%, 100.0%] | 1 localizable event |
| **Checkpoint Recovery Success** | 100.0% (1/1) | [2.5%, 100.0%] | 1 checkpointed incident |
| **Valve Isolation Verified** | 100.0% (1/1) | [2.5%, 100.0%] | 1 commanded close |

## Per split

| Split | n | Recall | FPR | Precision |
| :--- | :--- | :--- | :--- | :--- |
| all | 4 | 100.0% (2/2) | 0.0% (0/2) | 100.0% (2/2) |
| dev | 2 | 100.0% (1/1) | 0.0% (0/1) | 100.0% (1/1) |
| test | 2 | 100.0% (1/1) | 0.0% (0/1) | 100.0% (1/1) |

## Per clip

| Clip | Split | Ground truth | Predicted | Label | Rule | P | Detection (s) | Zone | QD match | Actions |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| sim_leak_qd4 | dev | CONFIRMED_LEAK | incident | CONFIRMED_LEAK | R2 | 0.853 | 7.60 | 0 | yes | 6 |
| sim_condensation | dev | BENIGN_CONDENSATION | none | NOMINAL | R7 | 0.035 | — | — | n/a | 0 |
| sim_glint | test | SUSPECT_LEAK | none | NOMINAL | R7 | 0.099 | — | — | n/a | 0 |
| sim_hidden_hydraulic | test | HIDDEN_HYDRAULIC_EVENT | incident | HIDDEN_HYDRAULIC_EVENT | R4 | 0.332 | 3.50 | — | n/a | 2 |

## Limits

- Every clip is synthetic (source: synthetic); no captured hardware exists in this repository.
- The manifest split is provisional: two clips per side, so each confidence interval is wide.
- The pipeline is rule-based and not trained on these clips; the split is reported, not used for fitting.
- PLC, BMC, NVML, Slurm, and telemetry are simulators; the in-memory transport was used for this run.
- Zone match is diagnostic: the simulated leak starts in Zone 0 and drifts toward Zone 2, so the fitting (QD-04) is the localization claim.
- SAM ran in cache mode from a fixture-seeded cache (provenance=fixture); no hosted SAM call was made.
- Detection latency is scenario time from clip start to the decision. Wall-clock stage samples are in latency.json.
- The sequence timing is in-memory transport; the Modbus/Redfish socket path is exercised by tests/test_closed_loop.py.
