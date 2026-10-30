"""Benchmark payloads and rendering (P8-T03, T-BEN-05).

``reports/benchmark.json`` and ``reports/benchmark.md`` are both generated from
one payload; the Markdown renderer is a pure function of the JSON. A test
re-renders the committed JSON and compares it to the committed Markdown, so a
hand-edited number cannot survive.

Wall-clock measurements are a separate file, ``reports/latency.json``, flagged
``deterministic: false``. Mixing them into the benchmark would make the required
byte-stable artifact change on every run.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from cryoguard.bench.metrics import clopper_pearson, mean_std, percentile
from cryoguard.bench.runner import FITTING_RADIUS_MM, BenchRun

BENCHMARK_TITLE = "Deterministic Regression & Safety Verification Suite (4 Physical Scenarios)"
SAMPLE_NOTICE = (
    "Notice: Sample size n=4 across four synthetic benchmark fixtures. These results "
    "establish deterministic reproducibility across specified physical failure modes, "
    "not statistical generalization across unmodeled data center environments."
)


def _rate(successes: int, trials: int, confidence: float) -> dict[str, Any]:
    interval = clopper_pearson(successes, trials, confidence)
    if trials == 0 or interval is None:
        return {
            "k": successes,
            "n": trials,
            "rate": None,
            "display": "n/a",
            "ciLow": None,
            "ciHigh": None,
            "ciDisplay": "—",
        }
    low, high = interval
    return {
        "k": successes,
        "n": trials,
        "rate": successes / trials,
        "display": f"{100.0 * successes / trials:.1f}% ({successes}/{trials})",
        "ciLow": low,
        "ciHigh": high,
        "ciDisplay": f"[{100.0 * low:.1f}%, {100.0 * high:.1f}%]",
    }


def _confusion(rows: list[dict[str, Any]]) -> dict[str, int]:
    tp = sum(1 for row in rows if row["expectedIncident"] and row["predictedIncident"])
    fp = sum(1 for row in rows if not row["expectedIncident"] and row["predictedIncident"])
    tn = sum(1 for row in rows if not row["expectedIncident"] and not row["predictedIncident"])
    fn = sum(1 for row in rows if row["expectedIncident"] and not row["predictedIncident"])
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def score(rows: list[dict[str, Any]], confidence: float) -> dict[str, Any]:
    counts = _confusion(rows)
    tp, fp, tn, fn = counts["tp"], counts["fp"], counts["tn"], counts["fn"]
    return {
        **counts,
        "positives": tp + fn,
        "negatives": fp + tn,
        "recall": _rate(tp, tp + fn, confidence),
        "falsePositiveRate": _rate(fp, fp + tn, confidence),
        "precision": _rate(tp, tp + fp, confidence),
    }


def verification(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Scenario pass/fail, not a population rate: each fixture has a known answer."""

    counts = _confusion(rows)
    passed = counts["tp"] + counts["tn"]
    total = sum(counts.values())
    positives = counts["tp"] + counts["fn"]
    negatives = counts["fp"] + counts["tn"]
    return {
        "passed": passed,
        "total": total,
        "positives": positives,
        "negatives": negatives,
        "display": (
            f"Automated Scenario Verification: {passed}/{total} passing "
            f"({positives} positive detection scenario{'s' if positives != 1 else ''}, "
            f"{negatives} negative control scenario{'s' if negatives != 1 else ''})"
        ),
    }


def _latency(values: list[float], unit: str) -> dict[str, Any]:
    mean, std = mean_std(values)
    if mean is None:
        display = "n/a"
    elif std is None:
        display = f"{mean:.2f} {unit}"
    else:
        display = f"{mean:.2f} {unit} ± {std:.2f} {unit}"
    return {
        "n": len(values),
        "values": [round(value, 3) for value in values],
        "mean": None if mean is None else round(mean, 3),
        "std": None if std is None else round(std, 3),
        "display": display,
    }


def _positive_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row["expectedIncident"] and row["predictedIncident"]]


def _localization(rows: list[dict[str, Any]], confidence: float) -> dict[str, Any]:
    localizable = [row for row in rows if row["localizable"]]
    hits = sum(1 for row in localizable if row["localizedWithinTolerance"])
    errors = [
        float(row["localizationErrorMm"])
        for row in localizable
        if row["localizationErrorMm"] is not None
    ]
    block = _rate(hits, len(localizable), confidence)
    mean_error = sum(errors) / len(errors) if errors else None
    block["meanErrorMm"] = None if mean_error is None else round(mean_error, 2)
    block["meanErrorDisplay"] = "n/a" if mean_error is None else f"{mean_error:.1f} mm"
    return block


def _checkpoint_recovery(rows: list[dict[str, Any]], confidence: float) -> dict[str, Any]:
    attempted = [row for row in rows if row["checkpointCommitted"] is not None]
    committed = sum(1 for row in attempted if row["checkpointCommitted"])
    return _rate(committed, len(attempted), confidence)


def _valve_isolation(rows: list[dict[str, Any]], confidence: float) -> dict[str, Any]:
    commanded = [row for row in rows if row["valveVerified"] is not None]
    verified = sum(1 for row in commanded if row["valveVerified"])
    return _rate(verified, len(commanded), confidence)


def clip_row_payload(run: BenchRun) -> dict[str, Any]:
    row = run.row.as_row()
    row["pDisplay"] = f"{float(row['p']):.3f}"
    row["detectionSDisplay"] = (
        "—" if row["detectionS"] is None else f"{float(row['detectionS']):.2f}"
    )
    localized = row["localizedWithinTolerance"]
    row["localizedDisplay"] = "n/a" if localized is None else ("yes" if localized else "no")
    error_mm = row["localizationErrorMm"]
    row["localizationErrorDisplay"] = "—" if error_mm is None else f"{float(error_mm):.1f}"
    row["valveVerifiedDisplay"] = (
        "n/a" if row["valveVerified"] is None else ("yes" if row["valveVerified"] else "no")
    )
    row["checkpointDisplay"] = (
        "n/a"
        if row["checkpointCommitted"] is None
        else ("committed" if row["checkpointCommitted"] else "failed")
    )
    return row


def build_payload(
    runs: list[BenchRun],
    *,
    cfg: dict[str, Any],
    split: str,
    config_hash: str,
    fusion_hash: str,
    git_sha: str,
    confidence: float = 0.95,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    rows = [clip_row_payload(run) for run in runs]
    by_split: dict[str, list[dict[str, Any]]] = {"all": rows}
    for name in sorted({row["split"] for row in rows}):
        by_split[name] = [row for row in rows if row["split"] == name]
    if split not in by_split:
        raise ValueError(f"split {split!r} has no rows")

    selected = by_split[split]
    detection_values = [
        float(row["detectionS"])
        for row in _positive_rows(selected)
        if row["detectionS"] is not None
    ]

    from cryoguard.bench.runner import run_hash

    deterministic_rows = [
        {
            key: value
            for key, value in row.items()
            if key
            not in {
                "pDisplay",
                "detectionSDisplay",
                "localizedDisplay",
                "localizationErrorDisplay",
                "valveVerifiedDisplay",
                "checkpointDisplay",
            }
        }
        for row in rows
    ]
    digest = run_hash(deterministic_rows, config_hash=config_hash, fusion_hash=fusion_hash)

    return {
        "schemaVersion": "1.0",
        "generatedAt": (generated_at or datetime.now(UTC)).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
        + "Z",
        "deterministic": True,
        "note": (
            "generatedAt is the only non-reproducible field; runHash covers the "
            "config, freeze, and every clip row. Wall-clock latency is in latency.json."
        ),
        "title": BENCHMARK_TITLE,
        "notice": SAMPLE_NOTICE,
        "configHash": config_hash,
        "fusionHash": fusion_hash,
        "gitSha": git_sha,
        "split": split,
        "confidence": confidence,
        "confidencePercent": int(round(confidence * 100)),
        "runHash": digest,
        "headline": score(selected, confidence),
        "verification": verification(selected),
        "metrics": {name: score(value, confidence) for name, value in by_split.items()},
        "detectionLatencyS": _latency(detection_values, "s"),
        "localization": _localization(selected, confidence),
        "localizationToleranceMm": FITTING_RADIUS_MM,
        "checkpointRecovery": _checkpoint_recovery(selected, confidence),
        "valveIsolation": _valve_isolation(selected, confidence),
        "clips": rows,
        "limits": [
            "Every clip is synthetic (source: synthetic); no captured hardware "
            "exists in this repository.",
            "The manifest split is provisional: two clips per side, so each "
            "confidence interval is wide.",
            "The pipeline is rule-based and not trained on these clips; the split "
            "is reported, not used for fitting.",
            "PLC, BMC, NVML, Slurm, and telemetry are simulators; the in-memory "
            "transport was used for this run.",
            "Localization is geometric: the candidate centroid in rectified chassis "
            "millimetres is compared with the configured QD-04 ROI centre. A "
            "matching origin string is no longer accepted as localization.",
            "Localization error is measured at the decision frame; the simulated "
            "leak drifts 11 mm/s from QD-04, so the error includes the spill that "
            "accumulated before confirmation.",
            "SAM ran in cache mode from a fixture-seeded cache (provenance=fixture); "
            "no hosted SAM call was made.",
            "Detection latency is scenario time from clip start to the decision. "
            "Wall-clock stage samples are in latency.json.",
            "The sequence timing is in-memory transport; the Modbus/Redfish socket "
            "path is exercised by tests/test_closed_loop.py.",
        ],
    }


def render_markdown(payload: dict[str, Any]) -> str:
    headline = payload["headline"]
    tolerance = f"{float(payload['localizationToleranceMm']):.0f}"
    lines = [
        f"# {payload['title']}",
        (
            f"Generated: {payload['generatedAt']} | Config Hash: {payload['configHash']} "
            f"| Git: {payload['gitSha']}"
        ),
        (
            f"Split: {payload['split']} (n={len(payload['clips'])}) | Fusion freeze: "
            f"{payload['fusionHash']} | Run hash: {payload['runHash']}"
        ),
        "",
        "> Every value below is computed from `reports/benchmark.json`; the Markdown",
        "> renderer is a pure function of that payload. Wall-clock samples are in",
        "> `reports/latency.json` and are explicitly not deterministic.",
        "",
        "## Headline",
        "",
        f"**{payload['verification']['display']}**",
        "",
        (
            f"| Metric | Measured Value | {payload['confidencePercent']}% Confidence Interval "
            "(Clopper-Pearson) | Sample Count |"
        ),
        "| :--- | :--- | :--- | :--- |",
        (
            f"| **Positive Scenario Detection** | {headline['recall']['display']} "
            f"| {headline['recall']['ciDisplay']} | {headline['positives']} "
            f"positive fixture{'s' if headline['positives'] != 1 else ''} |"
        ),
        (
            f"| **Negative Control False Positive Rate** | "
            f"{headline['falsePositiveRate']['display']} "
            f"| {headline['falsePositiveRate']['ciDisplay']} "
            f"| {headline['negatives']} negative fixture"
            f"{'s' if headline['negatives'] != 1 else ''} |"
        ),
        (
            f"| **Mean Detection Latency** | {payload['detectionLatencyS']['display']} "
            f"| — | {payload['detectionLatencyS']['n']} confirmed "
            f"event{'s' if payload['detectionLatencyS']['n'] != 1 else ''} |"
        ),
        (
            f"| **Fitting Localization (≤ {tolerance} mm)** | "
            f"{payload['localization']['display']} "
            f"| {payload['localization']['ciDisplay']} "
            f"| {payload['localization']['n']} localizable "
            f"event{'s' if payload['localization']['n'] != 1 else ''} |"
        ),
        (
            f"| **Mean Localization Error** | {payload['localization']['meanErrorDisplay']} "
            f"| — | {payload['localization']['n']} localizable "
            f"event{'s' if payload['localization']['n'] != 1 else ''} |"
        ),
        (
            f"| **Checkpoint Recovery Success** | {payload['checkpointRecovery']['display']} "
            f"| {payload['checkpointRecovery']['ciDisplay']} "
            f"| {payload['checkpointRecovery']['n']} checkpointed "
            f"incident{'s' if payload['checkpointRecovery']['n'] != 1 else ''} |"
        ),
        (
            f"| **Valve Isolation Verified** | {payload['valveIsolation']['display']} "
            f"| {payload['valveIsolation']['ciDisplay']} "
            f"| {payload['valveIsolation']['n']} commanded "
            f"close{'s' if payload['valveIsolation']['n'] != 1 else ''} |"
        ),
        "",
        f"> **{payload['notice']}**",
        "",
        "## Per split",
        "",
        "| Split | n | Recall | FPR | Precision |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ]
    for name in sorted(payload["metrics"]):
        block = payload["metrics"][name]
        lines.append(
            f"| {name} | {block['tp'] + block['fp'] + block['tn'] + block['fn']} "
            f"| {block['recall']['display']} | {block['falsePositiveRate']['display']} "
            f"| {block['precision']['display']} |"
        )

    lines += [
        "",
        "## Per clip",
        "",
        "| Clip | Split | Ground truth | Predicted | Label | Rule | P "
        "| Detection (s) | Zone | Loc. error (mm) | Within tolerance | Actions |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    for row in payload["clips"]:
        lines.append(
            f"| {row['clipId']} | {row['split']} | {row['groundTruthLabel']} "
            f"| {'incident' if row['predictedIncident'] else 'none'} | {row['label']} "
            f"| {row['ruleId']} | {row['pDisplay']} | {row['detectionSDisplay']} "
            f"| {row['zone'] if row['zone'] is not None else '—'} "
            f"| {row['localizationErrorDisplay']} | {row['localizedDisplay']} "
            f"| {row['actionCount']} |"
        )

    lines += ["", "## Limits", ""]
    lines += [f"- {limit}" for limit in payload["limits"]]
    lines.append("")
    return "\n".join(lines)


def build_latency(
    runs: list[BenchRun], *, config_hash: str, run_hash: str, git_sha: str
) -> dict[str, Any]:
    def samples(values: list[float]) -> dict[str, Any]:
        return {
            "n": len(values),
            "median": percentile(values, 0.5),
            "p95": percentile(values, 0.95),
        }

    per_clip: list[dict[str, Any]] = []
    perception_all: list[float] = []
    fusion_all: list[float] = []
    cached_all: list[float] = []
    live_all: list[float] = []
    sequence_all: list[float] = []

    for run in runs:
        timing = run.result.timing
        cached = list(run.result.sam.cached_ms) if run.result.sam is not None else []
        live = list(run.result.sam.live_ms) if run.result.sam is not None else []
        perception_all += list(timing.perception_ms)
        fusion_all += list(timing.fusion_ms)
        cached_all += cached
        live_all += live
        if timing.sequence_ms > 0:
            sequence_all.append(timing.sequence_ms)
        per_clip.append(
            {
                "clipId": run.row.clip_id,
                "perceptionTickMs": samples(list(timing.perception_ms)),
                "fusionMs": samples(list(timing.fusion_ms)),
                "samCachedMs": samples(cached),
                "samLiveMs": samples(live),
                "sequenceMs": timing.sequence_ms,
            }
        )

    return {
        "schemaVersion": "1.0",
        "deterministic": False,
        "note": (
            "Wall-clock samples from one local run. They are measured, not assumed, "
            "and are excluded from benchmark.json so that file stays reproducible."
        ),
        "configHash": config_hash,
        "runHash": run_hash,
        "gitSha": git_sha,
        "stages": {
            "perceptionTickMs": samples(perception_all),
            "fusionMs": samples(fusion_all),
            "samCachedMs": samples(cached_all),
            "samLiveMs": samples(live_all),
            "sequenceMs": samples(sequence_all),
        },
        "perClip": per_clip,
    }


def write_reports(
    *,
    payload: dict[str, Any],
    latency: dict[str, Any],
    out_dir: str | Path,
) -> tuple[Path, Path, Path]:
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)

    benchmark_json = directory / "benchmark.json"
    benchmark_md = directory / "benchmark.md"
    latency_json = directory / "latency.json"

    benchmark_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    benchmark_md.write_text(render_markdown(payload), encoding="utf-8")
    latency_json.write_text(json.dumps(latency, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return benchmark_json, benchmark_md, latency_json
