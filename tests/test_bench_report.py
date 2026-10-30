"""Benchmark payload and Markdown rendering (P8-T03, T-BEN-05)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from cryoguard.bench.report import build_latency, build_payload, render_markdown
from cryoguard.bench.runner import BenchRun, ClipRun
from cryoguard.pipeline import RunResult, TimingSamples

ROOT = Path(__file__).resolve().parents[1]


def _row(
    clip_id: str,
    *,
    split: str = "test",
    expected: bool,
    predicted: bool,
    label: str,
    p: float = 0.9,
    zone: int | None = 0,
    localized: bool | None = True,
    localizable: bool = True,
    checkpoint: bool | None = True,
    valve: bool | None = True,
    detection_s: float | None = 3.5,
) -> ClipRun:
    return ClipRun(
        clip_id=clip_id,
        scenario=clip_id,
        split=split,
        path=f"fixtures/{clip_id}.mp4",
        sha256="0" * 64,
        ground_truth_label=label,
        expected_incident=expected,
        predicted_incident=predicted,
        label=label,
        rule_id="R2",
        p=p,
        expected_qd="QD-04",
        suspected_origin="QD-04 (Zone 0)",
        localized_within_tolerance=localized,
        localization_error_mm=None if localized is None else (0.8 if localized else 40.0),
        localizable=localizable,
        zone=zone,
        expected_zone=0,
        detection_s=detection_s,
        ticks=10,
        frames=100,
        action_count=6 if predicted else 0,
        valve_verified=valve,
        checkpoint_committed=checkpoint,
        sequence_state="RESOLVED_PENDING_REPAIR" if predicted else None,
    )


def _run(row: ClipRun, *, perception: float = 2.0) -> BenchRun:
    result = RunResult(
        scenario=row.scenario,
        decision=None,
        report=None,
        path=None,
        frames=row.frames,
        ticks=row.ticks,
        tick_rate_hz=5.0,
        rate_changes=0,
        sam=None,
        timing=TimingSamples((perception,), (0.05,), 12.0 if row.predicted_incident else 0.0),
    )
    return BenchRun(row=row, result=result)


def payload(rows: list[ClipRun] | None = None) -> dict:
    rows = rows or [
        _row("leak", expected=True, predicted=True, label="CONFIRMED_LEAK"),
        _row(
            "glint",
            expected=False,
            predicted=False,
            label="NOMINAL",
            zone=None,
            localized=None,
            localizable=False,
            checkpoint=None,
            valve=None,
            detection_s=None,
        ),
    ]
    return build_payload(
        [_run(row) for row in rows],
        cfg={},
        split="all",
        config_hash="deadbeefdeadbeef",
        fusion_hash="164849b04a9afb87",
        git_sha="90ac543",
        generated_at=datetime(2026, 10, 30, 20, 0, 0, tzinfo=UTC),
    )


def test_the_payload_has_the_required_digests_and_metrics() -> None:
    data = payload()

    assert data["configHash"] == "deadbeefdeadbeef"
    assert data["fusionHash"] == "164849b04a9afb87"
    assert data["gitSha"] == "90ac543"
    assert data["runHash"]
    assert data["generatedAt"] == "2026-10-30T20:00:00.000Z"
    assert data["headline"]["recall"]["display"] == "100.0% (1/1)"
    assert data["headline"]["falsePositiveRate"]["display"] == "0.0% (0/1)"
    assert data["metrics"]["all"]["tp"] == 1
    assert data["metrics"]["test"]["tn"] == 1


def test_the_report_is_framed_as_a_scenario_verification_suite() -> None:
    data = payload()
    rendered = render_markdown(data)

    assert data["title"] == (
        "Deterministic Regression & Safety Verification Suite (4 Synthetic Scenarios)"
    )
    assert rendered.startswith(
        "# Deterministic Regression & Safety Verification Suite (4 Synthetic Scenarios)"
    )
    assert data["verification"]["display"] == (
        "Automated Scenario Verification: 2/2 passing "
        "(1 positive detection scenario, 1 negative control scenario)"
    )
    assert data["verification"]["display"] in rendered
    assert data["verification"]["passed"] == 2
    assert data["verification"]["total"] == 2
    assert "100% Recall" not in rendered


def test_the_sample_size_notice_sits_next_to_the_confidence_intervals() -> None:
    data = payload()
    rendered = render_markdown(data)

    assert data["notice"].startswith("Notice: Sample size n=4")
    assert "Clopper-Pearson" in rendered
    assert "Notice: Sample size n=4" in rendered
    # The notice follows the headline table, where the CI and counts live.
    assert rendered.index("Clopper-Pearson") < rendered.index("Notice: Sample size n=4")


def test_localization_reports_the_geometric_error_and_tolerance() -> None:
    data = payload()
    rendered = render_markdown(data)

    assert data["localizationToleranceMm"] == 25.0
    assert data["localization"]["meanErrorMm"] == pytest.approx(0.8)
    assert data["localization"]["meanErrorDisplay"] == "0.8 mm"
    assert data["localization"]["display"] == "100.0% (1/1)"
    assert data["clips"][0]["localizationErrorMm"] == pytest.approx(0.8)
    assert data["clips"][0]["localizedWithinTolerance"] is True
    assert "Loc. error (mm)" in rendered


def test_the_markdown_is_a_pure_function_of_the_json() -> None:
    data = payload()

    rendered = render_markdown(data)

    assert render_markdown(json.loads(json.dumps(data))) == rendered
    assert "Clopper-Pearson" in rendered
    assert "| leak |" in rendered


def test_every_number_in_the_headline_table_exists_in_the_json() -> None:
    data = payload()
    rendered = render_markdown(data)
    headline = rendered.split("## Headline", 1)[1].split("## Per split", 1)[0]
    numbers = {
        token
        for token in __import__("re").findall(r"\d+(?:\.\d+)?", headline)
        if token not in {"95"}
    }
    serialized = json.dumps(data, sort_keys=True)

    for token in numbers:
        assert token in serialized, f"Markdown number {token!r} is not in the JSON"


def test_a_changed_row_changes_the_run_hash() -> None:
    first = payload()
    second = payload(
        [
            _row("leak", expected=True, predicted=True, label="CONFIRMED_LEAK", p=0.5),
            _row(
                "glint",
                expected=False,
                predicted=False,
                label="NOMINAL",
                zone=None,
                localized=None,
                localizable=False,
                checkpoint=None,
                valve=None,
                detection_s=None,
            ),
        ]
    )

    assert first["runHash"] != second["runHash"]


def test_null_blocks_render_as_na() -> None:
    data = payload(
        [
            _row(
                "only",
                expected=True,
                predicted=False,
                label="NOMINAL",
                zone=None,
                localized=None,
                localizable=False,
                checkpoint=None,
                valve=None,
                detection_s=None,
            )
        ]
    )

    assert data["localization"]["display"] == "n/a"
    assert data["checkpointRecovery"]["display"] == "n/a"
    assert data["valveIsolation"]["display"] == "n/a"
    assert data["detectionLatencyS"]["display"] == "n/a"


def test_latency_is_flagged_as_not_deterministic() -> None:
    data = payload()
    runs = [
        _run(_row("leak", expected=True, predicted=True, label="CONFIRMED_LEAK")),
        _run(_row("glint", expected=False, predicted=False, label="NOMINAL")),
    ]

    latency = build_latency(
        runs, config_hash=data["configHash"], run_hash=data["runHash"], git_sha="90ac543"
    )

    assert latency["deterministic"] is False
    assert latency["stages"]["perceptionTickMs"]["median"] == pytest.approx(2.0)
    assert latency["stages"]["samCachedMs"]["n"] == 0
    assert latency["stages"]["samLiveMs"]["n"] == 0
    assert latency["stages"]["sequenceMs"]["n"] == 1


def test_a_missing_split_is_refused() -> None:
    with pytest.raises(ValueError, match="no rows"):
        build_payload(
            [_run(_row("leak", split="dev", expected=True, predicted=True, label="LEAK"))],
            cfg={},
            split="test",
            config_hash="x",
            fusion_hash="y",
            git_sha="z",
        )
