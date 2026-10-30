"""Deterministic fault-injection suite for the benchmark runner.

Each case exercises a bench runner path with mocked inputs — no video
fixtures, no network, no wall clock — and asserts the fault is recorded
honestly: an explicit failure, or unavailable (None), never a silent zero or
a false success. Counts are published separately from the scenario metrics
(recall X/Y over n=4 synthetic fixtures) and never enter ``score()`` or
``verification()``.
"""

from __future__ import annotations

import pytest

from cryoguard.bench.faults import FAULT_CASES, SUITE, run_fault_matrix


def _ids() -> list[str]:
    return [case.__name__.replace("case_", "").replace("_", "-") for case in FAULT_CASES]


def test_the_matrix_covers_the_required_fault_classes() -> None:
    ids = set(_ids())

    assert {
        "camera-degraded",
        "sam-timeout",
        "sam-malformed",
        "plc-disconnect",
        "valve-seal-leak",
        "bmc-poweroff-unverified",
        "stale-telemetry",
        "checkpoint-corrupt",
        "watchdog-fallback-failed",
    } <= ids


@pytest.mark.parametrize("case", FAULT_CASES, ids=_ids())
def test_each_fault_is_recorded_honestly(case) -> None:
    outcome = case()

    assert outcome.passed, f"{outcome.id}: {outcome.detail}"
    assert outcome.detail, f"{outcome.id} must explain what was verified"


def test_the_whole_matrix_passes_with_a_separate_count() -> None:
    summary = run_fault_matrix()

    assert summary["suite"] == SUITE
    assert summary["total"] == len(FAULT_CASES) > 0
    assert summary["passed"] == summary["total"]
    assert summary["separateFromScenarioMetrics"] is True
    assert len(summary["cases"]) == summary["total"]
    assert all(case["passed"] for case in summary["cases"])


def test_fault_cases_never_enter_scenario_metrics() -> None:
    from cryoguard.bench.report import score, verification

    summary = run_fault_matrix()
    scenario_rows = [
        {"expectedIncident": True, "predictedIncident": True},
        {"expectedIncident": False, "predictedIncident": False},
    ]

    # The scenario metric inputs contain no fault rows: fault outcomes carry
    # no expectedIncident/predictedIncident keys and are never scored.
    assert all(
        "expectedIncident" not in case and "predictedIncident" not in case
        for case in summary["cases"]
    )
    assert verification(scenario_rows)["total"] == 2
    assert score(scenario_rows, 0.95)["tp"] == 1


def test_the_matrix_is_deterministic() -> None:
    assert run_fault_matrix() == run_fault_matrix()
