"""Deterministic fault-injection matrix for the benchmark (no video fixtures).

Scenario metrics (recall X/Y with exact Clopper-Pearson intervals over n=4
synthetic fixtures) are regression verification, not population performance.
The cases below exercise the bench runner paths with adversarial, fully mocked
inputs — degraded camera, SAM timeouts, actuator and checkpoint failures,
stale and frozen telemetry, a failed watchdog primary with a verified fallback,
and total watchdog fallback loss — and assert each fault is recorded honestly
(explicit failure, or unavailable, never a silent zero or a false success).

Fault cases are counted separately from scenario metrics: they never enter
``score()`` or ``verification()``. The summary from :func:`run_fault_matrix`
is JSON-serializable and deterministic (no timings, no randomness, no I/O),
so a caller may publish it alongside ``benchmark.json`` without touching
``runHash``.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from cryoguard.bench.report import build_latency
from cryoguard.bench.runner import BenchRun, _row
from cryoguard.orchestrator.watchdog import (
    ThermalWatchdog,
    WatchdogActuationError,
    WatchdogReading,
    WatchdogTrip,
)
from cryoguard.perception.candidates import SourceEstimate
from cryoguard.pipeline import RunResult

SUITE = "tests/test_bench_faults.py"
TOPOLOGY = {"qd": [{"id": "QD-04", "roi_mm": [155, 95, 235, 195]}]}


@dataclass(frozen=True)
class FaultOutcome:
    """One fault case result: pass means the fault was recorded honestly."""

    id: str
    category: str
    passed: bool
    detail: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "category": self.category,
            "passed": self.passed,
            "detail": self.detail,
        }


def _clip(**overrides: Any) -> dict[str, Any]:
    clip: dict[str, Any] = {
        "id": "sim_leak_qd4",
        "scenario": "slow_leak_qd4",
        "split": "dev",
        "path": "fixtures/videos/simulated/sim_leak_qd4.mp4",
        "sha256": "0" * 64,
        "ground_truth_label": "CONFIRMED_LEAK",
        "qd_origin": "QD-04",
        "expected_zone": 2,
    }
    clip.update(overrides)
    return clip


def _evidence(zone: int | None = 0, label: str = "CONFIRMED_LEAK", p: float = 0.9) -> Any:
    return SimpleNamespace(
        deterministic_actions_taken=[],
        suspected_origin="QD-04 (Zone 0)",
        physical_evidence=SimpleNamespace(zone=zone, label=label, p_leak=p),
    )


def _decision(label: str = "NOMINAL", rule_id: str = "R7", p: float = 0.05) -> Any:
    return SimpleNamespace(label=SimpleNamespace(value=label), rule_id=rule_id, p=p)


def _action(command: str, status: str, t_s: float = 1.0) -> Any:
    return SimpleNamespace(command=command, status=status, t_s=t_s)


def _result(
    report: Any,
    centroid_mm: tuple[float, float] | None,
    *,
    decision: Any = None,
    actions: list[Any] | None = None,
    sam: Any = None,
    source: SourceEstimate | None = None,
    sequence_state: Any = None,
) -> RunResult:
    if report is not None and actions is not None:
        report.deterministic_actions_taken = actions
    return RunResult(
        scenario="slow_leak_qd4",
        decision=decision,
        report=report,
        path=None,
        frames=100,
        ticks=10,
        tick_rate_hz=5.0,
        rate_changes=0,
        sam=sam,
        centroid_mm=centroid_mm,
        source_estimate=source,
        sequence_state=sequence_state,
    )


def _check(outcome_id: str, category: str, failures: list[str], ok_detail: str) -> FaultOutcome:
    if failures:
        return FaultOutcome(
            id=outcome_id, category=category, passed=False, detail="; ".join(failures)
        )
    return FaultOutcome(id=outcome_id, category=category, passed=True, detail=ok_detail)


def case_camera_degraded() -> FaultOutcome:
    """Degraded camera: no candidate, no report. Nothing is localizable."""

    row = _row(_clip(), _result(None, None, decision=_decision()), TOPOLOGY)
    failures = []
    if row.predicted_incident is not False:
        failures.append("degraded frame must not predict an incident")
    if row.localizable is not False:
        failures.append("no candidate must not be localizable")
    if row.localization_error_mm is not None:
        failures.append(f"missing error must be None, not {row.localization_error_mm!r}")
    if row.source_available is not False or row.source_error_mm is not None:
        failures.append("missing source estimate must be unavailable, not 0")
    return _check(
        "camera-degraded", "perception", failures, "no incident, localization unavailable (None)"
    )


def case_sam_timeout() -> FaultOutcome:
    """SAM timeout: no model summary. The row and latency blocks still build."""

    row = _row(_clip(), _result(None, None, decision=_decision()), TOPOLOGY)
    bench = BenchRun(row=row, result=_result(None, None, decision=_decision()))
    latency = build_latency([bench], config_hash="0" * 16, run_hash="0" * 64, git_sha="test")
    failures = []
    if latency["deterministic"] is not False:
        failures.append("latency must stay flagged non-deterministic")
    if latency["stages"]["samCachedMs"]["n"] != 0:
        failures.append("absent SAM must report n=0 cached samples, not fabricated values")
    if latency["stages"]["samLiveMs"]["n"] != 0:
        failures.append("absent SAM must report n=0 live samples, not fabricated values")
    return _check(
        "sam-timeout",
        "perception",
        failures,
        "row built; SAM stages report n=0, latency non-deterministic",
    )


def case_sam_malformed() -> FaultOutcome:
    """Malformed SAM output dropped upstream: decision falls back to classical."""

    row = _row(_clip(), _result(None, None, decision=_decision(p=0.12)), TOPOLOGY)
    failures = []
    if row.label != "NOMINAL":
        failures.append(f"fallback label must come from the decision, got {row.label!r}")
    if abs(row.p - 0.12) > 1e-9:
        failures.append(f"fallback p must be preserved, got {row.p!r}")
    if row.predicted_incident is not False:
        failures.append("malformed SAM must not manufacture an incident")
    return _check(
        "sam-malformed", "perception", failures, "decision fallback preserved (NOMINAL, p=0.12)"
    )


def case_plc_disconnect() -> FaultOutcome:
    """PLC disconnect mid-sequence: the failed valve close is explicit."""

    actions = [_action("VALVE_CLOSE", "TRANSPORT_ERROR", t_s=7.6)]
    row = _row(_clip(), _result(_evidence(), (203.4, 145.0), actions=actions), TOPOLOGY)
    failures = []
    if row.valve_verified is not False:
        failures.append(f"failed valve close must be False, not {row.valve_verified!r}")
    if row.detection_s is None:
        failures.append("scenario-time detection must still be recorded")
    return _check(
        "plc-disconnect", "actuation", failures, "valve_verified=False with detection_s kept"
    )


def case_valve_seal_leak() -> FaultOutcome:
    """Valve commanded but never confirmed closed: unverified is failure."""

    actions = [_action("VALVE_CLOSE", "COMMAND_SENT", t_s=7.6)]
    row = _row(_clip(), _result(_evidence(), (203.4, 145.0), actions=actions), TOPOLOGY)
    failures = []
    if row.valve_verified is not False:
        failures.append(f"unconfirmed close must be False, not {row.valve_verified!r}")
    return _check(
        "valve-seal-leak",
        "actuation",
        failures,
        "valve_verified=False without limit-switch readback",
    )


def case_bmc_poweroff_unverified() -> FaultOutcome:
    """BMC accepted the power cut but the readback never confirmed it."""

    actions = [_action("FORCE_OFF", "UNVERIFIED", t_s=3.5)]
    state = SimpleNamespace(value="FAILED_SAFE")
    row = _row(
        _clip(),
        _result(_evidence(), None, actions=actions, sequence_state=state),
        TOPOLOGY,
    )
    failures = []
    if row.sequence_state != "FAILED_SAFE":
        failures.append(f"unverified trip must stay FAILED_SAFE, got {row.sequence_state!r}")
    if row.valve_verified is not None:
        failures.append("no valve action means unavailable (None), not a verdict")
    if row.action_count != 1:
        failures.append("the unverified attempt must remain on the record")
    return _check(
        "bmc-poweroff-unverified",
        "actuation",
        failures,
        "FAILED_SAFE preserved; unverified attempt kept, no success claimed",
    )


def case_stale_telemetry() -> FaultOutcome:
    """Frozen telemetry: ticks advance but no detection is claimed."""

    row = _row(_clip(), _result(None, None, decision=_decision()), TOPOLOGY)
    failures = []
    if row.detection_s is not None:
        failures.append(f"no decision means no detection time, got {row.detection_s!r}")
    if row.ticks != 10:
        failures.append("tick count must be preserved even without a detection")
    return _check("stale-telemetry", "telemetry", failures, "detection_s=None with ticks preserved")


def case_checkpoint_corrupt() -> FaultOutcome:
    """Corrupt checkpoint write: explicit failure, distinct from 'no checkpoint'."""

    corrupt = _row(
        _clip(),
        _result(_evidence(), (203.4, 145.0), actions=[_action("CHECKPOINT_VERIFY", "CORRUPT")]),
        TOPOLOGY,
    )
    missing = _row(_clip(), _result(_evidence(), (203.4, 145.0), actions=[]), TOPOLOGY)
    failures = []
    if corrupt.checkpoint_committed is not False:
        failures.append(f"corrupt checkpoint must be False, not {corrupt.checkpoint_committed!r}")
    if missing.checkpoint_committed is not None:
        failures.append("absent checkpoint action must be None (unavailable), not False")
    return _check(
        "checkpoint-corrupt",
        "recovery",
        failures,
        "corrupt=False distinguished from missing=None",
    )


def case_watchdog_fallback_failed() -> FaultOutcome:
    """Watchdog primary and independent fallback both failed: FAILED_SAFE."""

    actions = [
        _action("FORCE_OFF", "FAILED", t_s=1.0),
        _action("FORCE_OFF_FALLBACK", "FAILED", t_s=1.5),
    ]
    state = SimpleNamespace(value="FAILED_SAFE")
    row = _row(
        _clip(),
        _result(_evidence(), None, actions=actions, sequence_state=state),
        TOPOLOGY,
    )
    failures = []
    if row.sequence_state != "FAILED_SAFE":
        failures.append(f"must stay FAILED_SAFE, got {row.sequence_state!r}")
    if row.predicted_incident is not True:
        failures.append("the incident that tripped the watchdog must stay predicted")
    if row.action_count != 2:
        failures.append("both failed attempts must remain on the record")
    return _check(
        "watchdog-fallback-failed",
        "safety",
        failures,
        "FAILED_SAFE with both failed attempts recorded",
    )


def case_frozen_sensor() -> FaultOutcome:
    """Frozen sensor: a multi-channel stuck trace trips SENSOR_FROZEN, explicitly."""

    watchdog = ThermalWatchdog(
        sensor=SimpleNamespace(read=lambda: WatchdogReading(junction_c=65.0)),
        limit_c=90.0,
        period_s=0.0,
        actuate=_noop_actuate,
        max_frozen_ticks=3,
    )
    failure = None
    record = None
    # Sustained load is armed by the load channels: all three are stuck, which is
    # the corroborated case the policy may escalate.
    for tick in range(1, 9):
        reading = WatchdogReading(
            junction_c=65.0,
            utilization_pct=95.0,
            power_w=300.0,
            clock_mhz=1800.0,
            timestamp_s=float(tick) * 0.1,
            valid=True,
        )
        record = watchdog.observe(reading, float(tick) * 0.1)
        if record is not None:
            failure = record
            break
    failures = []
    if failure is None or failure.reason is not WatchdogTrip.SENSOR_FROZEN:
        failures.append(f"multi-channel stuck trace must trip SENSOR_FROZEN, got {failure!r}")
    return _check(
        "frozen-sensor",
        "telemetry",
        failures,
        "SENSOR_FROZEN tripped only after all channels stayed stuck",
    )


def case_watchdog_primary_failed() -> FaultOutcome:
    """Watchdog primary actuation failed; the independent fallback verified Off."""

    calls: list[str] = []

    async def primary(reason: WatchdogTrip, detail: str) -> None:
        calls.append("primary")
        raise RuntimeError("primary Redfish connection lost")

    async def fallback(reason: WatchdogTrip, detail: str) -> None:
        calls.append("fallback")
        # Only a verified Off readback is allowed to mark the fallback as good.

    watchdog = ThermalWatchdog(
        sensor=SimpleNamespace(read=lambda: WatchdogReading(junction_c=95.0)),
        limit_c=90.0,
        period_s=0.0,
        actuate=primary,
        fallback=fallback,
    )
    failures: list[str] = []
    try:
        # The primary failure is logged CRITICAL by design; the matrix reports it
        # through the outcome instead of echoing it into the benchmark output.
        logging.disable(logging.CRITICAL)
        asyncio.run(watchdog.check(WatchdogTrip.JUNCTION_OVER_TEMP, "junction=95.0 C"))
    except WatchdogActuationError:
        pass
    else:
        failures.append("a failed primary must raise WatchdogActuationError")
    finally:
        logging.disable(logging.NOTSET)
    if calls != ["primary", "fallback"]:
        failures.append(f"expected primary then fallback, got {calls!r}")
    if not watchdog.trip_initiated or watchdog.actuated:
        failures.append("trip must stay initiated, never reported actuated")
    if not watchdog.actuation_failed or not watchdog.fallback_attempted:
        failures.append("primary failure and fallback attempt must both be recorded")
    if not watchdog.fallback_ok:
        failures.append("a verified fallback must be recorded as successful")
    return _check(
        "watchdog-primary-failed",
        "safety",
        failures,
        "primary failed, fallback attempted and verified, actuated never claimed",
    )


async def _noop_actuate(reason: WatchdogTrip, detail: str) -> None:
    return None


FAULT_CASES: tuple[Callable[[], FaultOutcome], ...] = (
    case_camera_degraded,
    case_sam_timeout,
    case_sam_malformed,
    case_plc_disconnect,
    case_valve_seal_leak,
    case_bmc_poweroff_unverified,
    case_stale_telemetry,
    case_frozen_sensor,
    case_checkpoint_corrupt,
    case_watchdog_primary_failed,
    case_watchdog_fallback_failed,
)


def run_fault_matrix() -> dict[str, Any]:
    """Run every fault case in fixed order; deterministic, no I/O."""

    outcomes = [case() for case in FAULT_CASES]
    passed = sum(1 for outcome in outcomes if outcome.passed)
    return {
        "suite": SUITE,
        "total": len(outcomes),
        "passed": passed,
        "separateFromScenarioMetrics": True,
        "note": (
            "Deterministic fault-injection cases with mocked inputs (no video "
            "fixtures). Counted separately; never merged into scenario recall, "
            "localization, or runHash."
        ),
        "cases": [outcome.as_dict() for outcome in outcomes],
    }
