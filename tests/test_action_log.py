"""P6-T08: the action log is append-only, ordered, and sealed at incident close."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from cryoguard.models import IncidentState
from cryoguard.orchestrator.actions import ActionLog, ActionOrderError, IncidentClosed
from cryoguard.orchestrator.adapters import (
    ScenarioClock,
    SimulatedBmc,
    SimulatedCheckpoint,
    SimulatedNvml,
    SimulatedPlc,
    SimulatedScheduler,
)
from cryoguard.orchestrator.incident_fsm import ConfirmedLeakOrchestrator
from cryoguard.report.fallback import build_fallback_report
from cryoguard.report.schema import ActionModel, EvidenceModel
from cryoguard.report.writer import IncidentImmutableError, write_incident

ROOT = Path(__file__).resolve().parents[1]


class MutableClock:
    def __init__(self, start: float = 0.0) -> None:
        self._t = start

    def now(self) -> float:
        return self._t

    def advance(self, seconds: float) -> float:
        self._t += seconds
        return self._t

    def set(self, seconds: float) -> None:
        self._t = seconds


def _orchestrator(cfg: dict, clock: ScenarioClock) -> ConfirmedLeakOrchestrator:
    return ConfirmedLeakOrchestrator(
        clock=clock,
        nvml=SimulatedNvml(200.0),
        plc=SimulatedPlc(950.0),
        checkpoint=SimulatedCheckpoint(clock, settle_s=1.0, step=4200),
        bmc=SimulatedBmc(),
        scheduler=SimulatedScheduler(cfg["scheduler"]),
        clamp_latency_ms=40.0,
    )


def test_steps_keep_execution_order_and_carry_their_own_timestamp() -> None:
    clock = MutableClock(1.0)
    log = ActionLog(clock)

    first = log.append("NVML", "CLAMP", "READBACK_VERIFIED")
    clock.advance(0.5)
    second = log.append("PLC", "VALVE_CLOSE", "LIMIT_SWITCH_CLOSED")

    assert first.ts == 1.0
    assert second.ts == 1.5
    assert [step.command for step in log.steps] == ["CLAMP", "VALVE_CLOSE"]


def test_a_step_timestamped_before_its_predecessor_is_refused() -> None:
    clock = MutableClock(5.0)
    log = ActionLog(clock)
    log.append("NVML", "CLAMP", "READBACK_VERIFIED")

    clock.set(4.0)
    with pytest.raises(ActionOrderError, match="precedes CLAMP"):
        log.append("PLC", "VALVE_CLOSE", "LIMIT_SWITCH_CLOSED")


def test_a_sealed_log_cannot_grow() -> None:
    log = ActionLog(MutableClock())
    log.append("NVML", "CLAMP", "READBACK_VERIFIED")
    snapshot = log.seal()

    assert log.sealed is True
    assert [step.command for step in snapshot] == ["CLAMP"]
    with pytest.raises(IncidentClosed, match="refused to append REQUEUE"):
        log.append("SLURM", "REQUEUE", "REQUEUED")


def test_the_snapshot_is_detached_from_later_appends() -> None:
    log = ActionLog(MutableClock())
    log.append("NVML", "CLAMP", "READBACK_VERIFIED")
    snapshot = log.snapshot()

    log.append("PLC", "VALVE_CLOSE", "LIMIT_SWITCH_CLOSED")

    assert len(snapshot) == 1
    assert isinstance(snapshot, tuple)


def test_the_orchestrator_seals_its_log_on_the_success_path(cfg) -> None:
    orchestrator = _orchestrator(cfg, ScenarioClock(7.0))

    fsm = orchestrator.run()

    assert fsm.state is IncidentState.RESOLVED_PENDING_REPAIR
    assert orchestrator.log.sealed is True
    assert isinstance(orchestrator.steps, tuple)
    with pytest.raises(IncidentClosed):
        orchestrator.log.append("BMC", "FORCE_OFF", "ACCEPTED")


def test_the_orchestrator_seals_its_log_on_an_early_fault_path(cfg) -> None:
    clock = ScenarioClock(7.0)
    orchestrator = ConfirmedLeakOrchestrator(
        clock=clock,
        nvml=SimulatedNvml(200.0, fail=True),
        plc=SimulatedPlc(950.0),
        checkpoint=SimulatedCheckpoint(clock, settle_s=1.0, step=4200),
        bmc=SimulatedBmc(),
        scheduler=SimulatedScheduler(cfg["scheduler"]),
        clamp_latency_ms=40.0,
    )

    fsm = orchestrator.run()

    assert fsm.state is IncidentState.HARD_TRIP
    assert orchestrator.log.sealed is True


def test_an_action_record_cannot_be_mutated() -> None:
    action = ActionModel(seq=1, t_s=1.0, actor="PLC", command="VALVE_CLOSE", status="FAULT")

    with pytest.raises(ValidationError):
        action.status = "LIMIT_SWITCH_CLOSED"  # type: ignore[misc]


def _report(p_status: str = "LIMIT_SWITCH_CLOSED"):
    evidence = EvidenceModel(
        label="CONFIRMED_LEAK",
        zone=0,
        qd_origin="QD-04",
        p_leak=0.9,
        phi_visual=0.6,
        phi_hydraulic=0.4,
        phi_thermal=1.0,
        wetness=1.0,
        area_mm2=700.0,
        growth_mm2_per_s=80.0,
        mask_quality=1.0,
        pressure_psi=30.0,
        pressure_rate_psi_per_s=0.3,
        flow_imbalance_lpm=0.4,
        surface_c=40.0,
        dew_point_c=11.0,
    )
    from datetime import UTC, datetime

    return build_fallback_report(
        evidence=evidence,
        actions=[ActionModel(seq=1, t_s=7.8, actor="PLC", command="VALVE_CLOSE", status=p_status)],
        incident_time=datetime(2026, 10, 30, 0, 0, 7, 800_000, tzinfo=UTC),
        rack_id="RACK-08B",
        run_index=1,
        config_hash="deadbeefdeadbeef",
        clip_id="slow_leak_qd4",
        frame_range=(0, 10),
        fixture_source="synthetic",
    )


def test_an_identical_rewrite_is_accepted_and_the_file_is_read_only(tmp_path) -> None:
    first = write_incident(_report(), tmp_path)

    second = write_incident(_report(), tmp_path)

    assert second == first
    assert first.stat().st_mode & 0o222 == 0


def test_a_closed_incident_is_never_rewritten_with_different_content(tmp_path) -> None:
    path = write_incident(_report(), tmp_path)

    with pytest.raises(IncidentImmutableError, match="already closed"):
        write_incident(_report("FAULT"), tmp_path)

    assert "FAULT" not in path.read_text(encoding="utf-8")
