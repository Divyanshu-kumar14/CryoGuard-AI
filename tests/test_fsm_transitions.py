"""FSM and orchestration order. The two invariants are structural, not conventions."""

from __future__ import annotations

import pytest

from cryoguard.models import IncidentState, ValveStatus
from cryoguard.orchestrator.incident_fsm import (
    CheckpointResult,
    ClampResult,
    ConfirmedLeakOrchestrator,
    IllegalTransition,
    IncidentFSM,
    PowerResult,
    ValveResult,
)


class FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def now(self) -> float:
        return self.t

    def advance(self, seconds: float) -> float:
        self.t += max(0.0, seconds)
        return self.t


class FakeNvml:
    def __init__(self, calls: list[str], ok: bool = True, applied_w: float = 200.0) -> None:
        self.calls = calls
        self.ok = ok
        self.applied_w = applied_w

    def clamp(self) -> ClampResult:
        self.calls.append("clamp")
        return ClampResult(ok=self.ok, applied_w=self.applied_w, detail="fake")


class FakePlc:
    def __init__(self, calls: list[str], status: ValveStatus = ValveStatus.VERIFIED) -> None:
        self.calls = calls
        self.status = status
        self.ever_called = False

    def close_valve(self) -> ValveResult:
        self.calls.append("valve")
        self.ever_called = True
        return ValveResult(status=self.status, travel_ms=950.0, detail="fake", nominal_ms=950.0)


class FakeCheckpoint:
    def __init__(self, calls: list[str], clock: FakeClock, settle_after_s: float) -> None:
        self.calls = calls
        self.clock = clock
        self.settle_after_s = settle_after_s

    def start(self) -> CheckpointResult:
        self.calls.append("checkpoint")
        started = self.clock.now()
        return CheckpointResult(
            started_at=started,
            ready_at=started + self.settle_after_s,
            step=42,
            reason="fake checkpoint",
        )


class FakeBmc:
    def __init__(self, calls: list[str], ok: bool = True) -> None:
        self.calls = calls
        self.ok = ok

    def force_off(self) -> PowerResult:
        self.calls.append("bmc")
        return PowerResult(ok=self.ok, state="Off" if self.ok else "On", simulated=True)


class FakeScheduler:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls
        self.drained = False
        self.requeued = False
        self.checkpoint_step = 0

    def drain(self) -> str:
        self.calls.append("drain")
        self.drained = True
        return "DRAINED"

    def requeue(self, checkpoint_step: int) -> str:
        self.calls.append("requeue")
        self.requeued = True
        self.checkpoint_step = checkpoint_step
        return "gpu-node-09a"


def build(
    clock: FakeClock,
    *,
    clamp_ok: bool = True,
    valve: ValveStatus = ValveStatus.VERIFIED,
    settle: float = 30.0,
    bmc_ok: bool = True,
):
    calls: list[str] = []
    plc = FakePlc(calls, status=valve)
    bmc = FakeBmc(calls, ok=bmc_ok)
    scheduler = FakeScheduler(calls)
    orch = ConfirmedLeakOrchestrator(
        clock=clock,
        nvml=FakeNvml(calls, ok=clamp_ok),
        plc=plc,
        checkpoint=FakeCheckpoint(calls, clock, settle),
        bmc=bmc,
        scheduler=scheduler,
    )
    return orch, calls, plc, bmc, scheduler


def test_clamp_is_verified_before_the_valve_moves() -> None:
    orch, calls, _, _, _ = build(FakeClock())
    fsm = orch.run()

    assert calls[:3] == ["checkpoint", "clamp", "valve"]
    assert fsm.state is IncidentState.RESOLVED_PENDING_REPAIR


def test_isolation_does_not_wait_for_the_checkpoint() -> None:
    """Delta C-04: the valve is what stops the coolant. It cannot queue behind I/O."""

    clock = FakeClock()
    orch, _, _, _, _ = build(clock, settle=30.0)
    fsm = orch.run()

    assert fsm.timestamp("valve_closed") < fsm.timestamp("checkpoint_settled")
    assert fsm.timestamp("valve_closed") == pytest.approx(0.95)  # one valve travel
    assert fsm.timestamp("power_off") == pytest.approx(30.0)


def test_power_off_is_refused_until_the_checkpoint_settles() -> None:
    fsm = IncidentFSM()
    fsm.fire("decision", 0.0)
    fsm.fire("clamp_verified", 0.1)
    fsm.fire("valve_closed", 1.0)

    with pytest.raises(IllegalTransition, match="power-off"):
        fsm.fire("power_off", 1.1)


def test_failed_clamp_hard_trips_without_ever_commanding_the_valve() -> None:
    clock = FakeClock()
    orch, calls, plc, _, scheduler = build(clock, clamp_ok=False)
    fsm = orch.run()

    assert fsm.state is IncidentState.HARD_TRIP
    assert not plc.ever_called
    assert calls == ["checkpoint", "clamp", "drain", "bmc"]
    assert scheduler.drained
    assert "power_off" not in [name for name, _ in fsm.events]
    assert [step.command for step in orch.steps] == ["CLAMP", "SCHEDULER_DRAIN", "FORCE_OFF"]


def test_a_valve_fault_escalates_to_a_power_cut() -> None:
    """T-FSM-04: a failed close is not isolation; cut power, then ask why."""

    clock = FakeClock()
    orch, calls, _, _, _ = build(clock, valve=ValveStatus.FAULT, bmc_ok=True)
    fsm = orch.run()

    assert fsm.state is IncidentState.HARD_TRIP
    assert "valve_closed" not in [name for name, _ in fsm.events]
    assert "power_off" not in [name for name, _ in fsm.events]
    assert calls == ["checkpoint", "clamp", "valve", "drain", "bmc"]
    assert [step.status for step in orch.steps] == [
        "READBACK_VERIFIED",
        "FAULT",
        "DRAINED",
        "ACCEPTED",
    ]


def test_a_valve_fault_with_a_failed_power_cut_is_failed_safe() -> None:
    clock = FakeClock()
    orch, _, _, _, _ = build(clock, valve=ValveStatus.FAULT, bmc_ok=False)
    fsm = orch.run()

    assert fsm.state is IncidentState.FAILED_SAFE
    assert "valve_closed" not in [name for name, _ in fsm.events]
    assert [step.command for step in orch.steps] == [
        "CLAMP",
        "VALVE_CLOSE",
        "SCHEDULER_DRAIN",
        "FORCE_OFF",
    ]
    assert orch.steps[-1].status == "REJECTED"


def test_a_verified_close_with_a_depressurized_line_reaches_isolated() -> None:
    """VERIFIED is only emitted after stem closure *and* depressurization."""

    clock = FakeClock()
    orch, _, _, _, _ = build(clock)
    fsm = orch.run()

    assert fsm.state is IncidentState.RESOLVED_PENDING_REPAIR
    assert "valve_closed" in [name for name, _ in fsm.events]
    valve_step = next(step for step in orch.steps if step.command == "VALVE_CLOSE")
    assert valve_step.status == "LIMIT_SWITCH_CLOSED"


def test_a_seal_leak_hard_trips_and_cuts_power() -> None:
    """A closed stem over a still-pressurized line is not isolation."""

    clock = FakeClock()
    orch, calls, _, _, _ = build(clock, valve=ValveStatus.VALVE_SEAL_LEAK)
    fsm = orch.run()

    assert fsm.state is IncidentState.HARD_TRIP
    assert calls == ["checkpoint", "clamp", "valve", "drain", "bmc"]
    assert "valve_closed" not in [name for name, _ in fsm.events]
    assert [step.command for step in orch.steps] == [
        "CLAMP",
        "VALVE_CLOSE",
        "SCHEDULER_DRAIN",
        "FORCE_OFF",
    ]
    assert orch.steps[1].status == "VALVE_SEAL_LEAK"
    assert orch.steps[-1].status == "ACCEPTED"
    assert "seal leak" in orch.steps[-1].detail


def test_a_seal_leak_with_a_failed_power_cut_is_failed_safe() -> None:
    clock = FakeClock()
    orch, _, _, _, _ = build(clock, valve=ValveStatus.VALVE_SEAL_LEAK, bmc_ok=False)
    fsm = orch.run()

    assert fsm.state is IncidentState.FAILED_SAFE
    assert "valve_closed" not in [name for name, _ in fsm.events]
    assert orch.steps[-1].command == "FORCE_OFF"
    assert orch.steps[-1].status == "REJECTED"


def test_a_slow_valve_still_isolates_and_is_reported_as_slow() -> None:
    clock = FakeClock()
    orch, _, _, _, _ = build(clock, valve=ValveStatus.VERIFIED_SLOW)
    fsm = orch.run()

    assert fsm.state is IncidentState.RESOLVED_PENDING_REPAIR
    valve_step = next(step for step in orch.steps if step.command == "VALVE_CLOSE")
    assert valve_step.status == "LIMIT_SWITCH_CLOSED_SLOW"


def test_a_failed_power_cut_after_isolation_is_failed_safe() -> None:
    clock = FakeClock()
    orch, _, _, _, _ = build(clock, bmc_ok=False)
    fsm = orch.run()

    assert fsm.state is IncidentState.FAILED_SAFE
    assert "power_off" not in [name for name, _ in fsm.events]
    assert "requeued" not in [name for name, _ in fsm.events]


def test_actions_are_ordered_and_complete() -> None:
    clock = FakeClock()
    orch, _, _, _, scheduler = build(clock)
    fsm = orch.run()

    assert fsm.state is IncidentState.RESOLVED_PENDING_REPAIR
    assert [step.command for step in orch.steps] == [
        "CLAMP",
        "VALVE_CLOSE",
        "SCHEDULER_DRAIN",
        "CHECKPOINT_VERIFY",
        "FORCE_OFF",
        "REQUEUE",
    ]
    assert scheduler.checkpoint_step == 42
    timestamps = [step.ts for step in orch.steps]
    assert timestamps == sorted(timestamps)


def test_terminal_states_reject_further_events() -> None:
    fsm = IncidentFSM()
    fsm.fire("decision", 0.0)
    fsm.fire("hard_trip", 0.1)

    with pytest.raises(IllegalTransition, match="terminal"):
        fsm.fire("clamp_verified", 0.2)
