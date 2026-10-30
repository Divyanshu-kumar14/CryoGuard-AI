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
        return ClampResult(ok=self.ok, applied_w=self.applied_w)


class FakePlc:
    def __init__(self, calls: list[str], status: ValveStatus = ValveStatus.VERIFIED) -> None:
        self.calls = calls
        self.status = status
        self.ever_called = False

    def close_valve(self) -> ValveResult:
        self.calls.append("valve")
        self.ever_called = True
        return ValveResult(status=self.status, travel_ms=950.0)


class FakeCheckpoint:
    def __init__(self, calls: list[str], clock: FakeClock, settle_after_s: float) -> None:
        self.calls = calls
        self.clock = clock
        self.settle_after_s = settle_after_s

    def start(self) -> CheckpointResult:
        self.calls.append("checkpoint")
        started = self.clock.now()
        return CheckpointResult(started_at=started, ready_at=started + self.settle_after_s, step=42)


def build(
    clock: FakeClock,
    *,
    clamp_ok: bool = True,
    valve: ValveStatus = ValveStatus.VERIFIED,
    settle: float = 30.0,
):
    calls: list[str] = []
    orch = ConfirmedLeakOrchestrator(
        clock=clock,
        nvml=FakeNvml(calls, ok=clamp_ok),
        plc=FakePlc(calls, status=valve),
        checkpoint=FakeCheckpoint(calls, clock, settle),
    )
    return orch, calls


def test_clamp_is_verified_before_the_valve_moves() -> None:
    orch, calls = build(FakeClock())
    fsm = orch.run()

    assert calls == ["checkpoint", "clamp", "valve"]
    assert fsm.state is IncidentState.RESOLVED_PENDING_REPAIR


def test_isolation_does_not_wait_for_the_checkpoint() -> None:
    """Delta C-04: the valve is what stops the coolant. It cannot queue behind I/O."""

    clock = FakeClock()
    orch, _ = build(clock, settle=30.0)
    fsm = orch.run()

    assert fsm.timestamp("valve_closed") < fsm.timestamp("checkpoint_settled")
    assert fsm.timestamp("valve_closed") == pytest.approx(0.0)
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
    calls: list[str] = []
    plc = FakePlc(calls)
    orch = ConfirmedLeakOrchestrator(
        clock=clock,
        nvml=FakeNvml(calls, ok=False),
        plc=plc,
        checkpoint=FakeCheckpoint(calls, clock, 30.0),
    )
    fsm = orch.run()

    assert fsm.state is IncidentState.HARD_TRIP
    assert not plc.ever_called
    assert "power_off" not in [name for name, _ in fsm.events]


def test_faulty_valve_reports_failed_safe_and_never_claims_isolation() -> None:
    clock = FakeClock()
    orch, _ = build(clock, valve=ValveStatus.FAULT)
    fsm = orch.run()

    assert fsm.state is IncidentState.FAILED_SAFE
    assert "valve_closed" not in [name for name, _ in fsm.events]
    assert "power_off" not in [name for name, _ in fsm.events]


def test_terminal_states_reject_further_events() -> None:
    fsm = IncidentFSM()
    fsm.fire("decision", 0.0)
    fsm.fire("hard_trip", 0.1)

    with pytest.raises(IllegalTransition, match="terminal"):
        fsm.fire("clamp_verified", 0.2)
