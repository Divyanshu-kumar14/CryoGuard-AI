"""Incident state machine with the D-06 ordering made structural.

Two invariants, both tested:
  1. the valve is never commanded before the power clamp is verified by readback;
  2. isolation is reached on the limit switch alone - a pending checkpoint must
     not delay it (delta C-04). The checkpoint gates power-off, not isolation.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from cryoguard.models import IncidentState, ValveStatus

TERMINAL = {
    IncidentState.RESOLVED_PENDING_REPAIR,
    IncidentState.HARD_TRIP,
    IncidentState.FAILED_SAFE,
}

TRANSITIONS: dict[tuple[IncidentState, str], IncidentState] = {
    (IncidentState.OPEN, "decision"): IncidentState.CLAMPING,
    (IncidentState.CLAMPING, "clamp_verified"): IncidentState.ISOLATING,
    (IncidentState.ISOLATING, "valve_closed"): IncidentState.ISOLATED,
    (IncidentState.ISOLATED, "power_off"): IncidentState.NODE_POWERED_DOWN,
    (IncidentState.NODE_POWERED_DOWN, "requeued"): IncidentState.REQUEUED,
    (IncidentState.REQUEUED, "closed"): IncidentState.RESOLVED_PENDING_REPAIR,
}


class IllegalTransition(RuntimeError):
    pass


class IncidentFSM:
    def __init__(self) -> None:
        self.state = IncidentState.OPEN
        self.checkpoint_settled = False
        self.events: list[tuple[str, float]] = []

    def fire(self, event: str, ts: float) -> IncidentState:
        if self.state in TERMINAL:
            raise IllegalTransition(f"{event} after terminal state {self.state.value}")

        if event == "checkpoint_settled":
            self.checkpoint_settled = True
            self.events.append((event, ts))
            return self.state

        if event == "power_off" and not self.checkpoint_settled:
            raise IllegalTransition("power-off before the checkpoint settled")

        if event in ("hard_trip", "failed_safe"):
            self.state = (
                IncidentState.HARD_TRIP if event == "hard_trip" else IncidentState.FAILED_SAFE
            )
            self.events.append((event, ts))
            return self.state

        target = TRANSITIONS.get((self.state, event))
        if target is None:
            raise IllegalTransition(f"{self.state.value} has no transition for {event}")

        self.state = target
        self.events.append((event, ts))
        return self.state

    def timestamp(self, event: str) -> float:
        for name, ts in self.events:
            if name == event:
                return ts
        raise KeyError(event)


class Clock(Protocol):
    def now(self) -> float: ...
    def advance(self, seconds: float) -> float: ...


class NvmlPort(Protocol):
    def clamp(self) -> ClampResult: ...


class PlcPort(Protocol):
    def close_valve(self) -> ValveResult: ...


class CheckpointPort(Protocol):
    def start(self) -> CheckpointResult: ...


@dataclass(frozen=True)
class ClampResult:
    ok: bool
    applied_w: float


@dataclass(frozen=True)
class ValveResult:
    status: ValveStatus
    travel_ms: float


@dataclass(frozen=True)
class CheckpointResult:
    started_at: float
    ready_at: float
    step: int


class ConfirmedLeakOrchestrator:
    """Runs the CONFIRMED_LEAK sequence in physical order (D-06)."""

    def __init__(
        self,
        clock: Clock,
        nvml: NvmlPort,
        plc: PlcPort,
        checkpoint: CheckpointPort,
        on_action: Callable[[str, str, float], None] | None = None,
        clamp_latency_ms: float = 0.0,
    ) -> None:
        self.clock = clock
        self.nvml = nvml
        self.plc = plc
        self.checkpoint = checkpoint
        self.on_action = on_action or (lambda *_: None)
        self.clamp_latency_ms = clamp_latency_ms

    def run(self) -> IncidentFSM:
        fsm = IncidentFSM()
        t0 = self.clock.now()
        fsm.fire("decision", t0)

        # Started first, settles independently - never blocks isolation.
        checkpoint = self.checkpoint.start()

        clamp = self.nvml.clamp()
        self.on_action("NVML", "CLAMP", clamp.applied_w)
        if not clamp.ok:
            fsm.fire("hard_trip", self.clock.now())
            return fsm
        self.clock.advance(self.clamp_latency_ms / 1000.0)
        fsm.fire("clamp_verified", self.clock.now())

        valve = self.plc.close_valve()
        self.on_action("PLC", "VALVE_CLOSE", valve.travel_ms)

        # A FAULT means closure is unverified, so isolation was never achieved.
        # Claiming ISOLATED here would be the most dangerous lie in the sequence.
        if valve.status is ValveStatus.FAULT:
            fsm.fire("failed_safe", self.clock.now())
            return fsm
        self.clock.advance(valve.travel_ms / 1000.0)
        fsm.fire("valve_closed", self.clock.now())

        # Isolation is already achieved; the checkpoint gates the power-off below.
        self.clock.advance(max(0.0, checkpoint.ready_at - self.clock.now()))
        fsm.fire("checkpoint_settled", self.clock.now())
        fsm.fire("power_off", self.clock.now())
        fsm.fire("requeued", self.clock.now())
        fsm.fire("closed", self.clock.now())
        return fsm
