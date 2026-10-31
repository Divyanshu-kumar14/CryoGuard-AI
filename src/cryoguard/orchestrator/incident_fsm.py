"""Incident state machine with the D-06 ordering made structural.

Two invariants, both tested:
  1. the valve is never commanded before the power clamp is verified by readback;
  2. isolation is reached on verified closure - the limit switch *and* both
     depressurization witnesses agreeing (DI 1 and HR 0 with stabilization) -
     and a pending checkpoint must not delay it (delta C-04). The checkpoint
     gates power-off, not isolation.

Failure paths follow T-FSM-04: a clamp readback failure, a valve limit-switch
FAULT, a SENSOR_DISAGREEMENT (witnesses contradict or a witness is
unavailable), or a stem that closes onto a still-pressurized line
(VALVE_SEAL_LEAK) escalates to a node power cut. Only when that cut also fails
does the sequence exit ``FAILED_SAFE``. A valve that did not close, or whose
sensors disagree, is never reported as isolated, and a power cut that was not
accepted and read back as ``Off`` is never reported as ``NODE_POWERED_DOWN``.

Safety ordering (structural, preserved): clamp -> valve close (Close accepted,
Stem closed) -> isolation verify (Line depressurized, Isolation verified or
Verification failed) -> cordon -> checkpoint verify -> power-off -> requeue.
The checkpoint starts first but settles independently; it gates power-off,
never isolation.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from cryoguard.models import IncidentState, ValveStatus
from cryoguard.orchestrator.actions import ActionLog, Clock, SequenceStep, StepCallback

TERMINAL = {
    IncidentState.RESOLVED_PENDING_REPAIR,
    IncidentState.EMERGENCY_TRIP,
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


class NvmlPort(Protocol):
    def clamp(self) -> ClampResult: ...


class PlcPort(Protocol):
    def close_valve(self) -> ValveResult: ...


class CheckpointPort(Protocol):
    def start(self) -> CheckpointResult: ...


class BmcPort(Protocol):
    def force_off(self) -> PowerResult: ...


class SchedulerPort(Protocol):
    def drain(self) -> str: ...
    def requeue(self, checkpoint_step: int) -> str: ...


@dataclass(frozen=True)
class ClampResult:
    ok: bool
    applied_w: float
    detail: str = ""


@dataclass(frozen=True)
class ValveResult:
    status: ValveStatus
    travel_ms: float
    detail: str = ""
    nominal_ms: float = 0.0


VERIFIED_OFF = "VERIFIED_OFF"
VERIFICATION_TIMEOUT = "VERIFICATION_TIMEOUT"


@dataclass(frozen=True)
class PowerResult:
    ok: bool
    state: str
    attempts: int = 0
    status_code: int = 0
    simulated: bool = True
    detail: str = ""
    # Readback outcome, set only by a port that actually verified power off. An
    # empty value means the port reports acceptance (ok) alone.
    status: str = ""


@dataclass(frozen=True)
class CheckpointResult:
    started_at: float
    ready_at: float
    step: int
    reason: str = ""
    verified: bool = True


def power_verified(power: PowerResult) -> bool:
    """Off is only claimed from a readback, never from an accepted command.

    A port that reports acceptance alone (``ok`` with an empty readback status)
    is not evidence that the node lost power. The in-memory BMC has no socket to
    poll, so its deterministic ``ok + Off + simulated`` verdict counts.
    """

    return bool(
        power.ok and power.state == "Off" and (power.simulated or power.status == VERIFIED_OFF)
    )


class ConfirmedLeakOrchestrator:
    """Runs the CONFIRMED_LEAK sequence in physical order (D-06)."""

    def __init__(
        self,
        clock: Clock,
        nvml: NvmlPort,
        plc: PlcPort,
        checkpoint: CheckpointPort,
        bmc: BmcPort,
        scheduler: SchedulerPort,
        clamp_latency_ms: float = 0.0,
        on_step: StepCallback | None = None,
        abort: Callable[[], str | None] | None = None,
    ) -> None:
        self.clock = clock
        self.nvml = nvml
        self.plc = plc
        self.checkpoint = checkpoint
        self.bmc = bmc
        self.scheduler = scheduler
        self.clamp_latency_ms = clamp_latency_ms
        self.on_step = on_step
        self.abort = abort
        self.log = ActionLog(clock, on_step)

    @property
    def steps(self) -> tuple[SequenceStep, ...]:
        """Read-only view: the report service never gets the mutable log."""

        return self.log.snapshot()

    def _record(self, actor: str, command: str, status: str, detail: str = "") -> SequenceStep:
        return self.log.append(actor, command, status, detail)

    def _power_cut(self, reason: str) -> PowerResult:
        power = self.bmc.force_off()
        source = "SIMULATED" if power.simulated else "live"
        status = power.status or ("ACCEPTED" if power.ok else "REJECTED")
        detail = f"{reason}; PowerState={power.state} ({source} BMC)"
        if power.detail:
            detail = f"{detail}; {power.detail}"
        self._record("BMC", "FORCE_OFF", status, detail)
        return power

    def _cordon(self) -> None:
        """Best-effort scheduler cordon. It must never delay the power cut."""

        try:
            drain_state = self.scheduler.drain()
        except Exception as exc:  # the cordon is reversible; the power cut is not optional
            self._record("SLURM", "SCHEDULER_DRAIN", "FAILED", f"{type(exc).__name__}: {exc}")
            return
        self._record("SLURM", "SCHEDULER_DRAIN", "DRAINED", f"State={drain_state} (mock)")

    def run(self) -> IncidentFSM:
        """Run one incident; the log is sealed on every exit path, success or not."""

        self.log = ActionLog(self.clock, self.on_step)
        fsm = IncidentFSM()
        fsm.fire("decision", self.clock.now())
        try:
            return self._sequence(fsm)
        finally:
            self.log.seal()

    def _sequence(self, fsm: IncidentFSM) -> IncidentFSM:

        # Started first, settles independently - never blocks isolation. A corrupt
        # or missing checkpoint is a failed branch, not a reason to leave coolant
        # flowing; it gates the power-off later, not the clamp or the valve.
        try:
            checkpoint = self.checkpoint.start()
        except Exception as exc:
            checkpoint = CheckpointResult(
                started_at=self.clock.now(),
                ready_at=self.clock.now(),
                step=-1,
                reason=f"checkpoint verification failed: {type(exc).__name__}: {exc}",
                verified=False,
            )
            self._record("STORAGE", "CHECKPOINT_VERIFY", "FAILED", checkpoint.reason)

        clamp = self.nvml.clamp()
        clamp_status = "READBACK_VERIFIED" if clamp.ok else "READBACK_FAILED"
        self._record(
            "NVML",
            "CLAMP",
            clamp_status,
            clamp.detail or f"{clamp.applied_w:.0f} W",
        )
        if not clamp.ok:
            self._cordon()
            power = self._power_cut("clamp readback failed")
            fsm.fire("hard_trip" if power_verified(power) else "failed_safe", self.clock.now())
            return fsm
        self.clock.advance(self.clamp_latency_ms / 1000.0)
        fsm.fire("clamp_verified", self.clock.now())

        valve = self.plc.close_valve()
        if valve.status is ValveStatus.FAULT:
            valve_status = "FAULT"
        elif valve.status is ValveStatus.VALVE_SEAL_LEAK:
            valve_status = "VALVE_SEAL_LEAK"
        elif valve.status is ValveStatus.SENSOR_DISAGREEMENT:
            valve_status = "SENSOR_DISAGREEMENT"
        elif valve.status is ValveStatus.VERIFIED_SLOW:
            valve_status = "LIMIT_SWITCH_CLOSED_SLOW"
        else:
            valve_status = "LIMIT_SWITCH_CLOSED"
        if valve.status in (ValveStatus.VERIFIED, ValveStatus.VERIFIED_SLOW):
            fallback = f"limit switch; band {valve.status.value}"
            valve_detail = (
                "Close accepted; Stem closed; Line depressurized; "
                f"Isolation verified; {valve.detail or fallback}"
            )
        elif valve.status in (ValveStatus.VALVE_SEAL_LEAK, ValveStatus.SENSOR_DISAGREEMENT):
            fallback = f"limit switch; band {valve.status.value}"
            valve_detail = (
                "Verification failed; Stem closed; Line depressurized: unproven; "
                f"{valve.detail or fallback}"
            )
        else:
            fallback = f"limit switch; band {valve.status.value}"
            valve_detail = (
                "Verification failed; Stem closed: unproven; Line depressurized: unproven; "
                f"{valve.detail or fallback}"
            )
        self._record(
            "PLC",
            "VALVE_CLOSE",
            valve_status,
            valve_detail,
        )

        # A FAULT means the stem never closed (or reopened / transport failed);
        # a VALVE_SEAL_LEAK means it closed but the branch is still pressurized;
        # a SENSOR_DISAGREEMENT means the witnesses contradict each other or one
        # is unavailable. Anything short of a verified, depressurized close with
        # agreeing witnesses is not isolation: coolant may still be reaching the
        # chassis, so cut power now and claim nothing else. Never ISOLATED.
        if valve.status not in (ValveStatus.VERIFIED, ValveStatus.VERIFIED_SLOW):
            self._cordon()
            if valve.status is ValveStatus.VALVE_SEAL_LEAK:
                reason = "valve seal leak: line still pressurized"
            elif valve.status is ValveStatus.SENSOR_DISAGREEMENT:
                reason = "sensor disagreement: isolation unproven"
            else:
                reason = "valve limit switch fault"
            power = self._power_cut(reason)
            fsm.fire("hard_trip" if power_verified(power) else "failed_safe", self.clock.now())
            return fsm
        # The scenario clock advances by the configured nominal travel, not the
        # measured wall time, so two runs of the same clip produce the same
        # incident. The measured value still selects the VERIFIED/SLOW/FAULT band.
        self.clock.advance((valve.nominal_ms or valve.travel_ms) / 1000.0)
        fsm.fire("valve_closed", self.clock.now())

        # Cordon the node before the power cut so Slurm cannot reschedule the job
        # onto the node that is about to go down. Reversible, not a valve.
        self._cordon()

        if not checkpoint.verified:
            # Coolant is isolated, but there is no verified checkpoint to resume
            # from. Do not power off and do not requeue; an operator resolves this.
            fsm.fire("failed_safe", self.clock.now())
            return fsm

        # Isolation is already achieved; the checkpoint gates the power-off below.
        self.clock.advance(max(0.0, checkpoint.ready_at - self.clock.now()))
        fsm.fire("checkpoint_settled", self.clock.now())
        self._record(
            "STORAGE",
            "CHECKPOINT_VERIFY",
            "COMMITTED",
            checkpoint.reason or f"step {checkpoint.step} hash-verified",
        )

        power = self._power_cut("confirmed leak")
        if not power_verified(power):
            fsm.fire("failed_safe", self.clock.now())
            return fsm
        fsm.fire("power_off", self.clock.now())

        # The watchdog may have tripped while this sequence was in flight. Its
        # trip took precedence: do not reschedule a job onto a node that an
        # independent safety path has just powered down.
        if self.abort is not None:
            abort_reason = self.abort()
            if abort_reason:
                self._record("SLURM", "REQUEUE", "ABORTED", abort_reason)
                return fsm

        try:
            node = self.scheduler.requeue(checkpoint.step)
        except Exception as exc:
            # Power is verified off; a scheduler failure must not erase that fact
            # or turn into a crash that loses the incident report. Stay powered
            # down and leave the requeue visibly failed for an operator.
            self._record("SLURM", "REQUEUE", "FAILED", f"{type(exc).__name__}: {exc}")
            return fsm
        self._record("SLURM", "REQUEUE", "REQUEUED", f"node={node}; step={checkpoint.step}")
        fsm.fire("requeued", self.clock.now())
        fsm.fire("closed", self.clock.now())
        return fsm
