"""R1 emergency sequence: isolate and cut power, no checkpoint and no requeue.

This is the path the thermal watchdog and the R1 fusion label execute. It is
deliberately shorter than the confirmed-leak sequence: there is nothing to save,
because the trigger is a hardware condition rather than a measured leak with a
usable checkpoint. It never claims an isolation it did not verify.

Order: valve close (spring-return), scheduler cordon (reversible), power cut.
A valve ``FAULT`` still cuts power and ends ``HARD_TRIP``; if the cut also fails
the result is ``FAILED_SAFE``, never an optimistic ``EMERGENCY_TRIP``.
"""

from __future__ import annotations

from dataclasses import dataclass

from cryoguard.models import IncidentState, ValveStatus
from cryoguard.orchestrator.actions import ActionLog, Clock, SequenceStep
from cryoguard.orchestrator.incident_fsm import BmcPort, PlcPort, PowerResult, SchedulerPort


@dataclass(frozen=True)
class EmergencyResult:
    state: IncidentState
    steps: tuple[SequenceStep, ...]


class EmergencySequence:
    def __init__(
        self,
        clock: Clock,
        plc: PlcPort,
        bmc: BmcPort,
        scheduler: SchedulerPort,
    ) -> None:
        self.clock = clock
        self.plc = plc
        self.bmc = bmc
        self.scheduler = scheduler
        self.log = ActionLog(clock)

    def _record(self, actor: str, command: str, status: str, detail: str = "") -> SequenceStep:
        return self.log.append(actor, command, status, detail)

    def _power_cut(self, reason: str) -> PowerResult:
        power = self.bmc.force_off()
        source = "SIMULATED" if power.simulated else "live"
        detail = f"{reason}; PowerState={power.state} ({source} BMC)"
        if power.detail:
            detail = f"{detail}; {power.detail}"
        self._record("BMC", "FORCE_OFF", "ACCEPTED" if power.ok else "REJECTED", detail)
        return power

    def _cordon(self) -> None:
        try:
            state = self.scheduler.drain()
        except Exception as exc:  # reversible cordon; the power cut is not optional
            self._record("SLURM", "SCHEDULER_DRAIN", "FAILED", f"{type(exc).__name__}: {exc}")
            return
        self._record("SLURM", "SCHEDULER_DRAIN", "DRAINED", f"State={state} (mock)")

    def run(self, reason: str) -> EmergencyResult:
        self.log = ActionLog(self.clock)
        try:
            valve = self.plc.close_valve()
            if valve.status is ValveStatus.VERIFIED_SLOW:
                valve_status = "LIMIT_SWITCH_CLOSED_SLOW"
            elif valve.status is ValveStatus.FAULT:
                valve_status = "FAULT"
            else:
                valve_status = "LIMIT_SWITCH_CLOSED"
            self._record(
                "PLC",
                "VALVE_CLOSE",
                valve_status,
                valve.detail or f"limit switch; band {valve.status.value}",
            )
            if valve.status is not ValveStatus.FAULT:
                # Scenario time, not measured wall time: the incident stays reproducible.
                self.clock.advance((valve.nominal_ms or valve.travel_ms) / 1000.0)

            self._cordon()
            power = self._power_cut(reason)

            if not power.ok:
                return EmergencyResult(IncidentState.FAILED_SAFE, self.log.snapshot())
            if valve.status is ValveStatus.FAULT:
                return EmergencyResult(IncidentState.HARD_TRIP, self.log.snapshot())
            return EmergencyResult(IncidentState.EMERGENCY_TRIP, self.log.snapshot())
        finally:
            self.log.seal()
