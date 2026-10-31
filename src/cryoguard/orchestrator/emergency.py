"""R1 emergency sequence: isolate and cut power, no checkpoint and no requeue.

This is the path the thermal watchdog and the R1 fusion label execute. It is
deliberately shorter than the confirmed-leak sequence: there is nothing to save,
because the trigger is a hardware condition rather than a measured leak with a
usable checkpoint. It never claims an isolation it did not verify.

Order: valve close (spring-return), scheduler cordon (reversible), power cut.
A valve ``FAULT``, a ``SENSOR_DISAGREEMENT`` (witnesses contradict or a witness
is unavailable), or a ``VALVE_SEAL_LEAK`` (stem closed, line still pressurized)
still cuts power and ends ``HARD_TRIP``; if the cut also fails the result is
``FAILED_SAFE``, never an optimistic ``EMERGENCY_TRIP``.
"""

from __future__ import annotations

from dataclasses import dataclass

from cryoguard.models import IncidentState, ValveStatus
from cryoguard.orchestrator.actions import ActionLog, Clock, SequenceStep, StepCallback
from cryoguard.orchestrator.incident_fsm import (
    BmcPort,
    PlcPort,
    PowerResult,
    SchedulerPort,
    power_verified,
)


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
        on_step: StepCallback | None = None,
    ) -> None:
        self.clock = clock
        self.plc = plc
        self.bmc = bmc
        self.scheduler = scheduler
        self.on_step = on_step
        self.log = ActionLog(clock, on_step)

    def _record(self, actor: str, command: str, status: str, detail: str = "") -> SequenceStep:
        return self.log.append(actor, command, status, detail)

    def _power_cut(self, reason: str) -> PowerResult:
        power = self.bmc.force_off()
        source = "SIMULATED" if power.simulated else "live"
        detail = f"{reason}; PowerState={power.state} ({source} BMC)"
        if power.detail:
            detail = f"{detail}; {power.detail}"
        status = power.status or ("ACCEPTED" if power.ok else "REJECTED")
        self._record("BMC", "FORCE_OFF", status, detail)
        return power

    def _cordon(self) -> None:
        try:
            state = self.scheduler.drain()
        except Exception as exc:  # reversible cordon; the power cut is not optional
            self._record("SLURM", "SCHEDULER_DRAIN", "FAILED", f"{type(exc).__name__}: {exc}")
            return
        self._record("SLURM", "SCHEDULER_DRAIN", "DRAINED", f"State={state} (mock)")

    def run(self, reason: str) -> EmergencyResult:
        self.log = ActionLog(self.clock, self.on_step)
        try:
            valve = self.plc.close_valve()
            if valve.status is ValveStatus.VERIFIED_SLOW:
                valve_status = "LIMIT_SWITCH_CLOSED_SLOW"
            elif valve.status is ValveStatus.FAULT:
                valve_status = "FAULT"
            elif valve.status is ValveStatus.VALVE_SEAL_LEAK:
                valve_status = "VALVE_SEAL_LEAK"
            elif valve.status is ValveStatus.SENSOR_DISAGREEMENT:
                valve_status = "SENSOR_DISAGREEMENT"
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
            if valve.status in (ValveStatus.VERIFIED, ValveStatus.VERIFIED_SLOW):
                # Scenario time, not measured wall time: the incident stays reproducible.
                self.clock.advance((valve.nominal_ms or valve.travel_ms) / 1000.0)

            self._cordon()
            power = self._power_cut(reason)

            if not power_verified(power):
                return EmergencyResult(IncidentState.FAILED_SAFE, self.log.snapshot())
            if valve.status not in (ValveStatus.VERIFIED, ValveStatus.VERIFIED_SLOW):
                return EmergencyResult(IncidentState.HARD_TRIP, self.log.snapshot())
            return EmergencyResult(IncidentState.EMERGENCY_TRIP, self.log.snapshot())
        finally:
            self.log.seal()
