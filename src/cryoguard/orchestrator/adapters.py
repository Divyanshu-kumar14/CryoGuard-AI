"""In-memory adapters for unit tests and the ``memory`` protocol mode.

Every class here is SIMULATED. The live clients in ``cryoguard.adapters`` talk to
the simulators over real Modbus-TCP and Redfish sockets; these do not. Neither may
be presented as hardware.
"""

from __future__ import annotations

from dataclasses import dataclass

from cryoguard.models import ValveStatus
from cryoguard.orchestrator.incident_fsm import (
    CheckpointResult,
    ClampResult,
    PowerResult,
    ValveResult,
)
from cryoguard.sims.slurm_mock import SlurmMock


class ScenarioClock:
    """Monotonic clock driven by scenario time; ``advance`` is instant."""

    def __init__(self, start_s: float = 0.0) -> None:
        self._t = float(start_s)

    def now(self) -> float:
        return self._t

    def advance(self, seconds: float) -> float:
        self._t += max(0.0, float(seconds))
        return self._t


@dataclass(frozen=True)
class SimulatedPorts:
    clamp_w: float = 200.0
    valve_travel_ms: float = 950.0
    checkpoint_s: float = 30.0
    checkpoint_step: int = 4200


class SimulatedNvml:
    def __init__(self, clamp_w: float = 200.0, *, fail: bool = False) -> None:
        self.clamp_w = clamp_w
        self.fail = fail
        self.calls = 0

    def clamp(self) -> ClampResult:
        self.calls += 1
        if self.fail:
            return ClampResult(ok=False, applied_w=0.0, detail="injected readback failure")
        return ClampResult(ok=True, applied_w=self.clamp_w, detail=f"{self.clamp_w:.0f} W")


class SimulatedPlc:
    def __init__(
        self,
        travel_ms: float = 950.0,
        *,
        status: ValveStatus = ValveStatus.VERIFIED,
    ) -> None:
        self.travel_ms = travel_ms
        self.status = status
        self.calls = 0

    def close_valve(self) -> ValveResult:
        self.calls += 1
        return ValveResult(
            status=self.status,
            travel_ms=self.travel_ms,
            detail=f"travel {self.travel_ms:.0f} ms",
            nominal_ms=self.travel_ms,
        )


class SimulatedCheckpoint:
    def __init__(
        self,
        clock: ScenarioClock,
        settle_s: float = 30.0,
        step: int = 4200,
        reason: str = "simulated checkpoint",
    ) -> None:
        self.clock = clock
        self.settle_s = settle_s
        self.step = step
        self.reason = reason
        self.calls = 0

    def start(self) -> CheckpointResult:
        self.calls += 1
        started = self.clock.now()
        return CheckpointResult(
            started_at=started,
            ready_at=started + self.settle_s,
            step=self.step,
            reason=self.reason,
        )


class SimulatedBmc:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls = 0

    def force_off(self) -> PowerResult:
        self.calls += 1
        if self.fail:
            return PowerResult(
                ok=False,
                state="On",
                attempts=1,
                status_code=503,
                simulated=True,
                detail="injected BMC failure",
            )
        return PowerResult(
            ok=True,
            state="Off",
            attempts=1,
            status_code=202,
            simulated=True,
            detail="in-memory BMC",
        )


class SimulatedScheduler:
    def __init__(self, cfg: dict) -> None:
        self.slurm = SlurmMock(cfg)

    def drain(self) -> str:
        self.slurm.drain()
        return self.slurm.state.value

    def requeue(self, checkpoint_step: int) -> str:
        self.slurm.requeue()
        return self.slurm.node
