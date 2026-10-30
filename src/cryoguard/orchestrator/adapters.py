"""In-memory adapters for the vertical slice.

Every class here is SIMULATED. They exist so the safety sequence can be exercised
end to end before the Modbus, Redfish and NVML adapters land; none of them talks
to hardware and none may be presented as if it did.
"""

from __future__ import annotations

from dataclasses import dataclass

from cryoguard.models import ValveStatus
from cryoguard.orchestrator.incident_fsm import (
    CheckpointResult,
    ClampResult,
    ValveResult,
)


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
    def __init__(self, clamp_w: float = 200.0) -> None:
        self.clamp_w = clamp_w
        self.calls = 0

    def clamp(self) -> ClampResult:
        self.calls += 1
        return ClampResult(ok=True, applied_w=self.clamp_w)


class SimulatedPlc:
    def __init__(self, travel_ms: float = 950.0) -> None:
        self.travel_ms = travel_ms
        self.calls = 0

    def close_valve(self) -> ValveResult:
        self.calls += 1
        return ValveResult(status=ValveStatus.VERIFIED, travel_ms=self.travel_ms)


class SimulatedCheckpoint:
    def __init__(self, clock: ScenarioClock, settle_s: float = 30.0, step: int = 4200) -> None:
        self.clock = clock
        self.settle_s = settle_s
        self.step = step
        self.calls = 0

    def start(self) -> CheckpointResult:
        self.calls += 1
        started = self.clock.now()
        return CheckpointResult(
            started_at=started, ready_at=started + self.settle_s, step=self.step
        )
