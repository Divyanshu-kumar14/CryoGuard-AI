"""Adaptive tick policy (D-18): 2 Hz idle -> 5 Hz tracking -> 10 Hz suspect.

The policy is a pure function of the last label and whether a candidate exists, so
two replays of a clip choose the same cadence and the benchmark stays identical.
"""

from __future__ import annotations

from typing import Any

from cryoguard.models import SystemLabel

# Labels that must be tracked at the highest rate. CAMERA_DEGRADED is included
# so a recovering camera is noticed quickly, not because a blind frame is urgent.
SEVERE = frozenset(
    {
        SystemLabel.SUSPECT_LEAK,
        SystemLabel.UNVERIFIED_ANOMALY,
        SystemLabel.CONFIRMED_LEAK,
        SystemLabel.EMERGENCY_TRIP,
        SystemLabel.HIDDEN_HYDRAULIC_EVENT,
        SystemLabel.CAMERA_DEGRADED,
    }
)


class SamplingPolicy:
    def __init__(self, cfg: dict[str, Any]) -> None:
        self.idle_hz = float(cfg["idle_hz"])
        self.track_hz = float(cfg["track_hz"])
        self.suspect_hz = float(cfg["suspect_hz"])
        if not 0 < self.idle_hz <= self.track_hz <= self.suspect_hz:
            raise ValueError(
                f"tick rates must satisfy 0 < idle <= track <= suspect, got "
                f"{self.idle_hz}/{self.track_hz}/{self.suspect_hz}"
            )
        self.hz = self.idle_hz
        self.changes = 0

    def update(self, label: SystemLabel | None, has_candidate: bool) -> float:
        if label in SEVERE:
            target = self.suspect_hz
        elif has_candidate:
            target = self.track_hz
        else:
            target = self.idle_hz

        if target != self.hz:
            self.hz = target
            self.changes += 1
        return self.hz

    def period_s(self) -> float:
        return 1.0 / self.hz
