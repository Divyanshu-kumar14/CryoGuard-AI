"""Lumped thermal model (P4-T03, D-07).

``delta_T = P * t / C_th`` for a single thermal capacitance with no loss term.
That is the PRD's model, and the PRD's numbers follow from it exactly:

    40 K, 200 W, C_th = 350 J/K  ->  70 s
    40 K, 1000 W, C_th = 350 J/K ->  14 s

``C_th`` is an **assumption**, not a measurement. It lives in config and is
never described as calibrated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class ThermalError(ValueError):
    """A thermal parameter is outside its physical domain."""


def seconds_to_delta(delta_k: float, power_w: float, c_th_j_per_k: float) -> float:
    if delta_k < 0:
        raise ThermalError(f"delta must be non-negative: {delta_k}")
    if power_w <= 0:
        raise ThermalError(f"power must be positive: {power_w}")
    if c_th_j_per_k <= 0:
        raise ThermalError(f"thermal capacitance must be positive: {c_th_j_per_k}")
    return delta_k * c_th_j_per_k / power_w


@dataclass(frozen=True)
class ThermalSample:
    t_s: float
    temp_c: float
    watchdog_tripped: bool


class ThermalModel:
    def __init__(
        self,
        cfg: dict[str, Any],
        *,
        ambient_c: float = 25.0,
        initial_c: float | None = None,
    ) -> None:
        self.c_th = float(cfg["c_th_j_per_k"])
        self.watchdog_limit_c = float(cfg["watchdog_limit_c"])
        self.junction_max_c = float(cfg["t_junction_max_c"])
        if self.c_th <= 0:
            raise ThermalError(f"c_th must be positive: {self.c_th}")
        if not self.watchdog_limit_c < self.junction_max_c:
            raise ThermalError(
                f"watchdog {self.watchdog_limit_c} C must sit below the junction limit "
                f"{self.junction_max_c} C"
            )

        self.ambient_c = float(ambient_c)
        self.reset(initial_c)

    def reset(self, initial_c: float | None = None) -> None:
        self.t_s = 0.0
        self.temp_c = self.ambient_c if initial_c is None else float(initial_c)

    def step(self, dt_s: float, power_w: float) -> ThermalSample:
        if dt_s <= 0:
            raise ThermalError(f"dt must be positive: {dt_s}")
        if power_w < 0:
            raise ThermalError(f"power must be non-negative: {power_w}")

        self.temp_c += power_w * dt_s / self.c_th
        self.t_s += dt_s
        return ThermalSample(
            t_s=self.t_s,
            temp_c=self.temp_c,
            watchdog_tripped=self.temp_c >= self.watchdog_limit_c,
        )

    def time_to_watchdog_s(self, power_w: float) -> float:
        return seconds_to_delta(self.watchdog_limit_c - self.ambient_c, power_w, self.c_th)

    def run_to_watchdog(self, power_w: float, dt_s: float, max_s: float = 3600.0) -> ThermalSample:
        """Integrate until the watchdog trips; fail loudly if it never does."""

        while self.t_s < max_s:
            sample = self.step(dt_s, power_w)
            if sample.watchdog_tripped:
                return sample
        raise ThermalError(f"watchdog did not trip within {max_s} s at {power_w} W")
