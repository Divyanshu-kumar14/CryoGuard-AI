"""Scripted telemetry replay, synchronized to video time (FR-TEL-01/03).

Values are linear interpolations of the scenario's hand-authored channel tables.
Nothing is read from the wall clock, so a replay at the same video time returns
bit-identical values.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from cryoguard.telemetry.psychrometrics import dew_margin_c, dew_point_c


class TelemetryError(ValueError):
    """Raised when a scenario's channel table is malformed."""


@dataclass(frozen=True)
class TelemetrySample:
    t_s: float
    pressure_psi: float
    flow_in_lpm: float
    flow_out_lpm: float
    surface_c: float
    air_c: float
    rh_pct: float
    pressure_rate_psi_per_s: float  # negative dP/dt, positive on a drop
    flow_imbalance_lpm: float
    dew_point_c: float
    dew_margin_c: float


class TelemetryReplay:
    def __init__(self, scenario: dict[str, Any], pressure_window_s: float = 1.0) -> None:
        if pressure_window_s <= 0:
            raise TelemetryError(f"pressure window must be positive: {pressure_window_s}")
        self._window = pressure_window_s
        channels = scenario["channels"]

        self._pressure = self._table(channels["pressure"], 2, "pressure")
        self._flow = self._table(channels["flow"], 3, "flow")
        self._surface = self._table(channels["surface_temp"], 2, "surface_temp")
        self._ambient = self._table(channels["ambient"], 3, "ambient")

    @staticmethod
    def _table(rows: list[list[float]], width: int, name: str) -> np.ndarray:
        table = np.asarray(rows, dtype=np.float64)
        if table.ndim != 2 or table.shape[1] != width:
            raise TelemetryError(f"{name} must be Nx{width}, got {table.shape}")
        if len(table) < 2:
            raise TelemetryError(f"{name} needs at least two keyframes")
        if np.any(np.diff(table[:, 0]) <= 0):
            raise TelemetryError(f"{name} timestamps must be strictly increasing")
        return table

    @staticmethod
    def _at(table: np.ndarray, t_s: float) -> np.ndarray:
        times = table[:, 0]
        return np.array(
            [np.interp(t_s, times, table[:, column]) for column in range(1, table.shape[1])]
        )

    def pressure_at(self, t_s: float) -> float:
        return float(self._at(self._pressure, max(0.0, t_s))[0])

    def sample(self, t_s: float) -> TelemetrySample:
        t = max(0.0, float(t_s))
        pressure = self.pressure_at(t)
        earlier = self.pressure_at(max(0.0, t - self._window))
        span = min(self._window, t)
        rate = 0.0 if span <= 0.0 else (earlier - pressure) / span

        flow_in, flow_out = (float(v) for v in self._at(self._flow, t))
        surface = float(self._at(self._surface, t)[0])
        air, rh = (float(v) for v in self._at(self._ambient, t))

        dew = dew_point_c(air, rh)
        return TelemetrySample(
            t_s=t,
            pressure_psi=pressure,
            flow_in_lpm=flow_in,
            flow_out_lpm=flow_out,
            surface_c=surface,
            air_c=air,
            rh_pct=rh,
            pressure_rate_psi_per_s=rate,
            flow_imbalance_lpm=flow_in - flow_out,
            dew_point_c=dew,
            dew_margin_c=surface - dew,
        )

    def iter_series(self, duration_s: float, dt_s: float) -> list[TelemetrySample]:
        """Fixed-grid samples on the video clock, never the wall clock (P4-T02)."""

        if duration_s < 0:
            raise TelemetryError(f"duration must be non-negative: {duration_s}")
        if dt_s <= 0:
            raise TelemetryError(f"dt must be positive: {dt_s}")

        count = int(round(duration_s / dt_s))
        return [self.sample(index * dt_s) for index in range(count + 1)]

    def series_hash(self, duration_s: float, dt_s: float) -> str:
        """Stable hash of a replay series; two runs must match exactly."""

        payload = []
        for sample in self.iter_series(duration_s, dt_s):
            record = {
                key: (round(value, 9) if isinstance(value, float) else value)
                for key, value in asdict(sample).items()
            }
            payload.append(record)
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# Re-exported so callers do not need both modules for the common case.
__all__ = ["TelemetryError", "TelemetryReplay", "TelemetrySample", "dew_margin_c", "dew_point_c"]
