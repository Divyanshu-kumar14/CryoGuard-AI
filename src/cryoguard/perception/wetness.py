"""Wetness continuity W (FR-FUS-05).

W is 1 only when the candidate's fluorescent component is physically connected to
the originating QD ROI, for at least three consecutive ticks. A droplet that
appears across the chassis without a path back to the fitting is not coolant that
leaked there — it is either a reflection or a splash, and either way it must not
inherit W = 1.

Connectivity is checked on the full fluorescence mask, not the cropped candidate,
so a thin wet trail is enough to connect the two.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from cryoguard.capture.zones import ZoneMap
from cryoguard.perception.fast_tracker import Track


class WetnessTracker:
    def __init__(
        self,
        topology: dict[str, Any],
        px_per_mm: float,
        perception_cfg: dict[str, Any],
        zone_map: ZoneMap,
    ) -> None:
        self._ticks_required = int(perception_cfg["wetness_ticks"])
        self._streak: dict[int, int] = {}
        self._qd_mask = self._build_qd_mask(topology, px_per_mm, zone_map)

    @staticmethod
    def _build_qd_mask(topology: dict[str, Any], px_per_mm: float, zone_map: ZoneMap) -> np.ndarray:
        mask = np.zeros(zone_map.labels.shape, dtype=np.uint8)
        for qd in topology["qd"]:
            x0, y0, x1, y1 = (int(round(value * px_per_mm)) for value in qd["roi_mm"])
            mask[y0:y1, x0:x1] = 1
        return mask.astype(bool)

    def reset(self) -> None:
        self._streak.clear()

    def prune(self, active_ids: set[int]) -> None:
        """Drop streaks for candidates that were not matched this tick."""

        for track_id in list(self._streak):
            if track_id not in active_ids:
                del self._streak[track_id]

    def update(self, track: Track, full_mask: np.ndarray | None) -> bool:
        if full_mask is None or full_mask.shape != self._qd_mask.shape:
            return False

        _, labels = cv2.connectedComponents(full_mask, connectivity=8)
        qd_labels = set(np.unique(labels[self._qd_mask])) - {0}

        x0, y0 = track.offset_px
        height, width = track.mask.shape
        candidate = labels[y0 : y0 + height, x0 : x0 + width][track.mask]
        candidate_labels = set(np.unique(candidate)) - {0}

        connected = bool(qd_labels & candidate_labels)
        streak = self._streak.get(track.track_id, 0) + 1 if connected else 0
        self._streak[track.track_id] = streak
        return streak >= self._ticks_required
