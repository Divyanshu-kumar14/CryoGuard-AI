"""Assign a candidate to one risk zone (PRD §6.1).

Two rules from the PRD are encoded structurally, not documented and hoped for:

* a candidate belongs to a zone only if at least 5 % of its area lies inside it;
* when it overlaps several zones, the highest zone index wins.

Filling polygons in ascending id order achieves the second rule for free: a later
fill overwrites an earlier one, so the label image already stores the maximum.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


class ZoneError(ValueError):
    """Raised when the topology is unusable."""


@dataclass(frozen=True)
class ZoneMap:
    labels: np.ndarray
    px_per_mm: float
    zone_ids: tuple[int, ...]

    @classmethod
    def build(cls, topology: dict[str, Any], px_per_mm: float) -> ZoneMap:
        if px_per_mm <= 0:
            raise ZoneError(f"px_per_mm must be positive: {px_per_mm}")

        chassis = topology["chassis"]
        width = int(round(float(chassis["width_mm"]) * px_per_mm))
        height = int(round(float(chassis["height_mm"]) * px_per_mm))
        labels = np.full((height, width), -1, dtype=np.int32)

        zones = sorted(topology["zones"], key=lambda zone: int(zone["id"]))
        for zone in zones:
            points = np.asarray(zone["polygon_mm"], dtype=np.float64) * px_per_mm
            polygon = np.round(points).astype(np.int32)
            cv2.fillPoly(labels, [polygon], int(zone["id"]))

        return cls(labels=labels, px_per_mm=px_per_mm, zone_ids=tuple(int(z["id"]) for z in zones))

    def zone_at(self, x_mm: float, y_mm: float) -> int | None:
        x = int(round(x_mm * self.px_per_mm))
        y = int(round(y_mm * self.px_per_mm))
        if not (0 <= y < self.labels.shape[0] and 0 <= x < self.labels.shape[1]):
            return None
        value = int(self.labels[y, x])
        return None if value < 0 else value

    def zone_for_mask(self, mask: np.ndarray, min_fraction: float = 0.05) -> int | None:
        """Zone containing >= ``min_fraction`` of the mask; ties go to the higher id."""

        if mask.shape != self.labels.shape:
            raise ZoneError(f"mask {mask.shape} does not match zone map {self.labels.shape}")

        selected = mask.astype(bool)
        total = int(selected.sum())
        if total == 0:
            return None

        return self._best_zone(self.labels[selected], total, min_fraction)

    def zone_for_cropped(
        self, mask: np.ndarray, offset_px: tuple[int, int], min_fraction: float = 0.05
    ) -> int | None:
        """Zone for a candidate mask stored cropped at ``offset_px`` (x, y).

        Tracks keep cropped masks so a 1200 x 800 frame is not copied per
        candidate per tick. The offset is part of the track, so the lookup stays
        exact without that allocation.
        """

        x0, y0 = (int(v) for v in offset_px)
        height, width = mask.shape
        if (
            x0 < 0
            or y0 < 0
            or y0 + height > self.labels.shape[0]
            or x0 + width > self.labels.shape[1]
        ):
            raise ZoneError(f"candidate at {offset_px} with shape {mask.shape} leaves the zone map")

        selected = mask.astype(bool)
        total = int(selected.sum())
        if total == 0:
            return None
        covered = self.labels[y0 : y0 + height, x0 : x0 + width][selected]
        return self._best_zone(covered, total, min_fraction)

    def _best_zone(self, covered: np.ndarray, total: int, min_fraction: float) -> int | None:
        best: int | None = None
        for zone_id in self.zone_ids:  # ascending, so the last qualifying id wins
            fraction = float(np.count_nonzero(covered == zone_id)) / total
            if fraction >= min_fraction:
                best = zone_id
        return best
