"""Candidate lifecycle (P3-T04).

``FastTracker`` owns association and kinematics; ``WetnessTracker`` owns the
path-to-fitting test; this manager is the single place that turns a track into
the evidence-shaped object the fusion loop consumes. Keeping that composition in
one class is what stops the pipeline from re-deriving q, W and zone slightly
differently in two code paths.

W is advanced once per classical tick. A SAM re-anchor in the same tick must not
increment the streak a second time, so ``current()`` reads the stored value.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from cryoguard.capture.rectify import Rectification, warp_to_canvas
from cryoguard.capture.zones import ZoneMap
from cryoguard.perception.fast_tracker import FastTracker, Track
from cryoguard.perception.quality import mask_quality
from cryoguard.perception.wetness import WetnessTracker


@dataclass(frozen=True)
class Candidate:
    track_id: int
    centroid_mm: tuple[float, float]
    area_mm2: float
    growth_mm2_per_s: float
    mean_intensity: float
    intensity_var: float
    solidity: float
    clipped_fraction: float
    zone: int | None
    wetness: float
    quality: float
    mask: np.ndarray
    offset_px: tuple[int, int]
    age_ticks: int
    stale: bool
    sam_verified: bool


class CandidateManager:
    def __init__(
        self,
        perception_cfg: dict[str, Any],
        topology: dict[str, Any],
        px_per_mm: float,
        canvas_px: tuple[int, int],
        anchor_ttl_s: float = 1.2,
    ) -> None:
        self.cfg = perception_cfg
        self.px_per_mm = px_per_mm
        self.canvas_px = canvas_px
        self.anchor_ttl_s = float(anchor_ttl_s)
        self.zones = ZoneMap.build(topology, px_per_mm)
        self.tracker = FastTracker(perception_cfg)
        self.wetness = WetnessTracker(topology, px_per_mm, perception_cfg, self.zones)

    @property
    def last_warped(self) -> np.ndarray | None:
        return self.tracker.last_warped

    def update(
        self, image_bgr: np.ndarray, t_s: float, rect: Rectification
    ) -> tuple[Candidate, ...]:
        warped = warp_to_canvas(image_bgr, rect, self.canvas_px, self.px_per_mm)
        return self.update_warped(warped, t_s)

    def update_warped(self, warped_bgr: np.ndarray, t_s: float) -> tuple[Candidate, ...]:
        self.tracker.update_warped(warped_bgr, t_s, self.px_per_mm)
        return self._build(t_s, advance_wetness=True)

    def reanchor(self, mask_full: np.ndarray, t_s: float) -> None:
        if self.tracker.last_warped is None:
            raise RuntimeError("reanchor requires a classical tick first")
        self.tracker.reanchor(mask_full, self.tracker.last_warped, t_s, self.px_per_mm)

    def current(self, t_s: float) -> tuple[Candidate, ...]:
        return self._build(t_s, advance_wetness=False)

    def _build(self, t_s: float, *, advance_wetness: bool) -> tuple[Candidate, ...]:
        tracks = self.tracker.tracks
        self.wetness.prune({track.track_id for track in tracks})

        candidates = [
            self._candidate(track, t_s, advance_wetness=advance_wetness) for track in tracks
        ]
        # Stable ordering for selection: largest area first, then lower id.
        candidates.sort(key=lambda candidate: (-candidate.area_mm2, candidate.track_id))
        return tuple(candidates)

    def _candidate(self, track: Track, t_s: float, *, advance_wetness: bool) -> Candidate:
        if advance_wetness:
            wetness = 1.0 if self.wetness.update(track, self.tracker.last_mask) else 0.0
        else:
            wetness = self.wetness.value(track.track_id)

        stale = track.misses > 0
        sam_recent = track.sam_t_s is not None and (t_s - track.sam_t_s) <= self.anchor_ttl_s
        quality = mask_quality(track, self.cfg)
        if stale and not sam_recent:
            # A missed track may keep its identity and its mask for the TTL, but
            # an unverified old mask must never satisfy R1/R2's quality gate.
            quality = 0.0

        try:
            zone = self.zones.zone_for_cropped(track.mask, track.offset_px)
        except Exception as exc:  # a track outside the zone map is a bug, not a zone
            raise RuntimeError(f"candidate {track.track_id} left the zone map: {exc}") from exc

        return Candidate(
            track_id=track.track_id,
            centroid_mm=track.centroid_mm,
            area_mm2=track.area_mm2,
            growth_mm2_per_s=track.dA_dt_mm2_per_s,
            mean_intensity=track.mean_intensity,
            intensity_var=track.intensity_var,
            solidity=track.solidity,
            clipped_fraction=track.clipped_fraction,
            zone=zone,
            wetness=wetness,
            quality=quality,
            mask=track.mask,
            offset_px=track.offset_px,
            age_ticks=track.age_ticks,
            stale=stale,
            sam_verified=sam_recent,
        )

    @staticmethod
    def select(candidates: tuple[Candidate, ...]) -> Candidate | None:
        return candidates[0] if candidates else None
