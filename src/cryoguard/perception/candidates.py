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

import math
from collections.abc import Mapping, Sequence
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
    # Observed spill centroid in chassis mm at read time. This is where the
    # fluorescent mass is, not where it leaked from: the spill drifts
    # downstream (~11 mm/s observed), so the centroid must never be reported
    # as the leak source.
    centroid_mm: tuple[float, float]
    # Bounding-box geometry of the same observation; the bbox centre is a
    # distinct spill quantity (pure geometry vs the mask centroid).
    bbox_mm: tuple[float, float, float, float]
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

    @property
    def bbox_centre_mm(self) -> tuple[float, float]:
        x0, y0, x1, y1 = self.bbox_mm
        return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)


# Tolerance for the source-error flag. Mirrors the bench fitting radius
# (bench/runner.py FITTING_RADIUS_MM); kept as a named constant here so the
# estimator does not import the bench layer.
SOURCE_TOLERANCE_MM = 25.0
SOURCE_METHOD = "earliest-highq-backproject-v1"
# Reliability floor: a single gated blob could be a splash or a glint remnant
# with no drift baseline, so at least two high-quality ticks spanning one
# classical interval are required before any source claim.
SOURCE_MIN_HIGHQ_SAMPLES = 2
SOURCE_MIN_SPAN_S = 0.2
# Early-motion window for the velocity fit: the first seconds best approximate
# the initial spread direction before the wet trail dominates the mask.
SOURCE_VELOCITY_WINDOW_S = 2.0
SOURCE_VELOCITY_SAMPLES = 5


@dataclass(frozen=True)
class SourceEstimate:
    """Estimated leak-source position with honest uncertainty.

    Separate from the observed spill centroid by construction: ``x_mm``/``y_mm``
    is the back-projected origin guess, while ``centroid_mm`` and
    ``bbox_centre_mm`` record the spill observation it was derived from.

    Confidence semantics (heuristic, NOT a calibrated probability):
    0 is never returned (unreliable inputs yield None instead); ~0.3 is a
    barely-gated short track far from any fitting without corroboration;
    ~0.6 is a solid early track; >0.8 is an early high-quality track adjacent
    to the configured fitting with independent hydraulic corroboration.
    It is ``q_anchor * span_factor`` plus 0.1 each for fitting proximity and
    hydraulic corroboration, clamped to [0, 1].
    """

    x_mm: float
    y_mm: float
    t_s: float
    confidence: float
    method: str
    quality: float
    track_id: int
    centroid_mm: tuple[float, float]
    bbox_centre_mm: tuple[float, float]
    velocity_mm_per_s: tuple[float, float] | None = None
    # Configured fitting location (detector config: QD ROI centre). Used only
    # for the proximity flag; never treated as ground truth.
    fitting_mm: tuple[float, float] | None = None
    fitting_distance_mm: float | None = None
    near_fitting: bool | None = None
    hydraulic_corroborated: bool | None = None
    # Independent ground truth from the scenario manifest (leak_source_mm).
    # None when the manifest does not declare it; error fields are None then.
    ground_truth_mm: tuple[float, float] | None = None
    error_mm: float | None = None
    within_tolerance: bool | None = None
    tolerance_mm: float = SOURCE_TOLERANCE_MM


def ground_truth_source_mm(scenario: Mapping[str, Any]) -> tuple[float, float] | None:
    """Independent ground-truth source from the scenario manifest.

    Reads ``leak_source_mm`` (top level, or under ``expected`` for older
    manifests). Returns None when absent or malformed — never falls back to
    the detector's QD ROI, which would conflate the instrument with the truth.
    """

    raw: Any = scenario.get("leak_source_mm")
    if raw is None and isinstance(scenario.get("expected"), Mapping):
        raw = scenario["expected"].get("leak_source_mm")
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        return None
    try:
        x, y = float(raw[0]), float(raw[1])
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(x) and math.isfinite(y)):
        return None
    return (x, y)


def fitting_centre_mm(topology: Mapping[str, Any], qd_id: str) -> tuple[float, float] | None:
    """Configured fitting location: centre of the QD ROI in chassis mm.

    This is detector configuration from the topology file, used only for the
    proximity flag. It is explicitly NOT ground truth.
    """

    qds = topology.get("qd", [])
    if not isinstance(qds, list):
        return None
    for qd in qds:
        if isinstance(qd, Mapping) and str(qd.get("id")) == qd_id:
            roi = qd.get("roi_mm")
            if not isinstance(roi, (list, tuple)) or len(roi) != 4:
                return None
            try:
                x0, y0, x1, y1 = (float(v) for v in roi)
            except (TypeError, ValueError):
                return None
            if not all(math.isfinite(v) for v in (x0, y0, x1, y1)):
                return None
            return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
    return None


def estimate_leak_source(
    observations: Sequence[tuple[float, tuple[float, float], tuple[float, float], float]],
    perception_cfg: Mapping[str, Any],
    *,
    track_id: int = 0,
    fitting_mm: tuple[float, float] | None = None,
    ground_truth_mm: tuple[float, float] | None = None,
    tolerance_mm: float = SOURCE_TOLERANCE_MM,
    hydraulic_corroborated: bool | None = None,
) -> SourceEstimate | None:
    """Back-project the earliest reliable spill observation toward the source.

    Assumption: the earliest high-quality detection is the closest reliable
    glimpse of the spill to the leak onset; later centroids drift downstream
    and must not be reported as the origin. The initial velocity (least
    squares over the early high-quality prefix) corrects only the observed
    detection-delay drift between track birth and the first gated tick,
    capped at one link radius so noisy velocity cannot fling the estimate.

    Limits: assumes early motion is roughly linear and the earliest gated
    blob is source-adjacent. Splashes and secondaries violate this, which is
    why fitting proximity and hydraulic corroboration are reported as flags
    rather than trusted blindly. The first centroid is never taken as the
    origin without the quality gate (q_min) plus the time window
    (MIN_HIGHQ_SAMPLES over MIN_SPAN_S); anything weaker returns None.
    Bias is bounded by |v| * detection delay; the velocity is reported so
    callers can judge the drift themselves.
    """

    try:
        q_min = float(perception_cfg["q_min"])
        link_radius = float(perception_cfg.get("link_radius_mm", SOURCE_TOLERANCE_MM))
    except (KeyError, TypeError, ValueError):
        return None
    if not math.isfinite(q_min) or not math.isfinite(link_radius) or link_radius <= 0:
        return None

    ordered = sorted(observations, key=lambda obs: obs[0])
    if not ordered:
        return None
    t_birth = ordered[0][0]
    if not math.isfinite(t_birth):
        return None

    highq = [
        obs
        for obs in ordered
        if math.isfinite(obs[0])
        and math.isfinite(obs[1][0])
        and math.isfinite(obs[1][1])
        and math.isfinite(obs[2][0])
        and math.isfinite(obs[2][1])
        and math.isfinite(obs[3])
        and obs[3] >= q_min
    ]
    if len(highq) < SOURCE_MIN_HIGHQ_SAMPLES:
        return None
    span_s = highq[-1][0] - highq[0][0]
    if not math.isfinite(span_s) or span_s < SOURCE_MIN_SPAN_S:
        return None

    t0, c0, b0, q0 = highq[0]
    prefix = [obs for obs in highq if obs[0] - t0 <= SOURCE_VELOCITY_WINDOW_S][
        :SOURCE_VELOCITY_SAMPLES
    ]
    if len(prefix) < 2:
        prefix = highq[:SOURCE_VELOCITY_SAMPLES]
    times = np.array([o[0] - t0 for o in prefix], dtype=np.float64)
    xs = np.array([o[1][0] for o in prefix], dtype=np.float64)
    ys = np.array([o[1][1] for o in prefix], dtype=np.float64)
    if times[-1] <= 0.0 or not np.all(np.isfinite(times)):
        velocity: tuple[float, float] = (0.0, 0.0)
    else:
        design = np.vstack([times, np.ones_like(times)]).T
        vx, _ = np.linalg.lstsq(design, xs, rcond=None)[0]
        vy, _ = np.linalg.lstsq(design, ys, rcond=None)[0]
        velocity = (
            float(vx) if math.isfinite(vx) else 0.0,
            float(vy) if math.isfinite(vy) else 0.0,
        )

    delay = max(0.0, t0 - t_birth)
    shift_x, shift_y = velocity[0] * delay, velocity[1] * delay
    shift_norm = math.hypot(shift_x, shift_y)
    if shift_norm > link_radius:
        scale = link_radius / shift_norm
        shift_x, shift_y = shift_x * scale, shift_y * scale
    src_x, src_y = c0[0] - shift_x, c0[1] - shift_y
    if not (math.isfinite(src_x) and math.isfinite(src_y)):
        return None

    fitting_distance: float | None = None
    near_fitting: bool | None = None
    if fitting_mm is not None:
        fitting_distance = math.hypot(src_x - fitting_mm[0], src_y - fitting_mm[1])
        near_fitting = fitting_distance <= tolerance_mm

    error: float | None = None
    within_tolerance: bool | None = None
    if ground_truth_mm is not None:
        error = math.hypot(src_x - ground_truth_mm[0], src_y - ground_truth_mm[1])
        within_tolerance = error <= tolerance_mm

    span_factor = min(1.0, max(0.0, span_s) / SOURCE_VELOCITY_WINDOW_S)
    confidence = q0 * span_factor
    if near_fitting:
        confidence += 0.1
    if hydraulic_corroborated:
        confidence += 0.1
    confidence = max(0.0, min(1.0, confidence))

    return SourceEstimate(
        x_mm=src_x,
        y_mm=src_y,
        t_s=t0,
        confidence=confidence,
        method=SOURCE_METHOD,
        quality=q0,
        track_id=track_id,
        centroid_mm=(c0[0], c0[1]),
        bbox_centre_mm=(b0[0], b0[1]),
        velocity_mm_per_s=velocity,
        fitting_mm=fitting_mm,
        fitting_distance_mm=fitting_distance,
        near_fitting=near_fitting,
        hydraulic_corroborated=hydraulic_corroborated,
        ground_truth_mm=ground_truth_mm,
        error_mm=error,
        within_tolerance=within_tolerance,
        tolerance_mm=tolerance_mm,
    )


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
        # Per-track observation log for the source estimator:
        # track_id -> [(t_s, centroid_mm, bbox_centre_mm, quality)] in tick order.
        # Logged once per classical tick (advance_wetness=True) so a second
        # read in the same tick never double-counts; pruned with the wetness
        # streaks. Deterministic: _build iterates tracks sorted by id.
        self._source_obs: dict[
            int, list[tuple[float, tuple[float, float], tuple[float, float], float]]
        ] = {}

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
        for track_id in list(self._source_obs):
            if track_id not in {track.track_id for track in tracks}:
                del self._source_obs[track_id]

        candidates = [
            self._candidate(track, t_s, advance_wetness=advance_wetness) for track in tracks
        ]
        if advance_wetness:
            for candidate in candidates:
                self._source_obs.setdefault(candidate.track_id, []).append(
                    (
                        t_s,
                        candidate.centroid_mm,
                        candidate.bbox_centre_mm,
                        candidate.quality,
                    )
                )
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
            bbox_mm=track.bbox_mm,
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

    def estimate_source(
        self,
        track_id: int | None,
        *,
        fitting_mm: tuple[float, float] | None = None,
        ground_truth_mm: tuple[float, float] | None = None,
        tolerance_mm: float = SOURCE_TOLERANCE_MM,
        hydraulic_corroborated: bool | None = None,
    ) -> SourceEstimate | None:
        """Source estimate for one track from its gated observation history.

        Returns None for an unknown track or an unreliable (too short /
        too weak) history — never a guess. Ground truth, when supplied, is
        used only to score the error, never as an estimator input.
        """

        if track_id is None:
            return None
        observations = self._source_obs.get(track_id)
        if not observations:
            return None
        return estimate_leak_source(
            observations,
            self.cfg,
            track_id=track_id,
            fitting_mm=fitting_mm,
            ground_truth_mm=ground_truth_mm,
            tolerance_mm=tolerance_mm,
            hydraulic_corroborated=hydraulic_corroborated,
        )

    @staticmethod
    def select(candidates: tuple[Candidate, ...]) -> Candidate | None:
        return candidates[0] if candidates else None
