"""Fluorescence threshold tracker on the rectified chassis plane.

Deliberately simple: HSV gate, morphology, connected components, greedy
nearest-centroid association, and a least-squares area growth rate. No model, no
randomness, no wall clock. Determinism is a requirement, not a preference.

The gate is tuned for a long-pass filtered camera under UV excitation: coolant
tracer is the only saturated yellow-green emitter in the scene. Condensation and
specular glint are handled downstream by wetness continuity and temporal
persistence, not by pretending this threshold can tell them apart.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from cryoguard.capture.rectify import Rectification, warp_to_canvas


class TrackerError(ValueError):
    """Raised when the tracker is given an unusable frame or configuration."""


@dataclass(frozen=True)
class Detection:
    centroid_mm: tuple[float, float]
    area_mm2: float
    bbox_mm: tuple[float, float, float, float]
    mean_intensity: float
    intensity_var: float
    clipped_fraction: float
    compactness: float
    solidity: float
    mask: np.ndarray
    offset_px: tuple[int, int]


@dataclass
class Track:
    track_id: int
    centroid_mm: tuple[float, float]
    area_mm2: float
    dA_dt_mm2_per_s: float
    mean_intensity: float
    intensity_var: float
    clipped_fraction: float
    compactness: float
    solidity: float
    bbox_mm: tuple[float, float, float, float]
    mask: np.ndarray
    offset_px: tuple[int, int]
    first_t: float
    last_t: float
    hits: int = 1
    misses: int = 0
    sam_t_s: float | None = None
    _area_history: list[tuple[float, float]] = field(default_factory=list, repr=False)
    # Kinematic log of classical-tick centroids: (t_s, (x_mm, y_mm)).
    # Re-anchor refreshes geometry without appending (its t_s is the older
    # submission time, not a new observation), so this stays monotonic and
    # deterministic. The leak-source estimator consumes it to assess drift;
    # the centroid itself is the observed spill position, never the source.
    _centroid_history: list[tuple[float, tuple[float, float]]] = field(
        default_factory=list, repr=False
    )

    @property
    def age_ticks(self) -> int:
        return self.hits + self.misses

    @property
    def bbox_centre_mm(self) -> tuple[float, float]:
        """Centre of the bounding box, distinct from the mask centroid.

        The centroid is the intensity-connected-component centre (it slides
        downstream as the spill spreads); the bbox centre is pure geometry.
        Neither is the leak source — both are observed spill quantities the
        source estimator takes as input.
        """

        x0, y0, x1, y1 = self.bbox_mm
        return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)

    @property
    def velocity_mm_per_s(self) -> tuple[float, float]:
        """Least-squares drift velocity over the centroid history.

        Returns (0, 0) with fewer than two samples or a degenerate clock.
        Deterministic: closed-form least squares over time-ordered history.
        """

        if len(self._centroid_history) < 2:
            return (0.0, 0.0)
        times = np.array([t for t, _ in self._centroid_history], dtype=np.float64)
        xs = np.array([c[0] for _, c in self._centroid_history], dtype=np.float64)
        ys = np.array([c[1] for _, c in self._centroid_history], dtype=np.float64)
        times = times - times[0]
        if times[-1] <= 0.0 or not np.all(np.isfinite(times)):
            return (0.0, 0.0)
        design = np.vstack([times, np.ones_like(times)]).T
        vx, _ = np.linalg.lstsq(design, xs, rcond=None)[0]
        vy, _ = np.linalg.lstsq(design, ys, rcond=None)[0]
        if not (np.isfinite(vx) and np.isfinite(vy)):
            return (0.0, 0.0)
        return (float(vx), float(vy))

    def __str__(self) -> str:  # pragma: no cover - debugging aid
        x, y = self.centroid_mm
        return f"T{self.track_id:02d} ({x:.1f},{y:.1f}) {self.area_mm2:.0f}mm2"


def _solidity(component: np.ndarray) -> float:
    """Area over convex-hull area: 1 for a filled blob, low for a spidery mask.

    Compactness (4*pi*A/P^2) punishes elongation, which every leak with a wet
    trail has. Solidity measures the property the quality gate actually cares
    about: whether the mask is a coherent region rather than disconnected strands.
    """

    contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    area = float(cv2.contourArea(contours[0]))
    hull = float(cv2.contourArea(cv2.convexHull(contours[0])))
    if hull <= 0.0:
        return 0.0
    return max(0.0, min(1.0, area / hull))


def _compactness(component: np.ndarray) -> float:
    contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    area = float(cv2.contourArea(contours[0]))
    perimeter = float(cv2.arcLength(contours[0], True))
    if perimeter <= 0.0:
        return 0.0
    return max(0.0, min(1.0, 4.0 * np.pi * area / (perimeter * perimeter)))


class FastTracker:
    def __init__(self, perception_cfg: dict[str, Any]) -> None:
        self.cfg = perception_cfg
        self.centroid_window = int(perception_cfg.get("centroid_window", 50))
        self._next_id = 1
        self._tracks: list[Track] = []
        self.last_mask: np.ndarray | None = None
        self.last_warped: np.ndarray | None = None

        self._kernel_open = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (int(perception_cfg["morph_open_px"]), int(perception_cfg["morph_open_px"])),
        )
        self._kernel_close = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (int(perception_cfg["morph_close_px"]), int(perception_cfg["morph_close_px"])),
        )

    @property
    def tracks(self) -> tuple[Track, ...]:
        return tuple(self._tracks)

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1
        self.last_mask = None
        self.last_warped = None

    def mask_for(self, image_bgr: np.ndarray) -> np.ndarray:
        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise TrackerError(f"expected BGR HxWx3, got {image_bgr.shape}")

        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        lower = np.array(
            [self.cfg["hue_min"], self.cfg["sat_min"], self.cfg["value_min"]], dtype=np.uint8
        )
        upper = np.array([self.cfg["hue_max"], 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel_open)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel_close)
        return mask

    def update(
        self,
        image_bgr: np.ndarray,
        t_s: float,
        rect: Rectification,
        canvas_px: tuple[int, int],
        px_per_mm: float,
    ) -> list[Track]:
        warped = warp_to_canvas(image_bgr, rect, canvas_px, px_per_mm)
        return self.update_warped(warped, t_s, px_per_mm)

    def update_warped(self, warped_bgr: np.ndarray, t_s: float, px_per_mm: float) -> list[Track]:
        if px_per_mm <= 0:
            raise TrackerError(f"px_per_mm must be positive: {px_per_mm}")

        mask = self.mask_for(warped_bgr)
        self.last_mask = mask
        self.last_warped = warped_bgr
        detections = self._detections(mask, warped_bgr, px_per_mm)
        self._associate(detections, t_s)

        return [track for track in self._tracks if track.misses == 0]

    def reanchor(
        self, mask: np.ndarray, warped_bgr: np.ndarray, t_s: float, px_per_mm: float
    ) -> list[Track]:
        """Apply an external (SAM) mask without counting classical misses.

        This runs at <=2 Hz between high-rate threshold ticks, so it must not
        advance the miss counter for tracks the SAM mask does not mention. A
        matched track keeps its identity and its area history; the SAM mask only
        refreshes geometry, which is what makes it an *anchor* rather than a
        second, slower tracker.

        The anchor refresh intentionally does not append to the centroid
        history: its t_s is the (older) submission time, not a new classical
        observation, and appending it would break the monotonic kinematic log
        the source estimator relies on.

        Staleness is enforced one layer up: the SAM worker drops late results
        from older frames before they reach here, so an old mask can never
        overwrite newer tracker state.

        Score separation: only the mask *geometry* is consumed. The SAM
        segmentation confidence (``SamSample.score``) is display-only — it is
        never written onto a track and never enters fusion P, which is a
        function of classical evidence (phi_v, phi_h, phi_t, W) alone.
        """

        if px_per_mm <= 0:
            raise TrackerError(f"px_per_mm must be positive: {px_per_mm}")
        if not isinstance(mask, np.ndarray) or mask.ndim != 2:
            raise TrackerError(
                f"reanchor needs a 2-D full-canvas mask, got {type(mask).__name__} "
                f"with ndim={getattr(mask, 'ndim', '?')}"
            )
        if mask.shape != warped_bgr.shape[:2]:
            raise TrackerError(
                f"reanchor mask shape {mask.shape} does not match frame {warped_bgr.shape[:2]}"
            )
        binary = np.where(np.ascontiguousarray(mask).astype(bool), 255, 0).astype(np.uint8)
        detections = self._detections(binary, warped_bgr, px_per_mm)

        radius = float(self.cfg.get("reanchor_radius_mm", self.cfg["link_radius_mm"]))
        pairs: list[tuple[float, int, int]] = []
        for track_index, track in enumerate(self._tracks):
            for det_index, det in enumerate(detections):
                distance = float(
                    np.hypot(
                        det.centroid_mm[0] - track.centroid_mm[0],
                        det.centroid_mm[1] - track.centroid_mm[1],
                    )
                )
                if distance <= radius:
                    pairs.append((distance, track_index, det_index))
        pairs.sort()

        used_tracks: set[int] = set()
        used_dets: set[int] = set()
        for _, track_index, det_index in pairs:
            if track_index in used_tracks or det_index in used_dets:
                continue
            used_tracks.add(track_index)
            used_dets.add(det_index)
            track = self._tracks[track_index]
            self._apply_geometry(track, detections[det_index])
            track.last_t = max(track.last_t, t_s)
            track.misses = 0
            track.sam_t_s = t_s

        for det_index, det in enumerate(detections):
            if det_index not in used_dets:
                track = self._spawn(det, t_s)
                track.sam_t_s = t_s
                self._tracks.append(track)

        self._tracks.sort(key=lambda track: track.track_id)
        return [track for track in self._tracks if track.sam_t_s == t_s]

    def _detections(
        self, mask: np.ndarray, warped_bgr: np.ndarray, px_per_mm: float
    ) -> list[Detection]:
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        mm_per_px = 1.0 / px_per_mm
        value = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2HSV)[:, :, 2]

        found: list[Detection] = []
        for label in range(1, count):
            x, y, width, height, area_px = (int(v) for v in stats[label])
            area_mm2 = area_px * mm_per_px * mm_per_px
            if not self.cfg["min_area_mm2"] <= area_mm2 <= self.cfg["max_area_mm2"]:
                continue

            component = (labels[y : y + height, x : x + width] == label).astype(np.uint8)
            selected = component.astype(bool)
            intensities = value[y : y + height, x : x + width][selected].astype(np.float32) / 255.0
            cx, cy = centroids[label]

            found.append(
                Detection(
                    centroid_mm=(float(cx) * mm_per_px, float(cy) * mm_per_px),
                    area_mm2=area_mm2,
                    bbox_mm=(
                        x * mm_per_px,
                        y * mm_per_px,
                        (x + width) * mm_per_px,
                        (y + height) * mm_per_px,
                    ),
                    mean_intensity=float(intensities.mean()),
                    intensity_var=float(intensities.var()),
                    clipped_fraction=float((intensities >= 250.0 / 255.0).mean()),
                    compactness=_compactness(component),
                    solidity=_solidity(component),
                    mask=component.astype(bool),
                    offset_px=(x, y),
                )
            )

        # Stable ordering keeps association deterministic when distances tie.
        found.sort(key=lambda det: (round(det.centroid_mm[0], 6), round(det.centroid_mm[1], 6)))
        return found

    def _associate(self, detections: list[Detection], t_s: float) -> None:
        radius = float(self.cfg["link_radius_mm"])
        pairs: list[tuple[float, int, int]] = []
        for track_index, track in enumerate(self._tracks):
            for det_index, det in enumerate(detections):
                dx = det.centroid_mm[0] - track.centroid_mm[0]
                dy = det.centroid_mm[1] - track.centroid_mm[1]
                distance = float(np.hypot(dx, dy))
                if distance <= radius:
                    pairs.append((distance, track_index, det_index))
        pairs.sort()

        matched_tracks: set[int] = set()
        matched_dets: set[int] = set()
        for _, track_index, det_index in pairs:
            if track_index in matched_tracks or det_index in matched_dets:
                continue
            matched_tracks.add(track_index)
            matched_dets.add(det_index)
            self._apply(self._tracks[track_index], detections[det_index], t_s)

        survivors: list[Track] = []
        for track_index, track in enumerate(self._tracks):
            if track_index in matched_tracks:
                survivors.append(track)
                continue
            track.misses += 1
            if track.misses <= int(self.cfg["candidate_ttl_ticks"]):
                survivors.append(track)

        for det_index, det in enumerate(detections):
            if det_index not in matched_dets:
                survivors.append(self._spawn(det, t_s))

        survivors.sort(key=lambda track: track.track_id)
        self._tracks = survivors

    def _spawn(self, det: Detection, t_s: float) -> Track:
        track = Track(
            track_id=self._next_id,
            centroid_mm=det.centroid_mm,
            area_mm2=det.area_mm2,
            dA_dt_mm2_per_s=0.0,
            mean_intensity=det.mean_intensity,
            intensity_var=det.intensity_var,
            clipped_fraction=det.clipped_fraction,
            compactness=det.compactness,
            solidity=det.solidity,
            bbox_mm=det.bbox_mm,
            mask=det.mask,
            offset_px=det.offset_px,
            first_t=t_s,
            last_t=t_s,
            _area_history=[(t_s, det.area_mm2)],
            _centroid_history=[(t_s, det.centroid_mm)],
        )
        self._next_id += 1
        return track

    def _apply(self, track: Track, det: Detection, t_s: float) -> None:
        self._apply_geometry(track, det)
        track.last_t = t_s
        track.hits += 1
        track.misses = 0
        track._centroid_history.append((t_s, det.centroid_mm))
        if self.centroid_window > 0:
            track._centroid_history = track._centroid_history[-self.centroid_window :]

        window = int(self.cfg["growth_window"])
        track._area_history.append((t_s, det.area_mm2))
        track._area_history = track._area_history[-window:]
        track.dA_dt_mm2_per_s = self._slope(track._area_history)

    @staticmethod
    def _apply_geometry(track: Track, det: Detection) -> None:
        track.centroid_mm = det.centroid_mm
        track.area_mm2 = det.area_mm2
        track.mean_intensity = det.mean_intensity
        track.intensity_var = det.intensity_var
        track.clipped_fraction = det.clipped_fraction
        track.compactness = det.compactness
        track.solidity = det.solidity
        track.bbox_mm = det.bbox_mm
        track.mask = det.mask
        track.offset_px = det.offset_px

    @staticmethod
    def _slope(history: list[tuple[float, float]]) -> float:
        if len(history) < 2:
            return 0.0
        times = np.array([t for t, _ in history], dtype=np.float64)
        areas = np.array([a for _, a in history], dtype=np.float64)
        times = times - times[0]
        if times[-1] <= 0.0:
            return 0.0
        design = np.vstack([times, np.ones_like(times)]).T
        slope, _ = np.linalg.lstsq(design, areas, rcond=None)[0]
        return float(slope)
