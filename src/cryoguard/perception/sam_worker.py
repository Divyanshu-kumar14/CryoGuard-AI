"""Asynchronous SAM worker (P3-T03, D-02).

SAM answers at <=2 Hz with at most one request in flight; the classical tracker
runs at 5-10 Hz and never waits for it. A slow or dead API therefore costs mask
refinement, not ticks.

The worker is deliberately a *scheduler*, not a tracker: it crops the hydraulic
regions, calls the client, and hands back full-canvas masks. The pipeline applies
those masks through ``FastTracker.reanchor``.

Guarantees this scheduler owns:

* non-blocking: :meth:`submit` only enqueues (or serves a fast local cache
  read); live calls run in tasks bounded by ``timeout_sec`` and degrade to
  :class:`SamFailure`, never to a stalled safety loop;
* provenance: every sample carries ``cached`` *and* ``provenance``. A
  fixture-seeded cache hit stays ``cached=True, provenance="fixture"`` — it
  is never relabelled ``live`` or ``api``;
* monotonicity: a late result from an older frame cannot overwrite newer
  tracker state. Results at or below the emitted watermark are dropped and
  counted as ``stale``;
* mask hygiene: a malformed mask (wrong rank, wrong extent) is a recorded
  failure, not a tracker write;
* score separation: ``SamSample.score`` is the segmentation confidence for
  display only. It is never written onto a track and never enters fusion P.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from cryoguard.perception.sam_client import SamCacheMiss, SamClient, SamResult

logger = logging.getLogger(__name__)


class SamWorkerError(RuntimeError):
    """The worker was used before it could produce a usable result."""


@dataclass(frozen=True)
class Roi:
    region_id: str
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def shape(self) -> tuple[int, int]:
        return (self.y1 - self.y0, self.x1 - self.x0)

    def crop(self, image: np.ndarray) -> np.ndarray:
        return image[self.y0 : self.y1, self.x0 : self.x1]


@dataclass(frozen=True)
class SamSample:
    region_id: str
    mask_full: np.ndarray
    bbox_full: tuple[int, int, int, int]
    score: float
    provenance: str
    cached: bool
    frame_index: int
    t_s: float


@dataclass(frozen=True)
class SamFailure:
    region_id: str
    frame_index: int
    t_s: float
    error: str


@dataclass
class SamStats:
    submitted: int = 0
    applied: int = 0
    cached: int = 0
    live: int = 0
    fixture: int = 0
    api: int = 0
    failures: int = 0
    skipped: int = 0
    stale: int = 0
    regions: int = 0
    # Per-prompt audit: prompt -> {"submitted", "applied", "empty", "failures"}.
    # The configured default stays the default; these counts exist so a prompt
    # trial can be reported honestly, not so a prompt can be picked for yield.
    per_prompt: dict[str, dict[str, int]] = field(default_factory=dict)
    # Wall-clock samples for reports/latency.json; never part of a decision.
    cached_ms: list[float] = field(default_factory=list)
    live_ms: list[float] = field(default_factory=list)

    def note(self, prompt: str, key: str) -> None:
        bucket = self.per_prompt.setdefault(
            prompt, {"submitted": 0, "applied": 0, "empty": 0, "failures": 0}
        )
        bucket[key] = bucket.get(key, 0) + 1

    def as_dict(self) -> dict[str, int]:
        return {
            "submitted": self.submitted,
            "applied": self.applied,
            "cached": self.cached,
            "live": self.live,
            "fixture": self.fixture,
            "api": self.api,
            "failures": self.failures,
            "skipped": self.skipped,
            "stale": self.stale,
            "regions": self.regions,
        }


def build_rois(
    topology: dict[str, Any],
    px_per_mm: float,
    canvas_px: tuple[int, int],
    margin_mm: float,
) -> tuple[Roi, ...]:
    width, height = canvas_px
    rois: list[Roi] = []
    for qd in topology["qd"]:
        x0, y0, x1, y1 = (float(v) for v in qd["roi_mm"])
        margin = float(margin_mm)
        rois.append(
            Roi(
                region_id=str(qd["id"]),
                x0=max(0, int(round((x0 - margin) * px_per_mm))),
                y0=max(0, int(round((y0 - margin) * px_per_mm))),
                x1=min(width, int(round((x1 + margin) * px_per_mm))),
                y1=min(height, int(round((y1 + margin) * px_per_mm))),
            )
        )
    return tuple(rois)


class SamWorker:
    def __init__(
        self,
        client: SamClient,
        topology: dict[str, Any],
        sam_cfg: dict[str, Any],
        px_per_mm: float,
        canvas_px: tuple[int, int],
        fps: float,
    ) -> None:
        self.client = client
        self.prompt = str(sam_cfg["prompts"]["coolant"])
        self.canvas_px = canvas_px
        self.rois = build_rois(topology, px_per_mm, canvas_px, float(sam_cfg["roi_margin_mm"]))
        if not self.rois:
            raise SamWorkerError("topology declares no hydraulic ROI to crop")

        rate = float(sam_cfg["async_rate_hz"])
        if rate <= 0:
            raise SamWorkerError(f"sam.async_rate_hz must be positive: {rate}")
        if rate > 2.0:
            raise SamWorkerError(f"sam.async_rate_hz must stay <= 2 Hz (D-02): {rate}")
        self.stride = max(1, int(round(fps / rate)))
        self.in_flight = max(1, int(sam_cfg["in_flight"]))
        self.timeout_s = float(sam_cfg["timeout_sec"])
        self._next_region = 0
        self._tasks: set[asyncio.Task[None]] = set()
        self._ready: deque[SamSample] = deque()
        self._failures: deque[SamFailure] = deque()
        # Monotonicity watermark: the newest (frame_index, t_s) already emitted.
        # A task that finishes late with an older frame is dropped in _accept
        # so it can never overwrite newer tracker state.
        self._emitted_frame = -1
        self._emitted_t = float("-inf")
        self.stats = SamStats(regions=len(self.rois))

    def due(self, frame_index: int) -> bool:
        return frame_index % self.stride == 0

    def _pick_roi(self) -> Roi:
        roi = self.rois[self._next_region % len(self.rois)]
        self._next_region += 1
        return roi

    def submit(self, warped_bgr: np.ndarray, frame_index: int, t_s: float) -> None:
        """Offer a frame. Returns immediately; at most ``in_flight`` requests run."""

        if not self.due(frame_index):
            return
        if len(self._tasks) >= self.in_flight:
            self.stats.skipped += 1
            return

        roi = self._pick_roi()
        crop = np.ascontiguousarray(roi.crop(warped_bgr))
        if crop.size == 0:
            self.stats.skipped += 1
            return

        if self.client.mode == "cache":
            # A miss must surface now, not inside a task nobody awaits. Any
            # other failure (corrupt cache entry, malformed mask) is recorded,
            # never raised: submit runs inside the safety tick (pipeline.py),
            # so a bad entry must cost refinement, not the run.
            started = time.perf_counter()
            try:
                result = self.client.segment(crop, self.prompt)
            except SamCacheMiss:
                raise
            except Exception as exc:
                self.stats.failures += 1
                self.stats.note(self.prompt, "failures")
                self._failures.append(SamFailure(roi.region_id, frame_index, t_s, repr(exc)))
                logger.warning("SAM cache read for %s failed: %r", roi.region_id, exc)
                return
            self.stats.cached_ms.append((time.perf_counter() - started) * 1000.0)
            self.stats.submitted += 1
            self.stats.note(self.prompt, "submitted")
            try:
                self._accept(roi, result, frame_index, t_s)
            except Exception as exc:  # a bad mask must not crash the tick
                self.stats.failures += 1
                self.stats.note(self.prompt, "failures")
                self._failures.append(SamFailure(roi.region_id, frame_index, t_s, repr(exc)))
                logger.warning("SAM cache result for %s dropped: %r", roi.region_id, exc)
            return

        self.stats.submitted += 1
        self.stats.note(self.prompt, "submitted")
        task = asyncio.create_task(self._run(roi, crop, frame_index, t_s))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _run(self, roi: Roi, crop: np.ndarray, frame_index: int, t_s: float) -> None:
        started = time.perf_counter()
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(self.client.segment, crop, self.prompt),
                timeout=self.timeout_s,
            )
        except Exception as exc:  # a dead API must not stop the safety loop
            self.stats.failures += 1
            self.stats.note(self.prompt, "failures")
            self._failures.append(SamFailure(roi.region_id, frame_index, t_s, repr(exc)))
            logger.warning(
                "SAM live request for %s failed (prompt=%r): %r",
                roi.region_id,
                self.prompt,
                exc,
            )
            return

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        (self.stats.cached_ms if result.cached else self.stats.live_ms).append(elapsed_ms)
        try:
            self._accept(roi, result, frame_index, t_s)
        except Exception as exc:  # a malformed live mask must not corrupt the loop
            self.stats.failures += 1
            self.stats.note(self.prompt, "failures")
            self._failures.append(SamFailure(roi.region_id, frame_index, t_s, repr(exc)))
            logger.warning("SAM live result for %s dropped: %r", roi.region_id, exc)

    def _accept(self, roi: Roi, result: SamResult, frame_index: int, t_s: float) -> None:
        # Stale guard: tasks may finish out of order. An older frame must never
        # overwrite state a newer frame already anchored.
        if (frame_index, t_s) < (self._emitted_frame, self._emitted_t):
            self.stats.stale += 1
            return
        self._emitted_frame, self._emitted_t = frame_index, t_s

        if result.cached:
            self.stats.cached += 1
        else:
            self.stats.live += 1
        if result.provenance == "fixture":
            self.stats.fixture += 1
        else:
            self.stats.api += 1
        if not result.masks:
            # Zero matches is a valid outcome (the configured coolant prompt
            # returns it on live), not a failure — but it is recorded per
            # prompt so the limitation stays visible instead of silent.
            self.stats.note(result.prompt or self.prompt, "empty")
            logger.info(
                "SAM returned zero matches for region %s (prompt=%r provenance=%s)",
                roi.region_id,
                result.prompt,
                result.provenance,
            )
            return

        width, height = self.canvas_px
        expected = (roi.y1 - roi.y0, roi.x1 - roi.x0)
        for sam_mask in result.masks:
            mask = np.asarray(sam_mask.mask)
            if mask.ndim != 2 or mask.shape != expected:
                self.stats.failures += 1
                self.stats.note(result.prompt or self.prompt, "failures")
                self._failures.append(
                    SamFailure(
                        roi.region_id,
                        frame_index,
                        t_s,
                        f"SamResponseError: mask shape {mask.shape} != ROI {expected}",
                    )
                )
                logger.warning(
                    "SAM mask for %s dropped: shape %s != ROI %s",
                    roi.region_id,
                    mask.shape,
                    expected,
                )
                continue
            full = np.zeros((height, width), dtype=bool)
            try:
                full[roi.y0 : roi.y1, roi.x0 : roi.x1] = mask.astype(bool)
            except (TypeError, ValueError) as exc:
                self.stats.failures += 1
                self.stats.note(result.prompt or self.prompt, "failures")
                self._failures.append(SamFailure(roi.region_id, frame_index, t_s, repr(exc)))
                logger.warning("SAM mask for %s dropped: %r", roi.region_id, exc)
                continue
            bx0, by0, bx1, by1 = sam_mask.bbox_px
            self._ready.append(
                SamSample(
                    region_id=roi.region_id,
                    mask_full=full,
                    bbox_full=(roi.x0 + bx0, roi.y0 + by0, roi.x0 + bx1, roi.y0 + by1),
                    # Segmentation confidence, display only: never written to a
                    # track and never an input to fusion P.
                    score=sam_mask.score,
                    provenance=result.provenance,
                    cached=result.cached,
                    frame_index=frame_index,
                    t_s=t_s,
                )
            )
            self.stats.applied += 1
            self.stats.note(result.prompt or self.prompt, "applied")

    def poll(self) -> list[SamSample]:
        samples: list[SamSample] = []
        while self._ready:
            samples.append(self._ready.popleft())
        return samples

    def failures(self) -> list[SamFailure]:
        failures = list(self._failures)
        self._failures.clear()
        return failures

    async def close(self) -> None:
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
