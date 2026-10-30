"""Asynchronous SAM worker (P3-T03, D-02).

SAM answers at <=2 Hz with at most one request in flight; the classical tracker
runs at 5-10 Hz and never waits for it. A slow or dead API therefore costs mask
refinement, not ticks.

The worker is deliberately a *scheduler*, not a tracker: it crops the hydraulic
regions, calls the client, and hands back full-canvas masks. The pipeline applies
those masks through ``FastTracker.reanchor``.
"""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from cryoguard.perception.sam_client import SamClient, SamResult


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
    regions: int = 0

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
        self.stride = max(1, int(round(fps / rate)))
        self.in_flight = max(1, int(sam_cfg["in_flight"]))
        self.timeout_s = float(sam_cfg["timeout_sec"])
        self._next_region = 0
        self._tasks: set[asyncio.Task[None]] = set()
        self._ready: deque[SamSample] = deque()
        self._failures: deque[SamFailure] = deque()
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
            # A miss must surface now, not inside a task nobody awaits.
            result = self.client.segment(crop, self.prompt)
            self.stats.submitted += 1
            self._accept(roi, result, frame_index, t_s)
            return

        self.stats.submitted += 1
        task = asyncio.create_task(self._run(roi, crop, frame_index, t_s))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _run(self, roi: Roi, crop: np.ndarray, frame_index: int, t_s: float) -> None:
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(self.client.segment, crop, self.prompt),
                timeout=self.timeout_s,
            )
        except Exception as exc:  # a dead API must not stop the safety loop
            self.stats.failures += 1
            self._failures.append(SamFailure(roi.region_id, frame_index, t_s, repr(exc)))
            return

        self._accept(roi, result, frame_index, t_s)

    def _accept(self, roi: Roi, result: SamResult, frame_index: int, t_s: float) -> None:
        if result.cached:
            self.stats.cached += 1
        else:
            self.stats.live += 1
        if result.provenance == "fixture":
            self.stats.fixture += 1
        else:
            self.stats.api += 1
        if not result.masks:
            return

        width, height = self.canvas_px
        for sam_mask in result.masks:
            full = np.zeros((height, width), dtype=bool)
            full[roi.y0 : roi.y1, roi.x0 : roi.x1] = sam_mask.mask
            bx0, by0, bx1, by1 = sam_mask.bbox_px
            self._ready.append(
                SamSample(
                    region_id=roi.region_id,
                    mask_full=full,
                    bbox_full=(roi.x0 + bx0, roi.y0 + by0, roi.x0 + bx1, roi.y0 + by1),
                    score=sam_mask.score,
                    provenance=result.provenance,
                    cached=result.cached,
                    frame_index=frame_index,
                    t_s=t_s,
                )
            )
            self.stats.applied += 1

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
