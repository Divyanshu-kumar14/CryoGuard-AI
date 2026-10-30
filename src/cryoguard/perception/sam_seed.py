"""Seed the SAM cache from the deterministic fluorescence mask (fixture provenance).

This exists so the cache path can be exercised end to end without an API key. It
is *not* SAM: the mask comes from the classical HSV gate. Every cache entry is
written with ``provenance="fixture"`` and the CLI prints that, so no run can
mistake a seeded cache for a hosted response.

With a key, delete the seeded cache and run with ``--sam live`` once; the live
client writes ``provenance="api"`` entries in the same format.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from cryoguard.capture.rectify import estimate, warp_to_canvas
from cryoguard.capture.source import VideoSource
from cryoguard.perception.fast_tracker import FastTracker
from cryoguard.perception.sam_client import SamClient, SamMask, cache_key, encode_png
from cryoguard.perception.sam_worker import build_rois


def seed_fixture_cache(
    *,
    clip_path: str | Path,
    topology: dict[str, Any],
    cfg: dict[str, Any],
    cache_dir: str | Path,
) -> int:
    sam_cfg = cfg["sam"]
    perception = cfg["perception"]
    px_per_mm = float(cfg["camera"]["px_per_mm"])
    canvas = (
        int(round(float(topology["chassis"]["width_mm"]) * px_per_mm)),
        int(round(float(topology["chassis"]["height_mm"]) * px_per_mm)),
    )

    client = SamClient(sam_cfg, mode="cache", cache_dir=cache_dir)
    tracker = FastTracker(perception)
    prompt = str(sam_cfg["prompts"]["coolant"])
    written = 0

    with VideoSource(clip_path) as source:
        first = source.read(0)
        if first is None:
            raise RuntimeError(f"clip has no frames: {clip_path}")
        rect = estimate(first.image, topology["chassis"]["aruco"])
        rois = build_rois(topology, px_per_mm, canvas, float(sam_cfg["roi_margin_mm"]))
        stride = max(1, int(round(source.fps / float(sam_cfg["async_rate_hz"]))))

        for frame in source.frames():
            if frame.index % stride:
                continue
            warped = warp_to_canvas(frame.image, rect, canvas, px_per_mm)
            for roi in rois:
                crop = np.ascontiguousarray(roi.crop(warped))
                mask = tracker.mask_for(crop) > 0
                if not mask.any():
                    continue

                ys, xs = np.nonzero(mask)
                bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
                sam_mask = SamMask(mask=mask, bbox_px=bbox, score=1.0)
                key = cache_key(encode_png(crop), prompt, str(sam_cfg["model"]))
                shape = (int(crop.shape[0]), int(crop.shape[1]))
                client.store(key, prompt, (sam_mask,), "fixture", shape)
                written += 1

    return written
