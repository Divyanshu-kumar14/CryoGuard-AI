"""Audit SAM prompts against one synthetic crop and record every outcome.

This script performs live hosted requests, so it needs ``META_API_KEY`` (read
from the environment or the repository ``.env``). It writes a JSON artifact with
the prompt, model, provenance, cache state, mask count, mask areas, and any
error — including honest zero-match lanes. It does not pick a prompt based on
mask yield and it never labels a fixture result as live.

Usage::

    PYTHONPATH=src python scripts/audit_sam_prompts.py
"""

from __future__ import annotations

import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
import yaml

from cryoguard.env import load_env
from cryoguard.perception.sam_client import SamClient

ROOT = Path(__file__).resolve().parents[1]
CLIP = ROOT / "fixtures/videos/simulated/sim_leak_qd4.mp4"
OUT = ROOT / "artifacts/sam_live_audit.json"


def brightest_blob_crop(clip: Path, *, size: int = 200) -> tuple[np.ndarray, dict[str, int]]:
    """Pick the frame/crop with the most tracer-coloured pixels.

    The crop is a demo aid, not a ground-truth annotation: it only ensures the
    prompt audit is run against pixels that actually contain the simulated
    tracer instead of a random corner.
    """

    capture = cv2.VideoCapture(str(clip))
    best_count = -1
    best_frame: np.ndarray | None = None
    best_bbox = (0, 0, 0, 0)
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = np.array((25, 80, 60), dtype=np.uint8)
        upper = np.array((95, 255, 255), dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        count = int(mask.sum() // 255)
        if count > best_count:
            ys, xs = np.nonzero(mask)
            if len(xs) == 0:
                continue
            best_count = count
            best_frame = frame
            best_bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
    capture.release()
    if best_frame is None:
        raise SystemExit(f"no tracer-coloured pixels in {clip}")
    x0, y0, x1, y1 = best_bbox
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    half = size // 2
    crop = best_frame[max(0, cy - half) : cy + half, max(0, cx - half) : cx + half]
    return crop, {"x0": x0, "y0": y0, "x1": x1, "y1": y1, "tracerPixels": best_count}


def main() -> int:
    load_env()
    cfg = yaml.safe_load((ROOT / "configs/default.yaml").read_text(encoding="utf-8"))["sam"]
    crop, bbox = brightest_blob_crop(CLIP)
    # A fresh cache per audit: a rerun must be a live call, never a cached
    # response masquerading as fresh evidence.
    with tempfile.TemporaryDirectory(prefix="cryoguard-sam-audit-") as cache_dir:
        client = SamClient(cfg, mode="live", cache_dir=cache_dir)
        prompts = [
            cfg["prompts"]["coolant"],
            "bright green blob",
            "yellow-green glow",
        ]
        trials = []
        for trial in client.evaluate_prompts(crop, prompts):
            entry: dict[str, object] = {
                "prompt": trial.prompt,
                "model": trial.model,
                "ok": trial.ok,
                "maskCount": trial.mask_count,
            }
            if trial.result is not None:
                entry.update(
                    {
                        "provenance": trial.result.provenance,
                        "cached": trial.result.cached,
                        "cacheKey": trial.result.cache_key,
                        "maskAreasPx": [int((mask.mask > 0).sum()) for mask in trial.result.masks],
                    }
                )
            else:
                entry["error"] = trial.error
            trials.append(entry)

    artifact = {
        "ts": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "clip": str(CLIP.relative_to(ROOT)),
        "cropShape": list(crop.shape),
        "tracerBBoxPx": bbox,
        "model": str(cfg["model"]),
        "mode": "live",
        "note": (
            "Live hosted requests. Zero mask counts are valid empty lanes, not "
            "errors, and are disclosed rather than replaced with a cache result."
        ),
        "trials": trials,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
