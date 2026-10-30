"""Temporary CI diagnostic: fingerprint every crop the benchmark sends to SAM.

Prints one JSON line per submitted crop: frame index, region, decoded-frame sha,
warped-crop sha, cache key, and whether the key is present in the committed cache.
Run locally and on CI; the first line where the two diverge locates the bug.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
from pathlib import Path

import numpy as np
import yaml

import cryoguard.perception.sam_worker as sw
from cryoguard.bench.runner import run_benchmark
from cryoguard.config import load

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "artifacts" / "sam_cache"
cfg = load(ROOT / "configs/default.yaml")
topology = yaml.safe_load((ROOT / "configs/topology.yaml").read_text(encoding="utf-8"))


def sha(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()[:16]


original_submit = sw.SamWorker.submit


def submit(self, warped_bgr, frame_index, t_s):  # type: ignore[no-untyped-def]
    crop = self.rois[0].crop(warped_bgr) if self.rois else None
    record = {
        "frame": frame_index,
        "warp_sha": sha(warped_bgr),
        "shape": list(warped_bgr.shape),
        "edge_head": warped_bgr[0, :2].reshape(-1).tolist(),
        "edge_tail": warped_bgr[-1, -2:].reshape(-1).tolist(),
        "due": self.due(frame_index),
    }
    if record["due"] and crop is not None:
        from cryoguard.perception.sam_client import cache_key, encode_png

        key = cache_key(encode_png(np.ascontiguousarray(crop)), self.prompt, self.client.model)
        record["key"] = key
        record["present"] = (CACHE / f"{key}.json").is_file()
        record["crop_sha"] = sha(crop)
        record["crop_first"] = np.asarray(crop)[0, 0].tolist()
    print("DIAG " + json.dumps(record), flush=True)
    return original_submit(self, warped_bgr, frame_index, t_s)


sw.SamWorker.submit = submit  # type: ignore[method-assign]

import cv2

print(
    "DIAG ENV "
    + json.dumps(
        {
            "python": platform.python_version(),
            "machine": platform.machine(),
            "cpus": os.cpu_count(),
            "opencv": cv2.__version__,
            "opencv_file": cv2.__file__,
            "numpy": np.__version__,
            "cpu": [line.strip() for line in cv2.getBuildInformation().splitlines()
                    if "CPU/HW" in line or "CPU baseline" in line or "Parallel" in line],
        }
    ),
    flush=True,
)

runs = run_benchmark(
    manifest_path=ROOT / "fixtures" / "manifest.yaml",
    cfg=cfg,
    topology=topology,
    split="all",
    work_dir=Path("/tmp/cryoguard-diag"),
    sam_mode="cache",
)
print("DIAG done", len(runs), flush=True)
