#!/usr/bin/env python
"""Measure the tracker on one clip. Diagnostic only; it prints, it does not assert.

uv run python scripts/probe_clip.py fixtures/videos/simulated/sim_leak_qd4.mp4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from cryoguard.capture.rectify import estimate
from cryoguard.capture.source import VideoSource
from cryoguard.capture.zones import ZoneMap
from cryoguard.config import load
from cryoguard.perception.fast_tracker import FastTracker

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("clip", type=Path)
    parser.add_argument("--every", type=int, default=3, help="process one frame in N")
    args = parser.parse_args()

    cfg = load(ROOT / "configs" / "default.yaml")
    topology = yaml.safe_load((ROOT / "configs" / "topology.yaml").read_text(encoding="utf-8"))
    px_per_mm = float(cfg["camera"]["px_per_mm"])
    canvas = (
        int(topology["chassis"]["width_mm"] * px_per_mm),
        int(topology["chassis"]["height_mm"] * px_per_mm),
    )
    zones = ZoneMap.build(topology, px_per_mm)

    with VideoSource(args.clip) as source:
        first = source.read(0)
        rect = estimate(first.image, topology["chassis"]["aruco"])
        tracker = FastTracker(cfg["perception"])

        seen = 0
        with_candidate = 0
        max_area = 0.0
        max_growth = 0.0
        zone_hits: dict[int, int] = {}
        for frame in source.frames():
            if frame.index % args.every:
                continue
            tracks = tracker.update(frame.image, frame.t_s, rect, canvas, px_per_mm)
            seen += 1
            if not tracks:
                continue
            with_candidate += 1
            best = max(tracks, key=lambda track: track.area_mm2)
            max_area = max(max_area, best.area_mm2)
            max_growth = max(max_growth, best.dA_dt_mm2_per_s)
            zone = zones.zone_for_cropped(best.mask, best.offset_px)
            if zone is not None:
                zone_hits[zone] = zone_hits.get(zone, 0) + 1

    print(f"{args.clip.name}: rect rms {rect.reprojection_rms_px:.2f}px")
    print(f"  frames processed     {seen} (1 in {args.every})")
    print(
        f"  with >=1 candidate   {with_candidate} ({100.0 * with_candidate / max(seen, 1):.1f} %)"
    )
    print(f"  max area             {max_area:.1f} mm2")
    print(f"  max dA/dt            {max_growth:.1f} mm2/s")
    print(f"  candidate zones      {dict(sorted(zone_hits.items()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
