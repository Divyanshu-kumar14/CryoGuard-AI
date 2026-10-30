#!/usr/bin/env python
"""Generate the synthetic fixture clips and their manifest.

Everything this script writes is SIMULATED. It exists so the pipeline has a
deterministic signal before the physical rig is available, and so the tracker's
negative cases (condensation, glint, hidden hydraulic) are reproducible. It is
not evidence that the detector works on real coolant.

    uv run python scripts/make_sim_fixtures.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
SEEDS = {
    "sim_leak_qd4": 427,
    "sim_condensation": 428,
    "sim_glint": 429,
    "sim_hidden_hydraulic": 430,
}


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def base_scene(topology: dict, px_per_mm: float) -> np.ndarray:
    from cryoguard.capture.board import render_board

    plate = render_board(topology, px_per_mm)
    scene = cv2.cvtColor(plate, cv2.COLOR_GRAY2BGR)

    for zone in topology["zones"]:
        points = (np.asarray(zone["polygon_mm"], dtype=np.float64) * px_per_mm).astype(np.int32)
        cv2.polylines(scene, [points], True, (60, 60, 60), 1, lineType=cv2.LINE_AA)

    qd = topology["qd"][0]["roi_mm"]
    cv2.rectangle(
        scene,
        (int(qd[0] * px_per_mm), int(qd[1] * px_per_mm)),
        (int(qd[2] * px_per_mm), int(qd[3] * px_per_mm)),
        (70, 70, 70),
        1,
        lineType=cv2.LINE_AA,
    )
    return scene


def add_fluorescence(
    scene: np.ndarray,
    center_mm: tuple[float, float],
    radius_mm: float,
    px_per_mm: float,
    origin_mm: tuple[float, float] | None = None,
) -> np.ndarray:
    """Bright core, a wet trail back to the fitting, and a soft glow.

    The trail matters: FR-FUS-05 requires a continuous path to the QD ROI, and a
    leak that has moved away from its fitting still has one.
    """

    core = np.zeros_like(scene)
    center_px = (int(center_mm[0] * px_per_mm), int(center_mm[1] * px_per_mm))
    radius_px = max(1, int(radius_mm * px_per_mm))
    if origin_mm is not None:
        origin_px = (int(origin_mm[0] * px_per_mm), int(origin_mm[1] * px_per_mm))
        cv2.line(
            core,
            origin_px,
            center_px,
            (0, 220, 40),
            max(2, radius_px // 2),
            lineType=cv2.LINE_AA,
        )
    cv2.circle(core, center_px, radius_px, (0, 220, 40), -1, lineType=cv2.LINE_AA)

    # Halo first, then replace the core. An additive blend would push every
    # channel to 255 on the white marker quiet zone and the tracer would vanish
    # exactly where the QD sits.
    glow = cv2.GaussianBlur(core, (0, 0), max(1.0, radius_px * 0.7))
    scene = cv2.addWeighted(scene, 1.0, glow, 0.45, 0.0)

    covered = core.any(axis=2)
    mixed = (0.85 * core.astype(np.float32) + 0.15 * scene.astype(np.float32)).astype(np.uint8)
    scene[covered] = mixed[covered]
    return scene


def add_bead(
    scene: np.ndarray, center_mm: tuple[float, float], radius_mm: float, px_per_mm: float
) -> np.ndarray:
    """Water: bright but desaturated, so the fluorescence gate must ignore it."""

    center_px = (int(center_mm[0] * px_per_mm), int(center_mm[1] * px_per_mm))
    radius_px = max(1, int(radius_mm * px_per_mm))
    cv2.circle(scene, center_px, radius_px, (205, 195, 170), -1, lineType=cv2.LINE_AA)
    cv2.circle(
        scene,
        (center_px[0] - radius_px // 3, center_px[1] - radius_px // 3),
        max(1, radius_px // 3),
        (235, 230, 215),
        -1,
        lineType=cv2.LINE_AA,
    )
    return scene


def add_glint(
    scene: np.ndarray, center_mm: tuple[float, float], radius_mm: float, px_per_mm: float
) -> np.ndarray:
    center_px = (int(center_mm[0] * px_per_mm), int(center_mm[1] * px_per_mm))
    radius_px = max(1, int(radius_mm * px_per_mm))
    return cv2.circle(scene, center_px, radius_px, (255, 255, 255), -1, lineType=cv2.LINE_AA)


def render_clip(
    name: str, topology: dict, px_per_mm: float, fps: int, seconds: float
) -> list[np.ndarray]:
    rng = np.random.default_rng(SEEDS[name])
    frames: list[np.ndarray] = []
    total = int(round(fps * seconds))

    for index in range(total):
        t = index / fps
        scene = base_scene(topology, px_per_mm)

        if name == "sim_leak_qd4":
            # 3 -> 15 mm radius, drifting from QD-04 (x=100) into the electrical plane (x>=300).
            radius = 3.0 + 0.6 * t
            center = (195.0 + 11.0 * t, 145.0 + 1.0 * t)
            scene = add_fluorescence(scene, center, radius, px_per_mm, origin_mm=(195.0, 145.0))

        elif name == "sim_condensation":
            for bead in range(6):
                x = 70.0 + bead * 34.0
                y = 255.0 + (bead % 3) * 22.0
                scene = add_bead(scene, (x, y), 2.5 + 0.05 * t + 0.4 * (bead % 2), px_per_mm)

        elif name == "sim_glint":
            # Moving specular reflections: white, zero saturation.
            for glint in range(3):
                x = 320.0 + 40.0 * glint + 6.0 * t
                y = 100.0 + 45.0 * glint + 2.0 * t
                scene = add_glint(scene, (x, y), 2.0 + 0.5 * glint, px_per_mm)
            # A saturated yellow blink in Zone 2: passes the hue gate for 2 frames.
            if index % 9 in (0, 1):
                phase = (index // 9) * 37
                scene = cv2.circle(
                    scene,
                    (int((360 + phase % 180) * px_per_mm), int((150 + phase % 120) * px_per_mm)),
                    3 * int(px_per_mm),
                    (0, 255, 255),
                    -1,
                    lineType=cv2.LINE_AA,
                )

        elif name == "sim_hidden_hydraulic":
            # A dark wet patch: no fluorescence, so the camera cannot see the event.
            progress = min(1.0, t / 8.0)
            radius = 4.0 + 20.0 * progress
            cv2.circle(
                scene,
                (int(120 * px_per_mm), int(160 * px_per_mm)),
                int(radius * px_per_mm),
                (45, 45, 52),
                -1,
                lineType=cv2.LINE_AA,
            )
        else:  # pragma: no cover - guarded by the caller
            raise ValueError(name)

        noise = rng.normal(0.0, 1.2, scene.shape)
        frames.append(np.clip(scene.astype(np.float32) + noise, 0, 255).astype(np.uint8))

    return frames


def write_clip(path: Path, frames: list[np.ndarray], fps: int) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"cannot open writer for {path}")
    for frame in frames:
        writer.write(frame)
    writer.release()

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--px-per-mm", type=float, default=2.0)
    parser.add_argument(
        "--reuse", action="store_true", help="re-hash existing clips instead of rendering"
    )
    args = parser.parse_args()

    topology = load_yaml(ROOT / "configs" / "topology.yaml")
    output = ROOT / "fixtures" / "videos" / "simulated"
    output.mkdir(parents=True, exist_ok=True)

    specs = [
        (
            "sim_leak_qd4",
            "slow_leak_qd4",
            "CONFIRMED_LEAK",
            2,
            "dev",
            "growing fluorescent leak from QD-04 into Zone 2",
        ),
        (
            "sim_condensation",
            "condensation_bead",
            "BENIGN_CONDENSATION",
            1,
            "dev",
            "desaturated beads, no fluorescence",
        ),
        (
            "sim_glint",
            "glint_noise",
            "SUSPECT_LEAK",
            2,
            "test",
            "white specular glints plus a 2-frame saturated blink",
        ),
        (
            "sim_hidden_hydraulic",
            "hidden_hydraulic",
            "HIDDEN_HYDRAULIC_EVENT",
            0,
            "test",
            "dark wet patch, no visual evidence",
        ),
    ]

    clips = []
    for name, scenario, label, zone, split, note in specs:
        path = output / f"{name}.mp4"
        if args.reuse:
            if not path.is_file():
                raise SystemExit(f"--reuse set but {path} is missing")
            frame_count = int(round(args.fps * args.seconds))
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            height = int(topology["chassis"]["height_mm"] * args.px_per_mm)
            width = int(topology["chassis"]["width_mm"] * args.px_per_mm)
        else:
            frames = render_clip(name, topology, args.px_per_mm, args.fps, args.seconds)
            digest = write_clip(path, frames, args.fps)
            frame_count = len(frames)
            height, width = frames[0].shape[:2]
        clips.append(
            {
                "id": name,
                "path": str(path.relative_to(ROOT)),
                "scenario": scenario,
                "source": "synthetic",
                "fps": args.fps,
                "frames": frame_count,
                "resolution_px": [width, height],
                "px_per_mm": args.px_per_mm,
                "calibration": "first_frame",
                "ground_truth_label": label,
                "qd_origin": "QD-04",
                "expected_zone": zone,
                "split": split,
                "sha256": digest,
                "notes": note,
            }
        )

    manifest = {
        "version": 1,
        "provenance": "synthetic",
        "note": (
            "Every clip is generated by scripts/make_sim_fixtures.py. None is captured "
            "hardware. The dashboard must badge all of them SIMULATED."
        ),
        "split": {
            "policy": "provisional: one clip per class, so a per-class 60/40 split is not possible",
            "dev": [c["id"] for c in clips if c["split"] == "dev"],
            "test": [c["id"] for c in clips if c["split"] == "test"],
        },
        "clips": clips,
    }
    manifest_path = ROOT / "fixtures" / "manifest.yaml"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"wrote {len(clips)} clips to {output}")
    for clip in clips:
        print(f"  {clip['id']:22} {clip['frames']:4d} frames  {clip['sha256'][:12]}")
    print(f"manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
