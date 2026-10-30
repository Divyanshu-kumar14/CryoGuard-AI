"""Fast tracker: area, association, growth rate, and the rejections that matter.

The negative cases are the point. Condensation and specular glint are what make
this product hard; a tracker that only finds bright blobs proves nothing.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from cryoguard.capture.rectify import estimate
from cryoguard.capture.zones import ZoneMap
from cryoguard.perception.fast_tracker import FastTracker, TrackerError
from support import chassis_frame, render_board

ROOT = Path(__file__).resolve().parents[1]
PX_PER_MM = 1.0


@pytest.fixture(scope="module")
def perception_cfg() -> dict:
    return yaml.safe_load((ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))[
        "perception"
    ]


@pytest.fixture()
def tracker(perception_cfg: dict) -> FastTracker:
    return FastTracker(perception_cfg)


def test_single_blob_area_and_centroid(tracker: FastTracker) -> None:
    tracks = tracker.update_warped(chassis_frame([(10, 150, 120)]), 0.0, PX_PER_MM)

    assert len(tracks) == 1
    track = tracks[0]
    assert track.centroid_mm == pytest.approx((150.0, 120.0), abs=1.5)
    assert 250.0 <= track.area_mm2 <= 360.0  # pi*10^2 = 314
    assert track.compactness > 0.7
    assert track.mean_intensity > 200.0


def test_growth_rate_is_measured_not_assumed(tracker: FastTracker) -> None:
    tracker.update_warped(chassis_frame([(10, 150, 120)]), 0.0, PX_PER_MM)
    tracks = tracker.update_warped(chassis_frame([(12, 150, 120)]), 0.5, PX_PER_MM)

    area_delta = np.pi * (12**2 - 10**2)
    assert tracks[0].dA_dt_mm2_per_s == pytest.approx(area_delta / 0.5, rel=0.25)


def test_static_blob_has_zero_growth(tracker: FastTracker) -> None:
    for tick in range(4):
        tracks = tracker.update_warped(chassis_frame([(10, 150, 120)]), tick * 0.2, PX_PER_MM)

    assert abs(tracks[0].dA_dt_mm2_per_s) < 5.0


def test_slow_drift_keeps_one_identity(tracker: FastTracker) -> None:
    ids = set()
    for tick in range(5):
        tracks = tracker.update_warped(
            chassis_frame([(10, 150 + tick * 4, 120)]), tick * 0.2, PX_PER_MM
        )
        ids.add(tracks[0].track_id)

    assert len(ids) == 1


def test_two_blobs_get_two_identities(tracker: FastTracker) -> None:
    tracks = tracker.update_warped(chassis_frame([(8, 120, 100), (8, 400, 300)]), 0.0, PX_PER_MM)
    assert len(tracks) == 2
    assert {t.track_id for t in tracks} == {1, 2}


def test_candidate_survives_a_short_dropout_then_expires(tracker: FastTracker) -> None:
    tracker.update_warped(chassis_frame([(10, 150, 120)]), 0.0, PX_PER_MM)
    empty = chassis_frame([])

    for tick in range(1, 4):
        assert tracker.update_warped(empty, tick * 0.2, PX_PER_MM) == []
    assert tracker.tracks[0].misses == 3

    tracker.update_warped(empty, 4 * 0.2, PX_PER_MM)
    assert tracker.tracks == ()


def test_reappearing_candidate_keeps_its_id_within_ttl(tracker: FastTracker) -> None:
    tracker.update_warped(chassis_frame([(10, 150, 120)]), 0.0, PX_PER_MM)
    tracker.update_warped(chassis_frame([]), 0.2, PX_PER_MM)
    tracks = tracker.update_warped(chassis_frame([(10, 152, 121)]), 0.4, PX_PER_MM)

    assert tracks[0].track_id == 1


def test_area_limits_reject_noise_and_bathtub(tracker: FastTracker) -> None:
    speck = np.zeros((400, 600, 3), dtype=np.uint8)
    speck[100, 100] = (0, 255, 0)  # one pixel: 1 mm2 at 1 px/mm, removed by open
    assert tracker.update_warped(speck, 0.0, PX_PER_MM) == []

    assert tracker.update_warped(chassis_frame([(50, 100, 100)]), 0.2, PX_PER_MM) == []


def test_condensation_colour_is_not_fluorescence(tracker: FastTracker) -> None:
    """A water bead is bright but unsaturated. Saturation is the discriminator."""

    frame = np.zeros((400, 600, 3), dtype=np.uint8)
    frame[:] = (90, 70, 40)  # dim, slightly warm chassis
    cv2.circle(frame, (200, 150), 12, (210, 190, 150), -1)  # pale blue-white bead

    assert tracker.update_warped(frame, 0.0, PX_PER_MM) == []


def test_specular_glint_is_not_fluorescence(tracker: FastTracker) -> None:
    frame = np.zeros((400, 600, 3), dtype=np.uint8)
    cv2.circle(frame, (200, 150), 6, (255, 255, 255), -1)  # white, zero saturation

    assert tracker.update_warped(frame, 0.0, PX_PER_MM) == []


def test_replay_is_deterministic(perception_cfg: dict) -> None:
    frames = [(0.0, chassis_frame([(10, 150, 120)])), (0.2, chassis_frame([(11, 154, 121)]))]

    runs = []
    for _ in range(2):
        tracker = FastTracker(perception_cfg)
        seen = []
        for t_s, frame in frames:
            seen.append(
                [
                    (t.track_id, t.centroid_mm, t.area_mm2)
                    for t in tracker.update_warped(frame, t_s, 1.0)
                ]
            )
        runs.append(seen)

    assert runs[0] == runs[1]


def test_tracker_rejects_non_bgr_frames(tracker: FastTracker) -> None:
    with pytest.raises(TrackerError, match="BGR"):
        tracker.update_warped(np.zeros((400, 600), dtype=np.uint8), 0.0, PX_PER_MM)


def test_rectified_frame_maps_blob_to_chassis_millimetres(perception_cfg: dict) -> None:
    """End-to-end: board in camera pixels -> mm -> tracker -> zone."""

    topology = yaml.safe_load((ROOT / "configs" / "topology.yaml").read_text(encoding="utf-8"))
    board = render_board(topology, px_per_mm=2.0)
    rect = estimate(board, topology["chassis"]["aruco"])

    canvas_px = (1200, 800)
    zones = ZoneMap.build(topology, px_per_mm=2.0)
    tracker = FastTracker(perception_cfg)

    # 2 px/mm: a 10 mm radius disc centred at 140,150 mm.
    camera_view = chassis_frame([(20, 280, 300)], size=(1200, 800))
    tracks = tracker.update(camera_view, 0.0, rect, canvas_px, px_per_mm=2.0)

    assert len(tracks) == 1
    assert tracks[0].centroid_mm == pytest.approx((140.0, 150.0), abs=2.0)
    assert zones.zone_for_cropped(tracks[0].mask, tracks[0].offset_px) in (0, 1, 2)
