"""ArUco homography: synthetic board with a known ground-truth mapping."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from cryoguard.capture.rectify import (
    RectificationError,
    estimate,
    marker_corners_mm,
    object_points,
)
from support import render_board

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def topology() -> dict:
    return yaml.safe_load((ROOT / "configs" / "topology.yaml").read_text(encoding="utf-8"))


def test_marker_corner_order_matches_aruco() -> None:
    corners = marker_corners_mm((100.0, 100.0), 50.0)
    assert corners[0].tolist() == [75.0, 75.0]  # top-left
    assert corners[2].tolist() == [125.0, 125.0]  # bottom-right


def test_identity_scale_and_offset_recovered(topology: dict) -> None:
    image = render_board(topology, px_per_mm=2.0, offset_px=(100, 50))
    rect = estimate(image, topology["chassis"]["aruco"])

    assert rect.reprojection_rms_px < 1.0
    assert len(rect.markers_used) == 4

    for marker in topology["chassis"]["aruco"]["markers"]:
        px_x = marker["center_mm"][0] * 2.0 + 100
        px_y = marker["center_mm"][1] * 2.0 + 50
        mapped = rect.to_mm(np.array([[px_x, px_y]]))[0]
        assert mapped == pytest.approx(marker["center_mm"], abs=1.0)


def test_perspective_view_maps_to_millimetres(topology: dict) -> None:
    board = render_board(topology, px_per_mm=3.0)
    height, width = board.shape

    # A plausible oblique view: the right edge is further away.
    source = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    target = np.float32([[40, 60], [1180, 20], [1240, 760], [10, 800]])
    warp = cv2.getPerspectiveTransform(source, target)
    view = cv2.warpPerspective(board, warp, (1280, 820))

    rect = estimate(view, topology["chassis"]["aruco"])
    assert rect.reprojection_rms_px < 1.5

    rng = np.random.default_rng(7)
    points_px = rng.uniform([0, 0], [width, height], size=(40, 2))
    truth = points_px / 3.0
    view_points = cv2.perspectiveTransform(points_px.reshape(-1, 1, 2), warp).reshape(-1, 2)
    mapped = rect.to_mm(view_points)
    error = np.linalg.norm(mapped - truth, axis=1)
    assert float(error.max()) < 1.0


def test_too_few_markers_fails_loudly(topology: dict) -> None:
    empty = np.zeros((400, 600), dtype=np.uint8)
    with pytest.raises(RectificationError, match="need 2 markers"):
        estimate(empty, topology["chassis"]["aruco"])


def test_object_points_cover_every_configured_marker(topology: dict) -> None:
    known = object_points(topology["chassis"]["aruco"])
    assert sorted(known) == [0, 1, 2, 3]
    assert all(points.shape == (4, 2) for points in known.values())
