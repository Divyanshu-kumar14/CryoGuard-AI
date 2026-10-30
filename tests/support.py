"""Synthetic scene helpers shared by capture and perception tests.

These are test doubles, not fixtures that ship. The rendered board uses the real
marker layout from topology.yaml so a topology edit breaks the tests.
"""

from __future__ import annotations

import cv2
import numpy as np

from cryoguard.capture.board import render_board as render_board_impl


def render_board(topology: dict, px_per_mm: float, offset_px=(0, 0)):
    """Test alias for the production renderer: one layout definition, no drift."""

    return render_board_impl(topology, px_per_mm, offset_px)


def chassis_frame(blobs: list[tuple[int, int, int]], size=(600, 400)) -> np.ndarray:
    """Black metric frame with filled green discs: (radius_px, x_mm, y_mm)."""

    width, height = size
    image = np.zeros((height, width, 3), dtype=np.uint8)
    for radius, x, y in blobs:
        cv2.circle(image, (int(x), int(y)), int(radius), (0, 220, 40), -1, lineType=cv2.LINE_8)
    return image
