"""Render the physical marker board from topology.yaml.

A printed ArUco marker needs a white quiet zone; without one the black border
merges into a dark chassis face and the detector reports nothing. That is why
this module draws the quiet zone explicitly instead of trusting the caller.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

PLATE_GRAY = 90
QUIET_FRACTION = 0.2


def render_board(
    topology: dict[str, Any],
    px_per_mm: float,
    offset_px: tuple[int, int] = (0, 0),
    plate_gray: int = PLATE_GRAY,
) -> np.ndarray:
    """Top-down grayscale rendering of the chassis face with its markers."""

    chassis = topology["chassis"]
    width = int(round(chassis["width_mm"] * px_per_mm)) + 2 * offset_px[0]
    height = int(round(chassis["height_mm"] * px_per_mm)) + 2 * offset_px[1]
    image = np.full((height, width), plate_gray, dtype=np.uint8)

    aruco = chassis["aruco"]
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, aruco["dictionary"]))
    side = int(round(aruco["marker_size_mm"] * px_per_mm))
    quiet = max(4, int(round(side * QUIET_FRACTION)))

    for marker in aruco["markers"]:
        tile = cv2.aruco.generateImageMarker(dictionary, int(marker["id"]), side)
        cx = int(round(marker["center_mm"][0] * px_per_mm)) + offset_px[0]
        cy = int(round(marker["center_mm"][1] * px_per_mm)) + offset_px[1]
        half = side // 2
        cv2.rectangle(
            image,
            (cx - half - quiet, cy - half - quiet),
            (cx + half + quiet, cy + half + quiet),
            255,
            -1,
        )
        image[cy - half : cy - half + side, cx - half : cx - half + side] = tile
    return image
