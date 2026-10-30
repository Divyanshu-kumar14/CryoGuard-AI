"""Pixels to chassis millimetres via ArUco homography.

The chassis face is treated as a plane. That assumption is exact only for markers
and surfaces coplanar with the board; anything standing off the plane is mapped
with a perspective error proportional to its height. Delta D-19 records the limit
and the demo never claims otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


class RectificationError(RuntimeError):
    """Raised when there are too few markers to solve a homography."""


@dataclass(frozen=True)
class Rectification:
    """px -> mm homography for one camera setup."""

    homography: np.ndarray
    inverse: np.ndarray
    reprojection_rms_px: float
    markers_used: tuple[int, ...]
    image_size: tuple[int, int]

    def to_mm(self, points_px: np.ndarray) -> np.ndarray:
        pts = np.asarray(points_px, dtype=np.float64).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.homography).reshape(-1, 2)

    def to_px(self, points_mm: np.ndarray) -> np.ndarray:
        pts = np.asarray(points_mm, dtype=np.float64).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.inverse).reshape(-1, 2)


def marker_corners_mm(center: tuple[float, float], size_mm: float) -> np.ndarray:
    """Axis-aligned corners in ArUco order: top-left, top-right, bottom-right, bottom-left."""

    cx, cy = center
    half = size_mm / 2.0
    return np.array(
        [
            [cx - half, cy - half],
            [cx + half, cy - half],
            [cx + half, cy + half],
            [cx - half, cy + half],
        ],
        dtype=np.float64,
    )


def object_points(aruco_cfg: dict[str, Any]) -> dict[int, np.ndarray]:
    size = float(aruco_cfg["marker_size_mm"])
    return {
        int(marker["id"]): marker_corners_mm(tuple(marker["center_mm"]), size)
        for marker in aruco_cfg["markers"]
    }


def detect_markers(
    image: np.ndarray, aruco_cfg: dict[str, Any]
) -> tuple[dict[int, np.ndarray], int]:
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, aruco_cfg["dictionary"]))
    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(image)
    if ids is None:
        return {}, 0

    found = {
        int(marker_id): np.asarray(corner[0], dtype=np.float64)
        for corner, marker_id in zip(corners, ids.ravel(), strict=True)
    }
    return found, len(found)


def estimate(image: np.ndarray, aruco_cfg: dict[str, Any]) -> Rectification:
    """Solve px -> mm from every configured marker visible in the frame."""

    found, _ = detect_markers(image, aruco_cfg)
    known = object_points(aruco_cfg)
    usable = sorted(set(found) & set(known))

    min_markers = int(aruco_cfg.get("min_markers", 2))
    if len(usable) < min_markers:
        raise RectificationError(
            f"need {min_markers} markers for a stable homography, found {len(usable)}"
        )

    src = np.concatenate([found[i] for i in usable]).astype(np.float32)
    dst = np.concatenate([known[i] for i in usable]).astype(np.float32)

    homography, _ = cv2.findHomography(src, dst, method=0)
    if homography is None:
        raise RectificationError("homography solve failed")

    inverse = np.linalg.inv(homography)
    projected = cv2.perspectiveTransform(src.reshape(-1, 1, 2), homography)
    back = cv2.perspectiveTransform(projected, inverse).reshape(-1, 2)
    rms = float(np.sqrt(np.mean(np.sum((back - src.reshape(-1, 2)) ** 2, axis=1))))

    height, width = image.shape[:2]
    return Rectification(
        homography=homography,
        inverse=inverse,
        reprojection_rms_px=rms,
        markers_used=tuple(usable),
        image_size=(width, height),
    )


def warp_to_canvas(
    image: np.ndarray, rect: Rectification, canvas_px: tuple[int, int], px_per_mm: float
) -> np.ndarray:
    """Render the rectified view at a fixed scale so area in px maps to mm²."""

    width, height = canvas_px
    scale = np.array([[px_per_mm, 0.0, 0.0], [0.0, px_per_mm, 0.0], [0.0, 0.0, 1.0]])
    transform = scale @ rect.homography
    return cv2.warpPerspective(
        image,
        transform,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderValue=(0, 0, 0),
    )
