"""Camera-health signal (P3-T06, R8).

A blind camera is not a nominal chassis. Three cheap measurements catch the ways
a fluorescence camera fails on a rack: the light dies, the lens defocuses, or the
sensor is blown out by a specular source. The decision table turns any of them
into ``CAMERA_DEGRADED``, which is observational: it never actuates, and it never
lets a blind frame confirm a leak.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class CameraHealth:
    brightness: float
    focus: float
    saturation: float
    bright_ok: bool
    focus_ok: bool
    unsaturated: bool

    @property
    def ok(self) -> bool:
        return self.bright_ok and self.focus_ok and self.unsaturated

    @property
    def reasons(self) -> tuple[str, ...]:
        reasons: list[str] = []
        if not self.bright_ok:
            reasons.append("brightness")
        if not self.focus_ok:
            reasons.append("focus")
        if not self.unsaturated:
            reasons.append("saturation")
        return tuple(reasons)


def assess_health(image_bgr: np.ndarray, camera_cfg: dict[str, Any]) -> CameraHealth:
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError(f"expected BGR HxWx3, got {image_bgr.shape}")

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    focus = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    saturation = float((gray >= 250).mean())

    health = camera_cfg["health"]
    return CameraHealth(
        brightness=brightness,
        focus=focus,
        saturation=saturation,
        bright_ok=float(health["brightness_min"]) <= brightness <= float(health["brightness_max"]),
        focus_ok=focus >= float(health["focus_min"]),
        unsaturated=saturation <= float(health["saturation_max"]),
    )
