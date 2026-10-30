"""Mask quality q = q_area * q_clip * q_solid (delta v1.2 §6).

R1 and R2 fail closed below ``q_min``: an unreadable mask must not be allowed to
confirm a leak, and it must not be treated as evidence of absence either.
"""

from __future__ import annotations

from typing import Any

from cryoguard.perception.fast_tracker import Track


def mask_quality(track: Track, perception_cfg: dict[str, Any]) -> float:
    q_area = min(1.0, track.area_mm2 / float(perception_cfg["min_area_mm2"]))
    q_clip = 1.0 - min(1.0, max(0.0, track.clipped_fraction))
    q_solid = min(1.0, track.solidity / float(perception_cfg["solidity_ref"]))
    quality = q_area * q_clip * q_solid

    # A mask that is mostly sensor saturation cannot localise a leak, whatever
    # its shape. This is a hard gate, not a soft penalty: R1/R2 require q_min.
    if track.clipped_fraction > float(perception_cfg["exposure_clip_max"]):
        return 0.0
    return quality
