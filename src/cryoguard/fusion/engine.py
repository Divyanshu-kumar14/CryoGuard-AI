"""Prior-weighted sigmoid fusion.

This is a suspicion score, not a calibrated probability. There is no training
set, no held-out calibration split, and no ECE or Brier score in this repository,
so the words "calibrated" and "probability" are not used anywhere (D-03).
"""

from __future__ import annotations

import math
from typing import Any


def sigmoid(z: float) -> float:
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    exp_z = math.exp(z)
    return exp_z / (1.0 + exp_z)


def leak_score(
    phi_v: float, phi_h: float, phi_t: float, w: float, weights: dict[str, Any]
) -> float:
    """P = sigma(w_v*phi_v + w_h*phi_h + w_t*min(phi_t,0) + w_w*W + b).

    ``min(phi_t, 0)`` keeps D-10 true: a dry room (positive phi_t) adds exactly
    zero leak evidence; proximity to the dew point subtracts.
    """

    z = (
        weights["w_visual"] * phi_v
        + weights["w_hydraulic"] * phi_h
        + weights["w_thermal"] * min(phi_t, 0.0)
        + weights["w_wetness"] * w
        + weights["bias"]
    )
    return sigmoid(z)
