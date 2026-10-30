"""Magnus-Tetens dew point (PRD FR-TEL-02)."""

from __future__ import annotations

import math

# Magnus-Tetens coefficients used by the PRD.
A = 17.27
B = 237.7


def dew_point_c(temp_c: float, rh_pct: float) -> float:
    """Return the dew point in degrees Celsius.

    Raises ValueError outside the physical domain rather than returning a
    plausible-looking number (fail loudly at the trust boundary).
    """

    if not -50.0 <= temp_c <= 60.0:
        raise ValueError(f"temperature out of range: {temp_c}")
    if not 0.0 < rh_pct <= 100.0:
        raise ValueError(f"relative humidity must be in (0, 100]: {rh_pct}")

    gamma = (A * temp_c) / (B + temp_c) + math.log(rh_pct / 100.0)
    return (B * gamma) / (A - gamma)


def dew_margin_c(surface_c: float, temp_c: float, rh_pct: float) -> float:
    """Surface temperature minus dew point. Negative means condensation is possible."""

    return surface_c - dew_point_c(temp_c, rh_pct)
