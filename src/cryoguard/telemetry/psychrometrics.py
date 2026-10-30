"""Magnus-Tetens dew point, absolute humidity, and wet-bulb (PRD FR-TEL-02).

All three are closed-form and deterministic. Invalid inputs raise rather than
returning a plausible number, because every one of these values feeds a safety
decision: a silently wrong dew point turns condensation into a confirmed leak.
"""

from __future__ import annotations

import math

# Magnus-Tetens coefficients used by the PRD.
A = 17.27
B = 237.7

# Stull (2011) wet-bulb approximation, valid for -20..50 C and 5..99 % RH.
STULL_MIN_TEMP_C = -20.0
STULL_MAX_TEMP_C = 50.0
STULL_MIN_RH = 5.0
STULL_MAX_RH = 99.0


def _validate(temp_c: float, rh_pct: float) -> None:
    if not -50.0 <= temp_c <= 60.0:
        raise ValueError(f"temperature out of range: {temp_c}")
    if not 0.0 < rh_pct <= 100.0:
        raise ValueError(f"relative humidity must be in (0, 100]: {rh_pct}")


def saturation_vapour_pressure_hpa(temp_c: float) -> float:
    """Magnus saturation vapour pressure over water, in hPa."""

    if not -50.0 <= temp_c <= 60.0:
        raise ValueError(f"temperature out of range: {temp_c}")
    return 6.112 * math.exp((17.67 * temp_c) / (temp_c + 243.5))


def dew_point_c(temp_c: float, rh_pct: float) -> float:
    """Return the dew point in degrees Celsius.

    Raises ValueError outside the physical domain rather than returning a
    plausible-looking number (fail loudly at the trust boundary).
    """

    _validate(temp_c, rh_pct)
    gamma = (A * temp_c) / (B + temp_c) + math.log(rh_pct / 100.0)
    return (B * gamma) / (A - gamma)


def dew_margin_c(surface_c: float, temp_c: float, rh_pct: float) -> float:
    """Surface temperature minus dew point. Negative means condensation is possible."""

    return surface_c - dew_point_c(temp_c, rh_pct)


def absolute_humidity_g_m3(temp_c: float, rh_pct: float) -> float:
    """Water vapour density in g/m^3 from the Magnus vapour pressure."""

    _validate(temp_c, rh_pct)
    vapour_hpa = saturation_vapour_pressure_hpa(temp_c) * rh_pct / 100.0
    return 216.7 * vapour_hpa / (temp_c + 273.15)


def wet_bulb_c(temp_c: float, rh_pct: float) -> float:
    """Stull (2011) wet-bulb temperature.

    The correlation is an approximation, not a psychrometric solution; it is used
    for display and for the thermal narrative, never for a decision threshold.
    """

    _validate(temp_c, rh_pct)
    if not STULL_MIN_TEMP_C <= temp_c <= STULL_MAX_TEMP_C:
        raise ValueError(f"wet-bulb correlation is undefined at {temp_c} C")
    if not STULL_MIN_RH <= rh_pct <= STULL_MAX_RH:
        raise ValueError(f"wet-bulb correlation is undefined at {rh_pct} % RH")

    return (
        temp_c * math.atan(0.151977 * math.sqrt(rh_pct + 8.313659))
        + math.atan(temp_c + rh_pct)
        - math.atan(rh_pct - 1.676331)
        + 0.00391838 * (rh_pct**1.5) * math.atan(0.023101 * rh_pct)
        - 4.686035
    )
