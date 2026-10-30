"""Canonical decision table (PRD §6.3) made total.

Three holes in the PRD's table are closed here, each with a test:
  * R2b / R2c  — P >= 0.85 with unmet confirmation preconditions previously
                 matched no rule at all;
  * R8         — an unhealthy camera used to fall through to NOMINAL;
  * growth gate — a static reflection connected to a fitting scores 0.61 and
                 would have tripped Zone 2 in 600 ms (delta v1.2 §3, R-1).

Order is fixed and first-match wins.
"""

from __future__ import annotations

import math
from typing import Any

from cryoguard.fusion.engine import leak_score
from cryoguard.models import Decision, Evidence, SystemLabel

EMERGENCY = ("VALVE_CLOSE", "FORCE_OFF", "SCHEDULER_DRAIN")

# Evidence is produced by bounded feature functions. A value outside these
# ranges is a bug upstream, not a quiet tick: fail loudly instead of letting a
# NaN comparison fall through to NOMINAL.
_BOUNDED = (
    ("phi_v", 0.0, 1.0),
    ("phi_h", 0.0, 1.0),
    ("phi_t", -1.0, 1.0),
    ("W", 0.0, 1.0),
    ("q", 0.0, 1.0),
)


class DecisionError(ValueError):
    """Evidence is non-finite or outside its physical bounds."""


def _validate(ev: Evidence) -> None:
    for name, low, high in _BOUNDED:
        value = float(getattr(ev, name))
        if not math.isfinite(value) or not low <= value <= high:
            raise DecisionError(f"{name} must be finite in [{low}, {high}], got {value!r}")
    if not math.isfinite(ev.dA_dt):
        raise DecisionError(f"dA_dt must be finite, got {ev.dA_dt!r}")
    if not math.isfinite(ev.temp_c):
        raise DecisionError(f"temp_c must be finite, got {ev.temp_c!r}")
    if ev.zone < -1:
        raise DecisionError(f"zone must be -1 (no candidate) or a zone id, got {ev.zone!r}")
    if ev.ticks < 0:
        raise DecisionError(f"ticks must be non-negative, got {ev.ticks!r}")
    if ev.hydraulic_low_ms < 0:
        raise DecisionError(f"hydraulic_low_ms must be non-negative, got {ev.hydraulic_low_ms!r}")


def decide(ev: Evidence, cfg: dict[str, Any]) -> Decision:
    _validate(ev)
    fusion = cfg["fusion"]
    thresholds = fusion["thresholds"]
    rules = cfg["decision"]
    q_min = cfg["perception"]["q_min"]
    p = leak_score(ev.phi_v, ev.phi_h, ev.phi_t, ev.W, fusion["weights"])

    # 1. R1b — hard safety faults win over everything.
    if (
        ev.plc_fault
        or ev.temp_c >= cfg["thermal_model"]["watchdog_limit_c"]
        or not ev.clamp_readback_ok
    ):
        return Decision("R1b", SystemLabel.EMERGENCY_TRIP, EMERGENCY, p)

    # 2. R8 — blind is not nominal.
    if not ev.camera_healthy:
        return Decision("R8", SystemLabel.CAMERA_DEGRADED, (), p)

    # A leak grows, or hydraulics already corroborate it. A static reflection does neither.
    growth_ok = (
        ev.dA_dt > rules["growth_epsilon_mm2_per_s"] or ev.phi_h >= rules["hydraulic_confirm_phi"]
    )
    zone_ok = ev.zone == rules["zone"]["electrical"]
    quality_ok = ev.q >= q_min

    # 3. R1 — electrical plane.
    if (
        zone_ok
        and p >= thresholds["zone2_trip"]
        and ev.ticks >= rules["hysteresis_ticks"]
        and quality_ok
        and growth_ok
    ):
        return Decision("R1", SystemLabel.EMERGENCY_TRIP, EMERGENCY, p)

    # 4. R2 — confirmed leak away from the electrical plane.
    confirmable = (
        p >= thresholds["confirmed_leak"]
        and ev.zone in (0, 1)
        and ev.ticks >= rules["hysteresis_ticks"]
        and ev.phi_v >= 0.2
        and ev.phi_h >= 0.2
        and quality_ok
        and not ev.maintenance
        and growth_ok
    )
    if confirmable:
        return Decision(
            "R2",
            SystemLabel.CONFIRMED_LEAK,
            ("CLAMP", "VALVE_CLOSE", "CHECKPOINT_VERIFY", "SCHEDULER_DRAIN"),
            p,
        )

    # 5-6. High suspicion that cannot be confirmed is still an anomaly, never silence.
    # Both branches are observational. Draining coolant requires the 3 s R4 hold below;
    # a score alone must never move hardware (D-11).
    if p >= thresholds["confirmed_leak"] and ev.zone in (0, 1):
        return Decision(
            "R2b", SystemLabel.UNVERIFIED_ANOMALY, ("ALERT", "SAMPLE_10HZ", "CDU_QUERY"), p
        )

    if p >= thresholds["confirmed_leak"] and zone_ok:
        return Decision(
            "R2c", SystemLabel.UNVERIFIED_ANOMALY, ("ALERT", "SAMPLE_10HZ", "CDU_QUERY"), p
        )

    # 7. R4 — hydraulic event with no visual evidence. Soft cordon only, never a valve (D-11).
    if ev.phi_v <= 0.1 and ev.phi_h >= 0.6 and ev.hydraulic_low_ms >= rules["hydraulic_hold_ms"]:
        return Decision("R4", SystemLabel.HIDDEN_HYDRAULIC_EVENT, ("ALERT", "SCHEDULER_DRAIN"), p)

    # 8. R3 — escalation band. PRD §6.3 is explicit: no actuation.
    if thresholds["escalation_cutoff"] <= p < thresholds["confirmed_leak"]:
        return Decision(
            "R3", SystemLabel.UNVERIFIED_ANOMALY, ("ALERT", "SAMPLE_10HZ", "CDU_QUERY"), p
        )

    # 9. R5 — suspect band; condensation is a labelled outcome, not an actuation.
    if thresholds["suspect_cutoff"] <= p < thresholds["escalation_cutoff"]:
        label = (
            SystemLabel.BENIGN_CONDENSATION
            if ev.phi_t < 0 and ev.phi_h <= rules["benign_phi_h_max"]
            else SystemLabel.SUSPECT_LEAK
        )
        return Decision("R5", label, ("SAMPLE_10HZ", "LOG"), p)

    # 10. R6 — log only.
    if thresholds["nominal_cutoff"] <= p < thresholds["suspect_cutoff"]:
        return Decision("R6", SystemLabel.LOG_ONLY, ("LOG",), p)

    # 11. R7.
    return Decision("R7", SystemLabel.NOMINAL, (), p)
