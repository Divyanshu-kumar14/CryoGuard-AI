"""Evidence features. All outputs are bounded so the logistic cannot be dominated
by a single unnormalised term (see delta v1.2 §4)."""

from __future__ import annotations

from typing import Any


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def visual_phi(
    dA_dt_mm2_per_s: float, contrast: float, var_norm: float, cfg: dict[str, Any]
) -> float:
    """phi_v = alpha * growth + beta * contrast - gamma * Var(I), clamped to [0,1].

    ``growth`` is a RATE ratio. Dividing an area by a rate reference (the PRD's
    A_ref) makes the term numerically irrelevant; delta v1.2 §3 records the check.
    """

    growth = clamp(dA_dt_mm2_per_s / cfg["growth_ref_mm2_per_s"], 0.0, 1.0)
    contrast_norm = clamp(contrast / cfg["contrast_max"], 0.0, 1.0)
    penalty = cfg["gamma"] * clamp(var_norm, 0.0, 1.0)
    return clamp(cfg["alpha"] * growth + cfg["beta"] * contrast_norm - penalty, 0.0, 1.0)


def hydraulic_phi(
    pressure_drop_psi_per_s: float, flow_imbalance_lpm: float, cfg: dict[str, Any]
) -> float:
    """phi_h = w_p * psi(dP/dt) + w_q * psi(Qin - Qout), both saturated at their refs."""

    pressure_term = clamp(pressure_drop_psi_per_s / cfg["pressure_rate_ref_psi_per_s"], 0.0, 1.0)
    flow_term = clamp(flow_imbalance_lpm / cfg["flow_imbalance_ref_lpm"], 0.0, 1.0)
    return clamp(
        cfg["w_pressure"] * pressure_term + cfg["w_flow"] * flow_term,
        0.0,
        1.0,
    )


def thermal_phi(dew_margin_c: float, cfg: dict[str, Any]) -> float:
    """PRD FR-FUS-01: phi_t = (T_surface - T_dewpoint) / dT_scale, clamped to [-1, 1].

    Zero at the dew point exactly, negative below it. No additional margin: an
    offset would move the neutral point away from the physical onset of
    condensation and would not be traceable to the requirements.
    """

    return clamp(dew_margin_c / cfg["dt_scale_c"], -1.0, 1.0)
