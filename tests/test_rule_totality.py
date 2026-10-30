"""Rule totality (P5-T03): every valid evidence vector maps to exactly one rule.

The decision table is a total function. ``decide`` validates its input first, so
there are only two outcomes: a canonical rule, or a ``DecisionError`` for
evidence that cannot physically exist. Neither path is a silent fall-through.

The property test covers the space; ``test_every_rule_is_reachable`` proves the
space actually reaches all eleven rules, so no branch is dead code.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from cryoguard.fusion.decision import DecisionError, decide
from cryoguard.models import Decision, Evidence, SystemLabel

RULES = {"R1", "R1b", "R2", "R2b", "R2c", "R3", "R4", "R5", "R6", "R7", "R8"}
ACTUATING_RULES = {"R1", "R1b", "R2"}
UNIT = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)


@st.composite
def evidence(draw: st.DrawFn) -> Evidence:
    return Evidence(
        zone=draw(st.sampled_from([-1, 0, 1, 2, 3])),
        phi_v=draw(UNIT),
        phi_h=draw(UNIT),
        phi_t=draw(st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False)),
        W=draw(UNIT),
        dA_dt=draw(
            st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False)
        ),
        q=draw(UNIT),
        camera_healthy=draw(st.booleans()),
        plc_fault=draw(st.booleans()),
        temp_c=draw(
            st.floats(min_value=-20.0, max_value=120.0, allow_nan=False, allow_infinity=False)
        ),
        clamp_readback_ok=draw(st.booleans()),
        maintenance=draw(st.booleans()),
        ticks=draw(st.integers(min_value=0, max_value=10)),
        hydraulic_low_ms=draw(st.integers(min_value=0, max_value=10_000)),
    )


@settings(max_examples=400, deadline=None)
@given(ev=evidence())
def test_every_valid_evidence_vector_maps_to_one_canonical_rule(cfg, ev: Evidence) -> None:
    decision = decide(ev, cfg)

    assert isinstance(decision, Decision)
    assert decision.rule_id in RULES
    assert decision.label in set(SystemLabel)
    assert 0.0 <= decision.p <= 1.0
    assert decide(ev, cfg) == decision  # pure: same input, same output


@settings(max_examples=400, deadline=None)
@given(ev=evidence())
def test_only_r1_r1b_r2_ever_actuate(cfg, ev: Evidence) -> None:
    decision = decide(ev, cfg)

    assert decision.actuates is (decision.rule_id in ACTUATING_RULES)
    if not decision.actuates:
        assert "VALVE_CLOSE" not in decision.actions
        assert "FORCE_OFF" not in decision.actions
        assert "CLAMP" not in decision.actions


@settings(max_examples=200, deadline=None)
@given(ev=evidence())
def test_a_scheduler_drain_alone_is_not_actuation(cfg, ev: Evidence) -> None:
    decision = decide(ev, cfg)
    if decision.rule_id == "R4":
        assert "SCHEDULER_DRAIN" in decision.actions
        assert not decision.actuates


def test_every_rule_is_reachable() -> None:
    """One constructed witness per rule. A missing row means dead safety code."""

    cases = {
        "R1b": Evidence(zone=0, phi_v=0.0, phi_h=0.0, phi_t=0.0, W=0.0, dA_dt=0.0, plc_fault=True),
        "R8": Evidence(
            zone=0, phi_v=0.0, phi_h=0.0, phi_t=0.0, W=0.0, dA_dt=0.0, camera_healthy=False
        ),
        "R1": Evidence(zone=2, phi_v=0.4, phi_h=0.0, phi_t=0.0, W=1.0, dA_dt=2.0, ticks=3),
        "R2": Evidence(zone=0, phi_v=0.66, phi_h=0.41, phi_t=0.0, W=1.0, dA_dt=20.0, ticks=3),
        "R2b": Evidence(
            zone=0, phi_v=0.66, phi_h=0.41, phi_t=0.0, W=1.0, dA_dt=20.0, q=0.2, ticks=3
        ),
        "R2c": Evidence(
            zone=2, phi_v=0.66, phi_h=0.41, phi_t=0.0, W=1.0, dA_dt=20.0, q=0.2, ticks=3
        ),
        "R4": Evidence(
            zone=0,
            phi_v=0.0,
            phi_h=1.0,
            phi_t=0.0,
            W=0.0,
            dA_dt=0.0,
            ticks=3,
            hydraulic_low_ms=3000,
        ),
        "R3": Evidence(zone=0, phi_v=0.66, phi_h=0.2, phi_t=0.0, W=1.0, dA_dt=20.0, ticks=3),
        "R5": Evidence(zone=0, phi_v=0.5, phi_h=0.2, phi_t=0.0, W=0.8, dA_dt=0.0, ticks=1),
        "R6": Evidence(zone=0, phi_v=0.3, phi_h=0.2, phi_t=0.0, W=0.7, dA_dt=0.0, ticks=1),
        "R7": Evidence(zone=-1, phi_v=0.0, phi_h=0.0, phi_t=0.0, W=0.0, dA_dt=0.0),
    }

    # cfg is loaded here rather than injected so the witness table stays readable.
    from cryoguard.config import load

    cfg = load("configs/default.yaml")
    for rule_id, ev in cases.items():
        decision = decide(ev, cfg)
        assert decision.rule_id == rule_id, (rule_id, decision.rule_id, decision.p)
    assert set(cases) == RULES


def test_condensation_is_a_labelled_r5_outcome_not_a_leak(cfg) -> None:
    ev = Evidence(
        zone=0,
        phi_v=0.5,
        phi_h=0.05,
        phi_t=-0.5,
        W=1.0,
        dA_dt=0.0,
        ticks=1,
    )
    decision = decide(ev, cfg)

    assert decision.rule_id == "R5"
    assert decision.label is SystemLabel.BENIGN_CONDENSATION
    assert not decision.actuates


@pytest.mark.parametrize(
    "field, value",
    [
        ("phi_v", float("nan")),
        ("phi_v", 1.01),
        ("phi_h", -0.01),
        ("phi_t", -1.01),
        ("W", 1.01),
        ("q", -0.1),
        ("dA_dt", float("inf")),
        ("temp_c", float("nan")),
        ("ticks", -1),
        ("zone", -2),
        ("hydraulic_low_ms", -1),
    ],
)
def test_impossible_evidence_fails_loudly(cfg, field: str, value: float) -> None:
    valid = Evidence(zone=0, phi_v=0.2, phi_h=0.2, phi_t=0.0, W=0.0, dA_dt=0.0, ticks=1)
    with pytest.raises(DecisionError, match=field):
        decide(replace(valid, **{field: value}), cfg)


def test_maintenance_suppresses_zone_zero_but_never_zone_two(cfg) -> None:
    zone_zero = Evidence(
        zone=0,
        phi_v=0.66,
        phi_h=0.41,
        phi_t=0.0,
        W=1.0,
        dA_dt=20.0,
        ticks=3,
        maintenance=True,
    )
    decision = decide(zone_zero, cfg)
    assert decision.rule_id == "R2b"
    assert not decision.actuates

    zone_two = Evidence(
        zone=2,
        phi_v=0.4,
        phi_h=0.0,
        phi_t=0.0,
        W=1.0,
        dA_dt=2.0,
        ticks=3,
        maintenance=True,
    )
    assert decide(zone_two, cfg).rule_id == "R1"


def test_a_blind_camera_cannot_manufacture_a_hidden_hydraulic_event(cfg) -> None:
    ev = Evidence(
        zone=0,
        phi_v=0.0,
        phi_h=1.0,
        phi_t=0.0,
        W=0.0,
        dA_dt=0.0,
        ticks=10,
        hydraulic_low_ms=5000,
        camera_healthy=False,
    )
    assert decide(ev, cfg).rule_id == "R8"


@pytest.mark.parametrize(
    "score, quality, expected",
    [
        (0.399999, 0.2, "R7"),
        (0.40, 0.2, "R6"),
        (0.499999, 0.2, "R6"),
        (0.50, 0.2, "R5"),
        (0.699999, 0.2, "R5"),
        (0.70, 0.2, "R3"),
        (0.849999, 0.2, "R3"),
        (0.85, 0.2, "R2b"),
        (0.85, 1.0, "R2"),
    ],
)
def test_p_band_boundaries_are_half_open(
    monkeypatch, cfg, score: float, quality: float, expected: str
) -> None:
    """Every cut is [low, high): 0.40/0.50/0.70/0.85 each land in exactly one band."""

    monkeypatch.setattr("cryoguard.fusion.decision.leak_score", lambda *args, **kwargs: score)
    ev = Evidence(
        zone=0,
        phi_v=0.66,
        phi_h=0.41,
        phi_t=0.0,
        W=1.0,
        dA_dt=20.0,
        q=quality,
        ticks=3,
    )
    assert decide(ev, cfg).rule_id == expected


def test_zone_two_trip_boundary_is_half_open(monkeypatch, cfg) -> None:
    monkeypatch.setattr("cryoguard.fusion.decision.leak_score", lambda *args, **kwargs: 0.50)
    ev = Evidence(zone=2, phi_v=0.4, phi_h=0.0, phi_t=0.0, W=1.0, dA_dt=2.0, ticks=3)
    assert decide(ev, cfg).rule_id == "R1"

    monkeypatch.setattr("cryoguard.fusion.decision.leak_score", lambda *args, **kwargs: 0.499999)
    assert decide(ev, cfg).rule_id == "R6"


@pytest.mark.parametrize(
    "phi_v, phi_h",
    [(0.1, 1.0), (1.0, 0.1)],
)
def test_thermal_and_score_alone_never_confirm(cfg, phi_v: float, phi_h: float) -> None:
    """T-DEC-07: R2 needs phi_v >= 0.2 and phi_h >= 0.2, not just a high score."""

    ev = Evidence(
        zone=0,
        phi_v=phi_v,
        phi_h=phi_h,
        phi_t=0.0,
        W=1.0,
        dA_dt=20.0,
        q=1.0,
        ticks=3,
    )
    decision = decide(ev, cfg)
    assert decision.rule_id == "R2b"
    assert not decision.actuates
