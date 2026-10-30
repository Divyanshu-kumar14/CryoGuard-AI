"""Fusion arithmetic: bounded features, dry-air neutrality, and the anti-glint gate.

Row three is the one that matters. With the canonical weights a static connected
reflection scores 0.61, above the Zone-2 trip threshold of 0.50. The score alone
would trip in 600 ms. Only the growth-or-hydraulic gate prevents it.
"""

from __future__ import annotations

import pytest

from cryoguard.fusion.decision import decide
from cryoguard.fusion.engine import leak_score, sigmoid
from cryoguard.fusion.features import hydraulic_phi, thermal_phi, visual_phi
from cryoguard.models import Evidence, SystemLabel

THRESH = "fusion.thresholds"


def test_sigmoid_saturates_without_overflow() -> None:
    assert sigmoid(0.0) == 0.5
    assert sigmoid(-1000.0) == 0.0
    assert sigmoid(1000.0) == 1.0


def test_scores_stay_in_unit_interval(cfg) -> None:
    weights = cfg["fusion"]["weights"]
    for phi_v, phi_h, phi_t, wet in [
        (0.0, 0.0, 0.0, 0.0),
        (1.0, 1.0, 0.0, 1.0),
        (1.0, 0.0, -1.0, 0.0),
        (0.0, 1.0, 1.0, 1.0),
    ]:
        assert 0.0 <= leak_score(phi_v, phi_h, phi_t, wet, weights) <= 1.0


def test_dry_air_adds_exactly_zero_evidence(cfg) -> None:
    """D-10: positive phi_t must not raise the score, or every dry room leaks."""

    weights = cfg["fusion"]["weights"]
    assert leak_score(0.5, 0.3, 1.0, 0.0, weights) == leak_score(0.5, 0.3, 0.0, 0.0, weights)


def test_condensation_stays_nominal_only_because_fluorescence_gates_wetness(cfg) -> None:
    """Water beads are not fluorescent, so W is measured on positive pixels only.

    phi_t = (0.8 - 3.0) / 5.0 = -0.44: surface is 0.8 C above dew point.
    With W = 0 the clip stays below the 0.40 nominal cutoff; with W = 1 it does not.
    That gap is why the W gate is load-bearing rather than cosmetic.
    """

    weights = cfg["fusion"]["weights"]
    phi_t = thermal_phi(dew_margin_c=-2.09, cfg=cfg["fusion"]["thermal"])
    assert phi_t == pytest.approx(-0.418)

    clean = leak_score(0.35, 0.0, phi_t, 0.0, weights)
    leaky = leak_score(0.35, 0.0, phi_t, 1.0, weights)

    assert clean < cfg["fusion"]["thresholds"]["nominal_cutoff"]
    assert leaky > cfg["fusion"]["thresholds"]["nominal_cutoff"]


def test_static_reflection_scores_above_trip_threshold_but_never_actuates(cfg) -> None:
    """Delta R-1: the score is dangerous; the gate is what saves the rack."""

    ev = Evidence(
        zone=2,
        phi_v=0.40,  # bright, connected, static: contrast 1.0 * beta 0.4
        phi_h=0.0,
        phi_t=0.5,  # dry air
        W=1.0,
        dA_dt=0.0,
        q=1.0,
        ticks=50,
    )
    decision = decide(ev, cfg)

    assert decision.p >= cfg["fusion"]["thresholds"]["zone2_trip"]
    assert not decision.actuates
    assert decision.label is not SystemLabel.EMERGENCY_TRIP


def test_identical_evidence_with_real_growth_trips_zone_two(cfg) -> None:
    """The positive control for the test above: only dA/dt changed."""

    ev = Evidence(
        zone=2,
        phi_v=0.40,
        phi_h=0.0,
        phi_t=0.5,
        W=1.0,
        dA_dt=2.0,
        q=1.0,
        ticks=50,
    )
    decision = decide(ev, cfg)

    assert decision.rule_id == "R1"
    assert decision.label is SystemLabel.EMERGENCY_TRIP
    assert "VALVE_CLOSE" in decision.actions


def test_confirmed_leak_scenario_reaches_r2_with_fixture_telemetry(cfg) -> None:
    """Scenario 1: growth 1.2 mm2/s, pressure 0.29 psi/s, 0.9 L/min imbalance.

    These are the numbers the fixture script produces, not tuned constants.
    """

    hydraulic = hydraulic_phi(0.29, 0.9, cfg["fusion"]["hydraulic"])
    visual = visual_phi(1.2, 0.9, 0.05, cfg["fusion"]["visual"])
    assert hydraulic == pytest.approx(0.6 * (0.29 / 2.0) + 0.4 * 1.0)
    assert visual == pytest.approx(0.4 * (1.2 / 2.0) + 0.4 * 0.9 - 0.2 * 0.05)

    ev = Evidence(
        zone=0,
        phi_v=visual,
        phi_h=hydraulic,
        phi_t=0.2,
        W=1.0,
        dA_dt=1.2,
        q=0.95,
        ticks=10,
    )
    decision = decide(ev, cfg)

    assert decision.p >= cfg["fusion"]["thresholds"]["confirmed_leak"]
    assert decision.label is SystemLabel.CONFIRMED_LEAK


def test_hidden_hydraulic_event_drains_and_never_touches_the_valve(cfg) -> None:
    """R4 / D-11: no visual evidence means a human decides, not the PLC."""

    ev = Evidence(
        zone=0,
        phi_v=0.0,
        phi_h=1.0,
        phi_t=0.0,
        W=0.0,
        dA_dt=0.0,
        q=1.0,
        ticks=10,
        hydraulic_low_ms=3000,
    )
    decision = decide(ev, cfg)

    assert decision.rule_id == "R4"
    assert decision.label is SystemLabel.HIDDEN_HYDRAULIC_EVENT
    assert "SCHEDULER_DRAIN" in decision.actions
    assert "VALVE_CLOSE" not in decision.actions


def test_blind_camera_is_never_reported_as_nominal(cfg) -> None:
    """R8 closes the hole where a dead camera looked quiet and therefore safe."""

    ev = Evidence(
        zone=0,
        phi_v=0.0,
        phi_h=0.0,
        phi_t=0.0,
        W=0.0,
        dA_dt=0.0,
        camera_healthy=False,
        ticks=100,
    )
    decision = decide(ev, cfg)

    assert decision.rule_id == "R8"
    assert decision.label is SystemLabel.CAMERA_DEGRADED
    assert not decision.actuates


def test_observational_responses_are_not_actuation(cfg) -> None:
    """A 10 Hz sample request is not hardware movement, and must not be counted as one."""

    ev = Evidence(zone=0, phi_v=0.6, phi_h=0.2, phi_t=0.0, W=0.6, dA_dt=2.0, ticks=10)
    decision = decide(ev, cfg)

    assert decision.rule_id in {"R3", "R5"}
    assert "SAMPLE_10HZ" in decision.actions
    assert not decision.actuates


def test_hysteresis_cannot_be_bypassed_by_a_drain(cfg) -> None:
    """Zone 2, P above every threshold, but only one tick: PRD says no actuation."""

    ev = Evidence(
        zone=2,
        phi_v=1.0,
        phi_h=1.0,
        phi_t=0.0,
        W=1.0,
        dA_dt=2.0,
        q=1.0,
        ticks=1,
    )
    decision = decide(ev, cfg)

    assert decision.rule_id in {"R2c", "R3"}
    assert not decision.actuates
    assert "SCHEDULER_DRAIN" not in decision.actions


def test_only_a_confirmed_leak_or_electrical_breach_moves_hardware(cfg) -> None:
    """The safety claim as an executable invariant over the whole evidence grid."""

    allowed = {"R1", "R1b", "R2"}
    grid = [0.0, 0.05, 0.2, 0.4, 0.6, 0.9, 1.0]
    for phi_v in grid:
        for phi_h in grid:
            for wet in (0.0, 0.5, 1.0):
                for zone in (0, 1, 2):
                    ev = Evidence(
                        zone=zone,
                        phi_v=phi_v,
                        phi_h=phi_h,
                        phi_t=0.0,
                        W=wet,
                        dA_dt=2.0,
                        q=1.0,
                        ticks=10,
                        hydraulic_low_ms=5000,
                    )
                    decision = decide(ev, cfg)
                    if decision.actuates:
                        assert decision.rule_id in allowed, (decision.rule_id, decision.label)


def test_every_evidence_point_maps_to_exactly_one_rule(cfg) -> None:
    """The table must be total: no silent fall-through, no double match.

    Sampled on a grid; decide() is pure and first-match, so overlap is impossible
    by construction and only totality needs evidence.
    """

    seen: set[str] = set()
    grid = [0.0, 0.05, 0.2, 0.4, 0.6, 0.9, 1.0]
    for phi_v in grid:
        for phi_h in grid:
            for wet in (0.0, 0.5, 1.0):
                for zone in (0, 1, 2):
                    ev = Evidence(
                        zone=zone,
                        phi_v=phi_v,
                        phi_h=phi_h,
                        phi_t=0.0,
                        W=wet,
                        dA_dt=2.0,
                        ticks=10,
                        hydraulic_low_ms=5000,
                    )
                    decision = decide(ev, cfg)
                    assert decision.rule_id
                    seen.add(decision.rule_id)

    assert {"R1", "R2", "R3", "R5", "R7"} <= seen
