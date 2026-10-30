"""Temporal gating: three ticks to confirm, five to step down (PRD §6.4)."""

from __future__ import annotations

from pathlib import Path

import pytest

from cryoguard.config import load
from cryoguard.fusion.hysteresis import HysteresisDecision
from cryoguard.models import Evidence, SystemLabel

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def cfg() -> dict:
    return load(ROOT / "configs" / "default.yaml")


def leak_evidence(**overrides) -> Evidence:
    base = dict(
        zone=0,
        phi_v=0.66,
        phi_h=0.45,
        phi_t=1.0,
        W=1.0,
        dA_dt=20.0,
        q=1.0,
        temp_c=42.0,
        ticks=1,
    )
    base.update(overrides)
    return Evidence(**base)


def scored_evidence(phi_v: float = 0.0, W: float = 0.0, **overrides) -> Evidence:
    """Evidence with a real fusion score; no monkeypatching of the score."""

    base = dict(
        zone=0,
        phi_v=phi_v,
        phi_h=0.0,
        phi_t=0.0,
        W=W,
        dA_dt=0.0,
        q=0.0,
        temp_c=42.0,
        ticks=1,
    )
    base.update(overrides)
    return Evidence(**base)


# Real scores from the frozen weights: 0.750, 0.550, 0.650 and 0.640.
R3_075 = scored_evidence(phi_v=1.0, W=0.82)
R5_055 = scored_evidence(phi_v=1.0, W=0.46)
R5_065 = scored_evidence(phi_v=1.0, W=0.6276)
R6_064 = scored_evidence(phi_v=1.0, W=0.6102)
CALM = scored_evidence()


def test_one_tick_does_not_confirm(cfg) -> None:
    engine = HysteresisDecision(cfg)
    decision = engine.update(leak_evidence(), t_s=0.0)
    assert decision.label is SystemLabel.UNVERIFIED_ANOMALY


def test_three_consecutive_ticks_confirm(cfg) -> None:
    engine = HysteresisDecision(cfg)
    for tick in range(2):
        decision = engine.update(leak_evidence(), t_s=tick * 0.2)
    assert decision.label is SystemLabel.UNVERIFIED_ANOMALY

    decision = engine.update(leak_evidence(), t_s=0.4)
    assert decision.rule_id == "R2"
    assert decision.label is SystemLabel.CONFIRMED_LEAK


def test_zone_two_reaches_emergency_trip_after_three_ticks(cfg) -> None:
    engine = HysteresisDecision(cfg)
    for tick in range(3):
        decision = engine.update(leak_evidence(zone=2), t_s=tick * 0.2)
    assert decision.rule_id == "R1"
    assert decision.label is SystemLabel.EMERGENCY_TRIP


def test_a_single_calm_tick_holds_the_band_but_resets_the_streak(cfg) -> None:
    """De-escalation is slow; escalation is still M consecutive ticks."""

    engine = HysteresisDecision(cfg)
    engine.update(leak_evidence(), t_s=0.0)
    engine.update(leak_evidence(), t_s=0.2)

    held = engine.update(CALM, t_s=0.4)
    assert held.rule_id == "R2b"
    assert held.label is SystemLabel.UNVERIFIED_ANOMALY

    # The calm tick did not de-escalate the state, but it did break the
    # consecutive-tick streak, so confirmation starts over.
    assert engine.update(leak_evidence(), t_s=0.6).label is SystemLabel.UNVERIFIED_ANOMALY
    assert engine.update(leak_evidence(), t_s=0.8).label is SystemLabel.UNVERIFIED_ANOMALY
    decision = engine.update(leak_evidence(), t_s=1.0)
    assert decision.rule_id == "R2"
    assert decision.label is SystemLabel.CONFIRMED_LEAK


def test_high_band_deescalates_only_after_five_quiet_ticks(cfg) -> None:
    engine = HysteresisDecision(cfg)
    engine.update(leak_evidence(), t_s=0.0)
    engine.update(leak_evidence(), t_s=0.2)

    for tick in range(4):
        decision = engine.update(CALM, t_s=0.4 + tick * 0.2)
        assert decision.label is SystemLabel.UNVERIFIED_ANOMALY, tick
        assert engine.drop_streak == tick + 1

    decision = engine.update(CALM, t_s=1.2)
    assert decision.rule_id == "R7"
    assert decision.label is SystemLabel.NOMINAL


def test_twelve_tick_deescalation_sequence(cfg) -> None:
    """R3 holds through four R5 ticks, drops on the fifth, then re-escalates.

    Tick  1-2  R3 at 0.750
    Tick  3-7  R5 evidence at 0.550, held at R3 for four ticks, R5 on the fifth
    Tick    8  R3 again: a higher band is adopted immediately
    Tick 9-12  R5 evidence, held at R3 (drop count 1..4)
    """

    engine = HysteresisDecision(cfg)
    script = [R3_075] * 2 + [R5_055] * 5 + [R3_075] + [R5_055] * 4
    rules = [engine.update(ev, tick * 0.2).rule_id for tick, ev in enumerate(script)]

    assert rules == [
        "R3",
        "R3",
        "R3",
        "R3",
        "R3",
        "R3",
        "R5",
        "R3",
        "R3",
        "R3",
        "R3",
        "R3",
    ]


def test_deescalation_boundary_is_strictly_below_the_margin(cfg, monkeypatch) -> None:
    """At exactly floor - margin the state holds; one tick below it counts."""

    engine = HysteresisDecision(cfg)

    monkeypatch.setattr("cryoguard.fusion.decision.leak_score", lambda *args, **kwargs: 0.75)
    engine.update(R3_075, t_s=0.0)

    monkeypatch.setattr("cryoguard.fusion.decision.leak_score", lambda *args, **kwargs: 0.65)
    for tick in range(6):
        decision = engine.update(R5_065, t_s=0.2 + tick * 0.2)
        assert decision.rule_id == "R3", tick
        assert engine.drop_streak == 0

    monkeypatch.setattr("cryoguard.fusion.decision.leak_score", lambda *args, **kwargs: 0.6499)
    for tick in range(4):
        assert engine.update(R6_064, t_s=1.4 + tick * 0.2).rule_id == "R3"
    assert engine.update(R6_064, t_s=2.2).rule_id == "R5"


def test_camera_failure_and_recovery_are_not_held(cfg) -> None:
    engine = HysteresisDecision(cfg)
    engine.update(R3_075, t_s=0.0)

    blind = engine.update(scored_evidence(phi_v=1.0, W=0.82, camera_healthy=False), t_s=0.2)
    assert blind.rule_id == "R8"
    assert blind.label is SystemLabel.CAMERA_DEGRADED

    recovered = engine.update(CALM, t_s=0.4)
    assert recovered.rule_id == "R7"
    assert recovered.label is SystemLabel.NOMINAL


def test_high_band_survives_the_rule_refining(cfg) -> None:
    """R2b -> R2 is the same evidence becoming complete, not a new streak."""

    engine = HysteresisDecision(cfg)
    engine.update(leak_evidence(q=0.2), t_s=0.0)  # quality gate fails: R2b
    engine.update(leak_evidence(q=0.2), t_s=0.2)
    decision = engine.update(leak_evidence(q=1.0), t_s=0.4)  # gate now passes

    assert decision.rule_id == "R2"


def test_confirmed_leak_latches_and_never_walks_back(cfg) -> None:
    engine = HysteresisDecision(cfg)
    for tick in range(3):
        decision = engine.update(leak_evidence(), t_s=tick * 0.2)
    assert decision.label is SystemLabel.CONFIRMED_LEAK

    calm = engine.update(leak_evidence(phi_v=0.0, phi_h=0.0, W=0.0, dA_dt=0.0, q=1.0), t_s=1.0)
    assert calm.label is SystemLabel.CONFIRMED_LEAK
    assert engine.latched is not None

    engine.reset()
    assert engine.update(leak_evidence(phi_v=0.0, phi_h=0.0, W=0.0, dA_dt=0.0), 1.2).label is (
        SystemLabel.NOMINAL
    )


def test_emergency_trip_latches(cfg) -> None:
    engine = HysteresisDecision(cfg)
    for tick in range(3):
        decision = engine.update(leak_evidence(zone=2), t_s=tick * 0.2)
    assert decision.label is SystemLabel.EMERGENCY_TRIP

    calm = engine.update(leak_evidence(zone=2, phi_v=0.0, phi_h=0.0, W=0.0, dA_dt=0.0), 1.0)
    assert calm.label is SystemLabel.EMERGENCY_TRIP
