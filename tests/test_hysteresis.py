"""Temporal gating: three ticks to confirm, and a reset when the pattern changes."""

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


def test_interrupted_pattern_restarts_the_count(cfg) -> None:
    engine = HysteresisDecision(cfg)
    engine.update(leak_evidence(), t_s=0.0)
    engine.update(leak_evidence(), t_s=0.2)
    calm = engine.update(leak_evidence(phi_v=0.0, phi_h=0.0, W=0.0, dA_dt=0.0, q=0.0), t_s=0.4)
    assert calm.label is SystemLabel.NOMINAL

    first = engine.update(leak_evidence(), t_s=0.6)
    assert first.label is SystemLabel.UNVERIFIED_ANOMALY


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
