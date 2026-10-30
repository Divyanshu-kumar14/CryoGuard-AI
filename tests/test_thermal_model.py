"""Lumped thermal model: the PRD's two numbers, and the guards around them."""

from __future__ import annotations

import pytest

from cryoguard.config import load
from cryoguard.sims.thermal_model import (
    ThermalError,
    ThermalModel,
    seconds_to_delta,
)


@pytest.fixture(scope="module")
def cfg() -> dict:
    return load("configs/default.yaml")["thermal_model"]


def test_forty_kelvin_at_two_hundred_watts_takes_seventy_seconds() -> None:
    assert seconds_to_delta(40.0, 200.0, 350.0) == pytest.approx(70.0)


def test_forty_kelvin_at_one_kilowatt_takes_fourteen_seconds() -> None:
    assert seconds_to_delta(40.0, 1000.0, 350.0) == pytest.approx(14.0)


def test_integration_reaches_the_watchdog_within_one_percent(cfg) -> None:
    for power_w, expected_s in ((200.0, 70.0), (1000.0, 14.0)):
        model = ThermalModel(cfg, ambient_c=50.0)
        sample = model.run_to_watchdog(power_w, dt_s=0.05)
        assert sample.temp_c >= model.watchdog_limit_c
        assert sample.t_s == pytest.approx(expected_s, rel=0.01)


def test_closed_form_matches_integration(cfg) -> None:
    model = ThermalModel(cfg, ambient_c=50.0)
    assert model.time_to_watchdog_s(200.0) == pytest.approx(70.0)
    assert model.time_to_watchdog_s(1000.0) == pytest.approx(14.0)


def test_no_power_means_no_heating(cfg) -> None:
    model = ThermalModel(cfg, ambient_c=25.0)
    sample = model.step(10.0, 0.0)
    assert sample.temp_c == pytest.approx(25.0)


def test_watchdog_must_sit_below_the_junction_limit(cfg) -> None:
    bad = dict(cfg, watchdog_limit_c=cfg["t_junction_max_c"])
    with pytest.raises(ThermalError, match="below the junction limit"):
        ThermalModel(bad)


@pytest.mark.parametrize("power", [0.0, -1.0])
def test_seconds_to_delta_rejects_non_positive_power(power: float) -> None:
    with pytest.raises(ThermalError, match="power"):
        seconds_to_delta(40.0, power, 350.0)


def test_step_rejects_a_non_positive_dt(cfg) -> None:
    with pytest.raises(ThermalError, match="dt"):
        ThermalModel(cfg).step(0.0, 200.0)
