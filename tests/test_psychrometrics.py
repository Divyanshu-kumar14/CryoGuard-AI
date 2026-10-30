"""Psychrometrics. Wrong dew points turn condensation into a confirmed leak."""

from __future__ import annotations

import pytest

from cryoguard.fusion.features import thermal_phi
from cryoguard.telemetry.psychrometrics import dew_margin_c, dew_point_c


def test_dew_point_matches_the_published_magnus_value() -> None:
    # 25 C at 50 % RH -> 13.9 C. Reference value from the Magnus-Tetens tables.
    assert dew_point_c(25.0, 50.0) == pytest.approx(13.9, abs=0.1)


def test_saturated_air_dew_point_equals_air_temperature() -> None:
    assert dew_point_c(25.0, 100.0) == pytest.approx(25.0, abs=1e-6)


def test_dew_point_rises_with_humidity() -> None:
    values = [dew_point_c(24.0, rh) for rh in (20.0, 40.0, 60.0, 80.0)]
    assert values == sorted(values)
    assert len(set(values)) == len(values)


@pytest.mark.parametrize("rh", [0.0, -1.0, 101.0])
def test_relative_humidity_outside_its_domain_fails_loudly(rh: float) -> None:
    with pytest.raises(ValueError, match="relative humidity"):
        dew_point_c(25.0, rh)


@pytest.mark.parametrize("temp", [-60.0, 70.0])
def test_temperature_outside_the_magnus_domain_fails_loudly(temp: float) -> None:
    with pytest.raises(ValueError, match="temperature"):
        dew_point_c(temp, 50.0)


def test_scenario_two_is_below_the_dew_point_by_design(cfg, scenario_path) -> None:
    """The condensation clip must actually be condensing, not just labelled so."""

    scenario = scenario_path("condensation_bead")
    surface_c = scenario["channels"]["surface_temp"][0][1]
    air_c, rh = scenario["channels"]["ambient"][0][1:]

    margin = dew_margin_c(surface_c, air_c, rh)
    assert margin < 0.0
    assert thermal_phi(margin, cfg["fusion"]["thermal"]) < 0.0


def test_dry_scenario_has_positive_margin_but_adds_no_leak_evidence(cfg, scenario_path) -> None:
    scenario = scenario_path("glint_noise")
    surface_c = scenario["channels"]["surface_temp"][0][1]
    air_c, rh = scenario["channels"]["ambient"][0][1:]

    margin = dew_margin_c(surface_c, air_c, rh)
    assert margin > 0.0
    assert thermal_phi(margin, cfg["fusion"]["thermal"]) > 0.0
