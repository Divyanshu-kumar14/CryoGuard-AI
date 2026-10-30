"""Scripted telemetry: interpolation, derivative, and fail-loudly behaviour."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from cryoguard.telemetry.replay import TelemetryError, TelemetryReplay

ROOT = Path(__file__).resolve().parents[1]


def load(name: str) -> dict:
    return yaml.safe_load((ROOT / "configs" / "scenarios" / f"{name}.yaml").read_text())


def test_pressure_interpolates_between_keyframes() -> None:
    telemetry = TelemetryReplay(load("slow_leak_qd4"))
    assert telemetry.pressure_at(0.0) == pytest.approx(34.0)
    assert telemetry.pressure_at(4.0) == pytest.approx(33.8)
    assert telemetry.pressure_at(6.0) == pytest.approx(33.2)
    assert telemetry.pressure_at(100.0) == pytest.approx(29.4)  # clamped at the end


def test_pressure_rate_is_positive_on_a_drop_and_zero_at_t0() -> None:
    telemetry = TelemetryReplay(load("slow_leak_qd4"))
    assert telemetry.sample(0.0).pressure_rate_psi_per_s == 0.0
    assert telemetry.sample(6.0).pressure_rate_psi_per_s == pytest.approx(0.3)
    assert telemetry.sample(10.0).pressure_rate_psi_per_s == pytest.approx(0.425)


def test_flow_imbalance_tracks_supply_minus_return() -> None:
    telemetry = TelemetryReplay(load("slow_leak_qd4"))
    assert telemetry.sample(0.0).flow_imbalance_lpm == pytest.approx(0.0)
    assert telemetry.sample(8.0).flow_imbalance_lpm == pytest.approx(0.4)
    assert telemetry.sample(16.0).flow_imbalance_lpm == pytest.approx(0.9)


def test_condensation_scenario_is_physically_below_the_dew_point() -> None:
    sample = TelemetryReplay(load("condensation_bead")).sample(0.0)
    assert sample.dew_margin_c < 0.0
    assert sample.surface_c < sample.dew_point_c


def test_malformed_channel_fails_loudly() -> None:
    bad = {
        "channels": {
            "pressure": [[0, 34.0]],
            "flow": [[0, 9.0, 9.0], [1, 9.0, 9.0]],
            "surface_temp": [[0, 40.0], [1, 40.0]],
            "ambient": [[0, 24.0, 45.0], [1, 24.0, 45.0]],
        }
    }
    with pytest.raises(TelemetryError, match="at least two keyframes"):
        TelemetryReplay(bad)


def test_non_monotonic_time_fails_loudly() -> None:
    bad = {
        "channels": {
            "pressure": [[0, 34.0], [4, 33.0], [2, 30.0]],
            "flow": [[0, 9.0, 9.0], [4, 9.0, 9.0]],
            "surface_temp": [[0, 40.0], [4, 40.0]],
            "ambient": [[0, 24.0, 45.0], [4, 24.0, 45.0]],
        }
    }
    with pytest.raises(TelemetryError, match="strictly increasing"):
        TelemetryReplay(bad)


def test_series_is_time_aligned_to_the_video_clock() -> None:
    telemetry = TelemetryReplay(load("slow_leak_qd4"))
    series = telemetry.iter_series(duration_s=2.0, dt_s=0.2)

    assert len(series) == 11
    assert [sample.t_s for sample in series] == pytest.approx([index * 0.2 for index in range(11)])
    assert series[5].pressure_psi == pytest.approx(telemetry.pressure_at(1.0))


def test_series_hash_is_identical_across_replays() -> None:
    first = TelemetryReplay(load("slow_leak_qd4"))
    second = TelemetryReplay(load("slow_leak_qd4"))
    assert first.series_hash(20.0, 0.2) == second.series_hash(20.0, 0.2)


def test_a_different_grid_changes_the_hash() -> None:
    telemetry = TelemetryReplay(load("slow_leak_qd4"))
    assert telemetry.series_hash(20.0, 0.2) != telemetry.series_hash(20.0, 0.1)


@pytest.mark.parametrize("duration, dt", [(-1.0, 0.2), (1.0, 0.0)])
def test_an_invalid_series_grid_fails_loudly(duration: float, dt: float) -> None:
    with pytest.raises(TelemetryError):
        TelemetryReplay(load("slow_leak_qd4")).iter_series(duration, dt)
