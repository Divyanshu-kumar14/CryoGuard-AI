"""P6-T07 end to end: a hot GPU trips the watchdog although vision sees glint."""

from __future__ import annotations

import asyncio
import copy
from pathlib import Path

import pytest
import yaml

from cryoguard.config import load
from cryoguard.models import IncidentState
from cryoguard.pipeline import run_scenario

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def cfg() -> dict:
    return load(ROOT / "configs" / "default.yaml")


@pytest.fixture(scope="module")
def topology() -> dict:
    return yaml.safe_load((ROOT / "configs" / "topology.yaml").read_text(encoding="utf-8"))


@pytest.mark.slow
def test_a_hot_junction_trips_an_emergency_incident_while_the_clip_is_benign(
    cfg, topology, tmp_path
) -> None:
    hot = copy.deepcopy(cfg)
    hot["nvml"]["fake_junction_c"] = 96.0
    hot["watchdog"]["period_ms"] = 0  # the scenario is seconds of simulated time

    result = asyncio.run(
        run_scenario(
            "glint",
            hot,
            topology,
            out_dir=tmp_path,
            max_seconds=3.0,
            protocol_mode="memory",
        )
    )

    assert result.decision is not None
    assert result.report is not None, "a hardware trip must file an incident"
    report = result.report
    assert report.physical_evidence.label == "EMERGENCY_TRIP"
    assert report.trip_reason == "JUNCTION_OVER_TEMP"
    assert report.physical_evidence.junction_c == 96.0
    assert report.physical_evidence.p_leak < 0.40  # vision saw nothing; the trip is thermal
    assert [action.command for action in report.deterministic_actions_taken] == [
        "VALVE_CLOSE",
        "SCHEDULER_DRAIN",
        "FORCE_OFF",
    ]
    assert result.sequence_state is IncidentState.EMERGENCY_TRIP
    assert "CHECKPOINT_VERIFY" not in [a.command for a in report.deterministic_actions_taken]
    assert "REQUEUE" not in [a.command for a in report.deterministic_actions_taken]


@pytest.mark.slow
def test_a_cool_junction_leaves_the_benign_clip_alone(cfg, topology, tmp_path) -> None:
    cool = copy.deepcopy(cfg)
    cool["watchdog"]["period_ms"] = 0

    result = asyncio.run(
        run_scenario(
            "glint",
            cool,
            topology,
            out_dir=tmp_path,
            max_seconds=3.0,
            protocol_mode="memory",
        )
    )

    assert result.report is None


@pytest.mark.slow
def test_a_trip_is_filed_even_when_the_socket_sequence_outlives_the_clip(
    cfg, topology, tmp_path
) -> None:
    """The clip ends in ~0.1 s; the Modbus/Redfish sequence takes ~1 s."""

    hot = copy.deepcopy(cfg)
    hot["nvml"]["fake_junction_c"] = 96.4
    hot["watchdog"]["period_ms"] = 0

    result = asyncio.run(
        run_scenario(
            "glint",
            hot,
            topology,
            out_dir=tmp_path,
            max_seconds=3.0,
            protocol_mode="loopback",
        )
    )

    assert result.path is not None
    assert result.report is not None
    assert result.report.report_source == "deterministic_fallback"
    assert [a.command for a in result.report.deterministic_actions_taken] == [
        "VALVE_CLOSE",
        "SCHEDULER_DRAIN",
        "FORCE_OFF",
    ]
    assert "junction=96.4 C >= limit 90.0 C" in result.report.deterministic_actions_taken[-1].detail
