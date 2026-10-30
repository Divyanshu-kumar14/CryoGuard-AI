"""Incident schema: camelCase, UTC with milliseconds, and a lossless round trip."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from cryoguard.report.fallback import build_fallback_report
from cryoguard.report.schema import ActionModel, EvidenceModel, IncidentReport
from cryoguard.report.writer import INCIDENT_ID_PATTERN

TS = datetime(2026, 10, 30, 0, 0, 7, 800_000, tzinfo=UTC)


def make_report() -> IncidentReport:
    evidence = EvidenceModel(
        label="CONFIRMED_LEAK",
        zone=0,
        qd_origin="QD-04",
        p_leak=0.87,
        phi_visual=0.66,
        phi_hydraulic=0.42,
        phi_thermal=1.0,
        wetness=1.0,
        area_mm2=739.0,
        growth_mm2_per_s=91.25,
        mask_quality=1.0,
        pressure_psi=32.66,
        pressure_rate_psi_per_s=0.3,
        flow_imbalance_lpm=0.39,
        surface_c=41.61,
        dew_point_c=11.32,
    )
    actions = [
        ActionModel(seq=1, t_s=7.84, actor="NVML", command="CLAMP", status="READBACK_VERIFIED"),
        ActionModel(
            seq=2, t_s=8.79, actor="PLC", command="VALVE_CLOSE", status="LIMIT_SWITCH_CLOSED"
        ),
    ]
    return build_fallback_report(
        evidence=evidence,
        actions=actions,
        incident_time=TS,
        rack_id="RACK-08B",
        run_index=1,
        config_hash="deadbeefdeadbeef",
        clip_id="slow_leak_qd4",
        frame_range=(0, 251),
        fixture_source="synthetic",
    )


def test_json_uses_camel_case_and_a_z_timestamp() -> None:
    payload = json.loads(make_report().model_dump_json(by_alias=True))

    assert payload["incidentId"] == "INC-RACK-08B-20261030-01"
    assert payload["timestamp"] == "2026-10-30T00:00:07.800Z"
    assert payload["physicalEvidence"]["pLeak"] == pytest.approx(0.87)
    assert payload["deterministicActionsTaken"][1]["command"] == "VALVE_CLOSE"


def test_report_round_trips_from_json() -> None:
    original = make_report()
    restored = IncidentReport.model_validate_json(original.model_dump_json(by_alias=True))
    assert restored == original


def test_incident_id_matches_the_schema_pattern() -> None:
    assert INCIDENT_ID_PATTERN.match(make_report().incident_id)


def test_fallback_is_labelled_and_never_claims_calibration() -> None:
    report = make_report()
    assert report.report_source == "deterministic_fallback"
    assert report.simulated is True
    assert "calibrated" not in report.recommended_operator_action.lower()


def test_suspected_origin_is_specific() -> None:
    assert make_report().suspected_origin == "QD-04 (Zone 0)"
