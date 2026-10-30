"""Deterministic incident report, no model involved.

This is the path that must run with the LLM endpoint down (NFR-SAF-01). Every
sentence is built from measured values in the incident, so it cannot contradict
the evidence or invent a number.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from cryoguard.report.schema import ActionModel, EvidenceModel, IncidentReport
from cryoguard.report.writer import incident_id

_RECOMMENDATIONS = {
    "CONFIRMED_LEAK": (
        "Keep the branch isolated and the node powered down. Pressure-test the "
        "quick-disconnect and inspect the electrical plane before re-energising."
    ),
    "EMERGENCY_TRIP": (
        "Do not re-energise. Treat the electrical plane as wet until it has been "
        "inspected, and confirm valve closure at the limit switch."
    ),
    "HIDDEN_HYDRAULIC_EVENT": (
        "No visual evidence was available. Dispatch a technician to inspect the "
        "loop and the camera field of view before restarting the branch."
    ),
}


def build_fallback_report(
    *,
    evidence: EvidenceModel,
    actions: list[ActionModel],
    incident_time: datetime,
    rack_id: str,
    run_index: int,
    config_hash: str,
    clip_id: str,
    frame_range: tuple[int, int],
    fixture_source: str,
) -> IncidentReport:
    label = evidence.label
    zone = "unknown zone" if evidence.zone is None else f"Zone {evidence.zone}"
    origin = (
        f"{evidence.qd_origin} (no visual candidate)"
        if evidence.zone is None
        else f"{evidence.qd_origin} ({zone})"
    )

    return IncidentReport(
        incident_id=incident_id(rack_id, incident_time, run_index),
        timestamp=incident_time,
        suspected_origin=origin,
        physical_evidence=evidence,
        deterministic_actions_taken=actions,
        recommended_operator_action=_RECOMMENDATIONS.get(
            label, "Review the incident log and confirm the branch state."
        ),
        report_source="deterministic_fallback",
        config_hash=config_hash,
        clip_id=clip_id,
        frame_range=frame_range,
        fixture_source=fixture_source,  # type: ignore[arg-type]
        simulated=fixture_source == "synthetic",
    )


def scenario_clock(epoch: datetime, t_s: float) -> datetime:
    """Scenario time, not wall time: two runs of a clip produce the same stamp."""

    return epoch + timedelta(seconds=round(float(t_s), 6))
