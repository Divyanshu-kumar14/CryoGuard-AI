"""Incident report schema (FR-REP-02).

Field names are camelCase in JSON to match the PRD, snake_case in Python. The
timestamp is always UTC with a trailing ``Z``; a report that says ``+00:00`` would
fail the schema test the organisers can run.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_serializer
from pydantic.alias_generators import to_camel


class SchemaModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        extra="forbid",
    )


class ActionModel(SchemaModel):
    seq: int = Field(ge=1)
    t_s: float = Field(ge=0)
    actor: str
    command: str
    status: str
    detail: str = ""


class EvidenceModel(SchemaModel):
    label: str
    zone: int | None
    qd_origin: str
    p_leak: float = Field(ge=0.0, le=1.0)
    phi_visual: float
    phi_hydraulic: float
    phi_thermal: float
    wetness: float
    area_mm2: float = Field(ge=0.0)
    growth_mm2_per_s: float
    mask_quality: float = Field(ge=0.0, le=1.0)
    pressure_psi: float
    pressure_rate_psi_per_s: float
    flow_imbalance_lpm: float
    surface_c: float
    dew_point_c: float


class IncidentReport(SchemaModel):
    incident_id: str
    timestamp: datetime
    suspected_origin: str
    physical_evidence: EvidenceModel
    deterministic_actions_taken: list[ActionModel]
    recommended_operator_action: str
    report_source: Literal["llm", "deterministic_fallback"]
    config_hash: str
    clip_id: str
    frame_range: tuple[int, int]
    fixture_source: Literal["synthetic", "capture"]
    simulated: bool

    @field_serializer("timestamp")
    def _iso_z(self, value: datetime) -> str:
        return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
