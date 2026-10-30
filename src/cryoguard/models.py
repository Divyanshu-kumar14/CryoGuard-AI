"""Typed vocabulary shared by every subsystem (PRD §6.2, §6.5)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

RACK_ID = "RACK-08B"
BRANCH_ID = "BRANCH-SEC-02"
NODE_ID = "gpu-node-08b"
STANDBY_NODE_ID = "gpu-node-09a"
JOB_ID = "train-llm-0427"

# Commands that move hardware. ALERT, LOG, SAMPLE_10HZ, CDU_QUERY and
# SCHEDULER_DRAIN are not here: the PRD's "drain" is `scontrol ... State=DRAIN`,
# a reversible scheduling cordon, not a coolant valve.
ACTUATING_ACTIONS = frozenset({"CLAMP", "VALVE_CLOSE", "FORCE_OFF"})


class SystemLabel(StrEnum):
    """Canonical system labels. CAMERA_DEGRADED closes the blind-camera hole."""

    NOMINAL = "NOMINAL"
    LOG_ONLY = "LOG_ONLY"
    SUSPECT_LEAK = "SUSPECT_LEAK"
    BENIGN_CONDENSATION = "BENIGN_CONDENSATION"
    UNVERIFIED_ANOMALY = "UNVERIFIED_ANOMALY"
    HIDDEN_HYDRAULIC_EVENT = "HIDDEN_HYDRAULIC_EVENT"
    CONFIRMED_LEAK = "CONFIRMED_LEAK"
    EMERGENCY_TRIP = "EMERGENCY_TRIP"
    CAMERA_DEGRADED = "CAMERA_DEGRADED"


class IncidentState(StrEnum):
    OPEN = "OPEN"
    CLAMPING = "CLAMPING"
    ISOLATING = "ISOLATING"
    ISOLATED = "ISOLATED"
    NODE_POWERED_DOWN = "NODE_POWERED_DOWN"
    REQUEUED = "REQUEUED"
    RESOLVED_PENDING_REPAIR = "RESOLVED_PENDING_REPAIR"
    EMERGENCY_TRIP = "EMERGENCY_TRIP"
    HARD_TRIP = "HARD_TRIP"
    FAILED_SAFE = "FAILED_SAFE"


class ValveStatus(StrEnum):
    VERIFIED = "VERIFIED"
    VERIFIED_SLOW = "VERIFIED_SLOW"
    FAULT = "FAULT"


@dataclass(frozen=True)
class Evidence:
    """Everything the decision table is allowed to see at one tick."""

    zone: int
    phi_v: float
    phi_h: float
    phi_t: float
    W: float
    dA_dt: float
    q: float = 1.0
    camera_healthy: bool = True
    plc_fault: bool = False
    temp_c: float = 40.0
    clamp_readback_ok: bool = True
    maintenance: bool = False
    ticks: int = 3
    hydraulic_low_ms: int = 0


@dataclass(frozen=True)
class Decision:
    rule_id: str
    label: SystemLabel
    actions: tuple[str, ...] = ()
    p: float = 0.0

    @property
    def actuates(self) -> bool:
        """True only for physical commands, not for observation.

        "SAMPLE_10HZ + LOG" is a response, not an actuation. Conflating the two
        would make the project's central safety claim - that glint and
        condensation cause zero actuation - impossible to test.
        """

        return any(action in ACTUATING_ACTIONS for action in self.actions)


@dataclass
class ActionRecord:
    seq: int
    ts: float
    actor: str
    command: str
    status: str
    latency_ms: float = 0.0
    detail: str = ""


@dataclass
class Incident:
    incident_id: str = ""
    state: IncidentState = IncidentState.OPEN
    actions: list[ActionRecord] = field(default_factory=list)
    events: list[tuple[str, float]] = field(default_factory=list)
