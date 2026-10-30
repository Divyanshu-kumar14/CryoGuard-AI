"""Slurm mock (P4-T06, D-14).

One job on four GPUs, drained to a reversible cordon and requeued on the standby
node. Kubernetes is an interface stub only and is deliberately not implemented.

The transitions are recorded in order because the incident log quotes them.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class SlurmError(RuntimeError):
    """An invalid scheduler transition."""


class JobState(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    DRAINING = "DRAINING"
    DRAINED = "DRAINED"
    REQUEUED = "REQUEUED"


@dataclass(frozen=True)
class Transition:
    state: JobState
    node: str


class SlurmMock:
    def __init__(self, cfg: dict[str, Any]) -> None:
        self.job_id = str(cfg["job_id"])
        self.node = str(cfg["node"])
        self.standby_node = str(cfg["standby_node"])
        self.gpus = int(cfg["gpus"])
        if self.gpus <= 0:
            raise SlurmError(f"gpus must be positive: {self.gpus}")

        self.state = JobState.RUNNING
        self.transitions: list[Transition] = [Transition(JobState.RUNNING, self.node)]

    def drain(self) -> None:
        """Reversible scheduling cordon. This is not a coolant valve (D-11)."""

        if self.state is not JobState.RUNNING:
            raise SlurmError(f"cannot drain from {self.state}")
        self.state = JobState.DRAINING
        self.transitions.append(Transition(self.state, self.node))
        self.state = JobState.DRAINED
        self.transitions.append(Transition(self.state, self.node))

    def requeue(self) -> None:
        if self.state is not JobState.DRAINED:
            raise SlurmError(f"cannot requeue from {self.state}")
        self.state = JobState.REQUEUED
        self.transitions.append(Transition(self.state, self.standby_node))
        self.node = self.standby_node
        self.state = JobState.RUNNING
        self.transitions.append(Transition(self.state, self.node))

    def describe(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "state": self.state.value,
            "node": self.node,
            "gpus": self.gpus,
            "standby_node": self.standby_node,
            "simulated": True,
            "transitions": [
                {"state": transition.state.value, "node": transition.node}
                for transition in self.transitions
            ],
        }
