"""Slurm mock: drain, requeue, and the standby node."""

from __future__ import annotations

from pathlib import Path

import pytest

from cryoguard.config import load
from cryoguard.sims.slurm_mock import JobState, SlurmError, SlurmMock

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def cfg() -> dict:
    return load(ROOT / "configs" / "default.yaml")["scheduler"]


def test_job_starts_running_on_four_gpus(cfg) -> None:
    slurm = SlurmMock(cfg)
    assert slurm.state is JobState.RUNNING
    assert slurm.node == "gpu-node-08b"
    assert slurm.gpus == 4


def test_drain_then_requeue_lands_on_the_standby_node(cfg) -> None:
    slurm = SlurmMock(cfg)
    slurm.drain()
    assert slurm.state is JobState.DRAINED
    assert slurm.node == "gpu-node-08b"

    slurm.requeue()
    assert slurm.state is JobState.RUNNING
    assert slurm.node == "gpu-node-09a"

    states = [transition.state for transition in slurm.transitions]
    assert states == [
        JobState.RUNNING,
        JobState.DRAINING,
        JobState.DRAINED,
        JobState.REQUEUED,
        JobState.RUNNING,
    ]


def test_drain_is_not_a_valve_action(cfg) -> None:
    describe = SlurmMock(cfg).describe()
    assert describe["simulated"] is True
    assert describe["state"] == "RUNNING"


def test_invalid_transitions_fail_loudly(cfg) -> None:
    slurm = SlurmMock(cfg)
    with pytest.raises(SlurmError, match="cannot requeue"):
        slurm.requeue()

    slurm.drain()
    with pytest.raises(SlurmError, match="cannot drain"):
        slurm.drain()
