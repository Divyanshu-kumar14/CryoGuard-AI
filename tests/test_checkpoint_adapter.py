"""Checkpoint selection: newest verified, N-1 on corruption, fail closed if none."""

from __future__ import annotations

from pathlib import Path

import pytest

from cryoguard.adapters.scheduler import (
    CheckpointAdapter,
    SchedulerAdapter,
    select_verified_checkpoint,
)
from cryoguard.config import load
from cryoguard.orchestrator.adapters import ScenarioClock
from cryoguard.sims.checkpoint_store import (
    CheckpointError,
    list_checkpoints,
    verify_checkpoint,
    write_checkpoint,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def cfg() -> dict:
    return load(ROOT / "configs" / "default.yaml")


def test_an_empty_directory_gets_a_deterministic_verified_checkpoint(cfg, tmp_path) -> None:
    directory = tmp_path / "checkpoints"
    adapter = CheckpointAdapter(
        ScenarioClock(7.0),
        directory,
        step=4200,
        scheduler_cfg=cfg["scheduler"],
    )

    result = adapter.start()

    assert result.step == 4200
    assert result.started_at == 7.0
    assert result.ready_at == 7.0
    assert result.reason == "step 4200 hash-verified"
    manifests = list_checkpoints(directory)
    assert len(manifests) == 1
    assert verify_checkpoint(manifests[0])

    payload = (directory / "step_004200.bin").read_bytes()
    CheckpointAdapter(
        ScenarioClock(7.0),
        directory,
        step=4200,
        scheduler_cfg=cfg["scheduler"],
    ).start()
    assert (directory / "step_004200.bin").read_bytes() == payload


def test_a_corrupt_newest_checkpoint_resumes_the_previous_step(cfg, tmp_path) -> None:
    directory = tmp_path / "checkpoints"
    write_checkpoint(directory, 10, b"step-ten-payload")
    write_checkpoint(directory, 20, b"step-twenty-payload")
    (directory / "step_000020.bin").write_bytes(b"tampered")

    choice = select_verified_checkpoint(directory)
    assert choice is not None
    assert choice.step == 10
    assert "step_000020.json failed SHA-256" in choice.reason
    assert "resumed step 10" in choice.reason

    adapter = CheckpointAdapter(
        ScenarioClock(0.0),
        directory,
        step=4200,
        scheduler_cfg=cfg["scheduler"],
    )
    result = adapter.start()
    assert result.step == 10
    assert "resumed step 10" in result.reason
    assert adapter.last_choice is not None


def test_all_corrupt_checkpoints_fail_closed(cfg, tmp_path) -> None:
    directory = tmp_path / "checkpoints"
    write_checkpoint(directory, 10, b"step-ten-payload")
    write_checkpoint(directory, 20, b"step-twenty-payload")
    (directory / "step_000010.bin").write_bytes(b"bad")
    (directory / "step_000020.bin").write_bytes(b"bad")

    with pytest.raises(CheckpointError, match="no hash-verified checkpoint"):
        select_verified_checkpoint(directory)

    adapter = CheckpointAdapter(
        ScenarioClock(0.0),
        directory,
        step=4200,
        scheduler_cfg=cfg["scheduler"],
    )
    with pytest.raises(CheckpointError, match="no hash-verified checkpoint"):
        adapter.start()


def test_requeue_lands_on_the_standby_node_and_records_the_step(cfg) -> None:
    scheduler = SchedulerAdapter(cfg["scheduler"])

    assert scheduler.drain() == "DRAINED"
    node = scheduler.requeue(10)

    assert node == "gpu-node-09a"
    assert scheduler.last_checkpoint_step == 10
    describe = scheduler.describe()
    assert describe["state"] == "RUNNING"
    assert describe["node"] == "gpu-node-09a"
    assert [transition["state"] for transition in describe["transitions"]] == [
        "RUNNING",
        "DRAINING",
        "DRAINED",
        "REQUEUED",
        "RUNNING",
    ]
