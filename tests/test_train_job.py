"""Training-job simulator: deterministic checkpoints the incident can verify."""

from __future__ import annotations

from pathlib import Path

import pytest

from cryoguard.config import load
from cryoguard.sims.checkpoint_store import verify_checkpoint
from cryoguard.sims.train_job import TrainJob, TrainJobError, loss_at

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def cfg() -> dict:
    return load(ROOT / "configs" / "default.yaml")["scheduler"]


def test_thirty_seconds_emits_four_checkpoints(cfg, tmp_path: Path) -> None:
    job = TrainJob(cfg, tmp_path, duration_s=30.0, interval_s=1.0, checkpoint_every_s=10.0)
    checkpoints = job.run()

    assert [checkpoint.step for checkpoint in checkpoints] == [0, 10, 20, 30]
    assert all(verify_checkpoint(checkpoint.manifest_path) for checkpoint in checkpoints)


def test_checkpoint_payloads_are_byte_identical_across_runs(cfg, tmp_path: Path) -> None:
    first = TrainJob(cfg, tmp_path / "a", duration_s=20.0, interval_s=1.0).run()
    second = TrainJob(cfg, tmp_path / "b", duration_s=20.0, interval_s=1.0).run()

    assert [item.payload_path.read_bytes() for item in first] == [
        item.payload_path.read_bytes() for item in second
    ]


def test_the_loss_curve_is_a_fixed_function_of_the_step() -> None:
    assert loss_at(0) > loss_at(10) > loss_at(20)
    assert loss_at(10) == pytest.approx(loss_at(10))


def test_an_impossible_schedule_fails_loudly(cfg, tmp_path: Path) -> None:
    with pytest.raises(TrainJobError, match="at least one training interval"):
        TrainJob(cfg, tmp_path, interval_s=2.0, checkpoint_every_s=1.0)

    with pytest.raises(TrainJobError, match="interval must be positive"):
        TrainJob(cfg, tmp_path, interval_s=0.0)
