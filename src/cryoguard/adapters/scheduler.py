"""Slurm checkpoint selection and drain/requeue (P6-T06).

The real scheduler interface is ``scontrol``; here it is the in-process
``SlurmMock``. DRAIN is a reversible cordon, not a coolant valve. REQUEUE always
resumes from the newest *hash-verified* checkpoint: if step N is in flight or
corrupt, the adapter falls back to N-1 and records why.

The checkpoint verifier is real SHA-256 verification against the manifest. When
the demo directory is empty, a deterministic SIMULATED checkpoint is written so
``make demo`` stays self-contained; its manifest carries ``simulated: true``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cryoguard.orchestrator.incident_fsm import CheckpointResult
from cryoguard.sims.checkpoint_store import (
    CheckpointError,
    list_checkpoints,
    read_manifest,
    verify_checkpoint,
    write_checkpoint,
)
from cryoguard.sims.slurm_mock import SlurmMock
from cryoguard.sims.train_job import TrainJob


@dataclass(frozen=True)
class CheckpointChoice:
    step: int
    manifest: Path
    reason: str


def select_verified_checkpoint(directory: str | Path) -> CheckpointChoice | None:
    """Newest verified checkpoint, falling back to N-1. ``None`` only if empty."""

    manifests = list_checkpoints(directory)
    if not manifests:
        return None

    rejected: list[str] = []
    for path in reversed(manifests):
        if verify_checkpoint(path):
            step = int(read_manifest(path)["step"])
            if rejected:
                reason = f"{', '.join(rejected)} failed SHA-256; resumed step {step}"
            else:
                reason = f"step {step} hash-verified"
            return CheckpointChoice(step=step, manifest=path, reason=reason)
        rejected.append(path.name)

    raise CheckpointError(
        f"no hash-verified checkpoint in {directory}: {', '.join(rejected)} failed SHA-256"
    )


class CheckpointAdapter:
    def __init__(
        self,
        clock: Any,
        directory: str | Path,
        *,
        step: int,
        scheduler_cfg: dict[str, Any],
        create_if_missing: bool = True,
    ) -> None:
        self.clock = clock
        self.directory = Path(directory)
        self.step = int(step)
        self.scheduler_cfg = scheduler_cfg
        self.create_if_missing = create_if_missing
        self.last_choice: CheckpointChoice | None = None
        if self.step < 0:
            raise CheckpointError(f"checkpoint step must be non-negative: {self.step}")

    def start(self) -> CheckpointResult:
        started = self.clock.now()
        choice = select_verified_checkpoint(self.directory)

        if choice is None:
            if not self.create_if_missing:
                raise CheckpointError(f"no checkpoint in {self.directory}")
            job = TrainJob(self.scheduler_cfg, self.directory, duration_s=0.0)
            checkpoint = write_checkpoint(self.directory, self.step, job.payload(self.step))
            if not verify_checkpoint(checkpoint.manifest_path):
                raise CheckpointError(
                    f"newly written checkpoint failed verification: {checkpoint.manifest_path}"
                )
            choice = CheckpointChoice(
                step=self.step,
                manifest=checkpoint.manifest_path,
                reason=f"step {self.step} hash-verified",
            )

        self.last_choice = choice
        return CheckpointResult(
            started_at=started,
            ready_at=self.clock.now(),
            step=choice.step,
            reason=choice.reason,
        )


class SchedulerAdapter:
    def __init__(self, cfg: dict[str, Any]) -> None:
        self.slurm = SlurmMock(cfg)
        self.last_checkpoint_step: int | None = None

    def drain(self) -> str:
        self.slurm.drain()
        return self.slurm.state.value

    def requeue(self, checkpoint_step: int) -> str:
        if checkpoint_step < 0:
            raise ValueError(f"checkpoint_step must be non-negative: {checkpoint_step}")
        self.slurm.requeue()
        self.last_checkpoint_step = checkpoint_step
        return self.slurm.node

    def describe(self) -> dict[str, Any]:
        return self.slurm.describe()
