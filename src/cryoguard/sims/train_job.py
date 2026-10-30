"""Deterministic training-job simulator (P4-T08).

Emits checkpoints so the incident has something real to verify. The loss curve is
a fixed function of the step; no RNG, no wall clock in the payload, so two runs
produce byte-identical checkpoints.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Any

from cryoguard.sims.checkpoint_store import Checkpoint, write_checkpoint


class TrainJobError(ValueError):
    """An unusable training schedule."""


def loss_at(step: int) -> float:
    return 2.5 * math.exp(-step / 20.0) + 0.08


class TrainJob:
    def __init__(
        self,
        cfg: dict[str, Any],
        out_dir: str | Path,
        *,
        duration_s: float | None = None,
        interval_s: float | None = None,
        checkpoint_every_s: float | None = None,
    ) -> None:
        self.job_id = str(cfg["job_id"])
        self.gpus = int(cfg["gpus"])
        self.interval_s = float(interval_s if interval_s is not None else cfg["train_interval_s"])
        self.checkpoint_every_s = float(
            checkpoint_every_s if checkpoint_every_s is not None else cfg["checkpoint_every_s"]
        )
        self.duration_s = float(duration_s if duration_s is not None else 30.0)
        self.out_dir = Path(out_dir)
        if self.interval_s <= 0:
            raise TrainJobError(f"interval must be positive: {self.interval_s}")
        if self.checkpoint_every_s <= 0:
            raise TrainJobError(f"checkpoint interval must be positive: {self.checkpoint_every_s}")
        if self.duration_s < 0:
            raise TrainJobError(f"duration must be non-negative: {self.duration_s}")
        if self.checkpoint_every_s < self.interval_s:
            raise TrainJobError(
                "checkpoint interval must be at least one training interval: "
                f"{self.checkpoint_every_s} < {self.interval_s}"
            )
        self._steps_per_checkpoint = max(1, int(round(self.checkpoint_every_s / self.interval_s)))

    def payload(self, step: int) -> bytes:
        return (
            f"job={self.job_id}\nstep={step:06d}\ngpus={self.gpus}\nloss={loss_at(step):.6f}\n"
        ).encode()

    def steps(self) -> list[int]:
        count = int(math.floor(self.duration_s / self.interval_s + 1e-9))
        return [index for index in range(count + 1)]

    def run(self, *, realtime: bool = False) -> list[Checkpoint]:
        checkpoints: list[Checkpoint] = []
        for index in self.steps():
            if index % self._steps_per_checkpoint == 0:
                checkpoints.append(write_checkpoint(self.out_dir, index, self.payload(index)))
            if realtime:
                time.sleep(self.interval_s)
        return checkpoints


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SIMULATED training job")
    parser.add_argument("--out", default="artifacts/checkpoints")
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--interval", type=float, default=None)
    parser.add_argument("--checkpoint-every", type=float, default=None)
    parser.add_argument("--realtime", action="store_true")
    args = parser.parse_args(argv)

    from cryoguard.config import load

    root = Path(__file__).resolve().parents[3]
    cfg = load(root / "configs" / "default.yaml")["scheduler"]
    job = TrainJob(
        cfg,
        root / args.out,
        duration_s=args.duration,
        interval_s=args.interval,
        checkpoint_every_s=args.checkpoint_every,
    )
    checkpoints = job.run(realtime=args.realtime)
    print(f"SIMULATED train job {job.job_id}: {len(checkpoints)} checkpoints in {job.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
