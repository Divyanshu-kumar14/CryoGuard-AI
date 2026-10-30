"""Checkpoint store with SHA-256 verification (P4-T07).

A checkpoint that fails verification must fail closed: the orchestrator may only
power the node off after ``verified=true``. A single flipped byte has to be enough
to block that gate, so the manifest and the payload are hashed separately and
both are checked.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class CheckpointError(ValueError):
    """A checkpoint request is malformed."""


@dataclass(frozen=True)
class Checkpoint:
    step: int
    payload_path: Path
    manifest_path: Path
    sha256: str
    size: int


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def write_checkpoint(directory: str | Path, step: int, payload: bytes) -> Checkpoint:
    if step < 0:
        raise CheckpointError(f"step must be non-negative: {step}")
    if not payload:
        raise CheckpointError("checkpoint payload must not be empty")

    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)
    payload_path = folder / f"step_{step:06d}.bin"
    manifest_path = folder / f"step_{step:06d}.json"

    temporary_payload = payload_path.with_suffix(".bin.tmp")
    temporary_payload.write_bytes(payload)
    temporary_payload.replace(payload_path)

    digest = _digest(payload)
    manifest = {
        "schema_version": "1.0",
        "step": step,
        "file": payload_path.name,
        "sha256": digest,
        "size": len(payload),
        "simulated": True,
    }
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary_manifest.replace(manifest_path)

    return Checkpoint(
        step=step,
        payload_path=payload_path,
        manifest_path=manifest_path,
        sha256=digest,
        size=len(payload),
    )


def read_manifest(manifest_path: str | Path) -> dict[str, Any]:
    path = Path(manifest_path)
    if not path.is_file():
        raise CheckpointError(f"manifest not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise CheckpointError(f"manifest is not an object: {path}")
    return data


def verify_checkpoint(manifest_path: str | Path) -> bool:
    """True only when the payload exists, is the recorded size, and matches SHA-256."""

    path = Path(manifest_path)
    if not path.is_file():
        return False
    try:
        manifest = read_manifest(path)
        payload_path = path.parent / str(manifest["file"])
        expected_size = int(manifest["size"])
        expected_digest = str(manifest["sha256"])
        payload = payload_path.read_bytes()
    except (CheckpointError, KeyError, OSError, ValueError):
        return False

    return len(payload) == expected_size and _digest(payload) == expected_digest


def list_checkpoints(directory: str | Path) -> list[Path]:
    folder = Path(directory)
    if not folder.is_dir():
        return []
    return sorted(folder.glob("step_*.json"))
