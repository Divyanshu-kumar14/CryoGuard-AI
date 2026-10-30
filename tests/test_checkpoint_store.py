"""Checkpoint verification must fail closed on a single flipped byte."""

from __future__ import annotations

from pathlib import Path

import pytest

from cryoguard.sims.checkpoint_store import (
    CheckpointError,
    list_checkpoints,
    read_manifest,
    verify_checkpoint,
    write_checkpoint,
)


def test_a_written_checkpoint_verifies(tmp_path: Path) -> None:
    checkpoint = write_checkpoint(tmp_path, 42, b"weights: deterministic\n")

    assert checkpoint.payload_path.is_file()
    assert checkpoint.manifest_path.is_file()
    assert verify_checkpoint(checkpoint.manifest_path) is True


def test_one_flipped_byte_fails_verification(tmp_path: Path) -> None:
    checkpoint = write_checkpoint(tmp_path, 7, b"0123456789abcdef")
    payload = bytearray(checkpoint.payload_path.read_bytes())
    payload[3] ^= 0x01
    checkpoint.payload_path.write_bytes(bytes(payload))

    assert verify_checkpoint(checkpoint.manifest_path) is False


def test_a_truncated_payload_fails_verification(tmp_path: Path) -> None:
    checkpoint = write_checkpoint(tmp_path, 8, b"0123456789abcdef")
    checkpoint.payload_path.write_bytes(b"0123")

    assert verify_checkpoint(checkpoint.manifest_path) is False


def test_a_missing_payload_fails_verification(tmp_path: Path) -> None:
    checkpoint = write_checkpoint(tmp_path, 9, b"payload")
    checkpoint.payload_path.unlink()

    assert verify_checkpoint(checkpoint.manifest_path) is False


def test_a_manifest_without_a_hash_fails_verification(tmp_path: Path) -> None:
    checkpoint = write_checkpoint(tmp_path, 10, b"payload")
    checkpoint.manifest_path.write_text('{"file": "step_000010.bin", "size": 7}\n')

    assert verify_checkpoint(checkpoint.manifest_path) is False


def test_manifests_are_deterministic(tmp_path: Path) -> None:
    first = write_checkpoint(tmp_path / "a", 11, b"same bytes")
    second = write_checkpoint(tmp_path / "b", 11, b"same bytes")

    assert first.sha256 == second.sha256
    assert first.manifest_path.read_bytes() == second.manifest_path.read_bytes()


def test_listing_returns_manifests_in_step_order(tmp_path: Path) -> None:
    for step in (20, 5, 10):
        write_checkpoint(tmp_path, step, f"step {step}".encode())
    assert [path.name for path in list_checkpoints(tmp_path)] == [
        "step_000005.json",
        "step_000010.json",
        "step_000020.json",
    ]


def test_writing_an_empty_checkpoint_fails_loudly(tmp_path: Path) -> None:
    with pytest.raises(CheckpointError, match="must not be empty"):
        write_checkpoint(tmp_path, 1, b"")


def test_a_missing_manifest_verifies_as_false(tmp_path: Path) -> None:
    assert verify_checkpoint(tmp_path / "absent.json") is False
    with pytest.raises(CheckpointError, match="manifest not found"):
        read_manifest(tmp_path / "absent.json")
