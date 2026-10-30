"""Empirical benchmark runner (P8-T01, D-12).

Every clip in the manifest is hash-checked, run through the same pipeline with
the in-memory transports, and reduced to a deterministic row. No network call is
allowed: the socket guard turns any connection attempt into a failure, and the
Muse client is disabled for the run.

The manifest split is shown per clip; the headline metrics are computed for the
requested split *and* for every clip. The split policy is provisional (two clips
per side), so hiding the other half would be worse than showing n=2.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import socket
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from cryoguard.pipeline import RunResult, run_scenario

ROOT = Path(__file__).resolve().parents[3]
POSITIVE_LABELS = {"CONFIRMED_LEAK", "EMERGENCY_TRIP", "HIDDEN_HYDRAULIC_EVENT"}


class BenchError(RuntimeError):
    """The benchmark cannot make an honest claim."""


@dataclass(frozen=True)
class BenchRun:
    """The deterministic row plus the RunResult carrying the wall-clock samples."""

    row: ClipRun
    result: RunResult


@dataclass(frozen=True)
class ClipRun:
    clip_id: str
    scenario: str
    split: str
    path: str
    sha256: str
    ground_truth_label: str
    expected_incident: bool
    predicted_incident: bool
    label: str
    rule_id: str
    p: float
    expected_qd: str
    suspected_origin: str
    localized: bool | None
    localizable: bool
    zone: int | None
    expected_zone: int
    detection_s: float | None
    ticks: int
    frames: int
    action_count: int
    valve_verified: bool | None
    checkpoint_committed: bool | None
    sequence_state: str | None

    def as_row(self) -> dict[str, Any]:
        return {
            "clipId": self.clip_id,
            "scenario": self.scenario,
            "split": self.split,
            "groundTruthLabel": self.ground_truth_label,
            "expectedIncident": self.expected_incident,
            "predictedIncident": self.predicted_incident,
            "label": self.label,
            "ruleId": self.rule_id,
            "p": self.p,
            "expectedQd": self.expected_qd,
            "suspectedOrigin": self.suspected_origin,
            "localized": self.localized,
            "localizable": self.localizable,
            "zone": self.zone,
            "expectedZone": self.expected_zone,
            "detectionS": self.detection_s,
            "ticks": self.ticks,
            "frames": self.frames,
            "actionCount": self.action_count,
            "valveVerified": self.valve_verified,
            "checkpointCommitted": self.checkpoint_committed,
            "sequenceState": self.sequence_state,
        }


@contextmanager
def no_network() -> Iterator[None]:
    """Fail loudly if anything in the benchmark reaches a socket."""

    original_connect = socket.socket.connect
    original_create = socket.create_connection

    def refuse_connect(self: Any, address: Any, *args: Any, **kwargs: Any) -> None:
        raise BenchError(f"benchmark attempted a network connection to {address!r}")

    def refuse_create(address: Any, *args: Any, **kwargs: Any) -> None:
        raise BenchError(f"benchmark attempted a network connection to {address!r}")

    socket.socket.connect = refuse_connect  # type: ignore[method-assign]
    socket.create_connection = refuse_create  # type: ignore[assignment]
    try:
        yield
    finally:
        socket.socket.connect = original_connect  # type: ignore[method-assign]
        socket.create_connection = original_create


def load_manifest(path: str | Path) -> dict[str, Any]:
    manifest = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not isinstance(manifest.get("clips"), list):
        raise BenchError(f"manifest has no clips list: {path}")
    return manifest


def verify_clip(clip: dict[str, Any]) -> None:
    path = ROOT / str(clip["path"])
    if not path.is_file():
        raise BenchError(f"clip is missing: {path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != clip["sha256"]:
        raise BenchError(f"{path} does not match the manifest: sha256 {digest} != {clip['sha256']}")


def select_clips(manifest: dict[str, Any], split: str) -> list[dict[str, Any]]:
    clips = manifest["clips"]
    if split == "all":
        return list(clips)
    selected = [clip for clip in clips if clip.get("split") == split]
    if not selected:
        raise BenchError(f"manifest has no clips in split {split!r}")
    return selected


def _row(clip: dict[str, Any], result: RunResult) -> ClipRun:
    report = result.report
    decision = result.decision
    actions = report.deterministic_actions_taken if report is not None else []
    expected_qd = str(clip.get("qd_origin", ""))
    origin = report.suspected_origin if report is not None else ""
    zone = report.physical_evidence.zone if report is not None else None

    localizable = bool(report is not None and zone is not None and expected_qd)
    localized = (expected_qd in origin and zone is not None) if localizable else None
    valve_actions = [action for action in actions if action.command == "VALVE_CLOSE"]
    checkpoint_actions = [action for action in actions if action.command == "CHECKPOINT_VERIFY"]

    return ClipRun(
        clip_id=str(clip["id"]),
        scenario=str(clip["scenario"]),
        split=str(clip.get("split", "unknown")),
        path=str(clip["path"]),
        sha256=str(clip["sha256"]),
        ground_truth_label=str(clip["ground_truth_label"]),
        expected_incident=str(clip["ground_truth_label"]) in POSITIVE_LABELS,
        predicted_incident=report is not None,
        label=(
            report.physical_evidence.label
            if report is not None
            else (decision.label.value if decision is not None else "NONE")
        ),
        rule_id=decision.rule_id if decision is not None else "",
        p=float(report.physical_evidence.p_leak)
        if report is not None
        else (float(decision.p) if decision is not None else 0.0),
        expected_qd=expected_qd,
        suspected_origin=origin,
        localized=localized,
        localizable=localizable,
        zone=zone,
        expected_zone=int(clip.get("expected_zone", -1)),
        detection_s=float(actions[0].t_s) if actions else None,
        ticks=result.ticks,
        frames=result.frames,
        action_count=len(actions),
        valve_verified=(
            any(action.status.startswith("LIMIT_SWITCH_CLOSED") for action in valve_actions)
            if valve_actions
            else None
        ),
        checkpoint_committed=(
            any(action.status == "COMMITTED" for action in checkpoint_actions)
            if checkpoint_actions
            else None
        ),
        sequence_state=result.sequence_state.value if result.sequence_state else None,
    )


def run_clip(
    clip: dict[str, Any],
    cfg: dict[str, Any],
    topology: dict[str, Any],
    *,
    out_dir: str | Path,
    sam_mode: str,
    sam_cache_dir: str | Path | None = None,
) -> BenchRun:
    verify_clip(clip)
    result = asyncio.run(
        run_scenario(
            str(clip["scenario"]),
            cfg,
            topology,
            out_dir=out_dir,
            sam_mode=sam_mode,
            sam_cache_dir=sam_cache_dir,
            protocol_mode="memory",
            allow_llm=False,
        )
    )
    return BenchRun(row=_row(clip, result), result=result)


def run_benchmark(
    *,
    manifest_path: str | Path,
    cfg: dict[str, Any],
    topology: dict[str, Any],
    split: str,
    work_dir: str | Path,
    sam_mode: str,
    sam_cache_dir: str | Path | None = None,
) -> list[BenchRun]:
    manifest = load_manifest(manifest_path)
    select_clips(manifest, split)  # fail early on an empty split
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)

    # Manifest order, every clip: the split selects the headline, it does not
    # hide the other half of a 4-clip set.
    runs: list[BenchRun] = []
    with no_network():
        for clip in manifest["clips"]:
            runs.append(
                run_clip(
                    clip,
                    cfg,
                    topology,
                    out_dir=work / str(clip["id"]),
                    sam_mode=sam_mode,
                    sam_cache_dir=sam_cache_dir,
                )
            )
    return runs


def run_hash(rows: list[dict[str, Any]], *, config_hash: str, fusion_hash: str) -> str:
    """Digest of the deterministic body; generated_at and timings are excluded."""

    body = {
        "configHash": config_hash,
        "fusionHash": fusion_hash,
        "clips": rows,
    }
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
