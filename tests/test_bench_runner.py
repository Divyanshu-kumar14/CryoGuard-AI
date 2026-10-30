"""Benchmark runner: manifest integrity, offline guard, and reproduction."""

from __future__ import annotations

import json
import socket
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from cryoguard.bench.report import build_payload
from cryoguard.bench.runner import (
    FITTING_RADIUS_MM,
    BenchError,
    _row,
    load_manifest,
    no_network,
    qd_centroid_mm,
    run_benchmark,
    select_clips,
    verify_clip,
)
from cryoguard.config import config_hash, load
from cryoguard.pipeline import RunResult

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def cfg() -> dict:
    return load(ROOT / "configs" / "default.yaml")


@pytest.fixture(scope="module")
def topology() -> dict:
    return yaml.safe_load((ROOT / "configs" / "topology.yaml").read_text(encoding="utf-8"))


def test_the_manifest_selects_the_declared_split() -> None:
    manifest = load_manifest(ROOT / "fixtures" / "manifest.yaml")

    assert {clip["split"] for clip in select_clips(manifest, "all")} == {"dev", "test"}
    assert [clip["id"] for clip in select_clips(manifest, "test")] == [
        "sim_glint",
        "sim_hidden_hydraulic",
    ]
    with pytest.raises(BenchError, match="no clips"):
        select_clips(manifest, "holdout")


def _clip(qd: str = "QD-04") -> dict:
    return {
        "id": "sim_leak_qd4",
        "scenario": "slow_leak_qd4",
        "split": "dev",
        "path": "fixtures/videos/simulated/sim_leak_qd4.mp4",
        "sha256": "0" * 64,
        "ground_truth_label": "CONFIRMED_LEAK",
        "qd_origin": qd,
        "expected_zone": 0,
    }


def _result(centroid_mm: tuple[float, float] | None, *, origin: str = "QD-04 (Zone 0)"):
    report = (
        SimpleNamespace(
            deterministic_actions_taken=[],
            suspected_origin=origin,
            physical_evidence=SimpleNamespace(zone=0, label="CONFIRMED_LEAK", p_leak=0.9),
        )
        if centroid_mm is not None
        else None
    )
    return RunResult(
        scenario="slow_leak_qd4",
        decision=None,
        report=report,  # type: ignore[arg-type]
        path=None,
        frames=1,
        ticks=1,
        tick_rate_hz=5.0,
        rate_changes=0,
        sam=None,
        centroid_mm=centroid_mm,
    )


def test_the_qd_centroid_is_the_configured_roi_centre(topology) -> None:
    assert qd_centroid_mm(topology, "QD-04") == (195.0, 145.0)
    assert qd_centroid_mm(topology, "QD-99") is None


def test_localization_is_a_geometric_distance_not_an_origin_string(topology) -> None:
    clip = _clip()

    near = _row(clip, _result((203.4, 145.0)), topology)
    assert near.localization_error_mm == pytest.approx(8.4)
    assert near.localized_within_tolerance is True
    assert FITTING_RADIUS_MM == 25.0

    # Same "QD-04" origin string, but 50 mm away: not a localization.
    far = _row(clip, _result((195.0, 195.0), origin="QD-04 (Zone 0)"), topology)
    assert far.localization_error_mm == pytest.approx(50.0)
    assert far.localized_within_tolerance is False


def test_a_clip_without_a_candidate_centroid_is_not_localizable(topology) -> None:
    row = _row(_clip(), _result(None), topology)

    assert row.localizable is False
    assert row.localization_error_mm is None
    assert row.localized_within_tolerance is None


def test_a_tampered_clip_is_refused(tmp_path) -> None:
    clip = {
        "path": "fixtures/videos/simulated/sim_leak_qd4.mp4",
        "sha256": "0" * 64,
    }

    with pytest.raises(BenchError, match="does not match the manifest"):
        verify_clip(clip)


def test_an_unreachable_socket_fails_the_benchmark() -> None:
    original = socket.create_connection

    with pytest.raises(BenchError, match="attempted a network connection"):
        with no_network():
            socket.create_connection(("127.0.0.1", 9), timeout=0.01)

    assert socket.create_connection is original


def test_the_socket_guard_is_restored_after_an_error() -> None:
    original = socket.socket.connect

    with pytest.raises(BenchError):
        with no_network():
            socket.socket().connect(("127.0.0.1", 9))

    assert socket.socket.connect is original


@pytest.mark.slow
def test_the_committed_benchmark_reproduces(cfg, topology) -> None:
    committed_path = ROOT / "reports" / "benchmark.json"
    if not committed_path.is_file():
        pytest.skip("no committed benchmark yet")
    committed = json.loads(committed_path.read_text(encoding="utf-8"))
    freeze = json.loads((ROOT / "reports" / "fusion-freeze.json").read_text(encoding="utf-8"))

    with tempfile.TemporaryDirectory(prefix="cryoguard-bench-test-") as work:
        runs = run_benchmark(
            manifest_path=ROOT / "fixtures" / "manifest.yaml",
            cfg=cfg,
            topology=topology,
            split=str(committed["split"]),
            work_dir=work,
            sam_mode="cache",
        )

    payload = build_payload(
        runs,
        cfg=cfg,
        split=str(committed["split"]),
        config_hash=config_hash(cfg),
        fusion_hash=str(freeze["fusion_hash"]),
        git_sha="test",
        confidence=float(cfg["bench"]["confidence"]),
    )

    assert payload["runHash"] == committed["runHash"]
    assert payload["clips"] == committed["clips"]
    assert payload["metrics"] == committed["metrics"]
