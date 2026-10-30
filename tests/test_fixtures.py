"""Fixture manifest and the tracker's measured behaviour on the simulated clips.

The clips are synthetic and the manifest says so. These tests prove the tracker
separates the four cases *in simulation*; they are not evidence about real
coolant, and no result here may be quoted as if they were.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

from cryoguard.capture.rectify import estimate
from cryoguard.capture.source import VideoSource
from cryoguard.capture.zones import ZoneMap
from cryoguard.config import load
from cryoguard.perception.fast_tracker import FastTracker

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "fixtures" / "manifest.yaml"


@pytest.fixture(scope="module")
def manifest() -> dict:
    return yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def pipeline():
    cfg = load(ROOT / "configs" / "default.yaml")
    topology = yaml.safe_load((ROOT / "configs" / "topology.yaml").read_text(encoding="utf-8"))
    return cfg, topology


def test_manifest_declares_its_provenance(manifest: dict) -> None:
    assert manifest["provenance"] == "synthetic"
    for clip in manifest["clips"]:
        assert clip["source"] == "synthetic"
        assert "SIMULATED" in manifest["note"]


def test_every_clip_exists_with_the_recorded_hash(manifest: dict) -> None:
    for clip in manifest["clips"]:
        path = ROOT / clip["path"]
        assert path.is_file(), clip["path"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == clip["sha256"], clip["id"]


def test_every_clip_is_labelled_and_split_is_disjoint(manifest: dict) -> None:
    ids = {clip["id"] for clip in manifest["clips"]}
    dev = set(manifest["split"]["dev"])
    test = set(manifest["split"]["test"])

    assert dev.isdisjoint(test)
    assert dev | test == ids
    for clip in manifest["clips"]:
        assert clip["ground_truth_label"]
        assert clip["expected_zone"] in (0, 1, 2)


def test_scenario_configs_point_at_existing_simulated_clips() -> None:
    for path in sorted((ROOT / "configs" / "scenarios").glob("*.yaml")):
        scenario = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert scenario["provenance"] == "synthetic", path.name
        assert (ROOT / scenario["clip"]).is_file(), path.name


def _track_clip(clip_path: Path, every: int, cfg: dict, topology: dict):
    px_per_mm = float(cfg["camera"]["px_per_mm"])
    canvas = (
        int(topology["chassis"]["width_mm"] * px_per_mm),
        int(topology["chassis"]["height_mm"] * px_per_mm),
    )
    zones = ZoneMap.build(topology, px_per_mm)
    tracker = FastTracker(cfg["perception"])

    with VideoSource(clip_path) as source:
        rect = estimate(source.read(0).image, topology["chassis"]["aruco"])
        seen = 0
        hits = 0
        max_area = 0.0
        max_growth = 0.0
        zones_seen: set[int] = set()
        for frame in source.frames():
            if frame.index % every:
                continue
            seen += 1
            tracks = tracker.update(frame.image, frame.t_s, rect, canvas, px_per_mm)
            if not tracks:
                continue
            hits += 1
            best = max(tracks, key=lambda track: track.area_mm2)
            max_area = max(max_area, best.area_mm2)
            max_growth = max(max_growth, abs(best.dA_dt_mm2_per_s))
            zone = zones.zone_for_cropped(best.mask, best.offset_px)
            if zone is not None:
                zones_seen.add(zone)
    return seen, hits, max_area, max_growth, zones_seen


@pytest.mark.slow
def test_leak_clip_produces_a_persistent_candidate(pipeline) -> None:
    cfg, topology = pipeline
    seen, hits, max_area, max_growth, zones = _track_clip(
        ROOT / "fixtures/videos/simulated/sim_leak_qd4.mp4", every=20, cfg=cfg, topology=topology
    )

    assert hits / seen > 0.5, f"only {hits}/{seen} frames had a candidate"
    assert max_area > 100.0
    assert max_growth > 10.0  # growing, not a static reflection
    assert 2 in zones  # reaches the electrical plane before the clip ends


@pytest.mark.slow
def test_condensation_clip_produces_no_candidate(pipeline) -> None:
    cfg, topology = pipeline
    seen, hits, _, _, _ = _track_clip(
        ROOT / "fixtures/videos/simulated/sim_condensation.mp4",
        every=20,
        cfg=cfg,
        topology=topology,
    )
    assert seen > 0 and hits == 0


@pytest.mark.slow
def test_hidden_hydraulic_clip_produces_no_visual_candidate(pipeline) -> None:
    cfg, topology = pipeline
    seen, hits, _, _, _ = _track_clip(
        ROOT / "fixtures/videos/simulated/sim_hidden_hydraulic.mp4",
        every=20,
        cfg=cfg,
        topology=topology,
    )
    assert seen > 0 and hits == 0


@pytest.mark.slow
def test_glint_clip_never_looks_like_growth(pipeline) -> None:
    cfg, topology = pipeline
    seen, _, max_area, max_growth, _ = _track_clip(
        ROOT / "fixtures/videos/simulated/sim_glint.mp4", every=3, cfg=cfg, topology=topology
    )

    assert seen > 0
    assert max_area < 200.0  # a reflection, not a spreading puddle
    assert max_growth < 50.0  # no sustained growth signal
