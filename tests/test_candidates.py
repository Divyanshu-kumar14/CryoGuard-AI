"""Candidate lifecycle, W continuity, and the mask-quality gates (P3-T04/T07)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from cryoguard.capture.zones import ZoneMap
from cryoguard.config import load
from cryoguard.perception.candidates import CandidateManager
from cryoguard.perception.fast_tracker import FastTracker, Track
from cryoguard.perception.quality import mask_quality
from support import chassis_frame

ROOT = Path(__file__).resolve().parents[1]
PX_PER_MM = 1.0
CANVAS = (600, 400)


@pytest.fixture(scope="module")
def cfg() -> dict:
    return load(ROOT / "configs" / "default.yaml")


@pytest.fixture(scope="module")
def topology() -> dict:
    return yaml.safe_load((ROOT / "configs" / "topology.yaml").read_text(encoding="utf-8"))


def manager(cfg: dict, topology: dict) -> CandidateManager:
    return CandidateManager(cfg["perception"], topology, PX_PER_MM, CANVAS, anchor_ttl_s=1.2)


def connected_frame() -> np.ndarray:
    frame = chassis_frame([(10, 190, 230)])
    cv2.line(frame, (190, 150), (190, 230), (0, 220, 40), 4)
    return frame


def test_w_is_true_only_with_a_continuous_path_to_the_qd(cfg, topology) -> None:
    manager_ = manager(cfg, topology)
    for tick in range(4):
        candidates = manager_.update_warped(connected_frame(), t_s=tick * 0.2)

    assert candidates[0].wetness == 1.0


def test_a_disconnected_blob_never_gets_w(cfg, topology) -> None:
    manager_ = manager(cfg, topology)
    for tick in range(6):
        candidates = manager_.update_warped(chassis_frame([(10, 450, 300)]), t_s=tick * 0.2)

    assert candidates[0].wetness == 0.0


def test_merge_keeps_one_identity_and_retires_the_other(cfg) -> None:
    tracker = FastTracker(cfg["perception"])
    tracker.update_warped(chassis_frame([(10, 100, 100), (10, 140, 100)]), 0.0, PX_PER_MM)

    merged = tracker.update_warped(chassis_frame([(24, 120, 100)]), 0.2, PX_PER_MM)

    assert len(merged) == 1
    assert len(tracker.tracks) == 2
    assert sorted(track.misses for track in tracker.tracks) == [0, 1]


def test_split_spawns_a_second_candidate(cfg) -> None:
    tracker = FastTracker(cfg["perception"])
    tracker.update_warped(chassis_frame([(22, 120, 100)]), 0.0, PX_PER_MM)

    split = tracker.update_warped(chassis_frame([(9, 100, 100), (9, 140, 100)]), 0.2, PX_PER_MM)

    assert len(split) == 2
    assert len({track.track_id for track in split}) == 2


def test_a_missed_candidate_is_stale_and_cannot_pass_the_quality_gate(cfg, topology) -> None:
    manager_ = manager(cfg, topology)
    manager_.update_warped(chassis_frame([(10, 190, 230)]), t_s=0.0)
    manager_.update_warped(chassis_frame([]), t_s=0.2)

    candidates = manager_.current(0.2)
    assert candidates
    assert candidates[0].stale is True
    assert candidates[0].quality == 0.0


def test_a_sam_anchor_can_spawn_a_candidate_the_threshold_cannot_see(cfg, topology) -> None:
    manager_ = manager(cfg, topology)
    manager_.update_warped(chassis_frame([]), t_s=0.0)

    mask = np.zeros((CANVAS[1], CANVAS[0]), dtype=bool)
    cv2.circle(mask.view(np.uint8), (300, 200), 14, 1, -1)
    manager_.reanchor(mask, t_s=0.0)

    candidates = manager_.current(0.0)
    assert candidates
    assert candidates[0].sam_verified is True
    assert candidates[0].quality > 0.0
    assert candidates[0].zone is not None


def test_exposure_clipping_fails_the_quality_gate_hard(cfg) -> None:
    track = Track(
        track_id=1,
        centroid_mm=(10.0, 10.0),
        area_mm2=100.0,
        dA_dt_mm2_per_s=0.0,
        mean_intensity=1.0,
        intensity_var=0.0,
        clipped_fraction=float(cfg["perception"]["exposure_clip_max"]) + 0.01,
        compactness=0.9,
        solidity=0.9,
        bbox_mm=(0.0, 0.0, 10.0, 10.0),
        mask=np.ones((10, 10), dtype=bool),
        offset_px=(0, 0),
        first_t=0.0,
        last_t=0.0,
    )
    assert mask_quality(track, cfg["perception"]) == 0.0


def test_a_solid_candidate_passes_its_quality_gate(cfg, topology) -> None:
    manager_ = manager(cfg, topology)
    candidates = manager_.update_warped(chassis_frame([(12, 190, 200)]), t_s=0.0)
    assert candidates[0].quality >= cfg["perception"]["q_min"]
    assert candidates[0].zone == 0


def test_zone_lookup_still_tiles_the_chassis(cfg, topology) -> None:
    zones = ZoneMap.build(topology, PX_PER_MM)
    for x_mm, y_mm in ((20, 20), (299, 239), (300, 20), (579, 379)):
        assert zones.zone_at(x_mm, y_mm) is not None
