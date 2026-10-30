"""P1-4 leak-source estimation: the ten cases the claim must survive.

The estimator is judged on honesty as much as accuracy: weak evidence must
return None, and a wrong calibration must be reported as error rather than
absorbed by the estimate. Every observation here is synthetic and
deterministic — no video files, no network, no wall clock.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import yaml

from cryoguard.perception.candidates import (
    SOURCE_METHOD,
    SOURCE_TOLERANCE_MM,
    CandidateManager,
    estimate_leak_source,
    fitting_centre_mm,
    ground_truth_source_mm,
)
from support import chassis_frame

ROOT = Path(__file__).resolve().parents[1]
PX_PER_MM = 1.0
CANVAS = (600, 400)
TICK_S = 0.2


@pytest.fixture(scope="module")
def perception_cfg(cfg: dict) -> dict:
    """Minimal estimator config: only the two knobs the estimator reads."""

    section = cfg["perception"]
    q_min = float(section["q_min"])
    assert 0.0 <= q_min <= 1.0
    return {"q_min": q_min, "link_radius_mm": float(section["link_radius_mm"])}


@pytest.fixture(scope="module")
def topology() -> dict:
    return yaml.safe_load((ROOT / "configs" / "topology.yaml").read_text(encoding="utf-8"))


def obs(t_s: float, x_mm: float, y_mm: float, quality: float):
    """One estimator observation: (t_s, centroid_mm, bbox_centre_mm, quality)."""

    return (t_s, (x_mm, y_mm), (x_mm, y_mm), quality)


def high_q(perception_cfg: dict) -> float:
    return min(1.0, float(perception_cfg["q_min"]) + 0.4)


def low_q(perception_cfg: dict) -> float:
    return float(perception_cfg["q_min"]) / 2.0


def make_manager(cfg: dict, topology: dict) -> CandidateManager:
    return CandidateManager(cfg["perception"], topology, PX_PER_MM, CANVAS)


def test_static_leak_origin_is_estimated_near_the_origin(perception_cfg) -> None:
    q = high_q(perception_cfg)
    observations = [obs(t, 200.0, 150.0, q) for t in (0.0, 0.2, 0.4)]

    estimate = estimate_leak_source(
        observations,
        perception_cfg,
        track_id=7,
        ground_truth_mm=(201.0, 149.0),
    )

    assert estimate is not None
    assert (estimate.x_mm, estimate.y_mm) == pytest.approx((200.0, 150.0), abs=1e-9)
    assert estimate.method == SOURCE_METHOD
    assert estimate.track_id == 7
    assert estimate.centroid_mm == pytest.approx((200.0, 150.0))
    assert estimate.error_mm == pytest.approx(math.sqrt(2.0), abs=1e-9)
    assert estimate.within_tolerance is True
    assert 0.0 < estimate.confidence <= 1.0


def test_slowly_expanding_leak_drift_is_corrected(perception_cfg) -> None:
    q_hi, q_lo = high_q(perception_cfg), low_q(perception_cfg)
    origin = (150.0, 150.0)
    # The mask is not yet readable for the first 0.6 s; the high-quality ticks
    # start 0.6 mm downstream of a 1 mm/s spread and the back-projection removes it.
    observations = [obs(t, origin[0] + 1.0 * t, origin[1], q_lo) for t in (0.0, 0.2, 0.4)]
    observations += [
        obs(t, origin[0] + 1.0 * t, origin[1], q_hi) for t in (0.6, 0.8, 1.0, 1.2, 1.4, 1.6)
    ]

    estimate = estimate_leak_source(observations, perception_cfg, ground_truth_mm=origin)

    assert estimate is not None
    assert estimate.error_mm is not None and estimate.error_mm < 0.5
    assert estimate.velocity_mm_per_s is not None
    assert estimate.velocity_mm_per_s[0] == pytest.approx(1.0, abs=1e-9)
    assert estimate.within_tolerance is True


def test_fast_spill_back_projection_recovers_the_origin(perception_cfg) -> None:
    q_hi, q_lo = high_q(perception_cfg), low_q(perception_cfg)
    speed = 11.0
    origin = (150.0, 150.0)
    observations = [obs(t, origin[0] + speed * t, origin[1], q_lo) for t in (0.0, 0.2, 0.4, 0.6)]
    observations += [
        obs(t, origin[0] + speed * t, origin[1], q_hi) for t in (0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0)
    ]

    estimate = estimate_leak_source(observations, perception_cfg, ground_truth_mm=origin)

    assert estimate is not None
    assert estimate.velocity_mm_per_s == pytest.approx((speed, 0.0), abs=1e-6)
    assert estimate.error_mm == pytest.approx(0.0, abs=0.01)
    assert estimate.within_tolerance is True
    # The observed centroid is ~8.8 mm downstream; the estimate is not.
    assert estimate.centroid_mm[0] - estimate.x_mm == pytest.approx(speed * 0.8, abs=1e-6)


def test_two_candidates_keep_separate_source_histories(cfg, topology) -> None:
    manager = make_manager(cfg, topology)
    for index in range(4):
        t_s = index * TICK_S
        manager.update_warped(chassis_frame([(12, int(120 + 5 * t_s), 100), (12, 420, 300)]), t_s)

    tracks = sorted(manager.tracker.tracks, key=lambda track: track.centroid_mm[0])
    assert len(tracks) == 2
    near, far = tracks

    near_estimate = manager.estimate_source(near.track_id)
    far_estimate = manager.estimate_source(far.track_id)

    assert near_estimate is not None and far_estimate is not None
    assert near_estimate.track_id == near.track_id
    assert far_estimate.track_id == far.track_id
    assert near_estimate.track_id != far_estimate.track_id
    assert near_estimate.x_mm == pytest.approx(120.0, abs=2.0)
    assert far_estimate.x_mm == pytest.approx(420.0, abs=2.0)
    assert far_estimate.velocity_mm_per_s == pytest.approx((0.0, 0.0), abs=0.5)
    assert manager.estimate_source(99) is None


def test_leak_near_a_fitting_reports_proximity(perception_cfg, topology) -> None:
    q = high_q(perception_cfg)
    fitting = fitting_centre_mm(topology, "QD-04")
    assert fitting == pytest.approx((195.0, 145.0))

    near = estimate_leak_source(
        [obs(t, 190.0, 145.0, q) for t in (0.0, 0.2, 0.4)],
        perception_cfg,
        fitting_mm=fitting,
    )
    far = estimate_leak_source(
        [obs(t, 400.0, 300.0, q) for t in (0.0, 0.2, 0.4)],
        perception_cfg,
        fitting_mm=fitting,
    )

    assert near is not None and far is not None
    assert near.fitting_mm == fitting
    assert near.fitting_distance_mm == pytest.approx(5.0)
    assert near.near_fitting is True
    assert far.fitting_distance_mm == pytest.approx(math.hypot(205.0, 155.0))
    assert far.near_fitting is False
    assert fitting_centre_mm(topology, "QD-99") is None


def test_spill_crossing_a_zone_boundary_still_estimates(cfg, topology) -> None:
    manager = make_manager(cfg, topology)
    zones: list[int | None] = []
    track_ids: set[int] = set()
    # x = 285 .. 305 crosses the zone-0 / zone-2 boundary at x = 300 mm.
    for index in range(6):
        t_s = index * TICK_S
        candidates = manager.update_warped(chassis_frame([(12, 285 + 4 * index, 150)]), t_s)
        assert len(candidates) == 1
        zones.append(candidates[0].zone)
        track_ids.add(candidates[0].track_id)

    assert len(track_ids) == 1
    assert 0 in zones and 2 in zones  # the spill really crossed the boundary

    estimate = manager.estimate_source(next(iter(track_ids)))
    assert estimate is not None
    assert estimate.x_mm == pytest.approx(285.0, abs=2.0)


def test_no_visual_candidate_never_invents_a_source(perception_cfg, cfg, topology) -> None:
    assert estimate_leak_source([], perception_cfg) is None

    manager = make_manager(cfg, topology)
    for index in range(3):
        assert manager.update_warped(chassis_frame([]), index * TICK_S) == ()

    assert manager.estimate_source(None) is None
    assert manager.estimate_source(1) is None


def test_low_quality_samples_are_never_used(perception_cfg) -> None:
    q_hi, q_lo = high_q(perception_cfg), low_q(perception_cfg)

    weak = [obs(t, 300.0, 200.0, q_lo) for t in (0.0, 0.2, 0.4, 0.6)]
    assert estimate_leak_source(weak, perception_cfg) is None

    mixed = [obs(0.0, 500.0, 300.0, q_lo)] + [obs(t, 200.0, 150.0, q_hi) for t in (0.4, 0.6, 0.8)]
    estimate = estimate_leak_source(mixed, perception_cfg)

    assert estimate is not None
    assert estimate.quality == pytest.approx(q_hi)
    assert (estimate.x_mm, estimate.y_mm) == pytest.approx((200.0, 150.0), abs=1e-9)


def test_missing_ground_truth_is_reported_as_unknown(perception_cfg) -> None:
    assert ground_truth_source_mm({}) is None
    assert ground_truth_source_mm({"leak_source_mm": None}) is None
    assert ground_truth_source_mm({"leak_source_mm": [1.0, 2.0, 3.0]}) is None
    assert ground_truth_source_mm({"leak_source_mm": ["x", "y"]}) is None
    assert ground_truth_source_mm({"leak_source_mm": [math.nan, 150.0]}) is None
    nested = ground_truth_source_mm({"expected": {"leak_source_mm": [210.0, 150.0]}})
    assert nested == pytest.approx((210.0, 150.0))

    q = high_q(perception_cfg)
    estimate = estimate_leak_source(
        [obs(t, 200.0, 150.0, q) for t in (0.0, 0.2, 0.4)], perception_cfg
    )

    assert estimate is not None
    assert estimate.ground_truth_mm is None
    assert estimate.error_mm is None
    assert estimate.within_tolerance is None


def test_inaccurate_calibration_reports_a_large_error(perception_cfg) -> None:
    q = high_q(perception_cfg)
    observations = [obs(t, 200.0, 150.0, q) for t in (0.0, 0.2, 0.4)]

    # Markers are 30 mm off, so the manifest truth is 30 mm from the real source.
    estimate = estimate_leak_source(
        observations,
        perception_cfg,
        ground_truth_mm=(200.0, 180.0),
    )

    assert estimate is not None
    assert estimate.error_mm == pytest.approx(30.0, abs=1e-9)
    assert estimate.within_tolerance is False
    assert estimate.tolerance_mm == SOURCE_TOLERANCE_MM
    # The estimate is not nudged toward the (wrong) ground truth.
    assert (estimate.x_mm, estimate.y_mm) == pytest.approx((200.0, 150.0), abs=1e-9)
