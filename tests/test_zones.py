"""Zone assignment: the 5 % rule and the highest-index rule, both from PRD §6.1."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from cryoguard.capture.zones import ZoneError, ZoneMap

ROOT = Path(__file__).resolve().parents[1]
PX_PER_MM = 2.0


@pytest.fixture(scope="module")
def zone_map() -> ZoneMap:
    topology = yaml.safe_load((ROOT / "configs" / "topology.yaml").read_text(encoding="utf-8"))
    return ZoneMap.build(topology, PX_PER_MM)


def test_points_land_in_the_configured_zones(zone_map: ZoneMap) -> None:
    assert zone_map.zone_at(100, 100) == 0  # hydraulic origin
    assert zone_map.zone_at(150, 280) == 1  # containment buffer
    assert zone_map.zone_at(500, 200) == 2  # electrical plane
    assert zone_map.zone_at(10, 10) is None  # outside every polygon


def test_higher_zone_wins_when_area_is_shared(zone_map: ZoneMap) -> None:
    mask = np.zeros_like(zone_map.labels, dtype=bool)
    mask[500:510, 100:410] = True  # 1000 px in zone 1 ...
    mask[500:510, 600:620] = True  # 200 px in zone 2, 16 %

    assert zone_map.zone_for_mask(mask) == 2


def test_shared_area_below_five_percent_is_ignored(zone_map: ZoneMap) -> None:
    mask = np.zeros_like(zone_map.labels, dtype=bool)
    mask[500:510, 100:410] = True  # 1000 px in zone 1
    mask[500:503, 600:602] = True  # 6 px in zone 2, 0.6 %

    assert zone_map.zone_for_mask(mask) == 1


def test_empty_mask_has_no_zone(zone_map: ZoneMap) -> None:
    assert zone_map.zone_for_mask(np.zeros_like(zone_map.labels, dtype=bool)) is None


def test_mismatched_mask_shape_fails_loudly(zone_map: ZoneMap) -> None:
    with pytest.raises(ZoneError, match="does not match"):
        zone_map.zone_for_mask(np.zeros((10, 10), dtype=bool))


def test_canvas_matches_the_declared_chassis(zone_map: ZoneMap) -> None:
    assert zone_map.labels.shape == (800, 1200)  # 400 x 600 mm at 2 px/mm
    assert set(np.unique(zone_map.labels)) == {-1, 0, 1, 2}
