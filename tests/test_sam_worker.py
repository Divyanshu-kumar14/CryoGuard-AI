"""SAM worker: cadence, in-flight cap, latency independence, cache-miss behaviour."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import numpy as np
import pytest
import yaml

from cryoguard.config import load
from cryoguard.perception.sam_client import SamCacheMiss, SamClient
from cryoguard.perception.sam_worker import SamWorker, build_rois

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = json.loads((ROOT / "fixtures" / "sam" / "response_schema_fixture.json").read_text())


@pytest.fixture(scope="module")
def cfg() -> dict:
    return load(ROOT / "configs" / "default.yaml")


@pytest.fixture(scope="module")
def topology() -> dict:
    return yaml.safe_load((ROOT / "configs" / "topology.yaml").read_text(encoding="utf-8"))


class DelayedTransport:
    def __init__(self, delay_s: float, payload: dict) -> None:
        self.delay_s = delay_s
        self.payload = payload
        self.calls = 0

    def __call__(self, image_bytes: bytes, prompt: str) -> dict:
        self.calls += 1
        time.sleep(self.delay_s)
        return self.payload


def sam_cfg(cfg: dict, mode: str = "live") -> dict:
    return dict(cfg["sam"], mode=mode)


def warped() -> np.ndarray:
    return np.zeros((400, 600, 3), dtype=np.uint8)


def test_rois_include_the_configured_margin(cfg, topology) -> None:
    rois = build_rois(topology, px_per_mm=1.0, canvas_px=(600, 400), margin_mm=10.0)
    assert len(rois) == 1
    assert (rois[0].x0, rois[0].y0, rois[0].x1, rois[0].y1) == (145, 85, 245, 205)


def test_a_miss_in_cache_mode_surfaces_from_submit(cfg, topology, tmp_path) -> None:
    client = SamClient(sam_cfg(cfg, "cache"), cache_dir=tmp_path / "empty")
    worker = SamWorker(client, topology, sam_cfg(cfg, "cache"), 1.0, (600, 400), 30.0)

    with pytest.raises(SamCacheMiss):
        worker.submit(warped(), frame_index=0, t_s=0.0)


def test_the_loop_keeps_ticking_while_sam_is_in_flight(cfg, topology, tmp_path) -> None:
    transport = DelayedTransport(0.25, FIXTURE)
    client = SamClient(sam_cfg(cfg, "live"), cache_dir=tmp_path / "cache", transport=transport)
    worker = SamWorker(client, topology, sam_cfg(cfg, "live"), 1.0, (600, 400), 30.0)

    async def case() -> tuple[int, int]:
        worker.submit(warped(), frame_index=0, t_s=0.0)
        ticks = 0
        deadline = time.perf_counter() + 0.15
        while time.perf_counter() < deadline:
            ticks += 1
            await asyncio.sleep(0.01)
        assert worker.poll() == []  # still in flight, and the loop never blocked
        await worker.close()
        samples = worker.poll()
        assert len(samples) == 1
        return ticks, len(samples)

    ticks, samples = asyncio.run(case())
    assert ticks > 5
    assert samples == 1
    assert transport.calls == 1


def test_in_flight_cap_is_one(cfg, topology, tmp_path) -> None:
    transport = DelayedTransport(0.2, FIXTURE)
    client = SamClient(sam_cfg(cfg, "live"), cache_dir=tmp_path / "cache", transport=transport)
    worker = SamWorker(client, topology, sam_cfg(cfg, "live"), 1.0, (600, 400), 30.0)

    async def case() -> None:
        worker.submit(warped(), frame_index=0, t_s=0.0)
        worker.submit(warped(), frame_index=30, t_s=1.0)
        assert worker.stats.skipped == 1
        await worker.close()

    asyncio.run(case())
    assert transport.calls == 1


def test_a_completed_mask_is_placed_in_full_canvas_coordinates(cfg, topology, tmp_path) -> None:
    client = SamClient(
        sam_cfg(cfg, "live"),
        cache_dir=tmp_path / "cache",
        transport=DelayedTransport(0.0, FIXTURE),
    )
    worker = SamWorker(client, topology, sam_cfg(cfg, "live"), 1.0, (600, 400), 30.0)

    async def case():
        worker.submit(warped(), frame_index=0, t_s=0.0)
        await worker.close()
        return worker.poll()

    samples = asyncio.run(case())
    assert len(samples) == 1
    sample = samples[0]
    assert sample.mask_full.shape == (400, 600)
    assert sample.mask_full.any()
    # The ROI starts at (145, 85); the fixture bbox starts at (8, 10).
    assert sample.bbox_full[0] >= 145
    assert sample.bbox_full[1] >= 85
    assert sample.provenance == "api"


def test_live_failure_is_recorded_not_raised(cfg, topology, tmp_path) -> None:
    class BrokenTransport:
        def __call__(self, image_bytes: bytes, prompt: str) -> dict:
            raise TimeoutError("endpoint down")

    client = SamClient(
        sam_cfg(cfg, "live"), cache_dir=tmp_path / "cache", transport=BrokenTransport()
    )
    worker = SamWorker(client, topology, sam_cfg(cfg, "live"), 1.0, (600, 400), 30.0)

    async def case():
        worker.submit(warped(), frame_index=0, t_s=0.0)
        await worker.close()
        return worker.failures(), worker.poll()

    failures, samples = asyncio.run(case())
    assert samples == []
    assert len(failures) == 1
    assert "TimeoutError" in failures[0].error
    assert worker.stats.failures == 1
