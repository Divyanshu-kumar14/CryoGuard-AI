"""SAM guarantees: non-blocking, provenance, staleness, mask hygiene, prompt audit.

These are the guards behind the task packet: SAM must never stall the safety
loop, a fixture mask must never be relabelled live, an old frame must never
overwrite newer tracker state, a malformed mask must fail loudly at the
boundary, and prompt trials must be recorded — never silently ranked.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import time
from pathlib import Path

import numpy as np
import pytest
import yaml

from cryoguard.config import load
from cryoguard.fusion.engine import leak_score
from cryoguard.perception.fast_tracker import FastTracker
from cryoguard.perception.sam_client import SamAPIError, SamClient, SamMask, SamResult
from cryoguard.perception.sam_worker import SamWorker, SamWorkerError

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = json.loads((ROOT / "fixtures" / "sam" / "response_schema_fixture.json").read_text())


@pytest.fixture(scope="module")
def cfg() -> dict:
    return load(ROOT / "configs" / "default.yaml")


@pytest.fixture(scope="module")
def topology() -> dict:
    return yaml.safe_load((ROOT / "configs" / "topology.yaml").read_text(encoding="utf-8"))


class ScriptedTransport:
    """Per-call delays and payloads, so slow/fast completions can be ordered."""

    def __init__(self, payload: dict, delays: list[float]) -> None:
        self.payload = payload
        self.delays = delays
        self.calls = 0

    def __call__(self, image_bytes: bytes, prompt: str) -> dict:
        index = min(self.calls, len(self.delays) - 1)
        self.calls += 1
        time.sleep(self.delays[index])
        return self.payload


def sam_cfg(cfg: dict, mode: str = "live", **overrides: object) -> dict:
    return dict(cfg["sam"], mode=mode, **overrides)


def warped() -> np.ndarray:
    return np.zeros((400, 600, 3), dtype=np.uint8)


def roi_mask(worker: SamWorker) -> np.ndarray:
    shape = (worker.rois[0].y1 - worker.rois[0].y0, worker.rois[0].x1 - worker.rois[0].x0)
    mask = np.zeros(shape, dtype=bool)
    mask[5:15, 5:20] = True
    return mask


def test_submit_returns_immediately_while_sam_is_in_flight(cfg, topology, tmp_path) -> None:
    transport = ScriptedTransport(FIXTURE, [0.5])
    client = SamClient(sam_cfg(cfg, "live"), cache_dir=tmp_path / "cache", transport=transport)
    worker = SamWorker(client, topology, sam_cfg(cfg, "live"), 1.0, (600, 400), 30.0)

    async def case() -> float:
        started = time.perf_counter()
        worker.submit(warped(), frame_index=0, t_s=0.0)
        elapsed = time.perf_counter() - started
        await worker.close()
        return elapsed

    elapsed = asyncio.run(case())
    assert elapsed < 0.2, f"submit blocked for {elapsed:.3f}s with a 0.5s transport"


def test_a_live_timeout_costs_refinement_not_ticks(cfg, topology, tmp_path) -> None:
    transport = ScriptedTransport(FIXTURE, [0.5])
    client = SamClient(sam_cfg(cfg, "live"), cache_dir=tmp_path / "cache", transport=transport)
    worker = SamWorker(
        client, topology, sam_cfg(cfg, "live", timeout_sec=0.05), 1.0, (600, 400), 30.0
    )

    async def case() -> tuple[int, list, list]:
        worker.submit(warped(), frame_index=0, t_s=0.0)
        ticks = 0
        deadline = time.perf_counter() + 0.2
        while time.perf_counter() < deadline:
            ticks += 1
            await asyncio.sleep(0.01)
        mid_poll = worker.poll()
        await worker.close()
        return ticks, mid_poll, worker.failures()

    ticks, mid_poll, failures = asyncio.run(case())
    assert ticks > 5, "the safety loop stalled behind the SAM timeout"
    assert mid_poll == []
    assert worker.poll() == []
    assert len(failures) == 1
    assert "TimeoutError" in failures[0].error
    assert worker.stats.failures == 1


def test_in_flight_cap_holds_under_burst(cfg, topology, tmp_path) -> None:
    transport = ScriptedTransport(FIXTURE, [0.3])
    worker_cfg = sam_cfg(cfg, "live", in_flight=1)
    client = SamClient(sam_cfg(cfg, "live"), cache_dir=tmp_path / "cache", transport=transport)
    worker = SamWorker(client, topology, worker_cfg, 1.0, (600, 400), 30.0)

    async def case() -> None:
        worker.submit(warped(), frame_index=0, t_s=0.0)
        worker.submit(warped(), frame_index=30, t_s=1.0)
        worker.submit(warped(), frame_index=60, t_s=2.0)
        assert worker.stats.skipped == 2
        assert len(worker._tasks) <= 1
        await worker.close()

    asyncio.run(case())
    assert transport.calls == 1


def test_async_rate_above_2hz_is_rejected(cfg, topology, tmp_path) -> None:
    client = SamClient(sam_cfg(cfg, "cache"), cache_dir=tmp_path / "cache")
    with pytest.raises(SamWorkerError, match="2 Hz"):
        SamWorker(client, topology, sam_cfg(cfg, "cache", async_rate_hz=5.0), 1.0, (600, 400), 30.0)


def test_malformed_mask_is_a_failure_not_tracker_state(cfg, topology, tmp_path) -> None:
    class GarbageTransport:
        def __call__(self, image_bytes: bytes, prompt: str) -> dict:
            return {"masks": [{"mask_png_base64": "!!!not-base64!!!"}]}

    client = SamClient(
        sam_cfg(cfg, "live"), cache_dir=tmp_path / "cache", transport=GarbageTransport()
    )
    worker = SamWorker(client, topology, sam_cfg(cfg, "live"), 1.0, (600, 400), 30.0)
    before = worker.stats.applied

    async def case():
        worker.submit(warped(), frame_index=0, t_s=0.0)
        await worker.close()
        return worker.poll(), worker.failures()

    samples, failures = asyncio.run(case())
    assert samples == []
    assert len(failures) == 1
    assert worker.stats.applied == before
    assert worker.stats.failures == 1


def test_wrong_shaped_mask_is_dropped_at_the_boundary(cfg, topology, tmp_path) -> None:
    client = SamClient(sam_cfg(cfg, "cache"), cache_dir=tmp_path / "cache")
    worker = SamWorker(client, topology, sam_cfg(cfg, "cache"), 1.0, (600, 400), 30.0)
    roi = worker.rois[0]
    bad = SamResult(
        masks=(SamMask(mask=np.zeros((7, 7), dtype=bool), bbox_px=(0, 0, 7, 7), score=0.9),),
        provenance="api",
        cached=False,
        cache_key="k",
        prompt="p",
        model="sam-3.1",
    )
    worker._accept(roi, bad, frame_index=0, t_s=0.0)
    assert worker.poll() == []
    assert worker.stats.applied == 0
    assert worker.stats.failures == 1


def test_reanchor_rejects_a_mask_from_another_canvas(cfg) -> None:
    from cryoguard.perception.fast_tracker import TrackerError

    tracker = FastTracker(cfg["perception"])
    frame = np.zeros((400, 600, 3), dtype=np.uint8)
    tracker.update_warped(frame, 0.0, 1.0)
    with pytest.raises(TrackerError, match="does not match frame"):
        tracker.reanchor(np.zeros((100, 100), dtype=bool), frame, 0.0, 1.0)
    with pytest.raises(TrackerError, match="2-D"):
        tracker.reanchor(np.zeros((400, 600, 3), dtype=bool), frame, 0.0, 1.0)  # type: ignore[arg-type]


def test_fixture_cache_hit_is_never_labelled_live(cfg, topology, tmp_path) -> None:
    seeder = SamClient(sam_cfg(cfg, "cache"), cache_dir=tmp_path / "seed")
    prompt = str(cfg["sam"]["prompts"]["coolant"])

    # Seed one entry with fixture provenance, then serve it through the worker.
    worker_client = SamClient(sam_cfg(cfg, "cache"), cache_dir=tmp_path / "seed")
    worker = SamWorker(worker_client, topology, sam_cfg(cfg, "cache"), 1.0, (600, 400), 30.0)
    roi = worker.rois[0]
    shape = (roi.y1 - roi.y0, roi.x1 - roi.x0)
    crop = np.zeros((*shape, 3), dtype=np.uint8)
    crop[:] = (30, 40, 50)
    blob = np.zeros(shape, dtype=bool)
    blob[10:40, 10:50] = True
    ys, xs = np.nonzero(blob)
    bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    from cryoguard.perception.sam_client import cache_key

    key = cache_key(crop, prompt, str(cfg["sam"]["model"]))
    seeder.store(key, prompt, (SamMask(mask=blob, bbox_px=bbox, score=1.0),), "fixture", shape)

    hit = worker_client.load_cached(key, shape)
    assert hit is not None
    assert hit.cached is True and hit.provenance == "fixture"
    assert hit.model == str(cfg["sam"]["model"])
    assert hit.prompt == prompt

    worker._accept(roi, hit, frame_index=0, t_s=0.0)
    samples = worker.poll()
    assert len(samples) == 1
    sample = samples[0]
    assert sample.cached is True
    assert sample.provenance == "fixture"
    assert worker.stats.live == 0
    assert worker.stats.api == 0
    assert worker.stats.cached == 1
    assert worker.stats.fixture == 1


def test_live_cache_hit_counts_cached_not_live(cfg, topology, tmp_path) -> None:
    transport = ScriptedTransport(FIXTURE, [0.0])
    worker_cfg = sam_cfg(cfg, "live")
    client = SamClient(sam_cfg(cfg, "live"), cache_dir=tmp_path / "cache", transport=transport)
    worker = SamWorker(client, topology, worker_cfg, 1.0, (600, 400), 30.0)

    async def case():
        worker.submit(warped(), frame_index=0, t_s=0.0)
        await worker.close()
        worker.submit(warped(), frame_index=30, t_s=1.0)
        await worker.close()
        return worker.poll()

    samples = asyncio.run(case())
    assert transport.calls == 1, "the second identical crop must hit the cache"
    assert worker.stats.live == 1
    assert worker.stats.cached == 1
    assert worker.stats.api == 2
    assert worker.stats.fixture == 0
    assert len(worker.stats.live_ms) == 1
    assert len(worker.stats.cached_ms) == 1
    assert all(s.provenance == "api" for s in samples)
    assert all(s.cached is (i == 1) for i, s in enumerate(samples))


def test_old_frame_result_cannot_overwrite_newer_state(cfg, topology, tmp_path) -> None:
    transport = ScriptedTransport(FIXTURE, [0.4, 0.0])
    worker_cfg = sam_cfg(cfg, "live", in_flight=2)
    client = SamClient(sam_cfg(cfg, "live"), cache_dir=tmp_path / "cache", transport=transport)
    worker = SamWorker(client, topology, worker_cfg, 1.0, (600, 400), 30.0)

    async def case():
        worker.submit(warped(), frame_index=0, t_s=0.0)  # slow: finishes last
        worker.submit(warped(), frame_index=30, t_s=1.0)  # fast: anchors first
        await worker.close()
        return worker.poll()

    samples = asyncio.run(case())
    assert transport.calls == 2
    assert worker.stats.stale == 1
    assert worker.stats.applied == 1
    assert [s.frame_index for s in samples] == [30]


def test_synchronous_stale_accept_is_dropped(cfg, topology, tmp_path) -> None:
    client = SamClient(sam_cfg(cfg, "cache"), cache_dir=tmp_path / "cache")
    worker = SamWorker(client, topology, sam_cfg(cfg, "cache"), 1.0, (600, 400), 30.0)
    roi = worker.rois[0]
    shape = (roi.y1 - roi.y0, roi.x1 - roi.x0)

    def result() -> SamResult:
        m = np.zeros(shape, dtype=bool)
        m[5:15, 5:20] = True
        return SamResult(
            masks=(SamMask(mask=m, bbox_px=(5, 5, 20, 15), score=0.9),),
            provenance="api",
            cached=False,
            cache_key="k",
            prompt="p",
            model="sam-3.1",
        )

    worker._accept(roi, result(), frame_index=30, t_s=1.0)
    worker._accept(roi, result(), frame_index=0, t_s=0.0)
    assert worker.stats.stale == 1
    assert worker.stats.applied == 1
    assert [s.frame_index for s in worker.poll()] == [30]


def test_zero_matches_are_recorded_not_failed(cfg, topology, tmp_path) -> None:
    client = SamClient(sam_cfg(cfg, "cache"), cache_dir=tmp_path / "seed")
    worker = SamWorker(client, topology, sam_cfg(cfg, "cache"), 1.0, (600, 400), 30.0)
    roi = worker.rois[0]
    empty = SamResult(
        masks=(),
        provenance="api",
        cached=False,
        cache_key="k",
        prompt=worker.prompt,
        model="sam-3.1",
    )
    worker._accept(roi, empty, frame_index=0, t_s=0.0)
    assert worker.poll() == []
    assert worker.stats.failures == 0
    assert worker.stats.per_prompt[worker.prompt]["empty"] == 1


def test_evaluate_prompts_records_every_prompt_and_keeps_the_default(
    cfg, topology, tmp_path
) -> None:
    good, bad = "fluorescent droplet?", "no-such-prompt"

    class RoutedTransport:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def __call__(self, image_bytes: bytes, prompt: str) -> dict:
            self.calls.append(prompt)
            if prompt == bad:
                raise SamAPIError("TimeoutError: endpoint down")
            return FIXTURE

    transport = RoutedTransport()
    client = SamClient(sam_cfg(cfg, "live"), cache_dir=tmp_path / "cache", transport=transport)
    image = np.zeros((48, 64, 3), dtype=np.uint8)
    image[:, :] = (30, 40, 50)

    trials = client.evaluate_prompts(image, [good, bad])
    assert [t.prompt for t in trials] == [good, bad]
    assert all(t.model == "sam-3.1" for t in trials)
    assert trials[0].ok and trials[0].result is not None and trials[0].mask_count == 1
    assert trials[0].result.prompt == good
    assert not trials[1].ok and trials[1].error is not None and "TimeoutError" in trials[1].error

    worker = SamWorker(
        SamClient(sam_cfg(cfg, "cache"), cache_dir=tmp_path / "w"),
        topology,
        sam_cfg(cfg, "cache"),
        1.0,
        (600, 400),
        30.0,
    )
    assert worker.prompt == str(cfg["sam"]["prompts"]["coolant"])


def test_sam_score_never_enters_fusion_p(cfg) -> None:
    params = inspect.signature(leak_score).parameters
    assert "score" not in params and "sam" not in "".join(params)

    tracker = FastTracker(cfg["perception"])
    frame = np.zeros((400, 600, 3), dtype=np.uint8)
    frame[190:230, 280:330] = (0, 230, 250)
    tracker.update_warped(frame, 0.0, 1.0)
    canvas_mask = np.zeros((400, 600), dtype=bool)
    canvas_mask[190:230, 280:330] = True
    tracker.reanchor(canvas_mask, frame, 0.0, 1.0)
    for track in tracker.tracks:
        assert not hasattr(track, "sam_score")
        assert "score" not in vars(track)
    p = leak_score(0.5, 0.1, 0.0, 1.0, cfg["fusion"]["weights"])
    assert p == leak_score(0.5, 0.1, 0.0, 1.0, cfg["fusion"]["weights"])


def test_sam_summary_exposes_split_latencies_and_provenance(cfg, topology, tmp_path) -> None:
    from cryoguard.pipeline import SamSummary

    transport = ScriptedTransport(FIXTURE, [0.0])
    client = SamClient(sam_cfg(cfg, "live"), cache_dir=tmp_path / "cache", transport=transport)
    worker = SamWorker(client, topology, sam_cfg(cfg, "live"), 1.0, (600, 400), 30.0)

    async def case():
        worker.submit(warped(), frame_index=0, t_s=0.0)
        await worker.close()
        worker.submit(warped(), frame_index=30, t_s=1.0)
        await worker.close()

    asyncio.run(case())
    summary = SamSummary.from_stats("live", worker.stats)
    assert len(summary.live_ms) == 1
    assert len(summary.cached_ms) == 1
    assert summary.api == 2 and summary.fixture == 0
    assert summary.stale == 0
    assert summary.per_prompt, "per-prompt audit must be exposed in the summary"
