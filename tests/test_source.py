"""Replay determinism: the property the benchmark depends on."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from cryoguard.capture.source import SourceError, VideoSource


@pytest.fixture(scope="module")
def clip(tmp_path_factory) -> str:
    path = tmp_path_factory.mktemp("clips") / "ten_frames.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (64, 48))
    assert writer.isOpened()
    for index in range(10):
        writer.write(np.full((48, 64, 3), index * 20, dtype=np.uint8))
    writer.release()
    return str(path)


def test_two_replays_produce_identical_timestamps_and_pixels(clip: str) -> None:
    first = list(VideoSource(clip).frames())
    second = list(VideoSource(clip).frames())

    assert [f.t_s for f in first] == [f.t_s for f in second]
    assert [f.index for f in first] == list(range(10))
    for a, b in zip(first, second, strict=True):
        assert np.array_equal(a.image, b.image)


def test_timestamps_are_index_over_fps(clip: str) -> None:
    with VideoSource(clip) as source:
        assert source.timestamp(0) == 0.0
        assert source.timestamp(30) == pytest.approx(1.0)
        assert source.read(3).t_s == pytest.approx(3 / 30.0)


def test_fps_override_changes_timing_but_not_pixels(clip: str) -> None:
    with VideoSource(clip, fps_override=10.0) as source:
        frame = source.read(5)
        assert frame.t_s == pytest.approx(0.5)
        assert frame.image.shape == (48, 64, 3)


def test_missing_clip_fails_loudly(tmp_path) -> None:
    with pytest.raises(SourceError, match="not found"):
        VideoSource(tmp_path / "nope.mp4")


def test_negative_index_fails_loudly(clip: str) -> None:
    with VideoSource(clip) as source:
        with pytest.raises(SourceError, match="negative"):
            source.read(-1)
