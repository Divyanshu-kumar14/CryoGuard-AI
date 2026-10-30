"""Deterministic file replay.

Timestamps come from frame index and the declared frame rate. The wall clock is
never read, so two replays of the same file produce byte-identical timestamps
and frames. That property is what makes `make bench` reproducible.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


class SourceError(RuntimeError):
    """Raised when a clip cannot be opened or its timing is undeclared."""


@dataclass(frozen=True)
class Frame:
    index: int
    t_s: float
    image: np.ndarray


class VideoSource:
    """Sequential and random-access replay of one video file."""

    def __init__(self, path: str | Path, fps_override: float | None = None) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise SourceError(f"clip not found: {self.path}")

        self._capture = cv2.VideoCapture(str(self.path))
        if not self._capture.isOpened():
            raise SourceError(f"OpenCV cannot decode: {self.path}")

        declared = float(self._capture.get(cv2.CAP_PROP_FPS))
        self.fps = fps_override if fps_override is not None else declared
        if not self.fps or self.fps <= 0 or self.fps != self.fps:  # NaN-safe
            self._capture.release()
            raise SourceError(
                f"{self.path} declares no usable frame rate ({declared!r}); pass fps_override"
            )

        self.frame_count = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def timestamp(self, index: int) -> float:
        """Pure function of the index. No wall clock, no accumulated drift."""

        if index < 0:
            raise SourceError(f"negative frame index: {index}")
        return index / self.fps

    def read(self, index: int) -> Frame | None:
        if index < 0:
            raise SourceError(f"negative frame index: {index}")
        self._capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, image = self._capture.read()
        if not ok:
            return None
        return Frame(index=index, t_s=self.timestamp(index), image=image)

    def frames(self, start: int = 0, stop: int | None = None) -> Iterator[Frame]:
        self._capture.set(cv2.CAP_PROP_POS_FRAMES, start)
        index = start
        while stop is None or index < stop:
            ok, image = self._capture.read()
            if not ok:
                return
            yield Frame(index=index, t_s=self.timestamp(index), image=image)
            index += 1

    def __len__(self) -> int:
        return self.frame_count

    def close(self) -> None:
        self._capture.release()

    def __enter__(self) -> VideoSource:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
