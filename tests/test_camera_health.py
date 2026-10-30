"""Camera health: a blind frame must never be treated as a nominal chassis."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from cryoguard.capture.board import render_board
from cryoguard.config import load
from cryoguard.perception.health import assess_health

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def cfg() -> dict:
    return load(ROOT / "configs" / "default.yaml")


@pytest.fixture(scope="module")
def topology() -> dict:
    return yaml.safe_load((ROOT / "configs" / "topology.yaml").read_text(encoding="utf-8"))


def test_a_rendered_board_is_healthy(cfg, topology) -> None:
    board = cv2.cvtColor(render_board(topology, px_per_mm=2.0), cv2.COLOR_GRAY2BGR)
    health = assess_health(board, cfg["camera"])
    assert health.ok, health.reasons


def test_a_black_frame_is_not_healthy(cfg) -> None:
    health = assess_health(np.zeros((400, 600, 3), dtype=np.uint8), cfg["camera"])
    assert health.ok is False
    assert "brightness" in health.reasons


def test_a_blown_out_frame_is_not_healthy(cfg) -> None:
    health = assess_health(np.full((400, 600, 3), 255, dtype=np.uint8), cfg["camera"])
    assert health.ok is False
    assert "brightness" in health.reasons or "saturation" in health.reasons


def test_a_defocused_frame_is_not_healthy(cfg) -> None:
    # A single flat mid-grey image has no edges at all: focus variance is zero.
    frame = np.full((400, 600, 3), 128, dtype=np.uint8)
    health = assess_health(frame, cfg["camera"])
    assert health.ok is False
    assert "focus" in health.reasons


def test_assess_health_rejects_a_non_bgr_frame(cfg) -> None:
    with pytest.raises(ValueError, match="BGR"):
        assess_health(np.zeros((400, 600), dtype=np.uint8), cfg["camera"])
