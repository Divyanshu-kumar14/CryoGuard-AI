"""Live SAM special-token lane: box/mask parsing and official mask decoding.

The live Responses API streams one line of special-token text per frame:
``<Nf>id<|box;...|><|mask;...data=H,W,<enc>payload|>``. These tests use the
official parser's test-only encoder to build a tiny one_bit mask, so they stay
offline and do not pretend a fixture is an API response.
"""

from __future__ import annotations

import numpy as np
import pytest
from meta_sam_parser._mask_codec import _encode_raster

from cryoguard.perception.sam_client import SamResponseError, parse_sam_text

MASK = _encode_raster([0, 1, 1, 0], 2, 2)


def line(
    encoded: str = MASK,
    *,
    object_id: int = 0,
    box: tuple[int, int, int, int] = (0, 0, 2, 2),
    box_score: str | None = None,
    mask_score: str | None = None,
    mask_hw: tuple[int, int] = (2, 2),
) -> str:
    x0, y0, x1, y1 = box
    box_c = f";c={box_score}" if box_score else ""
    mask_c = f";c={mask_score}" if mask_score else ""
    return (
        f"text<0f>{object_id}<|box;x1={x0};y1={y0};x2={x1};y2={y1};w=10;h=10{box_c}|>"
        f"<|mask;x=0;y=0{mask_c};data={mask_hw[0]},{mask_hw[1]},{encoded}|>"
    )


def test_a_live_record_decodes_to_a_crop_sized_mask() -> None:
    masks = parse_sam_text(line(), (10, 12))

    assert len(masks) == 1
    assert masks[0].mask.shape == (10, 12)
    assert masks[0].mask.sum() == masks[0].mask.size // 2, "two of four source pixels are set"
    assert masks[0].bbox_px == (0, 0, 2, 2)
    assert masks[0].score == 1.0


def test_two_records_on_one_lane_are_both_returned() -> None:
    text = line(object_id=0) + "," + line(object_id=1, box=(4, 4, 6, 6))

    masks = parse_sam_text(text, (10, 10))

    assert len(masks) == 2
    assert [mask.bbox_px for mask in masks] == [(0, 0, 2, 2), (4, 4, 6, 6)]


def test_the_mask_confidence_wins_over_the_box_confidence() -> None:
    masks = parse_sam_text(line(box_score="0.99", mask_score="0.80"), (10, 10))

    assert masks[0].score == pytest.approx(0.80)


def test_the_box_is_clamped_to_the_crop() -> None:
    masks = parse_sam_text(line(box=(0, 0, 20, 20)), (10, 10))

    assert masks[0].bbox_px == (0, 0, 10, 10)


def test_zero_matches_is_a_valid_empty_lane() -> None:
    assert parse_sam_text("", (10, 10)) == ()
    assert parse_sam_text("   \n", (10, 10)) == ()


def test_non_empty_text_without_a_record_is_an_error() -> None:
    with pytest.raises(SamResponseError, match="no box/mask record"):
        parse_sam_text("text<0f>the model refused", (10, 10))


def test_a_malformed_mask_payload_is_an_error() -> None:
    with pytest.raises(SamResponseError, match="did not decode"):
        parse_sam_text(line("!not-a-valid-payload"), (10, 10))


def test_the_lane_is_not_position_dependent() -> None:
    """A sparse object id (0, 2) must parse; ids are not dense indices."""

    text = line(object_id=0) + "," + line(object_id=2, box=(3, 3, 5, 5))

    masks = parse_sam_text(text, (10, 10))

    assert len(masks) == 2
    assert np.count_nonzero(masks[1].mask) > 0
