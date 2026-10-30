"""SAM client: prompt guard, response parsing, and the NFR-DET-01 cache."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from cryoguard.perception.sam_client import (
    SamCacheMiss,
    SamClient,
    SamConfigError,
    SamPromptError,
    SamResponseError,
    cache_key,
    count_prompt_tokens,
    encode_png,
    parse_response,
    validate_prompt,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "fixtures" / "sam" / "response_schema_fixture.json"
MASK_SHAPE = (48, 64)


def sam_cfg(tmp_path: Path, mode: str = "cache") -> dict:
    return {
        "mode": mode,
        "model": "sam-3.1",
        "base_url": "https://api.meta.ai/v1",
        "timeout_sec": 4.0,
        "prompt_max_tokens": 32,
        "mask_threshold": 127,
        "max_retries": 0,
        "cache_dir": str(tmp_path / "sam_cache"),
    }


def crop() -> np.ndarray:
    image = np.zeros((*MASK_SHAPE, 3), dtype=np.uint8)
    image[:, :] = (30, 40, 50)
    return image


def fixture_payload() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


class CountingTransport:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.calls = 0

    def __call__(self, image_bytes: bytes, prompt: str) -> dict:
        self.calls += 1
        assert image_bytes.startswith(b"\x89PNG")
        assert prompt
        return self.payload


def test_a_forty_token_prompt_is_rejected_client_side() -> None:
    prompt = " ".join(["droplet"] * 40)
    assert count_prompt_tokens(prompt) == 40
    with pytest.raises(SamPromptError, match="40 tokens"):
        validate_prompt(prompt, 32)


def test_punctuation_cannot_smuggle_extra_tokens_past_the_guard() -> None:
    prompt = ".".join(["x"] * 33)
    with pytest.raises(SamPromptError):
        validate_prompt(prompt, 32)


def test_the_configured_prompt_passes_the_guard(cfg) -> None:
    for prompt in cfg["sam"]["prompts"].values():
        validate_prompt(prompt, cfg["sam"]["prompt_max_tokens"])


def test_schema_fixture_parses_into_a_mask() -> None:
    masks = parse_response(fixture_payload(), MASK_SHAPE)

    assert len(masks) == 1
    assert masks[0].mask.shape == MASK_SHAPE
    assert masks[0].bbox_px == (8, 10, 40, 30)
    assert masks[0].score == pytest.approx(0.93)
    assert int(masks[0].mask.sum()) == 32 * 20


def test_a_payload_without_masks_fails_loudly() -> None:
    with pytest.raises(SamResponseError, match="no masks found"):
        parse_response({"output": []}, MASK_SHAPE)


def test_cache_key_is_the_documented_concatenation() -> None:
    image = b"\x89PNG\r\n\x1a\nbody"
    expected = hashlib.sha256(image + b"fluorescent droplet" + b"sam-3.1").hexdigest()
    assert cache_key(image, "fluorescent droplet", "sam-3.1") == expected


def test_second_call_serves_from_cache_without_network(tmp_path: Path) -> None:
    transport = CountingTransport(fixture_payload())
    client = SamClient(sam_cfg(tmp_path, "live"), transport=transport)

    first = client.segment(crop(), "fluorescent yellow liquid droplet")
    second = client.segment(crop(), "fluorescent yellow liquid droplet")

    assert transport.calls == 1
    assert first.cached is False
    assert second.cached is True
    assert second.provenance == "api"
    assert np.array_equal(first.masks[0].mask, second.masks[0].mask)

    # The cache is keyed by image + prompt + model, so a different prompt is a miss.
    with pytest.raises(SamCacheMiss):
        SamClient(sam_cfg(tmp_path, "cache")).segment(crop(), "water bead on metal")


def test_cache_mode_never_falls_back_to_the_network(tmp_path: Path) -> None:
    client = SamClient(sam_cfg(tmp_path, "cache"), transport=CountingTransport(fixture_payload()))
    with pytest.raises(SamCacheMiss, match="seed_sam_cache"):
        client.segment(crop(), "fluorescent yellow liquid droplet")


def test_fixture_provenance_can_never_be_reported_as_api(tmp_path: Path) -> None:
    client = SamClient(sam_cfg(tmp_path, "live"), transport=CountingTransport(fixture_payload()))
    client.segment(crop(), "fluorescent yellow liquid droplet")
    key = cache_key(
        encode_png(crop()),
        "fluorescent yellow liquid droplet",
        "sam-3.1",
    )
    metadata = json.loads((tmp_path / "sam_cache" / f"{key}.json").read_text())
    assert metadata["provenance"] == "api"

    # A fixture-seeded entry is labelled as such and is not upgraded on read.
    fixture_client = SamClient(sam_cfg(tmp_path / "fixtures"))
    fixture_client.store("deadbeef", "fluorescent yellow liquid droplet", (), "fixture", MASK_SHAPE)
    entry = json.loads((tmp_path / "fixtures" / "sam_cache" / "deadbeef.json").read_text())
    assert entry["provenance"] == "fixture"
    loaded = fixture_client.load_cached("deadbeef", MASK_SHAPE)
    assert loaded is not None and loaded.provenance == "fixture"


def test_live_mode_without_a_key_or_transport_fails_at_construction(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("META_API_KEY", raising=False)
    with pytest.raises(SamConfigError, match="META_API_KEY"):
        SamClient(sam_cfg(tmp_path, "live"))
