"""The frozen fusion arithmetic and the documentation that describes it (P5-T06)."""

from __future__ import annotations

import copy
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from cryoguard.config import load
from cryoguard.fusion.freeze import PERCEPTION_GATE_KEYS, fusion_hash

ROOT = Path(__file__).resolve().parents[1]
FREEZE = ROOT / "reports" / "fusion-freeze.json"
README = ROOT / "src" / "cryoguard" / "fusion" / "README.md"


@pytest.fixture(scope="module")
def cfg() -> dict:
    return load(ROOT / "configs" / "default.yaml")


@pytest.fixture(scope="module")
def freeze() -> dict:
    assert FREEZE.is_file(), "run scripts/freeze_fusion.py"
    return json.loads(FREEZE.read_text(encoding="utf-8"))


def test_the_freeze_is_strict(freeze) -> None:
    assert freeze["strict_freeze"] is True
    assert freeze["schema_version"] == "1.0"


def test_the_freeze_matches_the_live_arithmetic(cfg, freeze) -> None:
    assert freeze["fusion_hash"] == fusion_hash(cfg)
    assert freeze["fusion"] == cfg["fusion"]
    assert freeze["decision"] == cfg["decision"]
    assert freeze["perception_gates"] == {
        key: cfg["perception"][key] for key in PERCEPTION_GATE_KEYS
    }
    assert freeze["camera_health"] == cfg["camera"]["health"]


def test_a_weight_change_invalidates_the_freeze(cfg, freeze) -> None:
    mutated = copy.deepcopy(cfg)
    mutated["fusion"]["weights"]["bias"] += 0.1
    assert fusion_hash(mutated) != freeze["fusion_hash"]


def test_a_threshold_change_invalidates_the_freeze(cfg, freeze) -> None:
    mutated = copy.deepcopy(cfg)
    mutated["fusion"]["thresholds"]["confirmed_leak"] = 0.84
    assert fusion_hash(mutated) != freeze["fusion_hash"]


def test_a_quality_gate_change_invalidates_the_freeze(cfg, freeze) -> None:
    mutated = copy.deepcopy(cfg)
    mutated["perception"]["q_min"] = 0.4
    assert fusion_hash(mutated) != freeze["fusion_hash"]


def test_the_timestamp_is_utc_and_in_the_past(freeze) -> None:
    stamp = freeze["frozen_at"]
    assert stamp.endswith("Z")
    parsed = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    assert parsed.tzinfo is not None
    assert parsed <= datetime.now(UTC)


def test_the_freeze_never_claims_measurement(freeze) -> None:
    text = FREEZE.read_text(encoding="utf-8").lower()
    assert "calibrated" not in text
    assert "trained" not in text
    assert "no fitting was performed" in text


def test_a_later_benchmark_must_post_date_the_freeze(freeze) -> None:
    """The freeze has to exist before the first test-split run, not after it."""

    benchmark = ROOT / "reports" / "benchmark.json"
    if not benchmark.is_file():
        pytest.skip("no benchmark yet; Phase 8 writes it")

    payload = json.loads(benchmark.read_text(encoding="utf-8"))
    generated = payload.get("generated_at")
    if not generated:
        pytest.skip("benchmark has no generated_at")

    frozen_at = datetime.fromisoformat(freeze["frozen_at"].replace("Z", "+00:00"))
    run_at = datetime.fromisoformat(str(generated).replace("Z", "+00:00"))
    assert run_at >= frozen_at


def test_the_readme_documents_the_configured_values(cfg) -> None:
    assert README.is_file()
    text = README.read_text(encoding="utf-8")

    for value in cfg["fusion"]["weights"].values():
        assert str(value) in text, f"weight {value} missing from fusion/README.md"

    for key in ("nominal_cutoff", "suspect_cutoff", "escalation_cutoff", "confirmed_leak"):
        assert str(cfg["fusion"]["thresholds"][key]) in text, key

    for rule_id in ("R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8"):
        assert rule_id in text, rule_id


def test_the_readme_never_claims_measurement() -> None:
    text = README.read_text(encoding="utf-8").lower()
    assert "calibrated" not in text
    assert "trained" not in text
