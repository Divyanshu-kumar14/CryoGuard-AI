"""The frozen fusion payload and its hash (P5-T06).

The freeze covers the arithmetic and every gate that changes a rule outcome. It
deliberately does not cover unrelated config (ports, report model, demo paths):
those may change without invalidating a benchmark, and pretending otherwise
would force noisy freeze updates.

``fusion_hash`` is the executable contract: a weight, threshold, gate, or mask
quality input that changes produces a different hash, and the test fails until
the freeze file is regenerated on purpose.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

PERCEPTION_GATE_KEYS = (
    "q_min",
    "exposure_clip_max",
    "solidity_ref",
    "wetness_ticks",
    "min_area_mm2",
    "max_area_mm2",
    "growth_window",
    "candidate_ttl_ticks",
)


def frozen_payload(cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "fusion": cfg["fusion"],
        "decision": cfg["decision"],
        "perception_gates": {key: cfg["perception"][key] for key in PERCEPTION_GATE_KEYS},
        "camera_health": cfg["camera"]["health"],
    }


def fusion_hash(cfg: dict[str, Any]) -> str:
    canonical = json.dumps(frozen_payload(cfg), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
