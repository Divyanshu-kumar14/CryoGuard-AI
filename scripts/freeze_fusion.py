"""Freeze the fusion priors into reports/fusion-freeze.json (P5-T06).

Run this only when a deliberate change has been made to the fusion arithmetic,
a threshold, a gate, a mask-quality input, or the camera-health bounds. The test
suite fails when the file and the config disagree, which is the point.

The timestamp is the only non-deterministic field. It must predate the first
test-split benchmark run; the benchmark records ``generated_at`` and checks it.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from cryoguard.config import config_hash, load
from cryoguard.fusion.freeze import frozen_payload, fusion_hash

ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Freeze the fusion priors")
    parser.add_argument("--out", default="reports/fusion-freeze.json")
    args = parser.parse_args(argv)

    cfg = load(ROOT / "configs" / "default.yaml")
    payload = frozen_payload(cfg)

    freeze = {
        "schema_version": "1.0",
        "strict_freeze": True,
        "frozen_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
        "source": "hand-set design priors (D-03); no fitting was performed",
        "spec": "PRD v1.1 §6.3 + prd-delta-v1.2 §4",
        "hash_note": (
            "fusion_hash covers the frozen subset and is enforced by "
            "tests/test_fusion_freeze.py; config_hash is the full config at freeze "
            "time, recorded for traceability only"
        ),
        **payload,
        "config_hash": config_hash(cfg),
        "fusion_hash": fusion_hash(cfg),
    }

    path = ROOT / args.out
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(freeze, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"froze fusion priors -> {path}")
    print(f"fusion_hash {freeze['fusion_hash']}  config_hash {freeze['config_hash']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
