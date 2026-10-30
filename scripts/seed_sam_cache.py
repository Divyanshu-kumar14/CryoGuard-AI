"""Seed the SAM cache from the SIMULATED clips' deterministic mask.

Every entry is provenance=fixture. See ``cryoguard.perception.sam_seed`` for why
that label matters.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from cryoguard.config import load
from cryoguard.perception.sam_seed import seed_fixture_cache
from cryoguard.pipeline import load_scenario

ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Seed the SAM cache (fixture provenance)")
    parser.add_argument("--scenario", default="leak")
    parser.add_argument("--cache-dir", default=None)
    args = parser.parse_args(argv)

    cfg = load(ROOT / "configs" / "default.yaml")
    topology = yaml.safe_load((ROOT / "configs" / "topology.yaml").read_text(encoding="utf-8"))
    scenario = load_scenario(args.scenario)
    cache_dir = args.cache_dir or cfg["sam"]["cache_dir"]

    written = seed_fixture_cache(
        clip_path=ROOT / scenario["clip"],
        topology=topology,
        cfg=cfg,
        cache_dir=cache_dir,
    )
    print(f"seeded {written} fixture entries into {cache_dir} (provenance=fixture)")
    print("these are NOT hosted SAM responses")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
