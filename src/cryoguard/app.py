"""Command-line entry point for the vertical slice."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from cryoguard.config import load
from cryoguard.pipeline import run_scenario

ROOT = Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CryoGuard AI vertical slice")
    parser.add_argument("--scenario", default="leak", help="leak | condensation | glint | hidden")
    parser.add_argument(
        "--headless", action="store_true", help="no dashboard (the only mode so far)"
    )
    parser.add_argument("--out", default=None, help="incident directory")
    parser.add_argument("--max-seconds", type=float, default=None, help="stop after N clip seconds")
    args = parser.parse_args(argv)

    if not args.headless:
        print("the dashboard lands in Phase 9; run with --headless", file=sys.stderr)
        return 2

    cfg = load(ROOT / "configs" / "default.yaml")
    topology = yaml.safe_load((ROOT / "configs" / "topology.yaml").read_text(encoding="utf-8"))

    result = run_scenario(
        args.scenario, cfg, topology, out_dir=args.out, max_seconds=args.max_seconds
    )

    print(f"scenario  {result.scenario} (SIMULATED)")
    if result.report is None or result.decision is None:
        print(f"outcome   no incident in {result.ticks} ticks ({result.frames} frames)")
        return 0

    decision = result.decision
    print(f"decision  {decision.rule_id} {decision.label.value} p={decision.p:.3f}")
    print(f"actions   {len(result.report.deterministic_actions_taken)}")
    print(f"incident  {result.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
