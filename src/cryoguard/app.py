"""Command-line entry point for the vertical slice."""

from __future__ import annotations

import argparse
import asyncio
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
    parser.add_argument(
        "--sam",
        choices=("off", "cache", "live"),
        default=None,
        help="SAM mode; default comes from configs/default.yaml (off)",
    )
    parser.add_argument(
        "--sam-cache",
        default=None,
        help="override sam.cache_dir (useful for a seeded fixture cache)",
    )
    parser.add_argument(
        "--protocols",
        choices=("memory", "loopback", "external"),
        default=None,
        help=(
            "orchestrator transports; loopback starts the SIMULATED PLC/BMC on "
            "ephemeral sockets, external uses the configured ports"
        ),
    )
    args = parser.parse_args(argv)

    if not args.headless:
        print("the dashboard lands in Phase 9; run with --headless", file=sys.stderr)
        return 2

    cfg = load(ROOT / "configs" / "default.yaml")
    topology = yaml.safe_load((ROOT / "configs" / "topology.yaml").read_text(encoding="utf-8"))

    result = asyncio.run(
        run_scenario(
            args.scenario,
            cfg,
            topology,
            out_dir=args.out,
            max_seconds=args.max_seconds,
            sam_mode=args.sam,
            sam_cache_dir=args.sam_cache,
            protocol_mode=args.protocols,
        )
    )

    print(f"scenario  {result.scenario} (SIMULATED)")
    print(f"protocols {result.protocol_mode} (SIMULATED network mocks)")
    print(
        f"ticks     {result.ticks} at {result.tick_rate_hz:.0f} Hz "
        f"({result.rate_changes} rate changes)"
    )
    if result.sam is not None:
        sam = result.sam
        source = "fixture" if sam.fixture else "api"
        print(
            f"sam       {sam.mode}: {sam.applied} masks, {sam.cached} cached, "
            f"{sam.live} live, {sam.failures} failures ({source})"
        )
        if sam.mode == "cache" and sam.fixture == sam.applied and sam.applied:
            print("sam       provenance=fixture (seeded from the deterministic mask; no API call)")

    if result.report is None or result.decision is None:
        print(f"outcome   no incident in {result.ticks} ticks ({result.frames} frames)")
        return 0

    decision = result.decision
    print(f"decision  {decision.rule_id} {decision.label.value} p={decision.p:.3f}")
    if result.sequence_state is not None:
        print(f"sequence  {result.sequence_state.value}")
    print(f"actions   {len(result.report.deterministic_actions_taken)}")
    print(f"report    {result.report.report_source}")
    print(f"incident  {result.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
