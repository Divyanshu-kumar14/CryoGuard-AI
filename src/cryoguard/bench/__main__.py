"""``make bench`` entry point (P8-T01..T05).

Runs every manifest clip offline, writes the deterministic benchmark and the
non-deterministic latency samples, and can re-check a committed baseline.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml

from cryoguard.bench.report import build_latency, build_payload, write_reports
from cryoguard.bench.runner import BenchError, load_manifest, run_benchmark
from cryoguard.config import config_hash, load

ROOT = Path(__file__).resolve().parents[3]


def git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def _freeze() -> dict[str, Any]:
    path = ROOT / "reports" / "fusion-freeze.json"
    if not path.is_file():
        raise BenchError("reports/fusion-freeze.json is missing; run scripts/freeze_fusion.py")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("strict_freeze") is not True:
        raise BenchError("the fusion freeze is not strict; benchmarks would be meaningless")
    return payload


def _summary(payload: dict[str, Any], latency: dict[str, Any]) -> str:
    lines = [
        f"split     {payload['split']} ({len(payload['clips'])} clips)",
        f"runHash   {payload['runHash']}",
    ]
    for name in sorted(payload["metrics"]):
        block = payload["metrics"][name]
        lines.append(
            f"{name:<9} recall {block['recall']['display']:<14} "
            f"fpr {block['falsePositiveRate']['display']:<14} "
            f"tp={block['tp']} fp={block['fp']} tn={block['tn']} fn={block['fn']}"
        )
    for row in payload["clips"]:
        lines.append(
            f"  {row['clipId']:<22} {row['split']:<5} "
            f"{'incident' if row['predictedIncident'] else 'none':<8} "
            f"{row['label']:<24} p={row['pDisplay']} "
            f"detect={row['detectionSDisplay']} actions={row['actionCount']}"
        )
    stages = latency["stages"]
    lines.append(
        "latency   perception p95 "
        f"{_fmt(stages['perceptionTickMs']['p95'])} ms | fusion p95 "
        f"{_fmt(stages['fusionMs']['p95'])} ms | sam cached p95 "
        f"{_fmt(stages['samCachedMs']['p95'])} ms | sequence p95 "
        f"{_fmt(stages['sequenceMs']['p95'])} ms"
    )
    return "\n".join(lines)


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CryoGuard offline benchmark")
    parser.add_argument("--split", default=None, help="all | dev | test")
    parser.add_argument("--manifest", default="fixtures/manifest.yaml")
    parser.add_argument("--out", default="reports")
    parser.add_argument("--sam", default="cache", choices=("off", "cache", "live"))
    parser.add_argument("--sam-cache", default=None)
    parser.add_argument(
        "--check",
        action="store_true",
        help="compare this run's hash with the committed benchmark and fail on drift",
    )
    args = parser.parse_args(argv)

    cfg = load(ROOT / "configs" / "default.yaml")
    topology = yaml.safe_load((ROOT / "configs" / "topology.yaml").read_text(encoding="utf-8"))
    split = args.split or str(cfg["bench"]["split"])
    freeze = _freeze()
    sha = git_sha()
    manifest = load_manifest(ROOT / args.manifest)
    if not manifest["clips"]:
        raise BenchError("the manifest has no clips")

    with tempfile.TemporaryDirectory(prefix="cryoguard-bench-") as work:
        runs = run_benchmark(
            manifest_path=ROOT / args.manifest,
            cfg=cfg,
            topology=topology,
            split=split,
            work_dir=work,
            sam_mode=args.sam,
            sam_cache_dir=args.sam_cache,
        )

    payload = build_payload(
        runs,
        cfg=cfg,
        split=split,
        config_hash=config_hash(cfg),
        fusion_hash=str(freeze["fusion_hash"]),
        git_sha=sha,
        confidence=float(cfg["bench"]["confidence"]),
    )
    latency = build_latency(
        runs,
        config_hash=payload["configHash"],
        run_hash=payload["runHash"],
        git_sha=sha,
    )

    if args.check:
        committed = Path(ROOT / args.out) / "benchmark.json"
        if not committed.is_file():
            raise BenchError(f"no committed benchmark to check: {committed}")
        baseline = json.loads(committed.read_text(encoding="utf-8"))
        if baseline.get("runHash") != payload["runHash"]:
            print("benchmark drift detected", file=sys.stderr)
            print(f"  committed {baseline.get('runHash')}", file=sys.stderr)
            print(f"  this run  {payload['runHash']}", file=sys.stderr)
            return 1
        if baseline.get("clips") != payload["clips"]:
            print("benchmark rows differ from the committed rows", file=sys.stderr)
            return 1
        print(f"benchmark matches the committed runHash {payload['runHash']}")
        return 0

    json_path, md_path, latency_path = write_reports(
        payload=payload, latency=latency, out_dir=ROOT / args.out
    )
    print(_summary(payload, latency))
    print(f"benchmark {json_path.relative_to(ROOT)}")
    print(f"markdown  {md_path.relative_to(ROOT)}")
    print(f"latency   {latency_path.relative_to(ROOT)} (not deterministic)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
