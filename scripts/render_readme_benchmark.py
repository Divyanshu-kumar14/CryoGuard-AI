"""Render the README benchmark block from ``reports/benchmark.json``.

The committed benchmark payload is the single authoritative source of the
numbers in the README. Hand-copying them is how a README goes stale, so this
script owns the block between the markers and a test asserts they agree.

Usage::

    PYTHONPATH=src python scripts/render_readme_benchmark.py
    PYTHONPATH=src python scripts/render_readme_benchmark.py --check
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "reports" / "benchmark.json"
README = ROOT / "README.md"
BEGIN = "<!-- BENCH:BEGIN (generated from reports/benchmark.json) -->"
END = "<!-- BENCH:END -->"


def _metric(payload: dict[str, object], name: str) -> dict[str, object]:
    value = payload.get(name)
    if not isinstance(value, dict):
        raise SystemExit(f"benchmark payload is missing the {name!r} block")
    return value


def _fault(payload: dict[str, object]) -> dict[str, object]:
    value = payload.get("faultInjection")
    if not isinstance(value, dict):
        raise SystemExit("benchmark payload is missing the 'faultInjection' block")
    return value


def render_block(payload: dict[str, object]) -> str:
    """Pure function: payload -> Markdown block. No clock, no I/O."""

    headline = _metric(payload, "headline")
    recall = _metric(headline, "recall")
    fpr = _metric(headline, "falsePositiveRate")
    latency = _metric(payload, "detectionLatencyS")
    centroid = _metric(payload, "localization")
    source = _metric(payload, "sourceLocalization")
    checkpoint = _metric(payload, "checkpointRecovery")
    valve = _metric(payload, "valveIsolation")
    verification = _metric(payload, "verification")
    faults = _fault(payload)
    confidence = int(payload.get("confidencePercent", 95))

    lines = [
        BEGIN,
        "",
        f"**{payload['title']}**",
        "",
        f"Config hash `{payload['configHash']}` | "
        f"run hash `{payload['runHash']}` | "
        f"fusion freeze `{payload['fusionHash']}` | split `{payload['split']}`",
        "",
        f"{verification['display']}.",
        "",
        f"| Metric | Measured | {confidence} % Clopper–Pearson CI | Count |",
        "| :--- | :--- | :--- | :--- |",
        f"| Leak detection recall | {recall['display']} | {recall['ciDisplay']} | "
        f"{recall['n']} positive clips |",
        f"| False positive rate | {fpr['display']} | {fpr['ciDisplay']} | "
        f"{fpr['n']} negative clips |",
        f"| Detection latency (scenario time) | {latency['display']} | — | "
        f"{latency['n']} confirmed events |",
        f"| Spill-centroid localization (≤ {payload['localizationToleranceMm']} mm) | "
        f"{centroid['display']} | {centroid['ciDisplay']} | {centroid['n']} localizable events |",
        f"| Mean spill-centroid error | {centroid['meanErrorDisplay']} | — | "
        f"{centroid['n']} localizable events |",
        f"| Source estimate within tolerance | {source['display']} | {source['ciDisplay']} | "
        f"{source['n']} of {source['total']} clips estimated |",
        f"| Mean source error | {source['meanErrorDisplay']} | — | "
        f"{source['n']} estimated events |",
        f"| Checkpoint recovery | {checkpoint['display']} | {checkpoint['ciDisplay']} | "
        f"{checkpoint['n']} checkpointed incidents |",
        f"| Valve isolation verified | {valve['display']} | {valve['ciDisplay']} | "
        f"{valve['n']} commanded closes |",
        "",
        f"Fault-injection matrix (`{faults['suite']}`): "
        f"**{faults['passed']}/{faults['total']} cases passing**, counted separately from "
        "the four scenario fixtures and never merged into the metrics above.",
        "",
        f"> {payload['notice']}",
        "",
        END,
    ]
    return "\n".join(lines)


def replace_block(readme: str, block: str) -> str:
    start = readme.find(BEGIN)
    end = readme.find(END)
    if start == -1 or end == -1 or end < start:
        raise SystemExit(f"README is missing the benchmark markers {BEGIN!r} / {END!r}")
    return readme[:start] + block + readme[end + len(END) :]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail when the README block differs from the committed benchmark",
    )
    args = parser.parse_args(argv)

    payload = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    block = render_block(payload)
    readme = README.read_text(encoding="utf-8")
    updated = replace_block(readme, block)

    if args.check:
        if updated != readme:
            print("README benchmark block is stale; run scripts/render_readme_benchmark.py")
            return 1
        print("README benchmark block matches reports/benchmark.json")
        return 0

    README.write_text(updated, encoding="utf-8")
    print(f"updated {README.relative_to(ROOT)} from {BENCHMARK.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
