"""The README benchmark block is generated; this keeps it from drifting."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.render_readme_benchmark import render_block, replace_block

ROOT = Path(__file__).resolve().parents[1]


def test_readme_benchmark_block_matches_the_committed_payload() -> None:
    payload = json.loads((ROOT / "reports" / "benchmark.json").read_text(encoding="utf-8"))
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    rendered = replace_block(readme, render_block(payload))
    assert rendered == readme, "README benchmark block is stale; run make readme-sync"


def test_readme_block_has_no_hand_copied_hash() -> None:
    """A hash pasted into prose would bypass the generated block."""

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    start = readme.index("<!-- BENCH:BEGIN (generated from reports/benchmark.json) -->")
    end = readme.index("<!-- BENCH:END -->") + len("<!-- BENCH:END -->")
    outside = readme[:start] + readme[end:]
    payload = json.loads((ROOT / "reports" / "benchmark.json").read_text(encoding="utf-8"))
    for key in ("runHash", "configHash"):
        assert payload[key] not in outside, f"{key} is hand-copied outside the generated block"


def test_generated_block_uses_the_synthetic_title_and_fault_counts() -> None:
    payload = json.loads((ROOT / "reports" / "benchmark.json").read_text(encoding="utf-8"))
    block = render_block(payload)
    assert "Synthetic Scenarios" in block
    faults = payload["faultInjection"]
    assert f"{faults['passed']}/{faults['total']} cases passing" in block
    assert "100% accuracy" not in block.lower()
