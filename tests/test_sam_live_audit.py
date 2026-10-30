"""The committed live-SAM audit must keep its provenance honest."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "artifacts" / "sam_live_audit.json"


def test_the_live_audit_is_present_and_records_every_prompt() -> None:
    payload = json.loads(AUDIT.read_text(encoding="utf-8"))
    assert payload["mode"] == "live"
    trials = payload["trials"]
    assert len(trials) >= 3
    assert any(trial["prompt"] == "fluorescent yellow liquid droplet" for trial in trials)
    for trial in trials:
        assert {"prompt", "model", "ok", "maskCount"} <= set(trial)
        if trial["ok"]:
            assert trial["provenance"] == "api"
            assert trial["cached"] is False
            assert isinstance(trial["maskAreasPx"], list)


def test_a_zero_match_lane_is_recorded_as_a_result_not_an_error() -> None:
    payload = json.loads(AUDIT.read_text(encoding="utf-8"))
    zero = [trial for trial in payload["trials"] if trial["ok"] and trial["maskCount"] == 0]
    assert zero, "the audit should preserve at least one zero-match result"
    for trial in zero:
        assert "error" not in trial
        assert trial["maskAreasPx"] == []


def test_no_fixture_mask_is_labelled_live() -> None:
    payload = json.loads(AUDIT.read_text(encoding="utf-8"))
    for trial in payload["trials"]:
        if trial.get("provenance") == "fixture":
            raise AssertionError("a fixture result must never appear in the live audit")
