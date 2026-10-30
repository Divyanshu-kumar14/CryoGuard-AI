"""Pipeline event emission and headless/UI parity (P9-T01).

The central claim is that observing a run cannot change it. Two runs of the same
clip, one with a bus and one without, must produce byte-identical incident files
and the same ordered action list.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
import yaml

from cryoguard.events import EventBus
from cryoguard.pipeline import run_scenario

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def topology() -> dict:
    return yaml.safe_load((ROOT / "configs" / "topology.yaml").read_text(encoding="utf-8"))


def _run(
    name: str,
    cfg: dict,
    topology: dict,
    out_dir: Path,
    *,
    bus: EventBus | None = None,
    sam: str = "off",
):  # type: ignore[no-untyped-def]
    return asyncio.run(
        run_scenario(
            name,
            cfg,
            topology,
            out_dir=out_dir,
            sam_mode=sam,
            protocol_mode="memory",
            allow_llm=False,
            bus=bus,
        )
    )


def test_the_event_stream_reports_the_same_actions_as_the_sealed_report(
    cfg: dict, topology: dict, tmp_path: Path
) -> None:
    bus = EventBus()
    result = _run("leak", cfg, topology, tmp_path, bus=bus)
    events = bus.snapshot()

    assert events[0].type == "RUN_STARTED"
    assert events[-1].type == "RUN_FINISHED"
    assert [event.seq for event in events] == list(range(1, len(events) + 1))

    assert events[0].data["simulated"] is True
    assert events[0].data["provenance"] == "synthetic"
    assert events[0].data["scenario"] == "slow_leak_qd4"
    assert events[0].data["canvas"]["widthPx"] == 1200

    actions = [event for event in events if event.type == "ACTION_EXECUTED"]
    report_actions = result.report.deterministic_actions_taken
    assert len(actions) == len(report_actions) == 6
    assert [
        (
            event.data["seq"],
            event.data["actor"],
            event.data["command"],
            event.data["status"],
            event.data["detail"],
            round(event.t_s, 3),
        )
        for event in actions
    ] == [
        (action.seq, action.actor, action.command, action.status, action.detail, action.t_s)
        for action in report_actions
    ]
    assert [event.data["seq"] for event in actions] == [1, 2, 3, 4, 5, 6]

    closed = [event for event in events if event.type == "INCIDENT_CLOSED"]
    assert len(closed) == 1
    assert closed[0].data["incidentId"] == result.report.incident_id
    assert closed[0].data["reportSource"] == "deterministic_fallback"
    assert closed[0].data["sequenceState"] == "RESOLVED_PENDING_REPAIR"
    assert closed[0].data["report"]["deterministicActionsTaken"] == [
        action.model_dump(by_alias=True) for action in report_actions
    ]

    frame_events = [event for event in events if event.type == "FRAME_TICK"]
    assert frame_events
    assert any(event.data["contours"] for event in frame_events)
    width = events[0].data["canvas"]["widthPx"]
    height = events[0].data["canvas"]["heightPx"]
    for event in frame_events:
        for polygon in event.data["contours"]:
            for x, y in polygon:
                assert 0 <= x <= width
                assert 0 <= y <= height

    decision_events = [event for event in events if event.type == "DECISION_CHANGE"]
    assert decision_events
    assert any(event.data["changed"] for event in decision_events)
    assert decision_events[-1].data["label"] == "CONFIRMED_LEAK"
    assert decision_events[-1].data["p"] == pytest.approx(0.853, abs=1e-3)

    finished = events[-1].data
    assert finished["outcome"] == "incident"
    assert finished["incidentId"] == result.report.incident_id
    assert finished["sequenceState"] == "RESOLVED_PENDING_REPAIR"


def test_the_bus_cannot_change_the_incident(cfg: dict, topology: dict, tmp_path: Path) -> None:
    with_bus = _run("leak", cfg, topology, tmp_path / "observed", bus=EventBus())
    headless = _run("leak", cfg, topology, tmp_path / "headless")

    assert with_bus.path is not None
    assert headless.path is not None
    assert with_bus.path.read_bytes() == headless.path.read_bytes()
    assert with_bus.report == headless.report
    assert with_bus.decision == headless.decision


def test_a_benign_clip_emits_no_actions_and_no_incident(
    cfg: dict, topology: dict, tmp_path: Path
) -> None:
    bus = EventBus()
    result = _run("glint", cfg, topology, tmp_path, bus=bus, sam="off")
    events = bus.snapshot()

    assert result.report is None
    assert not [event for event in events if event.type == "ACTION_EXECUTED"]
    assert not [event for event in events if event.type == "INCIDENT_CLOSED"]
    assert events[-1].type == "RUN_FINISHED"
    assert events[-1].data["outcome"] == "no_incident"
    assert events[-1].data["ticks"] > 0


def test_the_hidden_event_emits_its_two_soft_actions(
    cfg: dict, topology: dict, tmp_path: Path
) -> None:
    bus = EventBus()
    result = _run("hidden", cfg, topology, tmp_path, bus=bus)
    events = bus.snapshot()

    actions = [event.data for event in events if event.type == "ACTION_EXECUTED"]
    assert [(action["actor"], action["command"], action["status"]) for action in actions] == [
        ("CONSOLE", "ALERT", "RAISED"),
        ("SLURM", "SCHEDULER_DRAIN", "DRAINED"),
    ]
    assert result.report is not None
    assert events[-1].data["outcome"] == "incident"
    closed = [event for event in events if event.type == "INCIDENT_CLOSED"][0]
    assert closed.data["sequenceState"] is None
