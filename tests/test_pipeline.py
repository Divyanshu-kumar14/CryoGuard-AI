"""The M1 chain on the simulated fixtures: clip in, incident out."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
import yaml

from cryoguard.config import load
from cryoguard.models import SystemLabel
from cryoguard.perception.sam_seed import seed_fixture_cache
from cryoguard.pipeline import RunResult, load_scenario, run_scenario

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def cfg() -> dict:
    return load(ROOT / "configs" / "default.yaml")


@pytest.fixture(scope="module")
def topology() -> dict:
    return yaml.safe_load((ROOT / "configs" / "topology.yaml").read_text(encoding="utf-8"))


def run(
    name: str,
    cfg: dict,
    topology: dict,
    out_dir: Path,
    seconds: float = 10.0,
    **kwargs,
) -> RunResult:
    return asyncio.run(
        run_scenario(name, cfg, topology, out_dir=out_dir, max_seconds=seconds, **kwargs)
    )


@pytest.fixture(scope="module")
def leak(cfg, topology, tmp_path_factory):
    return run("leak", cfg, topology, tmp_path_factory.mktemp("leak"))


@pytest.fixture(scope="module")
def hidden(cfg, topology, tmp_path_factory):
    return run("hidden", cfg, topology, tmp_path_factory.mktemp("hidden"))


@pytest.mark.slow
def test_leak_clip_confirms_and_writes_an_incident(leak) -> None:
    assert leak.decision is not None
    assert leak.decision.rule_id == "R2"
    assert leak.decision.label is SystemLabel.CONFIRMED_LEAK
    assert leak.report is not None
    assert leak.path is not None and leak.path.is_file()
    assert leak.report.physical_evidence.zone == 0


@pytest.mark.slow
def test_safe_sequence_order_is_preserved_in_the_report(leak) -> None:
    actions = leak.report.deterministic_actions_taken
    commands = [action.command for action in actions]
    assert commands == [
        "CLAMP",
        "VALVE_CLOSE",
        "CHECKPOINT_VERIFY",
        "FORCE_OFF",
        "SCHEDULER_DRAIN",
        "REQUEUE",
    ]

    assert actions[0].t_s < actions[1].t_s  # clamp verified before the valve moves
    assert actions[1].t_s < actions[2].t_s  # isolation before the checkpoint gate
    assert actions[1].status == "LIMIT_SWITCH_CLOSED"


@pytest.mark.slow
def test_hidden_hydraulic_event_drains_without_moving_a_valve(hidden) -> None:
    assert hidden.decision is not None
    assert hidden.decision.rule_id == "R4"
    assert hidden.decision.label is SystemLabel.HIDDEN_HYDRAULIC_EVENT

    commands = [action.command for action in hidden.report.deterministic_actions_taken]
    assert commands == ["ALERT", "SCHEDULER_DRAIN"]
    assert "VALVE_CLOSE" not in commands


@pytest.mark.slow
def test_condensation_and_glint_produce_no_incident(cfg, topology, tmp_path_factory) -> None:
    for name in ("condensation", "glint"):
        result = run(name, cfg, topology, tmp_path_factory.mktemp(name), seconds=8.0)
        assert result.report is None, name
        assert result.decision is not None, name


@pytest.mark.slow
def test_incident_ids_are_stable_and_unique(cfg, topology, tmp_path_factory, leak, hidden) -> None:
    assert leak.report.incident_id.endswith("-01")
    assert hidden.report.incident_id.endswith("-04")

    first = leak.path.read_bytes()
    again = run("leak", cfg, topology, tmp_path_factory.mktemp("again"))
    assert again.path.read_bytes() == first


@pytest.mark.slow
def test_adaptive_sampling_changes_rate_on_evidence(leak) -> None:
    # The leak clip starts with a candidate (5 Hz) and escalates (10 Hz).
    assert leak.rate_changes >= 2
    assert leak.tick_rate_hz == pytest.approx(10.0)


@pytest.mark.slow
def test_seeded_sam_cache_reanchors_without_a_network_call(
    cfg, topology, tmp_path, tmp_path_factory
) -> None:
    cache = tmp_path / "sam_cache"
    scenario = load_scenario("leak")
    written = seed_fixture_cache(
        clip_path=ROOT / scenario["clip"],
        topology=topology,
        cfg=cfg,
        cache_dir=cache,
    )
    assert written > 0

    result = run(
        "leak",
        cfg,
        topology,
        tmp_path_factory.mktemp("sam"),
        seconds=10.0,
        sam_mode="cache",
        sam_cache_dir=cache,
    )

    assert result.sam is not None
    assert result.sam.mode == "cache"
    assert result.sam.applied > 0
    assert result.sam.cached == result.sam.applied
    assert result.sam.live == 0
    assert result.sam.fixture == result.sam.applied
    assert result.sam.failures == 0
    assert result.report is not None
