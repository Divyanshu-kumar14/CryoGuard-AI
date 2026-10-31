"""P6-T07 end to end: a hot GPU trips the watchdog although vision sees glint."""

from __future__ import annotations

import asyncio
import copy
from pathlib import Path

import pytest
import yaml

from cryoguard.adapters.nvml import build_nvml
from cryoguard.config import load
from cryoguard.models import IncidentState
from cryoguard.orchestrator.adapters import SimulatedBmc
from cryoguard.orchestrator.incident_fsm import VERIFIED_OFF, PowerResult
from cryoguard.orchestrator.watchdog import (
    NvmlJunctionSensor,
    ThermalWatchdog,
    WatchdogActuationError,
    WatchdogReading,
    WatchdogTrip,
)
from cryoguard.pipeline import (
    _failed_trip_actions,
    _fallback_verified,
    _watchdog_fallback,
    run_scenario,
)
from cryoguard.report.schema import ActionModel, IncidentReport

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def cfg() -> dict:
    return load(ROOT / "configs" / "default.yaml")


@pytest.fixture(scope="module")
def topology() -> dict:
    return yaml.safe_load((ROOT / "configs" / "topology.yaml").read_text(encoding="utf-8"))


@pytest.mark.slow
def test_a_hot_junction_trips_an_emergency_incident_while_the_clip_is_benign(
    cfg, topology, tmp_path
) -> None:
    hot = copy.deepcopy(cfg)
    hot["nvml"]["fake_junction_c"] = 96.0
    hot["watchdog"]["period_ms"] = 0  # the scenario is seconds of simulated time

    result = asyncio.run(
        run_scenario(
            "glint",
            hot,
            topology,
            out_dir=tmp_path,
            max_seconds=3.0,
            protocol_mode="memory",
        )
    )

    assert result.decision is not None
    assert result.report is not None, "a hardware trip must file an incident"
    report = result.report
    assert report.physical_evidence.label == "EMERGENCY_TRIP"
    assert report.trip_reason == "JUNCTION_OVER_TEMP"
    assert report.physical_evidence.junction_c == 96.0
    assert report.physical_evidence.p_leak < 0.40  # vision saw nothing; the trip is thermal
    assert [action.command for action in report.deterministic_actions_taken] == [
        "VALVE_CLOSE",
        "SCHEDULER_DRAIN",
        "FORCE_OFF",
    ]
    assert result.sequence_state is IncidentState.EMERGENCY_TRIP
    assert "CHECKPOINT_VERIFY" not in [a.command for a in report.deterministic_actions_taken]
    assert "REQUEUE" not in [a.command for a in report.deterministic_actions_taken]


@pytest.mark.slow
def test_a_cool_junction_leaves_the_benign_clip_alone(cfg, topology, tmp_path) -> None:
    cool = copy.deepcopy(cfg)
    cool["watchdog"]["period_ms"] = 0

    result = asyncio.run(
        run_scenario(
            "glint",
            cool,
            topology,
            out_dir=tmp_path,
            max_seconds=3.0,
            protocol_mode="memory",
        )
    )

    assert result.report is None


@pytest.mark.slow
def test_a_trip_is_filed_even_when_the_socket_sequence_outlives_the_clip(
    cfg, topology, tmp_path
) -> None:
    """The clip ends in ~0.1 s; the Modbus/Redfish sequence takes ~1 s."""

    hot = copy.deepcopy(cfg)
    hot["nvml"]["fake_junction_c"] = 96.4
    hot["watchdog"]["period_ms"] = 0

    result = asyncio.run(
        run_scenario(
            "glint",
            hot,
            topology,
            out_dir=tmp_path,
            max_seconds=3.0,
            protocol_mode="loopback",
        )
    )

    assert result.path is not None
    assert result.report is not None
    assert result.report.report_source == "deterministic_fallback"
    assert [a.command for a in result.report.deterministic_actions_taken] == [
        "VALVE_CLOSE",
        "SCHEDULER_DRAIN",
        "FORCE_OFF",
    ]
    assert "junction=96.4 C >= limit 90.0 C" in result.report.deterministic_actions_taken[-1].detail


# --- Pipeline fallback wiring: primary -> fallback -> verified record ---
#
# These tests drive the real composition pieces (``_watchdog_fallback``,
# ``_fallback_verified``, ``_failed_trip_actions`` plus ``ThermalWatchdog``),
# not the classes in isolation. Memory mode uses the in-memory BMC; fault
# injection is via monkeypatched ports, never real hardware or the network.


class _HotSensor:
    def __init__(self, junction_c: float = 96.4) -> None:
        self.reading = WatchdogReading(junction_c)

    def read(self) -> WatchdogReading:
        return self.reading


def _limited_dog(cfg: dict, sensor, *, primary, fallback=None) -> ThermalWatchdog:
    return ThermalWatchdog(
        sensor,
        limit_c=float(cfg["thermal_model"]["watchdog_limit_c"]),
        period_s=0.0,
        max_staleness_ms=float(cfg["watchdog"]["max_staleness_ms"]),
        max_frozen_ticks=int(cfg["watchdog"]["max_frozen_ticks"]),
        actuate=primary,
        fallback=fallback,
    )


def test_primary_success_never_touches_the_fallback(cfg) -> None:
    async def case():
        primary: list[tuple[WatchdogTrip, str]] = []
        fallback: list[tuple[WatchdogTrip, str]] = []

        async def actuate(reason: WatchdogTrip, detail: str) -> None:
            primary.append((reason, detail))

        async def fallback_actuate(reason: WatchdogTrip, detail: str) -> None:
            fallback.append((reason, detail))

        dog = _limited_dog(cfg, _HotSensor(), primary=actuate, fallback=fallback_actuate)
        trip = await asyncio.wait_for(dog.run(asyncio.Event()), timeout=2.0)
        return dog, trip, primary, fallback

    dog, trip, primary, fallback = asyncio.run(case())

    assert trip is not None
    assert trip.reason is WatchdogTrip.JUNCTION_OVER_TEMP
    assert dog.actuated is True
    assert len(primary) == 1
    assert fallback == []
    assert dog.fallback_attempted is False


def test_primary_failure_runs_the_memory_fallback_and_verifies_off(cfg) -> None:
    """The fallback is a direct BMC ForceOff whose readback must say Off."""

    async def case():
        async def actuate(reason: WatchdogTrip, detail: str) -> None:
            raise ConnectionError("primary BMC unreachable")

        dog = _limited_dog(
            cfg,
            _HotSensor(),
            primary=actuate,
            fallback=_watchdog_fallback("memory", cfg),
        )
        with pytest.raises(WatchdogActuationError):
            await asyncio.wait_for(dog.run(asyncio.Event()), timeout=5.0)
        return dog

    dog = asyncio.run(case())

    assert dog.trip is not None
    assert dog.trip.reason is WatchdogTrip.JUNCTION_OVER_TEMP
    assert dog.actuated is False
    assert dog.fallback_attempted is True
    assert dog.fallback_ok is True
    assert dog.fallback_error == ""


def test_fallback_verified_requires_a_readback_not_an_accept() -> None:
    assert _fallback_verified(PowerResult(True, "Off", 1, 202, True, "")) is True
    assert _fallback_verified(PowerResult(True, "Off", 1, 202, False, "", VERIFIED_OFF)) is True
    # Accepted but still On: the node may still be running, never verified.
    assert _fallback_verified(PowerResult(False, "On", 1, 202, True, "still On")) is False
    assert _fallback_verified(PowerResult(True, "On", 1, 202, True, "")) is False
    # Off without a readback verdict: an echo of intent, not evidence.
    assert _fallback_verified(PowerResult(True, "Off", 1, 202, False, "")) is False
    assert _fallback_verified(PowerResult(False, "Unknown", 1, 0, True, "down")) is False


def test_fallback_timeout_is_recorded_as_failed(cfg, monkeypatch) -> None:
    class _StuckBmc(SimulatedBmc):
        def force_off(self) -> PowerResult:
            return PowerResult(
                ok=False,
                state="On",
                attempts=1,
                status_code=202,
                simulated=True,
                detail="PowerState still On after 3000 ms",
                status="VERIFICATION_TIMEOUT",
            )

    monkeypatch.setattr("cryoguard.pipeline.SimulatedBmc", _StuckBmc)

    async def case():
        async def actuate(reason: WatchdogTrip, detail: str) -> None:
            raise ConnectionError("primary BMC unreachable")

        dog = _limited_dog(
            cfg,
            _HotSensor(),
            primary=actuate,
            fallback=_watchdog_fallback("memory", cfg),
        )
        with pytest.raises(WatchdogActuationError):
            await asyncio.wait_for(dog.run(asyncio.Event()), timeout=5.0)
        return dog

    dog = asyncio.run(case())

    assert dog.fallback_attempted is True
    assert dog.fallback_ok is False
    assert "PowerState still On" in dog.fallback_error
    actions = _failed_trip_actions(
        dog,
        1.5,
        [
            ActionModel(
                seq=1,
                t_s=1.5,
                actor="BMC",
                command="FORCE_OFF",
                status="FAILED",
                detail="ConnectionError: primary BMC unreachable",
            )
        ],
    )
    by_command = [(action.command, action.status) for action in actions]
    assert ("TRIP", "INITIATED") in by_command
    assert ("FORCE_OFF", "FAILED") in by_command
    assert ("FORCE_OFF_FALLBACK", "FAILED") in by_command
    assert not any(
        action.command == "FORCE_OFF_FALLBACK" and action.status == VERIFIED_OFF
        for action in actions
    )


def test_both_paths_failing_leaves_the_trip_unconfirmed(cfg, monkeypatch) -> None:
    class _DeadBmc(SimulatedBmc):
        def force_off(self) -> PowerResult:
            raise ConnectionError("fallback BMC unreachable")

    monkeypatch.setattr("cryoguard.pipeline.SimulatedBmc", _DeadBmc)

    async def case():
        async def actuate(reason: WatchdogTrip, detail: str) -> None:
            raise ConnectionError("primary BMC unreachable")

        dog = _limited_dog(
            cfg,
            _HotSensor(),
            primary=actuate,
            fallback=_watchdog_fallback("memory", cfg),
        )
        with pytest.raises(WatchdogActuationError):
            await asyncio.wait_for(dog.run(asyncio.Event()), timeout=5.0)
        return dog

    dog = asyncio.run(case())

    assert dog.trip is not None
    assert dog.trip_initiated is True
    assert dog.actuated is False
    assert dog.fallback_attempted is True
    assert dog.fallback_ok is False
    assert "ConnectionError" in dog.fallback_error


def test_duplicate_triggers_share_one_actuation(cfg) -> None:
    async def case():
        calls: list[tuple[WatchdogTrip, str]] = []

        async def actuate(reason: WatchdogTrip, detail: str) -> None:
            calls.append((reason, detail))
            await asyncio.sleep(0.05)

        dog = _limited_dog(cfg, _HotSensor(), primary=actuate)
        first, second = await asyncio.gather(
            dog.check(WatchdogTrip.JUNCTION_OVER_TEMP, "first"),
            dog.check(WatchdogTrip.JUNCTION_OVER_TEMP, "second"),
        )
        return dog, first, second, calls

    dog, first, second, calls = asyncio.run(case())

    assert len(calls) == 1
    assert first is second
    assert first.detail == "first"
    assert dog.trip is first


def test_pipeline_watchdog_stops_cleanly_without_a_trip(cfg) -> None:
    """The real composition (build_nvml -> NvmlJunctionSensor -> watchdog)."""

    async def case():
        sensor = NvmlJunctionSensor(
            build_nvml(
                cfg["nvml"],
                tolerance_w=float(cfg["orchestrator"]["clamp_tolerance_w"]),
            )
        )

        async def actuate(reason: WatchdogTrip, detail: str) -> None:  # pragma: no cover
            raise AssertionError("a cool GPU must not actuate")

        dog = _limited_dog(cfg, sensor, primary=actuate, fallback=_watchdog_fallback("memory", cfg))
        stop = asyncio.Event()
        stop.set()
        trip = await asyncio.wait_for(dog.run(stop), timeout=2.0)
        return dog, trip

    dog, trip = asyncio.run(case())

    assert trip is None
    assert dog.trip is None
    assert dog.checks == 0
    assert dog.actuated is False


@pytest.mark.slow
def test_a_failed_primary_files_failed_safe_with_a_verified_fallback(
    cfg, topology, tmp_path, monkeypatch
) -> None:
    """The trip is initiated but unconfirmed: file FAILED_SAFE, then re-raise."""

    async def _boom(*args, **kwargs):
        raise ConnectionError("primary BMC unreachable")

    monkeypatch.setattr("cryoguard.pipeline._emergency_actions", _boom)

    hot = copy.deepcopy(cfg)
    hot["nvml"]["fake_junction_c"] = 96.4
    hot["watchdog"]["period_ms"] = 0

    with pytest.raises(WatchdogActuationError):
        asyncio.run(
            run_scenario(
                "glint",
                hot,
                topology,
                out_dir=tmp_path,
                max_seconds=3.0,
                protocol_mode="memory",
            )
        )

    paths = sorted(tmp_path.glob("INC-*.json"))
    assert len(paths) == 1
    report = IncidentReport.model_validate_json(paths[0].read_text(encoding="utf-8"))
    assert report.trip_reason == "JUNCTION_OVER_TEMP"
    assert report.report_source == "deterministic_fallback"
    by_command = {(action.command, action.status) for action in report.deterministic_actions_taken}
    assert ("TRIP", "INITIATED") in by_command
    assert ("FORCE_OFF", "FAILED") in by_command
    assert ("FORCE_OFF_FALLBACK", VERIFIED_OFF) in by_command


@pytest.mark.slow
def test_watchdog_trip_before_first_frame_files_incident(cfg, topology, tmp_path) -> None:
    hot = copy.deepcopy(cfg)
    hot["nvml"]["fake_junction_c"] = 96.4
    hot["watchdog"]["period_ms"] = 0

    result = asyncio.run(
        run_scenario(
            "glint",
            hot,
            topology,
            out_dir=tmp_path,
            max_seconds=-1.0,
            protocol_mode="memory",
        )
    )

    assert result.report is not None
    assert result.report.trip_reason == "JUNCTION_OVER_TEMP"
    assert result.report.physical_evidence.label == "EMERGENCY_TRIP"
    paths = list(tmp_path.glob("INC-*.json"))
    assert len(paths) == 1
