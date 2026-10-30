"""R1b: a watchdog that is blind to stale or frozen telemetry is not a watchdog.

Every case drives the real ``ThermalWatchdog.run`` loop with a scripted sensor,
so the freshness and freeze counters are exercised through the same path the
pipeline uses.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import replace
from pathlib import Path

import pytest

from cryoguard.config import load
from cryoguard.orchestrator.watchdog import (
    ThermalWatchdog,
    WatchdogError,
    WatchdogReading,
    WatchdogTrip,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def cfg() -> dict:
    return load(ROOT / "configs" / "default.yaml")


class ScriptedSensor:
    """Serves scripted readings with a fresh timestamp, then stops the loop."""

    def __init__(self, readings: list[WatchdogReading], stop: asyncio.Event, reads: int) -> None:
        self.readings = readings
        self.stop = stop
        self.limit = reads
        self.reads = 0

    def read(self) -> WatchdogReading:
        reading = self.readings[min(self.reads, len(self.readings) - 1)]
        self.reads += 1
        if self.reads >= self.limit:
            self.stop.set()
        return replace(reading, timestamp_s=time.monotonic())


class LaggingSensor:
    """A live read whose sample timestamp is ``lag_s`` behind the watchdog."""

    def __init__(self, lag_s: float) -> None:
        self.lag_s = lag_s

    def read(self) -> WatchdogReading:
        return WatchdogReading(junction_c=42.0, timestamp_s=time.monotonic() - self.lag_s)


def _sustained_kwargs(cfg: dict) -> dict:
    section = cfg.get("watchdog", {})
    return {
        "sustained_min_util_pct": float(section.get("sustained_load_min_util_pct", 10.0)),
        "sustained_min_power_w": float(section.get("sustained_load_min_power_w", 70.0)),
        "sustained_ticks": int(section.get("sustained_load_ticks", 5)),
        "sustained_release_util_pct": float(section.get("sustained_load_release_util_pct", 5.0)),
        "sustained_release_power_w": float(section.get("sustained_load_release_power_w", 50.0)),
    }


async def _run(cfg: dict, make_sensor, *, period_s: float = 0.0):
    seen: list[tuple[WatchdogTrip, str]] = []

    async def actuate(reason: WatchdogTrip, detail: str) -> None:
        seen.append((reason, detail))

    stop = asyncio.Event()
    sensor = make_sensor(stop)
    dog = ThermalWatchdog(
        sensor,
        limit_c=float(cfg["thermal_model"]["watchdog_limit_c"]),
        period_s=period_s,
        max_staleness_ms=float(cfg["watchdog"]["max_staleness_ms"]),
        max_frozen_ticks=int(cfg["watchdog"]["max_frozen_ticks"]),
        **_sustained_kwargs(cfg),
        actuate=actuate,
    )
    trip = await asyncio.wait_for(dog.run(stop), timeout=2.0)
    return dog, trip, seen, sensor


def test_the_config_carries_the_staleness_budget(cfg) -> None:
    assert cfg["watchdog"]["max_staleness_ms"] == 500
    assert cfg["watchdog"]["max_frozen_ticks"] == 15
    assert cfg["watchdog"]["sustained_load_min_util_pct"] == 10.0
    assert cfg["watchdog"]["sustained_load_min_power_w"] == 70.0
    assert cfg["watchdog"]["sustained_load_ticks"] == 5
    assert cfg["watchdog"]["sustained_load_release_util_pct"] == 5.0
    assert cfg["watchdog"]["sustained_load_release_power_w"] == 50.0


def test_a_sample_older_than_500_ms_trips_stale(cfg) -> None:
    dog, trip, seen, _ = asyncio.run(_run(cfg, lambda stop: LaggingSensor(0.62)))

    assert trip is not None
    assert trip.reason is WatchdogTrip.SENSOR_STALE
    assert "ms old" in trip.detail
    assert "500 ms" in trip.detail
    assert dog.trip == trip
    assert seen == [(WatchdogTrip.SENSOR_STALE, trip.detail)]
    assert dog.checks == 1


def test_a_value_frozen_for_sixteen_ticks_under_load_trips_frozen(cfg) -> None:
    """Genuine freeze: temp AND corroborating channels stuck under sustained load."""

    frozen = WatchdogReading(42.0, utilization_pct=80.0, power_w=250.0)
    dog, trip, seen, sensor = asyncio.run(
        _run(cfg, lambda stop: ScriptedSensor([frozen], stop, reads=40))
    )

    assert trip is not None
    assert trip.reason is WatchdogTrip.SENSOR_FROZEN
    assert "16 ticks" in trip.detail
    assert dog.frozen_ticks == 16
    assert sensor.reads == 16  # trips on the first sample past the budget
    assert seen == [(WatchdogTrip.SENSOR_FROZEN, trip.detail)]


def test_a_constant_temp_with_varying_power_does_not_trip(cfg) -> None:
    """Integer quantization plus a thermal plateau is not a stuck register."""

    readings = [
        WatchdogReading(42.0, utilization_pct=80.0, power_w=240.0 + float(index))
        for index in range(40)
    ]
    dog, trip, seen, sensor = asyncio.run(
        _run(cfg, lambda stop: ScriptedSensor(readings, stop, reads=40))
    )

    assert trip is None
    assert seen == []
    assert dog.checks == 40
    assert sensor.reads == 40


def test_a_single_channel_freeze_without_corroboration_degrades_without_tripping(
    cfg,
) -> None:
    """Temp-only repeat under load is suspect but unproven: no emergency."""

    dog, trip, seen, sensor = asyncio.run(
        _run(cfg, lambda stop: ScriptedSensor([WatchdogReading(42.0)], stop, reads=40))
    )

    assert trip is None
    assert seen == []
    assert dog.checks == 40
    assert sensor.reads == 40
    assert dog.degraded_ticks == dog.checks - int(cfg["watchdog"]["sustained_load_ticks"]) + 1
    assert dog.last_degraded is not None
    assert dog.last_degraded.reason is WatchdogTrip.SENSOR_HEALTH_DEGRADED


def test_a_replayed_sample_sequence_is_stale(cfg) -> None:
    """A sensor that re-serves the same sample is stale even with a fresh clock."""

    _, trip, _, _ = asyncio.run(
        _run(
            cfg,
            lambda stop: ScriptedSensor([WatchdogReading(42.0, sequence=7)], stop, reads=10),
        )
    )

    assert trip is not None
    assert trip.reason is WatchdogTrip.SENSOR_STALE
    assert "sequence 7 did not advance" in trip.detail


def test_fresh_changing_telemetry_does_not_trip(cfg) -> None:
    readings = [WatchdogReading(40.0 + 0.1 * index) for index in range(40)]

    dog, trip, seen, sensor = asyncio.run(
        _run(cfg, lambda stop: ScriptedSensor(readings, stop, reads=40))
    )

    assert trip is None
    assert seen == []
    assert dog.checks == 40
    assert sensor.reads == 40
    assert dog.frozen_ticks == 1


def test_a_constant_value_while_idle_does_not_trip(cfg) -> None:
    """A GPU at rest holds its temperature; that is not a stuck register."""

    dog, trip, _, _ = asyncio.run(
        _run(
            cfg,
            lambda stop: ScriptedSensor([WatchdogReading(42.0, under_load=False)], stop, reads=40),
        )
    )

    assert trip is None
    assert dog.checks == 40
    assert dog.frozen_ticks == 0


def test_a_zero_staleness_budget_is_refused(cfg) -> None:
    with pytest.raises(WatchdogError, match="max_staleness_ms"):
        ThermalWatchdog(
            LaggingSensor(0.0),
            limit_c=90.0,
            period_s=0.1,
            max_staleness_ms=0.0,
            actuate=lambda reason, detail: asyncio.sleep(0),
        )
