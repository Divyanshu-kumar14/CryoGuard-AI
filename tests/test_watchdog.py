"""P6-T07: the thermal watchdog is independent of the perception loop."""

from __future__ import annotations

import asyncio
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


class FixedSensor:
    def __init__(self, readings: list[WatchdogReading] | WatchdogReading) -> None:
        self._readings = readings if isinstance(readings, list) else [readings]
        self.reads = 0

    def read(self) -> WatchdogReading:
        self.reads += 1
        index = min(self.reads - 1, len(self._readings) - 1)
        return self._readings[index]


class BrokenSensor:
    def read(self) -> WatchdogReading:
        raise RuntimeError("NVML handle is gone")


async def _run_watchdog(cfg: dict, sensor, *, period_s: float = 0.0):
    seen: list[tuple[WatchdogTrip, str]] = []

    async def actuate(reason: WatchdogTrip, detail: str) -> None:
        seen.append((reason, detail))

    dog = ThermalWatchdog(
        sensor,
        limit_c=float(cfg["thermal_model"]["watchdog_limit_c"]),
        period_s=period_s,
        actuate=actuate,
    )
    trip = await asyncio.wait_for(dog.run(asyncio.Event()), timeout=2.0)
    return dog, trip, seen


def test_the_limit_is_the_configured_watchdog_temperature(cfg) -> None:
    dog = ThermalWatchdog(
        FixedSensor(WatchdogReading(40.0)),
        limit_c=float(cfg["thermal_model"]["watchdog_limit_c"]),
        period_s=0.0,
        actuate=lambda reason, detail: asyncio.sleep(0),
    )

    assert dog.limit_c == 90.0
    assert dog.limit_c < float(cfg["thermal_model"]["t_junction_max_c"])
    assert dog.evaluate(WatchdogReading(89.9)) is None
    assert dog.evaluate(WatchdogReading(90.0)) is WatchdogTrip.JUNCTION_OVER_TEMP


def test_a_failed_clamp_readback_trips_even_at_a_cool_junction() -> None:
    dog = ThermalWatchdog(
        FixedSensor(WatchdogReading(40.0)),
        limit_c=90.0,
        period_s=0.0,
        actuate=lambda reason, detail: asyncio.sleep(0),
    )

    assert dog.evaluate(WatchdogReading(40.0, clamp_ok=False)) is (
        WatchdogTrip.CLAMP_READBACK_FAILED
    )


def test_a_plc_fault_trips() -> None:
    dog = ThermalWatchdog(
        FixedSensor(WatchdogReading(40.0)),
        limit_c=90.0,
        period_s=0.0,
        actuate=lambda reason, detail: asyncio.sleep(0),
    )

    assert dog.evaluate(WatchdogReading(40.0, plc_fault=True)) is WatchdogTrip.PLC_FAULT


def test_over_temperature_outranks_the_other_faults() -> None:
    dog = ThermalWatchdog(
        FixedSensor(WatchdogReading(40.0)),
        limit_c=90.0,
        period_s=0.0,
        actuate=lambda reason, detail: asyncio.sleep(0),
    )

    reading = WatchdogReading(96.0, clamp_ok=False, plc_fault=True)
    assert dog.evaluate(reading) is WatchdogTrip.JUNCTION_OVER_TEMP


def test_the_watchdog_still_trips_when_the_perception_loop_is_dead(cfg) -> None:
    async def case():
        perception_started = asyncio.Event()

        async def perception() -> None:
            perception_started.set()
            await asyncio.Event().wait()  # stalled forever: no decisions, no ticks

        stalled = asyncio.create_task(perception())
        await perception_started.wait()
        stalled.cancel()  # the loop is gone before the watchdog looks at the GPU
        with pytest.raises(asyncio.CancelledError):
            await stalled

        dog, trip, seen = await _run_watchdog(
            cfg, FixedSensor(WatchdogReading(96.4)), period_s=0.01
        )
        return dog, trip, seen

    dog, trip, seen = asyncio.run(case())

    assert trip is not None
    assert trip.reason is WatchdogTrip.JUNCTION_OVER_TEMP
    assert "96.4 C" in trip.detail
    assert dog.trip == trip
    assert seen == [(WatchdogTrip.JUNCTION_OVER_TEMP, trip.detail)]


def test_an_unreadable_sensor_trips_fail_safe(cfg) -> None:
    _, trip, seen = asyncio.run(_run_watchdog(cfg, BrokenSensor()))

    assert trip is not None
    assert trip.reason is WatchdogTrip.SENSOR_UNAVAILABLE
    assert "NVML handle is gone" in trip.detail
    assert len(seen) == 1


def test_a_healthy_watchdog_stops_without_tripping(cfg) -> None:
    async def case():
        seen: list[WatchdogTrip] = []

        async def actuate(reason: WatchdogTrip, detail: str) -> None:
            seen.append(reason)

        dog = ThermalWatchdog(
            FixedSensor(WatchdogReading(42.0)),
            limit_c=90.0,
            period_s=0.001,
            actuate=actuate,
        )
        stop = asyncio.Event()
        stop.set()
        return await dog.run(stop), seen, dog.checks

    trip, seen, checks = asyncio.run(case())

    assert trip is None
    assert seen == []
    assert checks == 0


def test_a_negative_period_is_refused() -> None:
    with pytest.raises(WatchdogError, match="period_s"):
        ThermalWatchdog(
            FixedSensor(WatchdogReading(40.0)),
            limit_c=90.0,
            period_s=-0.1,
            actuate=lambda reason, detail: asyncio.sleep(0),
        )


def test_a_zero_limit_is_refused() -> None:
    with pytest.raises(WatchdogError, match="limit_c"):
        ThermalWatchdog(
            FixedSensor(WatchdogReading(40.0)),
            limit_c=0.0,
            period_s=0.1,
            actuate=lambda reason, detail: asyncio.sleep(0),
        )
