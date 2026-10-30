"""P6-T07: the thermal watchdog is independent of the perception loop."""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

import pytest

from cryoguard.adapters.nvml import FakeNvml, GpuBackend, GpuObservation
from cryoguard.config import load
from cryoguard.orchestrator.watchdog import (
    EMERGENCY_TRIPS,
    NvmlJunctionSensor,
    ThermalWatchdog,
    WatchdogActuationError,
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
    assert dog.trip_initiated is True
    assert dog.actuated is True
    assert seen == [(WatchdogTrip.JUNCTION_OVER_TEMP, trip.detail)]


def test_an_unreadable_sensor_trips_fail_safe(cfg) -> None:
    _, trip, seen = asyncio.run(_run_watchdog(cfg, BrokenSensor()))

    assert trip is not None
    assert trip.reason is WatchdogTrip.SENSOR_UNAVAILABLE
    assert "NVML handle is gone" in trip.detail
    assert len(seen) == 1


def test_a_failing_actuation_callback_is_recorded_and_reraised(cfg, caplog) -> None:
    """A trip whose actuation raises must not leave the trip state ambiguous."""

    async def case():
        async def actuate(reason: WatchdogTrip, detail: str) -> None:
            raise ConnectionResetError("BMC socket reset")

        dog = ThermalWatchdog(
            FixedSensor(WatchdogReading(96.4)),
            limit_c=float(cfg["thermal_model"]["watchdog_limit_c"]),
            period_s=0.0,
            actuate=actuate,
        )
        with pytest.raises(WatchdogActuationError):
            await dog.run(asyncio.Event())
        return dog

    with caplog.at_level(logging.CRITICAL):
        dog = asyncio.run(case())

    assert dog.trip_initiated is True
    assert dog.trip_initiated_s is not None
    assert dog.trip is not None
    assert dog.trip.reason is WatchdogTrip.JUNCTION_OVER_TEMP
    assert dog.trip.initiated_s == dog.trip_initiated_s
    assert dog.actuation_failed is True
    assert "ConnectionResetError" in dog.actuation_error
    assert "BMC socket reset" in dog.actuation_error
    assert dog.actuated is False  # the action was never confirmed
    assert "Watchdog actuation callback failed: BMC socket reset" in caplog.text


def test_a_failed_actuation_still_attempts_the_fallback(cfg) -> None:
    """The fallback is a direct power cut, attempted once before the re-raise."""

    async def case():
        seen: list[tuple[WatchdogTrip, str]] = []

        async def actuate(reason: WatchdogTrip, detail: str) -> None:
            raise ConnectionResetError("BMC socket reset")

        async def fallback(reason: WatchdogTrip, detail: str) -> None:
            seen.append((reason, detail))

        dog = ThermalWatchdog(
            FixedSensor(WatchdogReading(96.4)),
            limit_c=float(cfg["thermal_model"]["watchdog_limit_c"]),
            period_s=0.0,
            actuate=actuate,
            fallback=fallback,
        )
        with pytest.raises(WatchdogActuationError):
            await dog.run(asyncio.Event())
        return dog, seen

    dog, seen = asyncio.run(case())

    assert dog.fallback_attempted is True
    assert dog.fallback_ok is True
    assert [reason for reason, _ in seen] == [WatchdogTrip.JUNCTION_OVER_TEMP]


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


# --- P0 regression: frozen / stale / unavailable telemetry (R1b) ---
#
# Production thresholds are reused verbatim (500 ms / 15 ticks / 10 % / 70 W /
# 5 arming ticks); only the sample values are scripted. Every case drives
# ``observe`` with explicit clocks or the real ``run`` loop, never wall sleeps.


def _p0_dog(sensor, **over) -> ThermalWatchdog:
    params: dict = {
        "limit_c": 90.0,
        "period_s": 0.0,
        "actuate": lambda reason, detail: asyncio.sleep(0),  # noqa: E731
        "max_staleness_ms": 500.0,
        "max_frozen_ticks": 15,
        "sustained_min_util_pct": 10.0,
        "sustained_min_power_w": 70.0,
        "sustained_ticks": 5,
        "sustained_release_util_pct": 5.0,
        "sustained_release_power_w": 50.0,
    }
    params.update(over)
    return ThermalWatchdog(sensor, **params)


class _VaryingGpu:
    """Sample injection: 65 C plateau while every corroborating channel moves."""

    def __init__(self) -> None:
        self.calls = 0

    def sample(self) -> GpuObservation:
        index = self.calls
        self.calls += 1
        return GpuObservation(
            temp_c=65.0,
            utilization_pct=75.0 + float(index % 3),
            power_w=240.0 + float(index),
            clock_mhz=1500.0 + float(index),
            device_index=0,
            sampled_monotonic_s=time.monotonic(),
            backend=GpuBackend.SIMULATED,
            detail="SIMULATED GPU temperature",
        )


class _ObsBackend:
    def __init__(self, obs: GpuObservation) -> None:
        self._obs = obs

    def sample(self) -> GpuObservation:
        return self._obs


class _ThrowingNvml:
    def sample(self) -> GpuObservation:
        raise RuntimeError("NVML handle is gone")


class _InvalidSensor:
    """Serves ``valid=False`` samples, then stops the loop."""

    def __init__(self, stop: asyncio.Event, reads: int) -> None:
        self.stop = stop
        self.limit = reads
        self.reads = 0

    def read(self) -> WatchdogReading:
        self.reads += 1
        if self.reads >= self.limit:
            self.stop.set()
        return WatchdogReading(65.0, valid=False, detail="simulated invalid sample")


def test_p0_constant_65c_with_changing_power_and_util_never_trips() -> None:
    """A thermal plateau under varying load is live telemetry, not a freeze."""

    sensor = NvmlJunctionSensor(_VaryingGpu())
    dog = _p0_dog(sensor)
    records = [dog.observe(sensor.read(), time.monotonic()) for _ in range(40)]

    assert all(record is None for record in records)
    assert dog.frozen_ticks == 1
    assert dog.trip is None
    assert sensor.nvml.calls == 40


def test_p0_frozen_all_channels_under_sustained_load_trips_frozen() -> None:
    frozen = WatchdogReading(65.0, utilization_pct=80.0, power_w=250.0, clock_mhz=1500.0)
    dog = _p0_dog(FixedSensor(frozen))
    now = 1000.0
    record = None
    for index in range(40):
        record = dog.observe(frozen, now + 0.1 * index)

    assert record is not None
    assert record.reason is WatchdogTrip.SENSOR_FROZEN
    assert record.reason in EMERGENCY_TRIPS
    assert dog.frozen_ticks == 40
    assert "40 ticks" in record.detail


def test_p0_changing_temp_under_load_never_trips() -> None:
    dog = _p0_dog(FixedSensor(WatchdogReading(42.0)))
    now = 1000.0
    records = [
        dog.observe(
            WatchdogReading(60.0 + 0.5 * index, utilization_pct=80.0, power_w=250.0),
            now + 0.1 * index,
        )
        for index in range(40)
    ]

    assert all(record is None for record in records)
    assert dog.trip is None
    assert dog.frozen_ticks == 1


def test_p0_idle_constant_temp_never_trips() -> None:
    dog = _p0_dog(FixedSensor(WatchdogReading(42.0)))
    records = [
        dog.observe(WatchdogReading(42.0, under_load=False), 1000.0 + 0.1 * index)
        for index in range(40)
    ]

    assert all(record is None for record in records)
    assert dog.frozen_ticks == 0
    assert dog.trip is None


def test_p0_repeated_upstream_sequence_is_stale() -> None:
    dog = _p0_dog(FixedSensor(WatchdogReading(42.0)))
    now = 2000.0
    assert dog.observe(WatchdogReading(42.0, timestamp_s=now, sequence=7), now) is None

    record = dog.observe(WatchdogReading(42.0, timestamp_s=now + 0.1, sequence=7), now + 0.1)

    assert record is not None
    assert record.reason is WatchdogTrip.SENSOR_STALE
    assert "sequence 7 did not advance" in record.detail


def test_p0_stale_source_timestamps_trip() -> None:
    replayed = _p0_dog(FixedSensor(WatchdogReading(42.0)))
    assert replayed.observe(WatchdogReading(42.0, source_timestamp_s=500.0), 2000.0) is None
    record = replayed.observe(WatchdogReading(42.0, source_timestamp_s=500.0), 2000.1)
    assert record is not None
    assert record.reason is WatchdogTrip.SENSOR_STALE
    assert "device sample timestamp did not advance" in record.detail

    lagging = _p0_dog(FixedSensor(WatchdogReading(42.0)))
    record = lagging.observe(
        WatchdogReading(42.0, timestamp_s=1000.0, source_timestamp_s=999.38), 1000.0
    )
    assert record is not None
    assert record.reason is WatchdogTrip.SENSOR_STALE
    assert "ms old" in record.detail

    old_poll = _p0_dog(FixedSensor(WatchdogReading(42.0)))
    record = old_poll.observe(WatchdogReading(42.0, timestamp_s=999.38), 1000.0)
    assert record is not None
    assert record.reason is WatchdogTrip.SENSOR_STALE
    assert "500 ms" in record.detail


def test_p0_nvml_exception_trips_sensor_unavailable() -> None:
    async def case():
        seen: list[tuple[WatchdogTrip, str]] = []

        async def actuate(reason: WatchdogTrip, detail: str) -> None:
            seen.append((reason, detail))

        dog = _p0_dog(NvmlJunctionSensor(_ThrowingNvml()), actuate=actuate)
        trip = await asyncio.wait_for(dog.run(asyncio.Event()), timeout=2.0)
        return dog, trip, seen

    dog, trip, seen = asyncio.run(case())

    assert trip is not None
    assert trip.reason is WatchdogTrip.SENSOR_UNAVAILABLE
    assert "NVML handle is gone" in trip.detail
    assert seen == [(WatchdogTrip.SENSOR_UNAVAILABLE, trip.detail)]
    assert dog.trip == trip


def test_p0_invalid_sample_degrades_without_false_verification() -> None:
    dog = _p0_dog(FixedSensor(WatchdogReading(65.0)))
    record = dog.observe(WatchdogReading(65.0, valid=False), 1000.0)

    assert record is not None
    assert record.reason is WatchdogTrip.SENSOR_HEALTH_DEGRADED
    assert record.reason not in EMERGENCY_TRIPS
    assert dog.trip is None

    bad = GpuObservation(
        temp_c=65.0,
        utilization_pct=None,
        power_w=None,
        clock_mhz=None,
        device_index=0,
        sampled_monotonic_s=time.monotonic(),
        valid=False,
        backend=GpuBackend.SIMULATED,
        detail="SIMULATED GPU temperature",
    )
    mapped = NvmlJunctionSensor(_ObsBackend(bad)).read()
    assert mapped.valid is False
    record = dog.observe(mapped, 1000.1)
    assert record is not None
    assert record.reason is WatchdogTrip.SENSOR_HEALTH_DEGRADED

    async def run_case():
        stop = asyncio.Event()
        loop_dog = _p0_dog(_InvalidSensor(stop, 10))
        trip = await asyncio.wait_for(loop_dog.run(stop), timeout=2.0)
        return loop_dog, trip

    loop_dog, trip = asyncio.run(run_case())
    assert trip is None
    assert loop_dog.trip is None
    assert loop_dog.actuated is False
    assert loop_dog.actuation_failed is False
    assert loop_dog.degraded_ticks == 10
    assert loop_dog.checks == 10


def test_p0_single_channel_repeat_degrades_without_frozen_trip() -> None:
    dog = _p0_dog(FixedSensor(WatchdogReading(65.0)))
    records = [dog.observe(WatchdogReading(65.0), 1000.0 + 0.1 * i) for i in range(40)]

    assert all(
        record is None or record.reason is WatchdogTrip.SENSOR_HEALTH_DEGRADED for record in records
    )
    assert not any(record is not None and record.reason in EMERGENCY_TRIPS for record in records)
    assert dog.last_degraded is not None
    assert dog.last_degraded.reason is WatchdogTrip.SENSOR_HEALTH_DEGRADED
    assert dog.trip is None


def test_p0_simulated_vs_real_backend_distinguished() -> None:
    fake = FakeNvml(200.0)
    simulated = fake.sample()
    assert simulated.backend is GpuBackend.SIMULATED
    assert "SIMULATED" in simulated.detail
    assert simulated.source_sequence is None  # never fabricated locally

    mapped = NvmlJunctionSensor(fake).read()
    assert mapped.sequence is None
    assert mapped.valid is True

    real = GpuObservation(
        temp_c=65.0,
        utilization_pct=80.0,
        power_w=250.0,
        clock_mhz=1500.0,
        device_index=0,
        sampled_monotonic_s=time.monotonic(),
        source_sequence=41,
        backend=GpuBackend.REAL,
        detail="NVML_TEMPERATURE_GPU",
    )
    assert real.backend is not simulated.backend
    real_mapped = NvmlJunctionSensor(_ObsBackend(real)).read()
    assert real_mapped.sequence == 41  # a genuine counter passes through
    assert real_mapped.utilization_pct == 80.0
    assert real_mapped.power_w == 250.0
    assert real_mapped.clock_mhz == 1500.0
