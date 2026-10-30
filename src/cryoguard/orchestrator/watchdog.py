"""Independent thermal watchdog (P6-T07, R1b).

The safety claim is that this task does **not** depend on the perception or
fusion loop. It reads hardware state on its own period, and if the junction
temperature crosses the limit - or the clamp readback failed, or the PLC raised
its fault coil - it runs the R1 emergency sequence and does not wait for a
decision. Killing perception cannot make the watchdog blind.

A sensor read that raises is treated as a trip: a watchdog that cannot see the
junction cannot certify that the GPUs are cool. The failure is recorded with
reason ``SENSOR_UNAVAILABLE`` rather than swallowed.

Freshness versus freeze, and what escalates:

* ``SENSOR_STALE`` (emergency): the poll timestamp is older than
  ``max_staleness_ms``; or a genuinely supplied upstream ``sequence`` did not
  advance (replay); or a genuinely supplied upstream ``source_timestamp_s``
  did not advance or lags the poll by more than the staleness budget.
  A locally generated poll counter is never used here — ``sequence=None``
  means "no upstream counter", and the check is skipped.
* ``SENSOR_FROZEN`` (emergency): the temperature repeats exactly for more
  than ``max_frozen_ticks`` consecutive samples **while** sustained load is
  proven by hysteresis **and** every other supplied channel
  (utilization/power/clock) repeats too. A constant integer temperature with
  changing power or utilization proves the sensor is live and never trips.
* ``SENSOR_HEALTH_DEGRADED`` (not an emergency): the sample is suspect but below
  the emergency bar — an explicitly invalid sample, or a temperature that
  repeats under sustained load without any corroborating channel. It is
  counted and logged, and polling continues; it never runs the R1 sequence.
* ``JUNCTION_OVER_TEMP`` / ``CLAMP_READBACK_FAILED`` / ``PLC_FAULT`` /
  ``SENSOR_UNAVAILABLE`` (emergency): unchanged severity order, same R1 path.

Sustained load uses hysteresis so one utilization spike cannot arm the freeze
detector: per-tick load is asserted when any supplied channel clears its
``sustained_*_min_*`` threshold (or the legacy ``under_load`` flag when no
channel is supplied); ``_sustained`` latches only after ``sustained_ticks``
consecutive asserted ticks and releases only when every supplied channel
drops below its release threshold.

The actuation path is fail-safe too: a trip is marked initiated with a timestamp
*before* the callback runs, a callback that raises is logged CRITICAL and given
one fallback attempt (a direct BMC ForceOff, if the composition root injects
one), and the failure is re-raised as ``WatchdogActuationError`` so a process
that cannot confirm the emergency action exits non-zero.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol

DEFAULT_MAX_STALENESS_MS = 500.0
DEFAULT_MAX_FROZEN_TICKS = 15
DEFAULT_SUSTAINED_MIN_UTIL_PCT = 10.0
DEFAULT_SUSTAINED_MIN_POWER_W = 70.0
DEFAULT_SUSTAINED_TICKS = 5
DEFAULT_SUSTAINED_RELEASE_UTIL_PCT = 5.0
DEFAULT_SUSTAINED_RELEASE_POWER_W = 50.0

logger = logging.getLogger(__name__)


class WatchdogError(ValueError):
    """An unusable watchdog configuration."""


class WatchdogTrip(StrEnum):
    JUNCTION_OVER_TEMP = "JUNCTION_OVER_TEMP"
    CLAMP_READBACK_FAILED = "CLAMP_READBACK_FAILED"
    PLC_FAULT = "PLC_FAULT"
    SENSOR_UNAVAILABLE = "SENSOR_UNAVAILABLE"
    SENSOR_STALE = "SENSOR_STALE"
    SENSOR_FROZEN = "SENSOR_FROZEN"
    SENSOR_HEALTH_DEGRADED = "SENSOR_HEALTH_DEGRADED"


#: Trips that run the R1 emergency sequence. ``SENSOR_HEALTH_DEGRADED`` is the only
#: non-emergency state: it is counted and logged, polling continues.
EMERGENCY_TRIPS = frozenset(
    {
        WatchdogTrip.JUNCTION_OVER_TEMP,
        WatchdogTrip.CLAMP_READBACK_FAILED,
        WatchdogTrip.PLC_FAULT,
        WatchdogTrip.SENSOR_UNAVAILABLE,
        WatchdogTrip.SENSOR_STALE,
        WatchdogTrip.SENSOR_FROZEN,
    }
)


class WatchdogActuationError(RuntimeError):
    """The trip was initiated but its actuation callback failed.

    Raised on every failed trip so the outer process exits non-zero instead of
    continuing after an actuation nobody can confirm.
    """

    def __init__(self, reason: WatchdogTrip, detail: str, error: BaseException) -> None:
        super().__init__(
            f"watchdog actuation failed for {reason.value}: {type(error).__name__}: {error}"
        )
        self.reason = reason
        self.detail = detail
        self.error = error


@dataclass(frozen=True)
class WatchdogReading:
    junction_c: float
    clamp_ok: bool = True
    plc_fault: bool = False
    detail: str = ""
    # Monotonic time the poll was taken. ``None`` means the watchdog stamps it
    # on receipt; a sensor serving a cached value must keep the original stamp.
    timestamp_s: float | None = None
    # Genuine upstream sample counter only. ``None`` means the device supplies
    # no counter — it is never filled with a local poll count, so replay
    # detection cannot be defeated by always-advancing fabrication.
    sequence: int | None = None
    # Legacy per-tick load hint. The sustained-load hysteresis below prefers
    # the multi-channel fields when any is supplied.
    under_load: bool = True
    # Corroborating channels; ``None`` means the backend cannot supply them.
    utilization_pct: float | None = None
    power_w: float | None = None
    clock_mhz: float | None = None
    # Genuine device sample clock only; ``None`` when the device has none.
    source_timestamp_s: float | None = None
    # False means the sensor itself reports the sample as unusable.
    valid: bool = True


class WatchdogSensor(Protocol):
    def read(self) -> WatchdogReading: ...


class JunctionReader(Protocol):
    def junction_c(self) -> float: ...
    def under_load(self) -> bool: ...


class NvmlJunctionSensor:
    """Hardware GPU temperature, read directly from the NVML backend.

    Contract: maps one :class:`GpuObservation` to one ``WatchdogReading``
    without fabricating an upstream sequence — ``sequence`` stays ``None``
    unless the observation genuinely carries ``source_sequence``.
    Accepts any backend exposing ``sample()`` (preferred, single-session
    poll) or the legacy ``junction_c()`` / ``under_load()`` pair.
    """

    def __init__(self, nvml: Any) -> None:
        self.nvml = nvml

    def read(self) -> WatchdogReading:
        sample = getattr(self.nvml, "sample", None)
        if callable(sample):
            obs = sample()
            under_load: bool
            if obs.utilization_pct is not None or obs.power_w is not None:
                under_load = (obs.utilization_pct or 0.0) > 0 or (obs.power_w or 0.0) > 0
            elif hasattr(self.nvml, "under_load"):
                under_load = bool(self.nvml.under_load())
            else:
                under_load = False
            return WatchdogReading(
                junction_c=float(obs.temp_c),
                timestamp_s=obs.sampled_monotonic_s,
                sequence=obs.source_sequence,
                source_timestamp_s=obs.source_timestamp_s,
                utilization_pct=obs.utilization_pct,
                power_w=obs.power_w,
                clock_mhz=obs.clock_mhz,
                valid=bool(obs.valid),
                under_load=under_load,
                detail=obs.detail,
            )
        junction_c = float(self.nvml.junction_c())  # legacy backend
        under_load = bool(self.nvml.under_load())
        return WatchdogReading(
            junction_c=junction_c,
            timestamp_s=time.monotonic(),
            sequence=None,
            under_load=under_load,
        )


@dataclass(frozen=True)
class TripRecord:
    """The incident itself is stamped on the scenario clock.

    ``initiated_s`` is a monotonic process timestamp taken *before* actuation,
    so a failed actuator cannot leave the trip undated.
    """

    reason: WatchdogTrip
    detail: str
    initiated_s: float | None = None


Actuate = Callable[[WatchdogTrip, str], Awaitable[None]]


class ThermalWatchdog:
    def __init__(
        self,
        sensor: WatchdogSensor,
        *,
        limit_c: float,
        period_s: float,
        actuate: Actuate,
        max_staleness_ms: float = DEFAULT_MAX_STALENESS_MS,
        max_frozen_ticks: int = DEFAULT_MAX_FROZEN_TICKS,
        sustained_min_util_pct: float = DEFAULT_SUSTAINED_MIN_UTIL_PCT,
        sustained_min_power_w: float = DEFAULT_SUSTAINED_MIN_POWER_W,
        sustained_ticks: int = DEFAULT_SUSTAINED_TICKS,
        sustained_release_util_pct: float = DEFAULT_SUSTAINED_RELEASE_UTIL_PCT,
        sustained_release_power_w: float = DEFAULT_SUSTAINED_RELEASE_POWER_W,
        fallback: Actuate | None = None,
    ) -> None:
        if limit_c <= 0:
            raise WatchdogError(f"limit_c must be positive: {limit_c}")
        if period_s < 0:
            raise WatchdogError(f"period_s must be non-negative: {period_s}")
        if max_staleness_ms <= 0:
            raise WatchdogError(f"max_staleness_ms must be positive: {max_staleness_ms}")
        if max_frozen_ticks < 1:
            raise WatchdogError(f"max_frozen_ticks must be at least 1: {max_frozen_ticks}")
        if sustained_min_util_pct < 0:
            raise WatchdogError(
                f"sustained_min_util_pct must be non-negative: {sustained_min_util_pct}"
            )
        if sustained_min_power_w < 0:
            raise WatchdogError(
                f"sustained_min_power_w must be non-negative: {sustained_min_power_w}"
            )
        if sustained_ticks < 1:
            raise WatchdogError(f"sustained_ticks must be at least 1: {sustained_ticks}")
        if sustained_release_util_pct < 0:
            raise WatchdogError(
                f"sustained_release_util_pct must be non-negative: {sustained_release_util_pct}"
            )
        if sustained_release_power_w < 0:
            raise WatchdogError(
                f"sustained_release_power_w must be non-negative: {sustained_release_power_w}"
            )
        if sustained_release_util_pct > sustained_min_util_pct:
            raise WatchdogError(
                "sustained_release_util_pct must not exceed sustained_min_util_pct "
                f"({sustained_release_util_pct} > {sustained_min_util_pct})"
            )
        if sustained_release_power_w > sustained_min_power_w:
            raise WatchdogError(
                "sustained_release_power_w must not exceed sustained_min_power_w "
                f"({sustained_release_power_w} > {sustained_min_power_w})"
            )
        self.sensor = sensor
        self.limit_c = float(limit_c)
        self.period_s = float(period_s)
        self.actuate = actuate
        self.fallback = fallback
        self.max_staleness_ms = float(max_staleness_ms)
        self.max_frozen_ticks = int(max_frozen_ticks)
        self.sustained_min_util_pct = float(sustained_min_util_pct)
        self.sustained_min_power_w = float(sustained_min_power_w)
        self.sustained_ticks = int(sustained_ticks)
        self.sustained_release_util_pct = float(sustained_release_util_pct)
        self.sustained_release_power_w = float(sustained_release_power_w)
        self.trip: TripRecord | None = None
        self.trip_initiated = False
        self.trip_initiated_s: float | None = None
        self.actuation_failed = False
        self.actuation_error = ""
        self.fallback_attempted = False
        self.fallback_ok = False
        self.fallback_error = ""
        self.actuated = False
        self.checks = 0
        self.last_sequence: int | None = None
        self.last_source_timestamp_s: float | None = None
        self.last_value: float | None = None
        self.last_utilization_pct: float | None = None
        self.last_power_w: float | None = None
        self.last_clock_mhz: float | None = None
        self.frozen_ticks = 0
        self._load_streak = 0
        self.sustained_load = False
        self.degraded_ticks = 0
        self.last_degraded: TripRecord | None = None
        # Idempotent latch guard: concurrent triggers share one actuation.
        # Created here (3.10+ locks bind to the running loop lazily on first
        # contention, so construction outside a loop is safe).
        self._latch = asyncio.Lock()

    def evaluate(self, reading: WatchdogReading) -> WatchdogTrip | None:
        """Pure value decision, in severity order. No clock, I/O, or history."""

        if reading.junction_c >= self.limit_c:
            return WatchdogTrip.JUNCTION_OVER_TEMP
        if not reading.clamp_ok:
            return WatchdogTrip.CLAMP_READBACK_FAILED
        if reading.plc_fault:
            return WatchdogTrip.PLC_FAULT
        return None

    def _load_active(self, reading: WatchdogReading) -> bool:
        if reading.utilization_pct is not None or reading.power_w is not None:
            util_hit = (
                reading.utilization_pct is not None
                and reading.utilization_pct >= self.sustained_min_util_pct
            )
            power_hit = (
                reading.power_w is not None and reading.power_w >= self.sustained_min_power_w
            )
            return util_hit or power_hit
        return bool(reading.under_load)

    def _load_released(self, reading: WatchdogReading) -> bool:
        if reading.utilization_pct is not None or reading.power_w is not None:
            util_low = (
                reading.utilization_pct is None
                or reading.utilization_pct < self.sustained_release_util_pct
            )
            power_low = reading.power_w is None or reading.power_w < self.sustained_release_power_w
            return util_low and power_low
        return not reading.under_load

    def _update_sustained(self, reading: WatchdogReading) -> bool:
        if self._load_active(reading):
            self._load_streak += 1
        else:
            self._load_streak = 0
        if not self.sustained_load and self._load_streak >= self.sustained_ticks:
            self.sustained_load = True
        elif self.sustained_load and self._load_released(reading):
            self.sustained_load = False
            self._load_streak = 0
        return self.sustained_load

    def _channels_changed(self, reading: WatchdogReading) -> bool:
        """True when any supplied corroborating channel moved since last poll."""

        if (
            reading.utilization_pct is not None
            and self.last_utilization_pct is not None
            and reading.utilization_pct != self.last_utilization_pct
        ):
            return True
        if (
            reading.power_w is not None
            and self.last_power_w is not None
            and reading.power_w != self.last_power_w
        ):
            return True
        if (
            reading.clock_mhz is not None
            and self.last_clock_mhz is not None
            and reading.clock_mhz != self.last_clock_mhz
        ):
            return True
        return False

    def _remember_channels(self, reading: WatchdogReading) -> None:
        self.last_value = reading.junction_c
        self.last_utilization_pct = reading.utilization_pct
        self.last_power_w = reading.power_w
        self.last_clock_mhz = reading.clock_mhz

    def observe(self, reading: WatchdogReading, now_s: float) -> TripRecord | None:
        """One stateful sample: freshness, the value limits, then a frozen trace.

        Contract: returns an emergency ``TripRecord`` (see ``EMERGENCY_TRIPS``)
        when the R1 sequence must run, a ``SENSOR_HEALTH_DEGRADED`` record when the
        sample is suspect but below the emergency bar, else ``None``.
        Normal: fresh, changing, loaded telemetry advances all counters.
        Empty: ``sequence=None`` / missing channels skip those checks, never
        trip on the absence. Invalid: ``valid=False`` degrades, never trips.
        """

        if not reading.valid:
            record = TripRecord(
                WatchdogTrip.SENSOR_HEALTH_DEGRADED,
                reading.detail or "sensor reported an invalid sample",
            )
            self.last_degraded = record
            return record

        age_ms = 0.0
        if reading.timestamp_s is not None:
            age_ms = max(0.0, (now_s - reading.timestamp_s) * 1000.0)
        replayed = (
            reading.sequence is not None
            and self.last_sequence is not None
            and reading.sequence <= self.last_sequence
        )
        source_replayed = (
            reading.source_timestamp_s is not None
            and self.last_source_timestamp_s is not None
            and reading.source_timestamp_s <= self.last_source_timestamp_s
        )
        source_lag_ms: float | None = None
        if reading.source_timestamp_s is not None and reading.timestamp_s is not None:
            source_lag_ms = max(0.0, (reading.timestamp_s - reading.source_timestamp_s) * 1000.0)
        self.last_sequence = reading.sequence
        self.last_source_timestamp_s = reading.source_timestamp_s
        if age_ms > self.max_staleness_ms:
            return TripRecord(
                WatchdogTrip.SENSOR_STALE,
                reading.detail or f"sample {age_ms:.0f} ms old > {self.max_staleness_ms:.0f} ms",
            )
        if replayed:
            return TripRecord(
                WatchdogTrip.SENSOR_STALE,
                reading.detail or f"sample sequence {reading.sequence} did not advance",
            )
        if source_replayed:
            return TripRecord(
                WatchdogTrip.SENSOR_STALE,
                reading.detail or "device sample timestamp did not advance",
            )
        if source_lag_ms is not None and source_lag_ms > self.max_staleness_ms:
            return TripRecord(
                WatchdogTrip.SENSOR_STALE,
                reading.detail
                or (f"device sample {source_lag_ms:.0f} ms old > {self.max_staleness_ms:.0f} ms"),
            )

        reason = self.evaluate(reading)
        if reason is WatchdogTrip.JUNCTION_OVER_TEMP:
            return TripRecord(
                reason, f"junction={reading.junction_c:.1f} C >= limit {self.limit_c:.1f} C"
            )
        if reason is WatchdogTrip.CLAMP_READBACK_FAILED:
            return TripRecord(reason, reading.detail or "clamp readback not verified")
        if reason is WatchdogTrip.PLC_FAULT:
            return TripRecord(reason, reading.detail or "PLC fault coil asserted")

        sustained = self._update_sustained(reading)
        active = self._load_active(reading)
        if not active:
            # No per-tick load: a constant temperature is idle physics, and the
            # freeze counters stay parked until load returns.
            self.last_value = None
            self.last_utilization_pct = None
            self.last_power_w = None
            self.last_clock_mhz = None
            self.frozen_ticks = 0
            return None

        temp_same = self.last_value is not None and reading.junction_c == self.last_value
        if not temp_same:
            self.frozen_ticks = 1
            self._remember_channels(reading)
            return None

        has_channels = (
            reading.utilization_pct is not None
            or reading.power_w is not None
            or reading.clock_mhz is not None
        )
        if not has_channels:
            # Single-channel repeat under load is suspect but unproven: an
            # integer-quantized plateau under steady load looks identical.
            # Count it, but only degrade loudly once sustained; before the
            # hysteresis arms this is still just a plateau.
            self.frozen_ticks += 1
            if not sustained:
                return None
            record = TripRecord(
                WatchdogTrip.SENSOR_HEALTH_DEGRADED,
                reading.detail
                or (
                    f"junction={reading.junction_c:.1f} C unchanged for "
                    f"{self.frozen_ticks} ticks under load without "
                    "corroborating power/utilization channels"
                ),
            )
            self.last_degraded = record
            return record
        if self._channels_changed(reading):
            # Another channel moved while the temperature held: the sensor is
            # live, so this is a thermal plateau, not a stuck register.
            self.frozen_ticks = 1
            self._remember_channels(reading)
            return None
        self.frozen_ticks += 1
        self._remember_channels(reading)
        if sustained and self.frozen_ticks > self.max_frozen_ticks:
            return TripRecord(
                WatchdogTrip.SENSOR_FROZEN,
                reading.detail
                or (
                    f"junction={reading.junction_c:.1f} C unchanged for "
                    f"{self.frozen_ticks} ticks under sustained load "
                    "with frozen power/utilization channels"
                ),
            )
        return None

    async def _read(self) -> WatchdogReading:
        # The sensor may be a blocking library call (NVML); keep it off the loop.
        return await asyncio.to_thread(self.sensor.read)

    async def run(self, stop: asyncio.Event) -> TripRecord | None:
        while not stop.is_set():
            try:
                reading = await self._read()
            except Exception as exc:
                return await self.check(
                    WatchdogTrip.SENSOR_UNAVAILABLE,
                    f"{type(exc).__name__}: {exc}",
                )

            self.checks += 1
            record = self.observe(reading, time.monotonic())
            if record is not None:
                if record.reason is WatchdogTrip.SENSOR_HEALTH_DEGRADED:
                    self.degraded_ticks += 1
                    logger.warning("Watchdog degraded sample: %s", record.detail)
                    if self.period_s == 0:
                        await asyncio.sleep(0)
                    else:
                        await asyncio.sleep(self.period_s)
                    continue
                return await self.check(record.reason, record.detail)

            if self.period_s == 0:
                await asyncio.sleep(0)
            else:
                await asyncio.sleep(self.period_s)
        return None

    async def check(self, reason: WatchdogTrip, detail: str) -> TripRecord:
        """Initiate one trip and run the actuation callback, fail-safe.

        The trip is marked initiated (with a monotonic timestamp) *before* the
        callback runs, so a callback that raises cannot leave the trip state
        ambiguous. A failure is logged CRITICAL, recorded, given one fallback
        attempt (a direct BMC ForceOff injected by the composition root, or an
        escalation to the supervisor), and re-raised as
        ``WatchdogActuationError`` so the process exits non-zero.

        Idempotent latch: the first trigger wins. A concurrent or repeated
        trigger while a trip is already initiated returns the existing record
        without re-running the actuator, so one thermal event can never issue
        two power cuts. The latch is held across the whole initiate-actuate
        sequence, which is safe because actuation is short and concurrent
        callers only ever wait for the single outcome.
        """

        async with self._latch:
            if self.trip is not None:
                logger.warning(
                    "Watchdog trip already initiated (%s); ignoring duplicate %s",
                    self.trip.reason.value,
                    reason.value,
                )
                return self.trip
            initiated_s = time.monotonic()
            record = TripRecord(reason=reason, detail=detail, initiated_s=initiated_s)
            self.trip = record
            self.trip_initiated = True
            self.trip_initiated_s = initiated_s
            try:
                await self.actuate(reason, detail)
            except Exception as exc:
                self.actuation_failed = True
                self.actuation_error = f"{type(exc).__name__}: {exc}"
                logger.critical("Watchdog actuation callback failed: %s", exc)
                await self._attempt_fallback(reason, detail)
                raise WatchdogActuationError(reason, detail, exc) from exc
            self.actuated = True
            return record

    async def _attempt_fallback(self, reason: WatchdogTrip, detail: str) -> None:
        """One direct actuation attempt after the primary callback failed."""

        if self.fallback is None:
            logger.critical("Watchdog has no fallback actuator; escalating to supervisor")
            return
        self.fallback_attempted = True
        try:
            await self.fallback(reason, detail)
        except Exception as exc:
            self.fallback_error = f"{type(exc).__name__}: {exc}"
            logger.critical("Watchdog fallback actuation failed: %s", exc)
            logger.critical("Escalating to supervisor: the node may still be running")
            return
        self.fallback_ok = True
