"""Independent thermal watchdog (P6-T07, R1b).

The safety claim is that this task does **not** depend on the perception or
fusion loop. It reads hardware state on its own period, and if the junction
temperature crosses the limit - or the clamp readback failed, or the PLC raised
its fault coil - it runs the R1 emergency sequence and does not wait for a
decision. Killing perception cannot make the watchdog blind.

A sensor read that raises is treated as a trip: a watchdog that cannot see the
junction cannot certify that the GPUs are cool. The failure is recorded with
reason ``SENSOR_UNAVAILABLE`` rather than swallowed. A sensor that keeps
answering while telling the truth about nothing is just as blind:

* a sample whose own timestamp is older than ``max_staleness_ms`` is stale
  (``SENSOR_STALE``), including a replayed sample whose sequence did not advance;
* a junction value that repeats byte-identically for more than
  ``max_frozen_ticks`` consecutive samples while the GPU reports load is a stuck
  register (``SENSOR_FROZEN``).

Both run the same R1 emergency sequence as an over-temperature.

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
from itertools import count
from typing import Protocol

DEFAULT_MAX_STALENESS_MS = 500.0
DEFAULT_MAX_FROZEN_TICKS = 15

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
    # Monotonic time the sample was taken. ``None`` means the watchdog stamps it
    # on receipt; a sensor serving a cached value must keep the original stamp.
    timestamp_s: float | None = None
    # The sensor's own sample counter. A value that does not advance is a replay.
    sequence: int | None = None
    # The GPU is doing work, so an unchanging junction trace is a stuck register.
    under_load: bool = True


class WatchdogSensor(Protocol):
    def read(self) -> WatchdogReading: ...


class JunctionReader(Protocol):
    def junction_c(self) -> float: ...
    def under_load(self) -> bool: ...


class NvmlJunctionSensor:
    """Hardware junction temperature, read directly from the NVML backend."""

    def __init__(self, nvml: JunctionReader) -> None:
        self.nvml = nvml
        self._sequence = count(1)

    def read(self) -> WatchdogReading:
        junction_c = float(self.nvml.junction_c())
        sampled_at = time.monotonic()
        return WatchdogReading(
            junction_c=junction_c,
            timestamp_s=sampled_at,
            sequence=next(self._sequence),
            under_load=self.nvml.under_load(),
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
        self.sensor = sensor
        self.limit_c = float(limit_c)
        self.period_s = float(period_s)
        self.actuate = actuate
        self.fallback = fallback
        self.max_staleness_ms = float(max_staleness_ms)
        self.max_frozen_ticks = int(max_frozen_ticks)
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
        self.last_value: float | None = None
        self.frozen_ticks = 0

    def evaluate(self, reading: WatchdogReading) -> WatchdogTrip | None:
        """Pure value decision, in severity order. No clock, I/O, or history."""

        if reading.junction_c >= self.limit_c:
            return WatchdogTrip.JUNCTION_OVER_TEMP
        if not reading.clamp_ok:
            return WatchdogTrip.CLAMP_READBACK_FAILED
        if reading.plc_fault:
            return WatchdogTrip.PLC_FAULT
        return None

    def observe(self, reading: WatchdogReading, now_s: float) -> TripRecord | None:
        """One stateful sample: freshness, the value limits, then a frozen trace."""

        age_ms = 0.0
        if reading.timestamp_s is not None:
            age_ms = max(0.0, (now_s - reading.timestamp_s) * 1000.0)
        replayed = (
            reading.sequence is not None
            and self.last_sequence is not None
            and reading.sequence <= self.last_sequence
        )
        self.last_sequence = reading.sequence
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

        reason = self.evaluate(reading)
        if reason is WatchdogTrip.JUNCTION_OVER_TEMP:
            return TripRecord(
                reason, f"junction={reading.junction_c:.1f} C >= limit {self.limit_c:.1f} C"
            )
        if reason is WatchdogTrip.CLAMP_READBACK_FAILED:
            return TripRecord(reason, reading.detail or "clamp readback not verified")
        if reason is WatchdogTrip.PLC_FAULT:
            return TripRecord(reason, reading.detail or "PLC fault coil asserted")

        if not reading.under_load:
            # An idle GPU holds a constant temperature; that is not a stuck register.
            self.last_value = None
            self.frozen_ticks = 0
            return None
        if self.last_value is not None and reading.junction_c == self.last_value:
            self.frozen_ticks += 1
        else:
            self.frozen_ticks = 1
        self.last_value = reading.junction_c
        if self.frozen_ticks > self.max_frozen_ticks:
            return TripRecord(
                WatchdogTrip.SENSOR_FROZEN,
                reading.detail
                or (
                    f"junction={reading.junction_c:.1f} C unchanged for "
                    f"{self.frozen_ticks} ticks while loaded"
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
        """

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
