"""Independent thermal watchdog (P6-T07, R1b).

The safety claim is that this task does **not** depend on the perception or
fusion loop. It reads hardware state on its own period, and if the junction
temperature crosses the limit - or the clamp readback failed, or the PLC raised
its fault coil - it runs the R1 emergency sequence and does not wait for a
decision. Killing perception cannot make the watchdog blind.

A sensor read that raises is treated as a trip: a watchdog that cannot see the
junction cannot certify that the GPUs are cool. The failure is recorded with
reason ``SENSOR_UNAVAILABLE`` rather than swallowed.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol


class WatchdogError(ValueError):
    """An unusable watchdog configuration."""


class WatchdogTrip(StrEnum):
    JUNCTION_OVER_TEMP = "JUNCTION_OVER_TEMP"
    CLAMP_READBACK_FAILED = "CLAMP_READBACK_FAILED"
    PLC_FAULT = "PLC_FAULT"
    SENSOR_UNAVAILABLE = "SENSOR_UNAVAILABLE"


@dataclass(frozen=True)
class WatchdogReading:
    junction_c: float
    clamp_ok: bool = True
    plc_fault: bool = False
    detail: str = ""


class WatchdogSensor(Protocol):
    def read(self) -> WatchdogReading: ...


class JunctionReader(Protocol):
    def junction_c(self) -> float: ...


class NvmlJunctionSensor:
    """Hardware junction temperature, read directly from the NVML backend."""

    def __init__(self, nvml: JunctionReader) -> None:
        self.nvml = nvml

    def read(self) -> WatchdogReading:
        return WatchdogReading(junction_c=float(self.nvml.junction_c()))


@dataclass(frozen=True)
class TripRecord:
    """No wall-clock field: the incident is stamped on the scenario clock."""

    reason: WatchdogTrip
    detail: str


Actuate = Callable[[WatchdogTrip, str], Awaitable[None]]


class ThermalWatchdog:
    def __init__(
        self,
        sensor: WatchdogSensor,
        *,
        limit_c: float,
        period_s: float,
        actuate: Actuate,
    ) -> None:
        if limit_c <= 0:
            raise WatchdogError(f"limit_c must be positive: {limit_c}")
        if period_s < 0:
            raise WatchdogError(f"period_s must be non-negative: {period_s}")
        self.sensor = sensor
        self.limit_c = float(limit_c)
        self.period_s = float(period_s)
        self.actuate = actuate
        self.trip: TripRecord | None = None
        self.actuated = False
        self.checks = 0

    def evaluate(self, reading: WatchdogReading) -> WatchdogTrip | None:
        """Pure decision, in severity order. No clock, no I/O, no fusion state."""

        if reading.junction_c >= self.limit_c:
            return WatchdogTrip.JUNCTION_OVER_TEMP
        if not reading.clamp_ok:
            return WatchdogTrip.CLAMP_READBACK_FAILED
        if reading.plc_fault:
            return WatchdogTrip.PLC_FAULT
        return None

    async def _read(self) -> WatchdogReading:
        # The sensor may be a blocking library call (NVML); keep it off the loop.
        return await asyncio.to_thread(self.sensor.read)

    async def run(self, stop: asyncio.Event) -> TripRecord | None:
        while not stop.is_set():
            try:
                reading = await self._read()
            except Exception as exc:
                return await self._trip(
                    WatchdogTrip.SENSOR_UNAVAILABLE,
                    f"{type(exc).__name__}: {exc}",
                )

            self.checks += 1
            reason = self.evaluate(reading)
            if reason is not None:
                detail = reading.detail or f"junction={reading.junction_c:.1f} C"
                if reason is WatchdogTrip.JUNCTION_OVER_TEMP:
                    detail = f"junction={reading.junction_c:.1f} C >= limit {self.limit_c:.1f} C"
                elif reason is WatchdogTrip.CLAMP_READBACK_FAILED:
                    detail = reading.detail or "clamp readback not verified"
                elif reason is WatchdogTrip.PLC_FAULT:
                    detail = reading.detail or "PLC fault coil asserted"
                return await self._trip(reason, detail)

            if self.period_s == 0:
                await asyncio.sleep(0)
            else:
                await asyncio.sleep(self.period_s)
        return None

    async def _trip(self, reason: WatchdogTrip, detail: str) -> TripRecord:
        record = TripRecord(reason=reason, detail=detail)
        self.trip = record
        await self.actuate(reason, detail)
        self.actuated = True
        return record
