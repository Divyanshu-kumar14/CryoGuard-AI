"""NVML power-clamp adapters (P6-T03, D-07).

The clamp target is never a literal: the real backend queries the device's
minimum power limit and reads the value back. If NVML is missing, the device is
unknown, or the readback does not land within tolerance, ``ClampResult.ok`` is
false and the FSM hard-trips. It never pretends a clamp succeeded.

Temperature honesty: the real backend reports ``NVML_TEMPERATURE_GPU`` — the
single GPU temperature NVML exposes. It is labelled as GPU temperature, never
as hotspot/junction, unless a future sensor proves a hotter domain.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import StrEnum
from importlib import import_module
from importlib.util import find_spec
from typing import Any, Protocol

from cryoguard.orchestrator.incident_fsm import ClampResult


class NvmlError(ValueError):
    """An unusable NVML configuration."""


class GpuBackend(StrEnum):
    REAL = "REAL"
    SIMULATED = "SIMULATED"


@dataclass(frozen=True)
class GpuObservation:
    """One consistent GPU poll.

    Contract: inputs are raw NVML (or simulated) reads; output is a typed
    snapshot with exactly one poll timestamp. ``source_sequence`` /
    ``source_timestamp_s`` stay ``None`` unless the device genuinely supplies
    a counter or sample clock — they are never fabricated locally, so a
    replay check on them cannot be defeated by a poll counter.
    ``utilization_pct`` / ``power_w`` / ``clock_mhz`` are ``None`` when the
    device or capability is missing, never zero-filled.
    """

    temp_c: float
    utilization_pct: float | None
    power_w: float | None
    clock_mhz: float | None
    device_index: int
    sampled_monotonic_s: float
    source_sequence: int | None = None
    source_timestamp_s: float | None = None
    valid: bool = True
    backend: GpuBackend = GpuBackend.SIMULATED
    detail: str = ""


class GpuSampler(Protocol):
    def sample(self) -> GpuObservation: ...


class FakeNvml:
    """SIMULATED backend. The floor comes from config, not from a hard-coded watt."""

    def __init__(
        self,
        min_limit_w: float,
        *,
        fail: bool = False,
        junction_c: float = 40.0,
        loaded: bool = False,
        utilization_pct: float | None = None,
        power_w: float | None = None,
        clock_mhz: float | None = None,
        device_index: int = 0,
    ) -> None:
        if min_limit_w <= 0:
            raise NvmlError(f"min_limit_w must be positive: {min_limit_w}")
        if device_index < 0:
            raise NvmlError(f"device_index must be non-negative: {device_index}")
        for name, value in (
            ("utilization_pct", utilization_pct),
            ("power_w", power_w),
            ("clock_mhz", clock_mhz),
        ):
            if value is not None and value < 0:
                raise NvmlError(f"{name} must be non-negative: {value}")
        self.min_limit_w = float(min_limit_w)
        self.fail = fail
        self._junction_c = float(junction_c)
        self.loaded = bool(loaded)
        self._utilization_pct = None if utilization_pct is None else float(utilization_pct)
        self._power_w = None if power_w is None else float(power_w)
        self._clock_mhz = None if clock_mhz is None else float(clock_mhz)
        self.device_index = int(device_index)
        self.calls = 0

    def sample(self) -> GpuObservation:
        """SIMULATED poll. No source counter exists, so it stays ``None``."""

        return GpuObservation(
            temp_c=float(self._junction_c),
            utilization_pct=self._utilization_pct,
            power_w=self._power_w,
            clock_mhz=self._clock_mhz,
            device_index=self.device_index,
            sampled_monotonic_s=time.monotonic(),
            source_sequence=None,
            source_timestamp_s=None,
            valid=True,
            backend=GpuBackend.SIMULATED,
            detail="SIMULATED GPU temperature",
        )

    def junction_c(self) -> float:
        """SIMULATED GPU temperature; the watchdog treats it as measured."""

        return self.sample().temp_c

    def under_load(self) -> bool:
        """SIMULATED load flag.

        A fixed junction is not a live thermal trace, so an unconfigured fake
        reports idle and never arms the watchdog's freeze detector. A test that
        wants a loaded GPU passes ``loaded=True`` and scripts the readings.
        """

        return self.loaded

    def clamp(self) -> ClampResult:
        self.calls += 1
        if self.fail:
            return ClampResult(ok=False, applied_w=0.0, detail="injected readback failure")
        return ClampResult(
            ok=True,
            applied_w=self.min_limit_w,
            detail=f"fake NVML min_limit={self.min_limit_w:.0f} W",
        )


class RealNvml:
    """NVML backend. Optional dependency: ``pynvml`` is imported lazily."""

    def __init__(self, device_index: int = 0, *, tolerance_w: float = 5.0) -> None:
        if device_index < 0:
            raise NvmlError(f"device_index must be non-negative: {device_index}")
        if tolerance_w <= 0:
            raise NvmlError(f"tolerance_w must be positive: {tolerance_w}")
        self.device_index = device_index
        self.tolerance_w = float(tolerance_w)
        self._initialized = False
        self._pynvml: Any = None

    @staticmethod
    def available() -> bool:
        return find_spec("pynvml") is not None

    def _ensure_init(self) -> Any:
        if self._pynvml is None:
            try:
                self._pynvml = import_module("pynvml")
            except ImportError as exc:
                raise NvmlError("pynvml is not installed") from exc
        if not self._initialized:
            try:
                self._pynvml.nvmlInit()
                self._initialized = True
            except Exception as exc:
                raise NvmlError(f"NVML init failed: {type(exc).__name__}: {exc}") from exc
        return self._pynvml

    def close(self) -> None:
        """Release the NVML driver handle session."""
        if self._initialized and self._pynvml is not None:
            try:
                self._pynvml.nvmlShutdown()
            except Exception:
                pass
            finally:
                self._initialized = False

    def __del__(self) -> None:
        self.close()

    def sample(self) -> GpuObservation:
        """One consistent poll in a managed NVML session.

        Contract: queries temperature, utilization, power, and clocks in one
        poll without reinitializing NVML on every tick. If the session was
        invalidated, recovers once before raising. A missing temperature or
        device handle raises ``NvmlError`` (fail-loud); a missing power/clock
        capability yields ``None`` for that channel only. The temperature is
        ``NVML_TEMPERATURE_GPU`` and is labelled as GPU temperature.
        """

        pynvml = self._ensure_init()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
        except Exception:
            # Attempt one session recovery if the session was invalidated
            self.close()
            try:
                pynvml = self._ensure_init()
                handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
            except Exception as retry_exc:
                raise NvmlError(
                    f"NVML error: unknown device index {self.device_index}: "
                    f"{type(retry_exc).__name__}: {retry_exc}"
                ) from retry_exc

        try:
            temp_c = float(pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU))
        except Exception as exc:
            raise NvmlError(
                f"NVML error: GPU temperature unavailable: {type(exc).__name__}: {exc}"
            ) from exc

        utilization: float | None = None
        try:
            utilization = float(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
        except Exception:
            utilization = None

        power_w: float | None = None
        try:
            power_w = float(pynvml.nvmlDeviceGetPowerUsage(handle)) / 1000.0
        except Exception:
            power_w = None

        clock_mhz: float | None = None
        try:
            clock_mhz = float(pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM))
        except Exception:
            clock_mhz = None

        return GpuObservation(
            temp_c=temp_c,
            utilization_pct=utilization,
            power_w=power_w,
            clock_mhz=clock_mhz,
            device_index=self.device_index,
            sampled_monotonic_s=time.monotonic(),
            source_sequence=None,
            source_timestamp_s=None,
            valid=True,
            backend=GpuBackend.REAL,
            detail="NVML_TEMPERATURE_GPU",
        )

    def junction_c(self) -> float:
        """GPU temperature in Celsius. Raises rather than inventing a value."""

        return self.sample().temp_c

    def under_load(self) -> bool:
        """True when the GPU reports non-zero utilization or power.

        Thin compatibility wrapper over :meth:`sample`; prefer ``sample()``
        so the temperature and load belong to the same poll.
        """

        obs = self.sample()
        if obs.utilization_pct is not None:
            return obs.utilization_pct > 0
        if obs.power_w is not None:
            return obs.power_w > 0
        return False

    def clamp(self) -> ClampResult:
        try:
            pynvml = self._ensure_init()
        except NvmlError as exc:
            return ClampResult(ok=False, applied_w=0.0, detail=str(exc))
        except Exception as exc:
            return ClampResult(ok=False, applied_w=0.0, detail=f"NVML init failed: {exc}")

        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
            minimum_mw, _maximum_mw = pynvml.nvmlDeviceGetPowerManagementLimitConstraints(handle)
            if int(minimum_mw) <= 0:
                return ClampResult(
                    ok=False, applied_w=0.0, detail=f"device reported min_limit={minimum_mw} mW"
                )
            pynvml.nvmlDeviceSetPowerManagementLimit(handle, int(minimum_mw))
            readback_mw = int(pynvml.nvmlDeviceGetPowerManagementLimit(handle))
        except Exception as exc:  # NVML raises its own exception type
            return ClampResult(
                ok=False,
                applied_w=0.0,
                detail=f"NVML error: {type(exc).__name__}: {exc}",
            )

        applied_w = readback_mw / 1000.0
        target_w = minimum_mw / 1000.0
        ok = abs(readback_mw - int(minimum_mw)) <= self.tolerance_w * 1000.0
        return ClampResult(
            ok=ok,
            applied_w=applied_w,
            detail=f"min_limit={target_w:.0f} W; readback={applied_w:.0f} W",
        )


def build_nvml(section: dict[str, Any], *, tolerance_w: float = 5.0) -> FakeNvml | RealNvml:
    backend = str(section.get("backend", "fake")).lower()
    if backend == "fake":
        loaded = bool(section.get("fake_loaded", section.get("loaded", False)))
        utilization = section.get("fake_utilization_pct", section.get("utilization_pct"))
        power = section.get("fake_power_w", section.get("power_w"))
        clock = section.get("fake_clock_mhz", section.get("clock_mhz"))
        return FakeNvml(
            float(section["fake_min_limit_w"]),
            junction_c=float(section.get("fake_junction_c", 40.0)),
            loaded=loaded,
            utilization_pct=None if utilization is None else float(utilization),
            power_w=None if power is None else float(power),
            clock_mhz=None if clock is None else float(clock),
            device_index=int(section.get("fake_device_index", section.get("device_index", 0))),
        )
    if backend == "real":
        device_index = int(section.get("device_index", section.get("real_device_index", 0)))
        return RealNvml(device_index=device_index, tolerance_w=tolerance_w)
    raise NvmlError(f"unknown nvml.backend {backend!r} (expected fake or real)")
