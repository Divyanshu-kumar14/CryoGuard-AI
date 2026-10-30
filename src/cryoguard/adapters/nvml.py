"""NVML power-clamp adapters (P6-T03, D-07).

The clamp target is never a literal: the real backend queries the device's
minimum power limit and reads the value back. If NVML is missing, the device is
unknown, or the readback does not land within tolerance, ``ClampResult.ok`` is
false and the FSM hard-trips. It never pretends a clamp succeeded.
"""

from __future__ import annotations

from importlib import import_module
from importlib.util import find_spec
from typing import Any

from cryoguard.orchestrator.incident_fsm import ClampResult


class NvmlError(ValueError):
    """An unusable NVML configuration."""


class FakeNvml:
    """SIMULATED backend. The floor comes from config, not from a hard-coded watt."""

    def __init__(
        self,
        min_limit_w: float,
        *,
        fail: bool = False,
        junction_c: float = 40.0,
        loaded: bool = False,
    ) -> None:
        if min_limit_w <= 0:
            raise NvmlError(f"min_limit_w must be positive: {min_limit_w}")
        self.min_limit_w = float(min_limit_w)
        self.fail = fail
        self._junction_c = float(junction_c)
        self.loaded = bool(loaded)
        self.calls = 0

    def junction_c(self) -> float:
        """SIMULATED junction temperature; the watchdog treats it as measured."""

        return self._junction_c

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

    @staticmethod
    def available() -> bool:
        return find_spec("pynvml") is not None

    def junction_c(self) -> float:
        """GPU temperature in Celsius. Raises rather than inventing a value."""

        try:
            pynvml: Any = import_module("pynvml")
        except ImportError as exc:
            raise NvmlError("pynvml is not installed") from exc

        initialized = False
        try:
            pynvml.nvmlInit()
            initialized = True
            handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
            return float(pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU))
        except Exception as exc:  # NVML raises its own exception type
            raise NvmlError(f"NVML error: {type(exc).__name__}: {exc}") from exc
        finally:
            if initialized:
                try:
                    pynvml.nvmlShutdown()
                except Exception:
                    pass

    def under_load(self) -> bool:
        """True when the GPU reports non-zero utilization.

        The watchdog only treats a byte-identical junction trace as frozen while
        the device is doing work; an idle GPU holds a constant temperature.
        """

        try:
            pynvml: Any = import_module("pynvml")
        except ImportError as exc:
            raise NvmlError("pynvml is not installed") from exc

        initialized = False
        try:
            pynvml.nvmlInit()
            initialized = True
            handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
            return int(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu) > 0
        except Exception as exc:  # NVML raises its own exception type
            raise NvmlError(f"NVML error: {type(exc).__name__}: {exc}") from exc
        finally:
            if initialized:
                try:
                    pynvml.nvmlShutdown()
                except Exception:
                    pass

    def clamp(self) -> ClampResult:
        try:
            pynvml: Any = import_module("pynvml")
        except ImportError:
            return ClampResult(ok=False, applied_w=0.0, detail="pynvml is not installed")

        initialized = False
        try:
            pynvml.nvmlInit()
            initialized = True
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
        finally:
            if initialized:
                try:
                    pynvml.nvmlShutdown()
                except Exception:
                    pass

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
        return FakeNvml(
            float(section["fake_min_limit_w"]),
            junction_c=float(section.get("fake_junction_c", 40.0)),
        )
    if backend == "real":
        return RealNvml(tolerance_w=tolerance_w)
    raise NvmlError(f"unknown nvml.backend {backend!r} (expected fake or real)")
