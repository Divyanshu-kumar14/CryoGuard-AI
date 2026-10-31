"""Tests for the RealNvml backend with mock pynvml."""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock

import pytest

from cryoguard.adapters.nvml import GpuBackend, NvmlError, RealNvml


class _MockPynvml(ModuleType):
    NVML_TEMPERATURE_GPU = 0
    NVML_CLOCK_SM = 1

    def __init__(self) -> None:
        super().__init__("pynvml")
        self.init_calls = 0
        self.shutdown_calls = 0
        self.temp = 55
        self.util = 42
        self.power = 150000
        self.clock = 1410
        self.min_mw = 100000
        self.max_mw = 300000
        self.set_limit = 0

    def nvmlInit(self) -> None:
        self.init_calls += 1

    def nvmlShutdown(self) -> None:
        self.shutdown_calls += 1

    def nvmlDeviceGetHandleByIndex(self, idx: int) -> int:
        if idx != 0:
            raise RuntimeError(f"invalid device index {idx}")
        return 12345

    def nvmlDeviceGetTemperature(self, handle: int, sensor_type: int) -> int:
        return self.temp

    def nvmlDeviceGetUtilizationRates(self, handle: int) -> MagicMock:
        return MagicMock(gpu=self.util)

    def nvmlDeviceGetPowerUsage(self, handle: int) -> int:
        return self.power

    def nvmlDeviceGetClockInfo(self, handle: int, clock_type: int) -> int:
        return self.clock

    def nvmlDeviceGetPowerManagementLimitConstraints(self, handle: int) -> tuple[int, int]:
        return (self.min_mw, self.max_mw)

    def nvmlDeviceSetPowerManagementLimit(self, handle: int, limit: int) -> None:
        self.set_limit = limit

    def nvmlDeviceGetPowerManagementLimit(self, handle: int) -> int:
        return self.set_limit or self.min_mw


@pytest.fixture
def mock_pynvml(monkeypatch):
    mock = _MockPynvml()
    monkeypatch.setitem(sys.modules, "pynvml", mock)
    return mock


def test_real_nvml_persists_session_across_samples(mock_pynvml) -> None:
    backend = RealNvml(device_index=0)

    # First sample initiates NVML session
    obs1 = backend.sample()
    assert obs1.temp_c == 55.0
    assert obs1.utilization_pct == 42.0
    assert obs1.power_w == 150.0
    assert obs1.clock_mhz == 1410.0
    assert obs1.backend == GpuBackend.REAL
    assert mock_pynvml.init_calls == 1
    assert mock_pynvml.shutdown_calls == 0

    # Second sample reuses the established session without init/shutdown thrashing
    obs2 = backend.sample()
    assert obs2.temp_c == 55.0
    assert mock_pynvml.init_calls == 1
    assert mock_pynvml.shutdown_calls == 0

    # Explicit close shuts down NVML cleanly
    backend.close()
    assert mock_pynvml.shutdown_calls == 1


def test_real_nvml_clamp_with_session(mock_pynvml) -> None:
    backend = RealNvml(device_index=0, tolerance_w=5.0)
    res = backend.clamp()
    assert res.ok is True
    assert res.applied_w == 100.0
    assert mock_pynvml.set_limit == 100000


def test_real_nvml_raises_on_invalid_device(mock_pynvml) -> None:
    backend = RealNvml(device_index=99)
    with pytest.raises(NvmlError, match="unknown device index 99"):
        backend.sample()
