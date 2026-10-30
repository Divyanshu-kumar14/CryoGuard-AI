"""Modbus-TCP client against the SIMULATED PLC over a real loopback socket."""

from __future__ import annotations

import asyncio
import socket
from pathlib import Path

import pytest

from cryoguard.adapters.loopback import loopback_simulators
from cryoguard.adapters.plc_client import PlcClient
from cryoguard.config import load
from cryoguard.models import ValveStatus

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def cfg() -> dict:
    return load(ROOT / "configs" / "default.yaml")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


async def _read_initial(cfg: dict):
    async with loopback_simulators(cfg) as endpoints:
        client = PlcClient(cfg["plc"], host=endpoints.plc_host, port=endpoints.plc_port)
        return await asyncio.to_thread(client.read)


async def _close_with_travel(cfg: dict, travel_ms: float | None):
    async with loopback_simulators(cfg, plc_travel_ms=travel_ms) as endpoints:
        client = PlcClient(cfg["plc"], host=endpoints.plc_host, port=endpoints.plc_port)
        result = await asyncio.to_thread(client.close_valve)
        reading = await asyncio.to_thread(client.read)
        return result, reading


def test_the_register_map_decodes_ints_not_floats(cfg) -> None:
    reading = asyncio.run(_read_initial(cfg))

    assert reading.solenoid_open is True
    assert reading.vacuum is False
    assert reading.fault is False
    assert reading.limit_closed is False
    assert reading.pressure_psi == pytest.approx(cfg["plc"]["open_pressure_psi"])
    assert reading.flow_lpm == pytest.approx(cfg["plc"]["nominal_flow_lpm"])


def test_a_950_ms_close_is_verified(cfg) -> None:
    result, reading = asyncio.run(_close_with_travel(cfg, None))

    assert result.status is ValveStatus.VERIFIED
    assert result.nominal_ms == pytest.approx(cfg["plc"]["valve_travel_ms"])
    assert result.travel_ms > 0
    assert reading.limit_closed is True
    assert reading.fault is False
    assert reading.pressure_psi == 0.0
    assert reading.flow_lpm == 0.0


def test_a_1300_ms_close_is_verified_slow(cfg) -> None:
    result, reading = asyncio.run(_close_with_travel(cfg, 1300))

    assert result.status is ValveStatus.VERIFIED_SLOW
    assert reading.limit_closed is True
    assert reading.fault is False


def test_a_1600_ms_close_raises_the_fault_alarm(cfg) -> None:
    result, reading = asyncio.run(_close_with_travel(cfg, 1600))

    assert result.status is ValveStatus.FAULT
    assert reading.fault is True
    assert reading.limit_closed is False
    assert reading.pressure_psi > 0.0


def test_an_unreachable_plc_is_a_fault_not_an_exception(cfg) -> None:
    client = PlcClient(cfg["plc"], port=_free_port())
    result = client.close_valve()

    assert result.status is ValveStatus.FAULT
    assert "connect" in result.detail.lower()
