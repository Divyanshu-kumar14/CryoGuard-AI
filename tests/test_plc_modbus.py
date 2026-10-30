"""PLC simulator: timing bands, the FSM, and the register map over a real socket."""

from __future__ import annotations

import asyncio
import socket
from pathlib import Path

import pytest
from pymodbus.client import AsyncModbusTcpClient

from cryoguard.config import load
from cryoguard.models import ValveStatus
from cryoguard.sims.plc_server import (
    PlcError,
    PlcModbusServer,
    PlcSimulator,
    ValveState,
    status_for_travel,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def cfg() -> dict:
    return load(ROOT / "configs" / "default.yaml")["plc"]


def test_the_three_travel_bands(cfg) -> None:
    assert status_for_travel(950, cfg["valve_slow_ms"], cfg["valve_fault_ms"]) is (
        ValveStatus.VERIFIED
    )
    assert status_for_travel(1200, cfg["valve_slow_ms"], cfg["valve_fault_ms"]) is (
        ValveStatus.VERIFIED
    )
    assert status_for_travel(1300, cfg["valve_slow_ms"], cfg["valve_fault_ms"]) is (
        ValveStatus.VERIFIED_SLOW
    )
    assert status_for_travel(1500, cfg["valve_slow_ms"], cfg["valve_fault_ms"]) is (
        ValveStatus.VERIFIED_SLOW
    )
    assert status_for_travel(1501, cfg["valve_slow_ms"], cfg["valve_fault_ms"]) is (
        ValveStatus.FAULT
    )


def test_closing_sequence_reaches_the_limit_switch(cfg) -> None:
    sim = PlcSimulator(cfg, travel_ms=950)
    sim.write_coil(0, False)  # spring-close command

    snapshot = sim.tick(0.5)
    assert 0.0 < snapshot.pressure_psi < cfg["open_pressure_psi"]
    assert snapshot.limit_closed is False

    snapshot = sim.tick(0.5)  # 1.0 s total >= 950 ms travel
    assert sim.state is ValveState.CLOSED
    assert snapshot.limit_closed is True
    assert snapshot.pressure_psi == 0.0
    assert snapshot.flow_lpm == 0.0
    assert snapshot.fault is False


def test_depressurisation_follows_the_limit_switch(cfg) -> None:
    sim = PlcSimulator(cfg, travel_ms=950)
    sim.write_coil(0, False)
    sim.tick(1.0)
    assert sim.tick(0.2).depressurized is False  # 200 ms < 500 ms
    assert sim.tick(0.4).depressurized is True


def test_a_valve_that_never_closes_faults(cfg) -> None:
    sim = PlcSimulator(cfg, travel_ms=1600)  # > 1500 ms fault band
    sim.write_coil(0, False)

    snapshot = sim.tick(1.0)
    assert snapshot.fault is False
    assert snapshot.limit_closed is False

    snapshot = sim.tick(0.6)  # crosses 1500 ms
    assert sim.state is ValveState.FAULT
    assert snapshot.fault is True
    assert snapshot.limit_closed is False
    assert snapshot.pressure_psi == pytest.approx(cfg["open_pressure_psi"])


def test_vacuum_coil_is_independent_of_the_valve(cfg) -> None:
    sim = PlcSimulator(cfg)
    sim.write_coil(1, True)
    assert sim.tick(0.1).vacuum is True


def test_an_unknown_coil_is_rejected(cfg) -> None:
    with pytest.raises(PlcError, match="no writable coil"):
        PlcSimulator(cfg).write_coil(7, True)


def test_invalid_bands_are_rejected(cfg) -> None:
    with pytest.raises(PlcError, match="slow <= fault"):
        PlcSimulator(dict(cfg, valve_slow_ms=2000))


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_register_map_over_a_real_modbus_socket(cfg) -> None:
    port = _free_port()
    simulator = PlcSimulator(cfg, travel_ms=250)
    server = PlcModbusServer(cfg, port=port, simulator=simulator)

    async def case() -> None:
        await server.start()
        client = AsyncModbusTcpClient("127.0.0.1", port=port)
        assert await client.connect()

        coils = await client.read_coils(0, count=3, device_id=1)
        assert coils.bits[:3] == [True, False, False]

        discrete = await client.read_discrete_inputs(0, count=2, device_id=1)
        assert discrete.bits[:2] == [False, False]

        registers = await client.read_holding_registers(0, count=2, device_id=1)
        assert registers.registers[0] == 340  # 34.0 psi x 10
        assert registers.registers[1] == 900  # 9.00 L/min x 100

        await client.write_coil(0, False, device_id=1)
        closed = False
        for _ in range(100):
            discrete = await client.read_discrete_inputs(0, count=2, device_id=1)
            if discrete.bits[0]:
                closed = True
                break
            await asyncio.sleep(0.02)
        assert closed, "limit switch never closed"

        registers = await client.read_holding_registers(0, count=2, device_id=1)
        assert registers.registers[0] == 0
        assert registers.registers[1] == 0

        client.close()
        await server.stop()

    asyncio.run(case())


def test_a_faulted_valve_reports_the_alarm_coil(cfg) -> None:
    port = _free_port()
    server = PlcModbusServer(cfg, port=port, simulator=PlcSimulator(cfg, travel_ms=1600))

    async def case() -> None:
        await server.start()
        client = AsyncModbusTcpClient("127.0.0.1", port=port)
        assert await client.connect()
        await client.write_coil(0, False, device_id=1)

        faulted = False
        for _ in range(200):
            coils = await client.read_coils(0, count=3, device_id=1)
            if coils.bits[2]:
                faulted = True
                break
            await asyncio.sleep(0.02)
        assert faulted, "fault alarm never raised"

        discrete = await client.read_discrete_inputs(0, count=2, device_id=1)
        assert discrete.bits[0] is False  # no limit switch on a failed close

        client.close()
        await server.stop()

    asyncio.run(case())
