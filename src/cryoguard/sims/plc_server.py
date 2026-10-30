"""Modbus-TCP PLC simulator (P4-T04, PRD §10.1).

Register map, exactly as the PRD names it (addresses are 0-based on the wire):

    Coil 0001        valve solenoid        0 = spring-close command
    Coil 0002        vacuum drain
    Coil 0003        VALVE_FAULT alarm     read-only output
    Discrete 1001    limit switch closed
    Discrete 1002    line depressurized
    HR 4001          pressure, int16 psi x 10
    HR 4002          flow, uint16 L/min x 100

The FSM is separated from the wire so the timing bands are testable without a
socket, and the socket test is a thin confirmation that the map is what the PRD
says it is.
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, cast

from pymodbus.datastore import (
    ModbusDeviceContext,
    ModbusSequentialDataBlock,
    ModbusServerContext,
)
from pymodbus.server import ModbusTcpServer

from cryoguard.adapters.plc_client import status_for_travel
from cryoguard.models import ValveStatus


class PlcError(ValueError):
    """An invalid coil write or an unusable PLC configuration."""


class ValveState(StrEnum):
    OPEN = "OPEN"
    CLOSING_COMMANDED = "CLOSING_COMMANDED"
    CLOSED = "CLOSED"
    FAULT = "FAULT"


@dataclass(frozen=True)
class PlcSnapshot:
    state: str
    valve_status: ValveStatus
    limit_closed: bool
    depressurized: bool
    pressure_psi: float
    flow_lpm: float
    fault: bool
    vacuum: bool


class PlcSimulator:
    def __init__(self, cfg: dict[str, Any], *, travel_ms: float | None = None) -> None:
        self.travel_ms = float(travel_ms if travel_ms is not None else cfg["valve_travel_ms"])
        self.slow_ms = float(cfg["valve_slow_ms"])
        self.fault_ms = float(cfg["valve_fault_ms"])
        self.depressurize_ms = float(cfg["depressurize_ms"])
        self.open_pressure_psi = float(cfg["open_pressure_psi"])
        self.nominal_flow_lpm = float(cfg["nominal_flow_lpm"])
        if not 0 < self.travel_ms:
            raise PlcError(f"valve travel must be positive: {self.travel_ms}")
        if not 0 < self.slow_ms <= self.fault_ms:
            raise PlcError(
                f"valve bands must satisfy 0 < slow <= fault, got {self.slow_ms}/{self.fault_ms}"
            )
        self.reset()

    def reset(self) -> None:
        self.elapsed_s = 0.0
        self.state = ValveState.OPEN
        self.vacuum = False
        self._closed_at: float | None = None
        self._closing_since: float | None = None

    @property
    def valve_status(self) -> ValveStatus:
        return status_for_travel(self.travel_ms, self.slow_ms, self.fault_ms)

    def write_coil(self, address: int, value: bool) -> None:
        """Apply a coil write. Coil 2 is an alarm output and ignores writes."""

        if address == 0:
            if not value and self.state is ValveState.OPEN:
                self.state = ValveState.CLOSING_COMMANDED
                self._closing_since = self.elapsed_s
            elif value and self.state in (ValveState.CLOSED, ValveState.FAULT):
                # Operator reset only; the safety sequence never reopens a valve.
                self.reset()
            return
        if address == 1:
            self.vacuum = bool(value)
            return
        if address == 2:
            return
        raise PlcError(f"no writable coil at address {address} (writable: 0, 1)")

    def tick(self, dt_s: float) -> PlcSnapshot:
        if dt_s < 0:
            raise PlcError(f"dt must be non-negative: {dt_s}")
        self.elapsed_s += dt_s

        if self.state is ValveState.CLOSING_COMMANDED:
            assert self._closing_since is not None
            since_ms = (self.elapsed_s - self._closing_since) * 1000.0
            if since_ms >= self.fault_ms:
                self.state = ValveState.FAULT
            elif since_ms >= self.travel_ms:
                self.state = ValveState.CLOSED
                self._closed_at = self.elapsed_s

        if self.state is ValveState.OPEN:
            return self._snapshot(
                limit=False,
                depressurized=False,
                pressure=self.open_pressure_psi,
                flow=self.nominal_flow_lpm,
                fault=False,
            )

        if self.state is ValveState.CLOSING_COMMANDED:
            assert self._closing_since is not None
            since_ms = max(0.0, (self.elapsed_s - self._closing_since) * 1000.0)
            progress = min(1.0, since_ms / self.travel_ms)
            return self._snapshot(
                limit=False,
                depressurized=False,
                pressure=self.open_pressure_psi * (1.0 - progress),
                flow=self.nominal_flow_lpm * (1.0 - progress),
                fault=False,
            )

        if self.state is ValveState.CLOSED:
            assert self._closed_at is not None
            depressurized = (self.elapsed_s - self._closed_at) * 1000.0 >= self.depressurize_ms
            return self._snapshot(
                limit=True,
                depressurized=depressurized,
                pressure=0.0,
                flow=0.0,
                fault=False,
            )

        # FAULT: the valve did not close; coolant is still flowing.
        return self._snapshot(
            limit=False,
            depressurized=False,
            pressure=self.open_pressure_psi,
            flow=self.nominal_flow_lpm,
            fault=True,
        )

    def _snapshot(
        self,
        *,
        limit: bool,
        depressurized: bool,
        pressure: float,
        flow: float,
        fault: bool,
    ) -> PlcSnapshot:
        return PlcSnapshot(
            state=self.state,
            valve_status=self.valve_status,
            limit_closed=limit,
            depressurized=depressurized,
            pressure_psi=pressure,
            flow_lpm=flow,
            fault=fault,
            vacuum=self.vacuum,
        )


def _scaled(snapshot: PlcSnapshot) -> tuple[int, int]:
    pressure = max(0, min(32767, int(round(snapshot.pressure_psi * 10.0))))
    flow = max(0, min(65535, int(round(snapshot.flow_lpm * 100.0))))
    return pressure, flow


class PlcModbusServer:
    def __init__(
        self,
        cfg: dict[str, Any],
        *,
        host: str | None = None,
        port: int | None = None,
        simulator: PlcSimulator | None = None,
    ) -> None:
        self.host = str(host if host is not None else cfg["host"])
        self.port = int(port if port is not None else cfg["port"])
        self.unit_id = int(cfg["unit_id"])
        self.tick_s = float(cfg["tick_ms"]) / 1000.0
        self.sim = simulator or PlcSimulator(cfg)

        # The first cell in a sequential block is the 1-based placeholder. The
        # initial register values come from the simulator, not from literals, so
        # a config change cannot leave the wire map out of sync with the FSM.
        initial = self.sim.tick(0.0)
        pressure, flow = _scaled(initial)
        solenoid_open = self.sim.state is ValveState.OPEN
        self._co = ModbusSequentialDataBlock(
            0, [False] + [solenoid_open, self.sim.vacuum, initial.fault] + [False] * 12
        )
        self._di = ModbusSequentialDataBlock(
            0, [False] + [initial.limit_closed, initial.depressurized] + [False] * 11
        )
        self._hr = ModbusSequentialDataBlock(0, [0] + [pressure, flow] + [0] * 10)

        self._server: ModbusTcpServer | None = None
        self._ticker: asyncio.Task[None] | None = None

    @property
    def bound_port(self) -> int:
        if self._server is None:
            raise PlcError("server is not running")
        sockets = getattr(self._server.transport, "sockets", None)
        if not sockets:
            raise PlcError("server has no bound socket")
        return int(sockets[0].getsockname()[1])

    async def start(self) -> None:
        if self._server is not None:
            raise PlcError("server is already running")
        device = ModbusDeviceContext(co=self._co, di=self._di, hr=self._hr)
        context = ModbusServerContext(devices={self.unit_id: device}, single=False)
        self._server = ModbusTcpServer(context, address=(self.host, self.port))
        await self._server.serve_forever(background=True)
        self._ticker = asyncio.create_task(self._tick())

    async def stop(self) -> None:
        if self._ticker is not None:
            self._ticker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._ticker
            self._ticker = None
        if self._server is not None:
            with contextlib.suppress(Exception):
                await self._server.shutdown()
            self._server = None

    async def _tick(self) -> None:
        loop = asyncio.get_running_loop()
        last = loop.time()
        while True:
            await asyncio.sleep(self.tick_s)
            now = loop.time()
            dt, last = now - last, now

            coils = cast(list[bool], self._co.getValues(1, 3))
            self.sim.write_coil(0, bool(coils[0]))
            self.sim.write_coil(1, bool(coils[1]))
            snapshot = self.sim.tick(dt)

            self._di.setValues(1, [snapshot.limit_closed, snapshot.depressurized])
            pressure, flow = _scaled(snapshot)
            self._hr.setValues(1, [pressure, flow])
            self._co.setValues(3, [snapshot.fault])
