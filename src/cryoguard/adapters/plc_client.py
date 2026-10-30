"""Modbus-TCP PLC client (P6-T04, D-13, D-15).

Wire map, 0-based on the wire and matching ``sims/plc_server.py``:

    coil 0      valve solenoid     0 = spring-close command
    coil 1      vacuum drain
    coil 2      VALVE_FAULT alarm  read-only output
    DI 0        limit switch closed
    DI 1        line depressurized
    HR 0        pressure, int16 psi x 10
    HR 1        flow, uint16 L/min x 100

``close_valve`` commands the spring-return close, then polls the limit switch and
the fault alarm. The travel time is measured from the coil write, not from
process start, so a slow TCP handshake cannot turn a VERIFIED close into
VERIFIED_SLOW. A transport failure is returned as ``FAULT``: the FSM must cut
power rather than proceed as if coolant had stopped.
"""

from __future__ import annotations

import time
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

from pymodbus.client import ModbusTcpClient

from cryoguard.models import ValveStatus
from cryoguard.orchestrator.incident_fsm import ValveResult


class PlcClientError(ValueError):
    """An invalid PLC configuration or an unusable response."""


def status_for_travel(travel_ms: float, slow_ms: float, fault_ms: float) -> ValveStatus:
    """D-15 bands: <= slow VERIFIED, <= fault VERIFIED_SLOW, above fault FAULT."""

    if travel_ms <= 0:
        raise PlcClientError(f"valve travel must be positive: {travel_ms}")
    if slow_ms <= 0 or fault_ms < slow_ms:
        raise PlcClientError(f"invalid bands: slow={slow_ms}, fault={fault_ms}")
    if travel_ms <= slow_ms:
        return ValveStatus.VERIFIED
    if travel_ms <= fault_ms:
        return ValveStatus.VERIFIED_SLOW
    return ValveStatus.FAULT


@dataclass(frozen=True)
class PlcReading:
    solenoid_open: bool
    vacuum: bool
    fault: bool
    limit_closed: bool
    depressurized: bool
    pressure_psi: float
    flow_lpm: float
    simulated: bool = True


class PlcClient:
    def __init__(
        self,
        cfg: dict[str, Any],
        *,
        host: str | None = None,
        port: int | None = None,
    ) -> None:
        self.host = str(host if host is not None else cfg["host"])
        self.port = int(port if port is not None else cfg["port"])
        self.unit_id = int(cfg["unit_id"])
        self.slow_ms = float(cfg["valve_slow_ms"])
        self.fault_ms = float(cfg["valve_fault_ms"])
        self.nominal_ms = float(cfg["valve_travel_ms"])
        self.timeout_s = float(cfg.get("client_timeout_ms", 2000)) / 1000.0
        self.poll_s = float(cfg.get("poll_ms", 10)) / 1000.0
        if self.port <= 0 or self.port > 65535:
            raise PlcClientError(f"invalid PLC port: {self.port}")
        if self.poll_s <= 0:
            raise PlcClientError(f"poll interval must be positive: {self.poll_s}")

    def _connect(self) -> ModbusTcpClient:
        client = ModbusTcpClient(self.host, port=self.port, timeout=self.timeout_s)
        if not client.connect():
            with suppress(Exception):
                client.close()
            raise ConnectionError(f"modbus connect failed to {self.host}:{self.port}")
        return client

    def close_valve(self) -> ValveResult:
        started = time.monotonic()
        client: ModbusTcpClient | None = None
        try:
            client = self._connect()
            write = client.write_coil(0, False, device_id=self.unit_id)
            if write.isError():
                return ValveResult(
                    status=ValveStatus.FAULT,
                    travel_ms=0.0,
                    detail=f"coil write failed: {write}",
                    nominal_ms=self.nominal_ms,
                )

            command_t = time.monotonic()
            deadline = command_t + self.fault_ms / 1000.0 + 0.5
            while time.monotonic() < deadline:
                coils = client.read_coils(0, count=3, device_id=self.unit_id)
                discrete = client.read_discrete_inputs(0, count=2, device_id=self.unit_id)
                elapsed_ms = (time.monotonic() - command_t) * 1000.0
                if coils.isError() or discrete.isError():
                    return ValveResult(
                        status=ValveStatus.FAULT,
                        travel_ms=elapsed_ms,
                        detail="read failed while polling the valve",
                        nominal_ms=self.nominal_ms,
                    )
                if coils.bits[2]:
                    return ValveResult(
                        status=ValveStatus.FAULT,
                        travel_ms=elapsed_ms,
                        detail="fault alarm; band FAULT",
                        nominal_ms=self.nominal_ms,
                    )
                if discrete.bits[0]:
                    status = status_for_travel(max(1.0, elapsed_ms), self.slow_ms, self.fault_ms)
                    return ValveResult(
                        status=status,
                        travel_ms=elapsed_ms,
                        detail=f"limit switch closed; band {status.value}",
                        nominal_ms=self.nominal_ms,
                    )
                time.sleep(self.poll_s)

            return ValveResult(
                status=ValveStatus.FAULT,
                travel_ms=(time.monotonic() - command_t) * 1000.0,
                detail=f"no limit switch within {self.fault_ms:.0f} ms",
                nominal_ms=self.nominal_ms,
            )
        except Exception as exc:  # transport errors are a failed close, not a crash
            elapsed_ms = (time.monotonic() - started) * 1000.0
            return ValveResult(
                status=ValveStatus.FAULT,
                travel_ms=elapsed_ms,
                detail=f"{type(exc).__name__}: {exc}",
                nominal_ms=self.nominal_ms,
            )
        finally:
            if client is not None:
                with suppress(Exception):
                    client.close()

    def read(self) -> PlcReading:
        """Decode the register map. Used for diagnostics and assertions."""

        client: ModbusTcpClient | None = None
        try:
            client = self._connect()
            coils = client.read_coils(0, count=3, device_id=self.unit_id)
            discrete = client.read_discrete_inputs(0, count=2, device_id=self.unit_id)
            registers = client.read_holding_registers(0, count=2, device_id=self.unit_id)
            for response in (coils, discrete, registers):
                if response.isError():
                    raise PlcClientError(f"modbus read failed: {response}")
        finally:
            if client is not None:
                with suppress(Exception):
                    client.close()

        pressure_raw = int(registers.registers[0])
        if pressure_raw > 32767:  # int16 psi x 10 on the wire
            pressure_raw -= 65536
        return PlcReading(
            solenoid_open=bool(coils.bits[0]),
            vacuum=bool(coils.bits[1]),
            fault=bool(coils.bits[2]),
            limit_closed=bool(discrete.bits[0]),
            depressurized=bool(discrete.bits[1]),
            pressure_psi=pressure_raw / 10.0,
            flow_lpm=int(registers.registers[1]) / 100.0,
        )

    def reset(self) -> None:
        """Operator reset: reopen the valve. Never called by the safety sequence."""

        client: ModbusTcpClient | None = None
        try:
            client = self._connect()
            response = client.write_coil(0, True, device_id=self.unit_id)
            if response.isError():
                raise PlcClientError(f"reset coil write failed: {response}")
        finally:
            if client is not None:
                with suppress(Exception):
                    client.close()
