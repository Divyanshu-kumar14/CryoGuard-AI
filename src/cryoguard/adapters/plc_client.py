"""Modbus-TCP PLC client (P6-T04, D-13, D-15).

Wire map, 0-based on the wire and matching ``sims/plc_server.py``:

    coil 0      valve solenoid     0 = spring-close command
    coil 1      vacuum drain
    coil 2      VALVE_FAULT alarm  read-only output
    DI 0        limit switch closed
    DI 1        line depressurized
    HR 0        pressure, int16 psi x 10
    HR 1        flow, uint16 L/min x 100

``close_valve`` commands the spring-return close, then waits for isolation: the
limit switch alone only proves the stem moved. A valve with blown seals or a
charged bypass can sit on its limit switch while coolant keeps reaching the
chassis, so the close is not confirmed until DI 1 reads depressurized or HR 0
reads at or below ``depressurized_max_psi``. A stem that closes onto a still
pressurized line is ``VALVE_SEAL_LEAK`` and the FSM cuts node power.

The travel time is measured from the coil write, not from process start, so a
slow TCP handshake cannot turn a VERIFIED close into VERIFIED_SLOW. A transport
failure is returned as ``FAULT``: the FSM must cut power rather than proceed as
if coolant had stopped.
"""

from __future__ import annotations

import time
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

from pymodbus.client import ModbusTcpClient

from cryoguard.models import ValveStatus
from cryoguard.orchestrator.incident_fsm import ValveResult

DEFAULT_DEPRESSURIZED_MAX_PSI = 5.0
DEFAULT_DEPRESSURIZE_TIMEOUT_MS = 1500


class PlcClientError(ValueError):
    """An invalid PLC configuration or an unusable response."""


def _pressure_psi(raw: int) -> float:
    """HR 4001 is int16 psi x 10 on the wire."""

    value = int(raw)
    if value > 32767:
        value -= 65536
    return value / 10.0


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
        self.depressurized_max_psi = float(
            cfg.get("depressurized_max_psi", DEFAULT_DEPRESSURIZED_MAX_PSI)
        )
        self.depressurize_timeout_s = (
            float(cfg.get("depressurize_timeout_ms", DEFAULT_DEPRESSURIZE_TIMEOUT_MS)) / 1000.0
        )
        if self.port <= 0 or self.port > 65535:
            raise PlcClientError(f"invalid PLC port: {self.port}")
        if self.poll_s <= 0:
            raise PlcClientError(f"poll interval must be positive: {self.poll_s}")
        if self.depressurized_max_psi < 0:
            raise PlcClientError(
                f"depressurized_max_psi must be non-negative: {self.depressurized_max_psi}"
            )
        if self.depressurize_timeout_s < 0:
            raise PlcClientError(
                "depressurize_timeout_ms must be non-negative: "
                f"{self.depressurize_timeout_s * 1000}"
            )

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

            return self.wait_for_closed(client, time.monotonic())
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

    def wait_for_closed(self, client: ModbusTcpClient, command_t: float) -> ValveResult:
        """Wait for the limit switch, then prove the branch lost pressure.

        Called by ``close_valve`` once the spring-close coil write returns;
        ``command_t`` is the monotonic time of that write, so travel is measured
        from the command rather than from process start.

        Stem closure and depressurization are separate facts. A valve with blown
        internal seals or a charged bypass closes its limit switch while the line
        stays pressurized, so only DI 1 (depressurized) or a pressure readback at
        or below ``depressurized_max_psi`` confirms isolation. If the pressure
        stays above the limit for ``depressurize_timeout_ms`` the close is a
        ``VALVE_SEAL_LEAK``, never a VERIFIED close.
        """

        # Phase 1: the stem must reach its limit switch inside the travel window.
        travel_deadline = command_t + self.fault_ms / 1000.0 + 0.5
        while True:
            if time.monotonic() >= travel_deadline:
                return ValveResult(
                    status=ValveStatus.FAULT,
                    travel_ms=(time.monotonic() - command_t) * 1000.0,
                    detail=f"no limit switch within {self.fault_ms:.0f} ms",
                    nominal_ms=self.nominal_ms,
                )
            coils = client.read_coils(0, count=3, device_id=self.unit_id)
            discrete = client.read_discrete_inputs(0, count=2, device_id=self.unit_id)
            travel_ms = (time.monotonic() - command_t) * 1000.0
            if coils.isError() or discrete.isError():
                return ValveResult(
                    status=ValveStatus.FAULT,
                    travel_ms=travel_ms,
                    detail="read failed while polling the valve",
                    nominal_ms=self.nominal_ms,
                )
            if coils.bits[2]:
                return ValveResult(
                    status=ValveStatus.FAULT,
                    travel_ms=travel_ms,
                    detail="fault alarm; band FAULT",
                    nominal_ms=self.nominal_ms,
                )
            if discrete.bits[0]:
                break
            time.sleep(self.poll_s)

        status = status_for_travel(max(1.0, travel_ms), self.slow_ms, self.fault_ms)
        if status is ValveStatus.FAULT:
            return ValveResult(
                status=status,
                travel_ms=travel_ms,
                detail=f"limit switch closed; band {status.value}",
                nominal_ms=self.nominal_ms,
            )

        # Phase 2: the stem is closed, the line is not yet proven isolated.
        deadline = time.monotonic() + self.depressurize_timeout_s
        while True:
            coils = client.read_coils(0, count=3, device_id=self.unit_id)
            discrete = client.read_discrete_inputs(0, count=2, device_id=self.unit_id)
            registers = client.read_holding_registers(0, count=1, device_id=self.unit_id)
            if coils.isError() or discrete.isError() or registers.isError():
                return ValveResult(
                    status=ValveStatus.FAULT,
                    travel_ms=travel_ms,
                    detail="read failed while verifying depressurization",
                    nominal_ms=self.nominal_ms,
                )
            if coils.bits[2]:
                return ValveResult(
                    status=ValveStatus.FAULT,
                    travel_ms=travel_ms,
                    detail="fault alarm while verifying depressurization",
                    nominal_ms=self.nominal_ms,
                )
            if not discrete.bits[0]:
                # The stem reopened: any pressure drop came from somewhere else.
                return ValveResult(
                    status=ValveStatus.FAULT,
                    travel_ms=travel_ms,
                    detail="limit switch reopened while verifying pressure",
                    nominal_ms=self.nominal_ms,
                )
            pressure_psi = _pressure_psi(registers.registers[0])
            if discrete.bits[1] or pressure_psi <= self.depressurized_max_psi:
                return ValveResult(
                    status=status,
                    travel_ms=travel_ms,
                    detail=f"limit switch closed and line depressurized; band {status.value}",
                    nominal_ms=self.nominal_ms,
                )
            if time.monotonic() >= deadline:
                return ValveResult(
                    status=ValveStatus.VALVE_SEAL_LEAK,
                    travel_ms=travel_ms,
                    detail=(
                        f"limit switch closed but pressure {pressure_psi:.1f} PSI > "
                        f"{self.depressurized_max_psi:.1f} PSI after "
                        f"{self.depressurize_timeout_s * 1000:.0f} ms"
                    ),
                    nominal_ms=self.nominal_ms,
                )
            time.sleep(self.poll_s)

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

        return PlcReading(
            solenoid_open=bool(coils.bits[0]),
            vacuum=bool(coils.bits[1]),
            fault=bool(coils.bits[2]),
            limit_closed=bool(discrete.bits[0]),
            depressurized=bool(discrete.bits[1]),
            pressure_psi=_pressure_psi(registers.registers[0]),
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
