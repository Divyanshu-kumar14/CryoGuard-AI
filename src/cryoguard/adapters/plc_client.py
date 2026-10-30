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
chassis, so the close is not confirmed until DI 1 reads depressurized AND HR 0
reads at or below ``depressurized_max_psi`` on ``depressurize_confirm_reads``
consecutive polls. Either witness alone is not isolation. A stem that closes
onto a still pressurized line is ``VALVE_SEAL_LEAK`` and the FSM cuts node
power.

Isolation policy truth table (limit = DI 0, di = DI 1, p = HR 0 pressure,
lo = ``depressurized_max_psi``, hi = lo + ``depressurize_hysteresis_psi``):

    1. limit open                    -> not isolated, keep waiting (Phase 1
                                        deadline) / reopen FAULT in Phase 2
    2. limit closed, !di, p >= hi    -> both agree pressurized: waiting; on
                                        timeout VALVE_SEAL_LEAK, never VERIFIED
    3. limit closed, di, p >= hi     -> DI claims depressurized, pressure
                                        refutes it: SENSOR_DISAGREEMENT (FAULT
                                        class, never VERIFIED), immediate
    4. limit closed, !di, p <= lo    -> pressure claims depressurized, DI
                                        refutes it: SENSOR_DISAGREEMENT,
                                        immediate, never VERIFIED
    5. limit closed, di, p <= lo     -> both agree depressurized: confirm vote;
                                        VERIFIED only after N consecutive
                                        agreeing reads (stabilization)
    6. DI read error (unavailable)   -> SENSOR_DISAGREEMENT, never VERIFIED
    7. pressure read error (unavail) -> SENSOR_DISAGREEMENT, never VERIFIED
    8. transport error / disconnect  -> FAULT via read-error / exception path,
                                        never VERIFIED
    9. p in (lo, hi) fluctuation     -> ambiguous band: reset the confirm
                                        counter, keep waiting; hysteresis
                                        (confirm <= lo, refute >= hi) stops a
                                        single threshold touch from verifying;
                                        timeout without N confirms is
                                        VALVE_SEAL_LEAK, never VERIFIED
   10. limit reopens mid-verify      -> FAULT ("limit switch reopened"),
                                        never VERIFIED (already handled, kept)

DI 1 is treated only as an independent depressurization witness that must agree
with HR 0; no new physical meaning is assigned to it.

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
DEFAULT_DEPRESSURIZE_CONFIRM_READS = 3
DEFAULT_DEPRESSURIZE_HYSTERESIS_PSI = 2.0


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
        self.depressurize_confirm_reads = int(
            cfg.get("depressurize_confirm_reads", DEFAULT_DEPRESSURIZE_CONFIRM_READS)
        )
        self.depressurize_hysteresis_psi = float(
            cfg.get("depressurize_hysteresis_psi", DEFAULT_DEPRESSURIZE_HYSTERESIS_PSI)
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
        if self.depressurize_confirm_reads < 1:
            raise PlcClientError(
                f"depressurize_confirm_reads must be >= 1: {self.depressurize_confirm_reads}"
            )
        if self.depressurize_hysteresis_psi < 0:
            raise PlcClientError(
                "depressurize_hysteresis_psi must be non-negative: "
                f"{self.depressurize_hysteresis_psi}"
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
        stays pressurized. Isolation requires the limit switch closed AND both
        depressurization witnesses agreeing: DI 1 reads depressurized AND HR 0
        reads at or below ``depressurized_max_psi``, on
        ``depressurize_confirm_reads`` consecutive polls. Either witness alone
        is never isolation.

        Disagreement policy (lo = max, hi = max + hysteresis):

        * DI claims depressurized while pressure refutes it (p >= hi) is a
          hard contradiction: it cannot be a normal depressurization transient
          (a healthy close drives pressure low before DI flips), so it returns
          ``SENSOR_DISAGREEMENT`` immediately.
        * Pressure claims depressurized (p <= lo) while DI still disagrees can
          be the normal transient: the simulator drops pressure to 0.0 at stem
          closure while DI flips ``depressurize_ms`` later. It is therefore
          treated as "not yet proven": the confirm counter resets and polling
          continues. If it persists until the timeout it returns
          ``SENSOR_DISAGREEMENT``, never VERIFIED.
        * Both agree pressurized (!di, p > lo) is "not yet isolated": polling
          continues; on timeout it is ``VALVE_SEAL_LEAK``, never VERIFIED.
        * The hysteresis deadband (lo, hi) is ambiguous: it resets the confirm
          counter and keeps waiting, so a single threshold touch or fluctuation
          cannot verify. Timeout without N consecutive agreeing reads is
          ``VALVE_SEAL_LEAK`` (or ``SENSOR_DISAGREEMENT`` when DI was claiming
          depressurized), never VERIFIED.
        * A single-channel read error (DI or pressure unavailable) is
          ``SENSOR_DISAGREEMENT``, never VERIFIED. A control-path read error
          (coils) or a transport exception/disconnect is ``FAULT``.
        * A reopened limit switch mid-verify is ``FAULT``: any pressure drop
          came from somewhere else.
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
        lo_psi = self.depressurized_max_psi
        hi_psi = lo_psi + self.depressurize_hysteresis_psi
        needed = self.depressurize_confirm_reads
        confirm_count = 0
        last_pressure_psi = float("nan")
        last_di = False
        while True:
            coils = client.read_coils(0, count=3, device_id=self.unit_id)
            discrete = client.read_discrete_inputs(0, count=2, device_id=self.unit_id)
            registers = client.read_holding_registers(0, count=1, device_id=self.unit_id)
            if coils.isError():
                return ValveResult(
                    status=ValveStatus.FAULT,
                    travel_ms=travel_ms,
                    detail="read failed while verifying depressurization",
                    nominal_ms=self.nominal_ms,
                )
            if discrete.isError():
                return ValveResult(
                    status=ValveStatus.SENSOR_DISAGREEMENT,
                    travel_ms=travel_ms,
                    detail="DI unavailable while verifying depressurization; "
                    "isolation unproven, never VERIFIED",
                    nominal_ms=self.nominal_ms,
                )
            if registers.isError():
                return ValveResult(
                    status=ValveStatus.SENSOR_DISAGREEMENT,
                    travel_ms=travel_ms,
                    detail="pressure unavailable while verifying depressurization; "
                    "isolation unproven, never VERIFIED",
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
            last_pressure_psi = pressure_psi
            di_depressurized = bool(discrete.bits[1])
            last_di = di_depressurized
            pressure_low = pressure_psi <= lo_psi
            pressure_high = pressure_psi >= hi_psi

            if di_depressurized and pressure_high:
                # Hard contradiction: DI claims depressurized while pressure
                # refutes it. Never VERIFIED.
                return ValveResult(
                    status=ValveStatus.SENSOR_DISAGREEMENT,
                    travel_ms=travel_ms,
                    detail=(
                        "sensor disagreement: DI claims depressurized but pressure "
                        f"{pressure_psi:.1f} PSI >= {hi_psi:.1f} PSI "
                        f"(max {lo_psi:.1f} + hyst {self.depressurize_hysteresis_psi:.1f}); "
                        "isolation unproven, never VERIFIED"
                    ),
                    nominal_ms=self.nominal_ms,
                )
            if di_depressurized and pressure_low:
                confirm_count += 1
                if confirm_count >= needed:
                    return ValveResult(
                        status=status,
                        travel_ms=travel_ms,
                        detail=(
                            f"limit switch closed and line depressurized; band {status.value}; "
                            f"{confirm_count} consecutive agreeing reads "
                            f"(pressure {pressure_psi:.1f} PSI <= {lo_psi:.1f} PSI, DI agreed)"
                        ),
                        nominal_ms=self.nominal_ms,
                    )
            else:
                # Either both agree pressurized, the deadband, or the normal
                # transient (pressure low before DI flips): not proven, keep
                # waiting. A single touch cannot verify.
                confirm_count = 0
            if time.monotonic() >= deadline:
                if last_di or last_pressure_psi <= lo_psi:
                    # A witness claimed depressurized but agreement never
                    # stabilized (fluctuation, stuck DI, or pressure-low/DI-low
                    # persistence). Disagreement, never VERIFIED.
                    return ValveResult(
                        status=ValveStatus.SENSOR_DISAGREEMENT,
                        travel_ms=travel_ms,
                        detail=(
                            "sensor disagreement: depressurization never stabilized with "
                            f"{needed} consecutive agreeing reads "
                            f"(last pressure {last_pressure_psi:.1f} PSI, "
                            f"DI={'depressurized' if last_di else 'pressurized'}); "
                            "isolation unproven, never VERIFIED"
                        ),
                        nominal_ms=self.nominal_ms,
                    )
                return ValveResult(
                    status=ValveStatus.VALVE_SEAL_LEAK,
                    travel_ms=travel_ms,
                    detail=(
                        f"limit switch closed but pressure {last_pressure_psi:.1f} PSI > "
                        f"{lo_psi:.1f} PSI after "
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
