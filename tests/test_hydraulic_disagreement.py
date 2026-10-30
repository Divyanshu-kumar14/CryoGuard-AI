"""Hydraulic isolation truth table: both witnesses must agree, or no isolation.

DI 1 (line depressurized) and HR 0 (pressure) are independent witnesses. Either
witness alone is never isolation: contradiction or an unavailable witness is
``SENSOR_DISAGREEMENT`` (FAULT class, never VERIFIED), and a closed stem over a
pressurized line is ``VALVE_SEAL_LEAK``. A stub Modbus client drives
``PlcClient.wait_for_closed`` directly: deterministic, no sockets, no sleeps
beyond the client's own millisecond polls.
"""

from __future__ import annotations

import socket
import time
from pathlib import Path

import pytest

from cryoguard.adapters.plc_client import PlcClient
from cryoguard.config import load
from cryoguard.models import IncidentState, ValveStatus
from cryoguard.orchestrator.adapters import (
    ScenarioClock,
    SimulatedBmc,
    SimulatedPlc,
    SimulatedScheduler,
)
from cryoguard.orchestrator.emergency import EmergencySequence

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def cfg() -> dict:
    return load(ROOT / "configs" / "default.yaml")


class _Ok:
    def __init__(self, bits=(), registers=()) -> None:
        self.bits = list(bits)
        self.registers = list(registers)

    def isError(self) -> bool:
        return False


class _Err:
    bits: list = []
    registers: list = []

    def isError(self) -> bool:
        return True


def _script(value):
    """A fixed response, a callable, or a per-call queue (last entry repeats)."""

    if callable(value) or not isinstance(value, list):
        return value
    queue = list(value)

    def pop():
        if len(queue) > 1:
            return queue.pop(0)
        return queue[0]

    return pop


class _StubModbus:
    """Deterministic stand-in for the Modbus TCP client."""

    def __init__(self, *, coils=None, discrete=None, registers=None) -> None:
        self._coils = _script(coils if coils is not None else _Ok([True, False, False]))
        self._discrete = _script(discrete if discrete is not None else _Ok([True, False]))
        self._registers = _script(registers if registers is not None else _Ok([], [0]))
        self.coil_reads = 0
        self.di_reads = 0
        self.hr_reads = 0

    def _resolve(self, value):
        return value() if callable(value) else value

    def read_coils(self, *args, **kwargs):
        self.coil_reads += 1
        return self._resolve(self._coils)

    def read_discrete_inputs(self, *args, **kwargs):
        self.di_reads += 1
        return self._resolve(self._discrete)

    def read_holding_registers(self, *args, **kwargs):
        self.hr_reads += 1
        return self._resolve(self._registers)


def _psi(value: float) -> int:
    return int(round(value * 10.0))


def _client(cfg: dict, **over) -> PlcClient:
    section = dict(cfg["plc"])
    section.setdefault("poll_ms", 5)
    section.update(over)
    return PlcClient(section)


def _closed_client(
    cfg: dict,
    *,
    limit: bool = True,
    di: bool = True,
    pressures_psi=(),
    coils=None,
    depressurize_timeout_ms: float = 1500,
    poll_ms: float = 5,
    confirm_reads: int = 3,
) -> tuple[PlcClient, _StubModbus]:
    registers = (
        [_Ok([], [_psi(p)]) if not hasattr(p, "isError") else p for p in pressures_psi]
        if pressures_psi
        else _Ok([], [_psi(0.0)])
    )
    client = _client(
        cfg,
        depressurize_timeout_ms=depressurize_timeout_ms,
        poll_ms=poll_ms,
        depressurize_confirm_reads=confirm_reads,
    )
    stub = _StubModbus(coils=coils, discrete=_Ok([limit, di]), registers=registers)
    return client, stub


def _close(client: PlcClient, stub: _StubModbus):
    return client.wait_for_closed(stub, time.monotonic())  # type: ignore[arg-type]


def test_limit_open_never_verifies(cfg) -> None:
    client = _client(cfg, valve_fault_ms=120, poll_ms=5)
    stub = _StubModbus(discrete=_Ok([False, False]))

    result = _close(client, stub)

    assert result.status is ValveStatus.FAULT
    assert result.status is not ValveStatus.VERIFIED
    assert "no limit switch" in result.detail


def test_both_agree_pressurized_is_a_seal_leak(cfg) -> None:
    client, stub = _closed_client(
        cfg, limit=True, di=False, pressures_psi=[34.0], depressurize_timeout_ms=150
    )

    result = _close(client, stub)

    assert result.status is ValveStatus.VALVE_SEAL_LEAK
    assert result.status is not ValveStatus.VERIFIED
    assert "34.0 PSI" in result.detail


def test_di_claims_depressurized_while_pressure_refutes_it(cfg) -> None:
    client, stub = _closed_client(cfg, limit=True, di=True, pressures_psi=[34.0])

    result = _close(client, stub)

    assert result.status is ValveStatus.SENSOR_DISAGREEMENT
    assert result.status is not ValveStatus.VERIFIED
    assert result.status is not ValveStatus.VERIFIED_SLOW
    assert "sensor disagreement" in result.detail
    assert "never VERIFIED" in result.detail


def test_pressure_low_while_di_disagrees_never_stabilizes(cfg) -> None:
    client, stub = _closed_client(
        cfg, limit=True, di=False, pressures_psi=[0.0], depressurize_timeout_ms=150
    )

    result = _close(client, stub)

    assert result.status is ValveStatus.SENSOR_DISAGREEMENT
    assert result.status is not ValveStatus.VERIFIED
    assert "never stabilized" in result.detail
    assert "never VERIFIED" in result.detail


def test_both_agree_depressurized_verifies(cfg) -> None:
    client, stub = _closed_client(cfg, limit=True, di=True, pressures_psi=[0.0])

    result = _close(client, stub)

    assert result.status is ValveStatus.VERIFIED
    assert "3 consecutive agreeing reads" in result.detail
    assert "line depressurized" in result.detail


def test_di_unavailable_is_disagreement(cfg) -> None:
    client, stub = _closed_client(cfg, depressurize_timeout_ms=500)
    stub._discrete = _script([_Ok([True, False]), _Err()])

    result = _close(client, stub)

    assert result.status is ValveStatus.SENSOR_DISAGREEMENT
    assert result.status is not ValveStatus.VERIFIED
    assert "DI unavailable" in result.detail
    assert "never VERIFIED" in result.detail


def test_pressure_unavailable_is_disagreement(cfg) -> None:
    client, stub = _closed_client(cfg, depressurize_timeout_ms=500)
    stub._registers = _script([_Err()])

    result = _close(client, stub)

    assert result.status is ValveStatus.SENSOR_DISAGREEMENT
    assert result.status is not ValveStatus.VERIFIED
    assert "pressure unavailable" in result.detail
    assert "never VERIFIED" in result.detail


def test_disconnect_is_a_fault_never_verified(cfg) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = int(sock.getsockname()[1])

    client = PlcClient(dict(cfg["plc"]), port=port)
    result = client.close_valve()

    assert result.status is ValveStatus.FAULT
    assert result.status is not ValveStatus.VERIFIED
    assert result.status is not ValveStatus.SENSOR_DISAGREEMENT


def test_fluctuation_around_threshold_needs_three_confirms(cfg) -> None:
    # lo=5, hi=7: the 6.0 touch sits in the deadband and resets the vote, so
    # verification needs three consecutive reads *after* the reset (6 reads).
    pressures = [0.0, 0.0, 6.0, 0.0, 0.0, 0.0]
    client, stub = _closed_client(
        cfg,
        limit=True,
        di=True,
        pressures_psi=pressures,
        depressurize_timeout_ms=2000,
        poll_ms=2,
    )

    result = _close(client, stub)

    assert result.status is ValveStatus.VERIFIED
    assert stub.hr_reads == 6
    assert "3 consecutive agreeing reads" in result.detail


def test_limit_reopen_mid_verify_is_a_fault(cfg) -> None:
    client = _client(cfg, poll_ms=5)
    stub = _StubModbus(
        discrete=[_Ok([True, False]), _Ok([False, False])],
        registers=_Ok([], [_psi(0.0)]),
    )

    result = _close(client, stub)

    assert result.status is ValveStatus.FAULT
    assert result.status is not ValveStatus.VERIFIED
    assert "reopened" in result.detail


def test_disagreement_outcomes_are_never_verified(cfg) -> None:
    disagreements = [
        _closed_client(cfg, limit=True, di=True, pressures_psi=[34.0]),
        _closed_client(cfg, limit=True, di=False, pressures_psi=[0.0], depressurize_timeout_ms=150),
    ]
    client, stub = _closed_client(cfg, depressurize_timeout_ms=500)
    stub._discrete = _script([_Ok([True, False]), _Err()])
    disagreements.append((client, stub))
    client, stub = _closed_client(cfg, depressurize_timeout_ms=500)
    stub._registers = _script([_Err()])
    disagreements.append((client, stub))

    for plc, stub_client in disagreements:
        result = _close(plc, stub_client)
        assert result.status is ValveStatus.SENSOR_DISAGREEMENT
        assert result.status is not ValveStatus.VERIFIED
        assert result.status is not ValveStatus.VERIFIED_SLOW


def _emergency(cfg: dict, status: ValveStatus):
    plc = SimulatedPlc(950.0)
    plc.status = status
    return EmergencySequence(
        ScenarioClock(10.0),
        plc,
        SimulatedBmc(),
        SimulatedScheduler(cfg["scheduler"]),
    )


def test_verified_log_claims_isolation(cfg) -> None:
    result = _emergency(cfg, ValveStatus.VERIFIED).run("junction 96.4 C")

    assert result.state is IncidentState.EMERGENCY_TRIP
    assert result.steps[0].status == "LIMIT_SWITCH_CLOSED"
    assert "Close accepted" in result.steps[0].detail
    assert "Stem closed" in result.steps[0].detail
    assert "Line depressurized" in result.steps[0].detail
    assert "Isolation verified" in result.steps[0].detail


def test_disagreement_log_withholds_isolation_and_still_cuts_power(cfg) -> None:
    result = _emergency(cfg, ValveStatus.SENSOR_DISAGREEMENT).run("junction 96.4 C")

    assert result.state is IncidentState.HARD_TRIP
    assert result.steps[0].status == "SENSOR_DISAGREEMENT"
    assert result.steps[0].status != "VERIFIED"
    assert "Verification failed" in result.steps[0].detail
    assert "unproven" in result.steps[0].detail
    assert "Isolation verified" not in result.steps[0].detail
    assert result.steps[-1].command == "FORCE_OFF"
    assert result.steps[-1].status == "ACCEPTED"


def test_seal_leak_log_withholds_isolation(cfg) -> None:
    result = _emergency(cfg, ValveStatus.VALVE_SEAL_LEAK).run("junction 96.4 C")

    assert result.state is IncidentState.HARD_TRIP
    assert result.steps[0].status == "VALVE_SEAL_LEAK"
    assert "Verification failed" in result.steps[0].detail
    assert "Isolation verified" not in result.steps[0].detail
