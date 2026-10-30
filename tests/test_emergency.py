"""R1 emergency path: isolate, cordon, cut power - no checkpoint, no requeue."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from cryoguard.adapters.loopback import loopback_simulators
from cryoguard.adapters.plc_client import PlcClient
from cryoguard.adapters.redfish_client import RedfishClient
from cryoguard.config import load
from cryoguard.models import IncidentState, ValveStatus
from cryoguard.orchestrator.actions import IncidentClosed
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


def _sequence(cfg: dict, *, bmc_fail: bool = False, plc_status: ValveStatus = ValveStatus.VERIFIED):
    clock = ScenarioClock(10.0)
    plc = SimulatedPlc(950.0)
    plc.status = plc_status
    return EmergencySequence(
        clock,
        plc,
        SimulatedBmc(fail=bmc_fail),
        SimulatedScheduler(cfg["scheduler"]),
    )


def test_the_emergency_path_isolates_cordons_and_cuts_power(cfg) -> None:
    sequence = _sequence(cfg)

    result = sequence.run("junction 96.4 C >= limit 90.0 C")

    assert result.state is IncidentState.EMERGENCY_TRIP
    assert [step.command for step in result.steps] == [
        "VALVE_CLOSE",
        "SCHEDULER_DRAIN",
        "FORCE_OFF",
    ]
    assert [step.actor for step in result.steps] == ["PLC", "SLURM", "BMC"]
    assert "CHECKPOINT_VERIFY" not in [step.command for step in result.steps]
    assert "REQUEUE" not in [step.command for step in result.steps]
    assert result.steps[0].ts == 10.0
    assert result.steps[1].ts == 10.95  # nominal travel, not measured wall time
    assert sequence.log.sealed is True
    with pytest.raises(IncidentClosed):
        sequence.log.append("BMC", "FORCE_OFF", "ACCEPTED")


def test_a_failed_valve_still_cuts_power_and_reports_a_hard_trip(cfg) -> None:
    result = _sequence(cfg, plc_status=ValveStatus.FAULT).run("junction 96.4 C")

    assert result.state is IncidentState.HARD_TRIP
    assert result.steps[0].status == "FAULT"
    assert result.steps[-1].command == "FORCE_OFF"
    assert result.steps[-1].status == "ACCEPTED"


def test_a_failed_valve_and_a_failed_power_cut_is_failed_safe(cfg) -> None:
    result = _sequence(cfg, plc_status=ValveStatus.FAULT, bmc_fail=True).run("junction 96.4 C")

    assert result.state is IncidentState.FAILED_SAFE
    assert result.steps[-1].status == "REJECTED"


def test_a_seal_leak_still_cuts_power_and_reports_a_hard_trip(cfg) -> None:
    """Emergency path: a closed stem over a pressurized line is not a safe trip."""

    result = _sequence(cfg, plc_status=ValveStatus.VALVE_SEAL_LEAK).run("junction 96.4 C")

    assert result.state is IncidentState.HARD_TRIP
    assert result.steps[0].status == "VALVE_SEAL_LEAK"
    assert result.steps[1].ts == 10.0  # no travel advance: isolation was never claimed
    assert result.steps[-1].command == "FORCE_OFF"
    assert result.steps[-1].status == "ACCEPTED"


def test_a_good_isolation_with_a_failed_power_cut_is_failed_safe(cfg) -> None:
    result = _sequence(cfg, bmc_fail=True).run("junction 96.4 C")

    assert result.state is IncidentState.FAILED_SAFE


def test_the_emergency_path_runs_over_real_modbus_and_redfish_sockets(cfg) -> None:
    async def case():
        async with loopback_simulators(cfg) as endpoints:
            plc = PlcClient(cfg["plc"], host=endpoints.plc_host, port=endpoints.plc_port)
            redfish = RedfishClient(cfg["bmc"], base_url=endpoints.bmc_url)

            def run() -> tuple:
                sequence = EmergencySequence(
                    ScenarioClock(10.0), plc, redfish, SimulatedScheduler(cfg["scheduler"])
                )
                return sequence.run("junction 96.4 C"), plc.read(), redfish.power_state()

            return await asyncio.to_thread(run)

    result, reading, power = asyncio.run(case())

    assert result.state is IncidentState.EMERGENCY_TRIP
    assert reading.solenoid_open is False
    assert reading.limit_closed is True
    assert reading.pressure_psi == 0.0
    assert power == "Off"
    assert "(SIMULATED BMC)" in result.steps[-1].detail
