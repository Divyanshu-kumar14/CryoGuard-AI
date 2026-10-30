"""M3 closed loop: clamp -> valve -> checkpoint -> power off -> requeue.

Every case here uses the live Modbus-TCP and Redfish clients against the
SIMULATED servers on loopback sockets. The only in-process piece is the Slurm
mock, whose real interface is ``scontrol`` rather than HTTP.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from cryoguard.adapters.loopback import loopback_simulators
from cryoguard.adapters.nvml import FakeNvml
from cryoguard.adapters.plc_client import PlcClient, PlcReading
from cryoguard.adapters.redfish_client import RedfishClient
from cryoguard.adapters.scheduler import CheckpointAdapter, SchedulerAdapter
from cryoguard.config import load
from cryoguard.models import IncidentState
from cryoguard.orchestrator.adapters import ScenarioClock, SimulatedBmc
from cryoguard.orchestrator.incident_fsm import (
    ConfirmedLeakOrchestrator,
    IncidentFSM,
    SequenceStep,
)
from cryoguard.sims.checkpoint_store import verify_checkpoint, write_checkpoint

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def cfg() -> dict:
    return load(ROOT / "configs" / "default.yaml")


async def _run(
    cfg: dict,
    checkpoint_dir: Path,
    *,
    clamp_ok: bool = True,
    plc_travel_ms: float | None = None,
    failing_bmc: bool = False,
) -> tuple[IncidentFSM, list[SequenceStep], PlcReading, str, dict]:
    async with loopback_simulators(cfg, plc_travel_ms=plc_travel_ms) as endpoints:

        def sequence() -> tuple[IncidentFSM, list[SequenceStep], PlcReading, str, dict]:
            clock = ScenarioClock(10.0)
            nvml = FakeNvml(cfg["nvml"]["fake_min_limit_w"], fail=not clamp_ok)
            plc = PlcClient(cfg["plc"], host=endpoints.plc_host, port=endpoints.plc_port)
            redfish = RedfishClient(cfg["bmc"], base_url=endpoints.bmc_url)
            checkpoint = CheckpointAdapter(
                clock,
                checkpoint_dir,
                step=int(cfg["orchestrator"]["checkpoint_step"]),
                scheduler_cfg=cfg["scheduler"],
            )
            scheduler = SchedulerAdapter(cfg["scheduler"])
            bmc = SimulatedBmc(fail=True) if failing_bmc else redfish
            orchestrator = ConfirmedLeakOrchestrator(
                clock=clock,
                nvml=nvml,
                plc=plc,
                checkpoint=checkpoint,
                bmc=bmc,
                scheduler=scheduler,
                clamp_latency_ms=float(cfg["orchestrator"]["clamp_latency_ms"]),
            )
            fsm = orchestrator.run()
            return (
                fsm,
                list(orchestrator.steps),
                plc.read(),
                redfish.power_state(),
                scheduler.describe(),
            )

        return await asyncio.to_thread(sequence)


def test_the_full_sequence_closes_over_real_sockets(cfg, tmp_path) -> None:
    fsm, steps, reading, power, slurm = asyncio.run(_run(cfg, tmp_path / "checkpoints"))

    assert fsm.state is IncidentState.RESOLVED_PENDING_REPAIR
    assert [step.command for step in steps] == [
        "CLAMP",
        "VALVE_CLOSE",
        "SCHEDULER_DRAIN",
        "CHECKPOINT_VERIFY",
        "FORCE_OFF",
        "REQUEUE",
    ]
    assert reading.solenoid_open is False
    assert reading.limit_closed is True
    assert reading.fault is False
    assert reading.pressure_psi == 0.0
    assert power == "Off"
    assert slurm["node"] == "gpu-node-09a"
    assert slurm["state"] == "RUNNING"

    assert verify_checkpoint(tmp_path / "checkpoints" / "step_004200.json")
    assert steps[3].detail == "step 4200 hash-verified"


def test_the_timestamps_prove_the_ordering(cfg, tmp_path) -> None:
    fsm, _, _, _, _ = asyncio.run(_run(cfg, tmp_path / "checkpoints"))

    assert fsm.timestamp("clamp_verified") < fsm.timestamp("valve_closed")
    assert fsm.timestamp("valve_closed") <= fsm.timestamp("checkpoint_settled")
    assert fsm.timestamp("checkpoint_settled") == fsm.timestamp("power_off")
    assert fsm.timestamp("power_off") <= fsm.timestamp("requeued")


def test_a_clamp_readback_failure_hard_trips_and_cuts_power(cfg, tmp_path) -> None:
    fsm, steps, reading, power, _ = asyncio.run(_run(cfg, tmp_path / "checkpoints", clamp_ok=False))

    assert fsm.state is IncidentState.HARD_TRIP
    assert [step.command for step in steps] == ["CLAMP", "SCHEDULER_DRAIN", "FORCE_OFF"]
    assert steps[0].status == "READBACK_FAILED"
    assert reading.solenoid_open is True
    assert reading.limit_closed is False
    assert reading.pressure_psi > 0.0
    assert power == "Off"
    assert "valve_closed" not in [name for name, _ in fsm.events]


def test_a_valve_fault_hard_trips_and_cuts_power(cfg, tmp_path) -> None:
    """T-FSM-04: a failed close escalates to the BMC, it does not claim isolation."""

    fsm, steps, reading, power, _ = asyncio.run(
        _run(cfg, tmp_path / "checkpoints", plc_travel_ms=1600)
    )

    assert fsm.state is IncidentState.HARD_TRIP
    assert [step.command for step in steps] == [
        "CLAMP",
        "VALVE_CLOSE",
        "SCHEDULER_DRAIN",
        "FORCE_OFF",
    ]
    assert steps[1].status == "FAULT"
    assert reading.fault is True
    assert reading.limit_closed is False
    assert power == "Off"
    assert "valve_closed" not in [name for name, _ in fsm.events]


def test_a_valve_fault_with_a_failed_power_cut_is_failed_safe(cfg, tmp_path) -> None:
    fsm, steps, reading, power, _ = asyncio.run(
        _run(cfg, tmp_path / "checkpoints", plc_travel_ms=1600, failing_bmc=True)
    )

    assert fsm.state is IncidentState.FAILED_SAFE
    assert steps[-1].command == "FORCE_OFF"
    assert steps[-1].status == "REJECTED"
    assert reading.fault is True
    assert power == "On"  # the real BMC was never asked, because the injected one failed


def test_a_corrupt_checkpoint_does_not_block_isolation(cfg, tmp_path) -> None:
    """C-04: the checkpoint gates power-off, never the clamp or the valve."""

    directory = tmp_path / "corrupt"
    write_checkpoint(directory, 10, b"step-ten-payload")
    (directory / "step_000010.bin").write_bytes(b"tampered")

    fsm, steps, reading, power, _ = asyncio.run(_run(cfg, directory))

    assert fsm.state is IncidentState.FAILED_SAFE
    assert [step.command for step in steps] == [
        "CHECKPOINT_VERIFY",
        "CLAMP",
        "VALVE_CLOSE",
        "SCHEDULER_DRAIN",
    ]
    assert steps[0].status == "FAILED"
    assert "failed SHA-256" in steps[0].detail
    assert reading.limit_closed is True
    assert reading.pressure_psi == 0.0
    assert power == "On"  # no verified checkpoint, so no power-off
    assert "power_off" not in [name for name, _ in fsm.events]
    assert "requeued" not in [name for name, _ in fsm.events]
