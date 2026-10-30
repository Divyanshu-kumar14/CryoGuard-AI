"""Adversarial safety regressions: failure paths that must never look healthy.

Every case uses the production classes with injected fakes. No test talks to real
hardware, the network, or a real BMC/PLC; the only real sockets in this project
are exercised by the adapter-specific suites. The assertions are about claims:
an accepted command is not a physical outcome, a blind camera is not a quiet
chassis, and a failed recovery step is never filed as a clean resolution.
"""

from __future__ import annotations

import asyncio
import copy
import time
from dataclasses import replace
from pathlib import Path

import httpx
import numpy as np
import pytest
import yaml

import cryoguard.pipeline as pipeline
from cryoguard.adapters.nvml import GpuBackend, GpuObservation
from cryoguard.adapters.plc_client import PlcClient
from cryoguard.adapters.redfish_client import RedfishClient
from cryoguard.config import load
from cryoguard.events import EventBus
from cryoguard.fusion.decision import decide
from cryoguard.models import Evidence, IncidentState, SystemLabel, ValveStatus
from cryoguard.orchestrator.actions import ActionLog, SequenceStep
from cryoguard.orchestrator.adapters import (
    ScenarioClock,
    SimulatedBmc,
    SimulatedCheckpoint,
    SimulatedNvml,
    SimulatedPlc,
    SimulatedScheduler,
)
from cryoguard.orchestrator.emergency import EmergencySequence
from cryoguard.orchestrator.incident_fsm import (
    VERIFICATION_TIMEOUT,
    VERIFIED_OFF,
    ClampResult,
    ConfirmedLeakOrchestrator,
    IllegalTransition,
    IncidentFSM,
    PowerResult,
    power_verified,
)
from cryoguard.orchestrator.watchdog import (
    ThermalWatchdog,
    WatchdogActuationError,
    WatchdogReading,
    WatchdogTrip,
)
from cryoguard.perception.health import assess_health
from cryoguard.pipeline import _step_emitter, _steps_to_actions
from cryoguard.report.schema import IncidentReport

ROOT = Path(__file__).resolve().parents[1]
EMERGENCY_ACTIONS = ("VALVE_CLOSE", "FORCE_OFF", "SCHEDULER_DRAIN")


@pytest.fixture(scope="module")
def cfg() -> dict:
    return load(ROOT / "configs" / "default.yaml")


@pytest.fixture(scope="module")
def topology() -> dict:
    return yaml.safe_load((ROOT / "configs" / "topology.yaml").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Camera health: blind is not nominal, and bad frames carry no confirmatory
# visual evidence into the decision table.
# ---------------------------------------------------------------------------


def _leak_evidence(**overrides) -> Evidence:
    evidence = Evidence(
        zone=0,
        phi_v=0.9,
        phi_h=0.9,
        phi_t=0.0,
        W=1.0,
        dA_dt=20.0,
        q=1.0,
        ticks=10,
    )
    return replace(evidence, **overrides)


def test_a_degraded_camera_cannot_confirm_a_visually_supported_leak(cfg) -> None:
    blind = _leak_evidence(camera_healthy=False)
    healthy = _leak_evidence(camera_healthy=True)

    # The same numbers would confirm a leak with a working camera.
    assert decide(healthy, cfg).rule_id == "R2"

    decision = decide(blind, cfg)
    assert decision.rule_id == "R8"
    assert decision.label is SystemLabel.CAMERA_DEGRADED
    assert decision.actuates is False
    assert "CLAMP" not in decision.actions
    assert "VALVE_CLOSE" not in decision.actions


def test_a_bad_frame_assessed_as_unhealthy_cannot_contribute_visual_evidence(cfg) -> None:
    frame = np.zeros((400, 600, 3), dtype=np.uint8)
    health = assess_health(frame, cfg["camera"])
    assert health.ok is False

    decision = decide(_leak_evidence(camera_healthy=health.ok), cfg)

    assert decision.rule_id == "R8"
    assert decision.label is SystemLabel.CAMERA_DEGRADED
    assert decision.actuates is False


@pytest.mark.parametrize(
    "overrides",
    [
        {"phi_v": 0.9, "phi_h": 0.9},  # would be R2 with a healthy camera
        {"phi_v": 0.0, "phi_h": 1.0},  # hydraulic-only evidence
        {"phi_v": 0.0, "phi_h": 0.0},  # a quiet-looking frame
        {"maintenance": True},
    ],
)
def test_a_camera_fault_is_never_labelled_nominal(cfg, overrides: dict) -> None:
    decision = decide(_leak_evidence(camera_healthy=False, **overrides), cfg)

    assert decision.label is SystemLabel.CAMERA_DEGRADED
    assert decision.label is not SystemLabel.NOMINAL
    assert decision.rule_id == "R8"


def test_a_camera_fault_with_a_hard_fault_is_emergency_not_nominal(cfg) -> None:
    decision = decide(
        _leak_evidence(camera_healthy=False, plc_fault=True, temp_c=96.0),
        cfg,
    )

    assert decision.rule_id == "R1b"
    assert decision.label is SystemLabel.EMERGENCY_TRIP
    assert decision.label is not SystemLabel.NOMINAL


# ---------------------------------------------------------------------------
# Incident lifecycle: duplicates, restarts, cancellation, timeouts, and
# recovery failures.
# ---------------------------------------------------------------------------


def _confirmed_orchestrator(
    cfg: dict,
    *,
    clock: ScenarioClock | None = None,
    bmc=None,
    scheduler=None,
    nvml=None,
    plc=None,
    checkpoint=None,
    on_step=None,
    abort=None,
) -> ConfirmedLeakOrchestrator:
    clock = clock or ScenarioClock(10.0)
    return ConfirmedLeakOrchestrator(
        clock=clock,
        nvml=nvml or SimulatedNvml(200.0),
        plc=plc or SimulatedPlc(950.0),
        checkpoint=checkpoint or SimulatedCheckpoint(clock, settle_s=1.0, step=4200),
        bmc=bmc or SimulatedBmc(),
        scheduler=scheduler or SimulatedScheduler(cfg["scheduler"]),
        on_step=on_step,
        abort=abort,
    )


def test_duplicate_incident_events_cannot_start_or_close_twice() -> None:
    fsm = IncidentFSM()
    fsm.fire("decision", 0.0)
    fsm.fire("clamp_verified", 0.1)
    fsm.fire("valve_closed", 1.0)

    with pytest.raises(IllegalTransition, match="no transition for decision"):
        fsm.fire("decision", 1.1)
    with pytest.raises(IllegalTransition, match="no transition for valve_closed"):
        fsm.fire("valve_closed", 1.2)

    assert [name for name, _ in fsm.events] == ["decision", "clamp_verified", "valve_closed"]
    assert fsm.state is IncidentState.ISOLATED


def test_a_restart_during_an_incident_starts_fresh_and_cannot_resume_power_off() -> None:
    before_restart = IncidentFSM()
    before_restart.fire("decision", 0.0)
    before_restart.fire("clamp_verified", 0.1)
    before_restart.fire("valve_closed", 1.0)
    assert before_restart.state is IncidentState.ISOLATED
    assert before_restart.checkpoint_settled is False

    restarted = IncidentFSM()
    restarted_log = ActionLog(ScenarioClock(50.0))

    assert restarted.state is IncidentState.OPEN
    assert restarted.events == []
    assert restarted.checkpoint_settled is False
    assert restarted_log.steps == ()
    with pytest.raises(IllegalTransition, match="power-off before the checkpoint settled"):
        restarted.fire("power_off", 50.0)
    assert restarted.state is IncidentState.OPEN


class _HangingSensor:
    def __init__(self, started: asyncio.Event) -> None:
        self.started = started

    def read(self) -> WatchdogReading:
        self.started.set()
        return WatchdogReading(42.0)


def test_event_loop_cancellation_never_claims_an_actuation(cfg) -> None:
    async def case():
        started = asyncio.Event()
        calls: list[WatchdogTrip] = []

        async def actuate(reason: WatchdogTrip, detail: str) -> None:
            calls.append(reason)

        dog = ThermalWatchdog(
            _HangingSensor(started),
            limit_c=float(cfg["thermal_model"]["watchdog_limit_c"]),
            period_s=0.05,
            actuate=actuate,
        )
        task = asyncio.create_task(dog.run(asyncio.Event()))
        await asyncio.wait_for(started.wait(), timeout=1.0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return dog, calls

    dog, calls = asyncio.run(case())

    assert calls == []
    assert dog.trip is None
    assert dog.actuated is False
    assert dog.actuation_failed is False


def test_an_emergency_power_off_timeout_stays_failed_safe(cfg) -> None:
    class _TimedOutBmc:
        def __init__(self) -> None:
            self.calls = 0

        def force_off(self) -> PowerResult:
            self.calls += 1
            return PowerResult(
                ok=False,
                state="On",
                attempts=1,
                status_code=202,
                simulated=True,
                detail="PowerState still On after 3000 ms",
                status=VERIFICATION_TIMEOUT,
            )

    bmc = _TimedOutBmc()
    result = EmergencySequence(
        ScenarioClock(10.0),
        SimulatedPlc(950.0),
        bmc,
        SimulatedScheduler(cfg["scheduler"]),
    ).run("junction 96.4 C")

    assert result.state is IncidentState.FAILED_SAFE
    assert bmc.calls == 1
    assert result.steps[-1].command == "FORCE_OFF"
    assert result.steps[-1].status == VERIFICATION_TIMEOUT
    assert result.steps[-1].status != "ACCEPTED"


class _BrokenDrain:
    """DRAIN fails while REQUEUE remains independently possible."""

    def __init__(self) -> None:
        self.requeued = False
        self.checkpoint_step: int | None = None

    def drain(self) -> str:
        raise ConnectionError("slurm controller unreachable")

    def requeue(self, checkpoint_step: int) -> str:
        self.requeued = True
        self.checkpoint_step = checkpoint_step
        return "gpu-node-09a"


class _BrokenRequeue(SimulatedScheduler):
    def requeue(self, checkpoint_step: int) -> str:
        raise ConnectionError("slurm controller unreachable")


def test_a_failed_scheduler_cordon_never_blocks_the_power_cut(cfg) -> None:
    bmc = SimulatedBmc()
    scheduler = _BrokenDrain()
    orchestrator = _confirmed_orchestrator(cfg, bmc=bmc, scheduler=scheduler)

    fsm = orchestrator.run()

    commands = [step.command for step in orchestrator.steps]
    assert fsm.state is IncidentState.RESOLVED_PENDING_REPAIR
    assert commands == [
        "CLAMP",
        "VALVE_CLOSE",
        "SCHEDULER_DRAIN",
        "CHECKPOINT_VERIFY",
        "FORCE_OFF",
        "REQUEUE",
    ]
    drain = next(step for step in orchestrator.steps if step.command == "SCHEDULER_DRAIN")
    assert drain.status == "FAILED"
    assert "ConnectionError" in drain.detail
    assert bmc.calls == 1
    assert scheduler.requeued is True
    assert scheduler.checkpoint_step == 4200
    assert "power_off" in [name for name, _ in fsm.events]


def test_a_failed_emergency_cordon_still_cuts_power(cfg) -> None:
    bmc = SimulatedBmc()
    result = EmergencySequence(
        ScenarioClock(10.0),
        SimulatedPlc(950.0),
        bmc,
        _BrokenDrain(),
    ).run("junction 96.4 C")

    assert result.state is IncidentState.EMERGENCY_TRIP
    assert [(step.command, step.status) for step in result.steps] == [
        ("VALVE_CLOSE", "LIMIT_SWITCH_CLOSED"),
        ("SCHEDULER_DRAIN", "FAILED"),
        ("FORCE_OFF", "ACCEPTED"),
    ]
    assert bmc.calls == 1


def test_a_failed_requeue_keeps_the_node_powered_down_and_records_the_failure(cfg) -> None:
    bmc = SimulatedBmc()
    orchestrator = _confirmed_orchestrator(cfg, bmc=bmc, scheduler=_BrokenRequeue(cfg["scheduler"]))

    fsm = orchestrator.run()

    assert fsm.state is IncidentState.NODE_POWERED_DOWN
    assert fsm.state is not IncidentState.RESOLVED_PENDING_REPAIR
    assert "requeued" not in [name for name, _ in fsm.events]
    assert "closed" not in [name for name, _ in fsm.events]
    assert bmc.calls == 1
    assert orchestrator.log.sealed is True

    last = orchestrator.steps[-1]
    assert (last.actor, last.command, last.status) == ("SLURM", "REQUEUE", "FAILED")
    assert "ConnectionError" in last.detail
    assert all(step.status != "REQUEUED" for step in orchestrator.steps)


def test_an_abort_after_power_off_never_requeues_the_node(cfg) -> None:
    scheduler = _BrokenDrain()
    orchestrator = _confirmed_orchestrator(
        cfg,
        scheduler=scheduler,
        abort=lambda: "watchdog JUNCTION_OVER_TEMP took precedence",
    )

    fsm = orchestrator.run()

    assert fsm.state is IncidentState.NODE_POWERED_DOWN
    assert scheduler.requeued is False
    assert "requeued" not in [name for name, _ in fsm.events]
    assert orchestrator.steps[-1].command == "REQUEUE"
    assert orchestrator.steps[-1].status == "ABORTED"
    assert "watchdog JUNCTION_OVER_TEMP" in orchestrator.steps[-1].detail


class _AcceptedOnlyBmc:
    """Reports HTTP acceptance but never a readback: not a power-off."""

    def __init__(self, state: str = "On", *, simulated: bool = True, status: str = "") -> None:
        self.state = state
        self.simulated = simulated
        self.status = status
        self.calls = 0

    def force_off(self) -> PowerResult:
        self.calls += 1
        return PowerResult(
            ok=True,
            state=self.state,
            attempts=1,
            status_code=202,
            simulated=self.simulated,
            detail="accepted only",
            status=self.status,
        )


def test_power_verified_requires_a_readback_not_an_accept() -> None:
    assert power_verified(PowerResult(True, "Off", 1, 202, True, "")) is True
    assert power_verified(PowerResult(True, "Off", 1, 202, False, "", VERIFIED_OFF)) is True

    assert power_verified(PowerResult(True, "On", 1, 202, True, "")) is False
    assert power_verified(PowerResult(True, "Off", 1, 202, False, "")) is False
    assert power_verified(PowerResult(False, "Off", 1, 202, True, "")) is False
    assert power_verified(PowerResult(False, "Unknown", 1, 0, True, "timeout")) is False


def test_an_accepted_but_unverified_power_cut_is_never_reported_as_success(cfg) -> None:
    bmc = _AcceptedOnlyBmc(state="On")
    orchestrator = _confirmed_orchestrator(cfg, bmc=bmc)

    fsm = orchestrator.run()

    assert fsm.state is IncidentState.FAILED_SAFE
    assert bmc.calls == 1
    assert "power_off" not in [name for name, _ in fsm.events]
    assert "requeued" not in [name for name, _ in fsm.events]
    assert "closed" not in [name for name, _ in fsm.events]
    assert orchestrator.steps[-1].command == "FORCE_OFF"
    assert orchestrator.steps[-1].status == "ACCEPTED"
    assert "REQUEUE" not in [step.command for step in orchestrator.steps]


def test_an_accepted_but_unverified_emergency_cut_is_failed_safe(cfg) -> None:
    result = EmergencySequence(
        ScenarioClock(10.0),
        SimulatedPlc(950.0),
        _AcceptedOnlyBmc(state="On"),
        SimulatedScheduler(cfg["scheduler"]),
    ).run("junction 96.4 C")

    assert result.state is IncidentState.FAILED_SAFE
    assert result.state is not IncidentState.EMERGENCY_TRIP
    assert result.steps[-1].status == "ACCEPTED"


def test_multiple_simultaneous_safety_triggers_take_one_emergency_path(cfg) -> None:
    decision = decide(
        _leak_evidence(
            zone=2,
            plc_fault=True,
            temp_c=96.0,
            clamp_readback_ok=False,
            camera_healthy=False,
        ),
        cfg,
    )

    assert decision.rule_id == "R1b"
    assert decision.label is SystemLabel.EMERGENCY_TRIP
    assert set(decision.actions) == set(EMERGENCY_ACTIONS)

    actions, state = asyncio.run(pipeline._actions_for(decision, 7.6, cfg, None, "memory", None))

    assert state is IncidentState.EMERGENCY_TRIP
    commands = [action.command for action in actions]
    assert commands == ["VALVE_CLOSE", "SCHEDULER_DRAIN", "FORCE_OFF"]
    assert commands.count("VALVE_CLOSE") == 1
    assert commands.count("FORCE_OFF") == 1


# ---------------------------------------------------------------------------
# Watchdog during a confirmed leak: the emergency trip must not be lost.
# ---------------------------------------------------------------------------


class _ScriptedNvml:
    """Cool until the confirmed-leak sequence starts, then over the limit."""

    def __init__(self) -> None:
        self.hot = False
        self.samples = 0

    def sample(self) -> GpuObservation:
        self.samples += 1
        return GpuObservation(
            temp_c=96.4 if self.hot else 42.0,
            utilization_pct=None,
            power_w=None,
            clock_mhz=None,
            device_index=0,
            sampled_monotonic_s=time.monotonic(),
            backend=GpuBackend.SIMULATED,
            detail="SIMULATED GPU temperature",
        )

    def clamp(self) -> ClampResult:
        return ClampResult(ok=True, applied_w=200.0, detail="scripted clamp")


@pytest.mark.slow
def test_a_watchdog_trip_during_a_confirmed_leak_is_not_lost(
    cfg, topology, tmp_path, monkeypatch
) -> None:
    sensor = _ScriptedNvml()
    emergency_done = asyncio.Event()
    original_confirmed = pipeline._confirmed_leak_actions
    original_emergency = pipeline._emergency_actions

    def build_scripted_nvml(*args, **kwargs) -> _ScriptedNvml:
        return sensor

    async def observed_emergency(*args, **kwargs):
        try:
            return await original_emergency(*args, **kwargs)
        finally:
            emergency_done.set()

    async def trip_then_confirm(*args, **kwargs):
        sensor.hot = True
        await asyncio.wait_for(emergency_done.wait(), timeout=2.0)
        await asyncio.sleep(0.01)  # let the watchdog latch `actuated`
        return await original_confirmed(*args, **kwargs)

    monkeypatch.setattr(pipeline, "build_nvml", build_scripted_nvml)
    monkeypatch.setattr(pipeline, "_emergency_actions", observed_emergency)
    monkeypatch.setattr(pipeline, "_confirmed_leak_actions", trip_then_confirm)

    hot = copy.deepcopy(cfg)
    hot["watchdog"]["period_ms"] = 0

    result = asyncio.run(
        pipeline.run_scenario(
            "leak",
            hot,
            topology,
            out_dir=tmp_path,
            max_seconds=10.0,
            protocol_mode="memory",
            allow_llm=False,
        )
    )

    assert result.report is not None
    assert result.report.physical_evidence.label == "EMERGENCY_TRIP"
    assert result.report.trip_reason == "JUNCTION_OVER_TEMP"
    assert result.sequence_state is IncidentState.EMERGENCY_TRIP
    commands = [action.command for action in result.report.deterministic_actions_taken]
    assert commands == ["VALVE_CLOSE", "SCHEDULER_DRAIN", "FORCE_OFF"]
    assert "REQUEUE" not in commands
    assert sensor.samples > 0


@pytest.mark.slow
def test_a_failed_watchdog_actuation_during_a_confirmed_leak_files_failed_safe(
    cfg, topology, tmp_path, monkeypatch
) -> None:
    sensor = _ScriptedNvml()
    emergency_done = asyncio.Event()
    original_confirmed = pipeline._confirmed_leak_actions

    def build_scripted_nvml(*args, **kwargs) -> _ScriptedNvml:
        return sensor

    async def failing_emergency(*args, **kwargs):
        try:
            raise ConnectionError("primary emergency actuation unreachable")
        finally:
            emergency_done.set()

    async def trip_then_confirm(*args, **kwargs):
        sensor.hot = True
        await asyncio.wait_for(emergency_done.wait(), timeout=2.0)
        await asyncio.sleep(0.01)
        return await original_confirmed(*args, **kwargs)

    monkeypatch.setattr(pipeline, "build_nvml", build_scripted_nvml)
    monkeypatch.setattr(pipeline, "_emergency_actions", failing_emergency)
    monkeypatch.setattr(pipeline, "_confirmed_leak_actions", trip_then_confirm)

    hot = copy.deepcopy(cfg)
    hot["watchdog"]["period_ms"] = 0

    with pytest.raises(WatchdogActuationError):
        asyncio.run(
            pipeline.run_scenario(
                "leak",
                hot,
                topology,
                out_dir=tmp_path,
                max_seconds=10.0,
                protocol_mode="memory",
                allow_llm=False,
            )
        )

    paths = sorted(tmp_path.glob("INC-*.json"))
    assert len(paths) == 1
    report = IncidentReport.model_validate_json(paths[0].read_text(encoding="utf-8"))
    assert report.trip_reason == "JUNCTION_OVER_TEMP"
    assert report.physical_evidence.label == "EMERGENCY_TRIP"
    by_command = {(action.command, action.status) for action in report.deterministic_actions_taken}
    assert ("TRIP", "INITIATED") in by_command
    assert ("FORCE_OFF", "FAILED") in by_command
    assert ("FORCE_OFF_FALLBACK", VERIFIED_OFF) in by_command
    assert ("REQUEUE", "REQUEUED") not in by_command


# ---------------------------------------------------------------------------
# Redfish: accepted is not off; only a readback can end the sequence.
# ---------------------------------------------------------------------------


def _power_body(power: str) -> dict:
    return {"PowerState": power, "Oem": {"CryoGuard": {"simulated": True}}}


def _redfish(cfg: dict, handler, **overrides) -> RedfishClient:
    section = dict(cfg["bmc"], **overrides)
    transport = httpx.MockTransport(handler)
    return RedfishClient(section, client=httpx.Client(transport=transport))


@pytest.mark.parametrize("status", [200, 202, 204])
def test_every_accepted_http_status_is_verified_by_readback(cfg, status: int) -> None:
    polls = {"get": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            polls["get"] += 1
            return httpx.Response(200, json=_power_body("On" if polls["get"] == 1 else "Off"))
        if status == 204:
            return httpx.Response(204)
        return httpx.Response(status, json=_power_body("On"))

    result = _redfish(cfg, handler, poll_interval_ms=1, power_off_timeout_ms=100).force_off()

    assert result.ok is True
    assert result.status == VERIFIED_OFF
    assert result.state == "Off"
    assert result.status_code == status
    assert polls["get"] == 2


def test_a_200_whose_power_state_never_flips_is_a_timeout(cfg) -> None:
    polls = {"get": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            polls["get"] += 1
            return httpx.Response(200, json=_power_body("On"))
        return httpx.Response(200, json=_power_body("Off"))

    result = _redfish(cfg, handler, poll_interval_ms=1, power_off_timeout_ms=20).force_off()

    assert result.ok is False
    assert result.status == VERIFICATION_TIMEOUT
    assert result.status_code == 200
    assert result.state == "On"
    assert polls["get"] > 1


def test_an_accepted_command_with_a_failing_readback_is_never_off(cfg) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            raise httpx.ConnectError("bmc readback unreachable", request=request)
        return httpx.Response(202, json=_power_body("Off"))

    result = _redfish(cfg, handler, poll_interval_ms=1, power_off_timeout_ms=20).force_off()

    assert result.ok is False
    assert result.status == VERIFICATION_TIMEOUT
    assert result.state == "Unknown"
    assert result.status != VERIFIED_OFF


def test_a_malformed_system_response_is_unknown_not_off(cfg) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, content=b"not-json")
        return httpx.Response(202, json=_power_body("Off"))

    result = _redfish(cfg, handler, poll_interval_ms=1, power_off_timeout_ms=20).force_off()

    assert result.ok is False
    assert result.status == VERIFICATION_TIMEOUT
    assert result.state == "Unknown"


@pytest.mark.parametrize("status", [401, 403])
def test_auth_rejection_is_not_retried_and_never_succeeds(cfg, status: int) -> None:
    calls = {"post": 0, "get": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            calls["get"] += 1
            return httpx.Response(200, json=_power_body("Off"))
        calls["post"] += 1
        return httpx.Response(status, json={"error": {"message": "unauthorized"}})

    result = _redfish(cfg, handler, retry_backoff_ms=0).force_off()

    assert result.ok is False
    assert result.attempts == 1
    assert result.status_code == status
    assert calls["post"] == 1
    assert calls["get"] == 0
    assert str(status) in result.detail


def test_a_client_timeout_on_the_reset_is_retried_then_failed(cfg) -> None:
    calls = {"post": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["post"] += 1
        raise httpx.ReadTimeout("bmc did not answer", request=request)

    result = _redfish(cfg, handler, retry_backoff_ms=0, max_retries=2).force_off()

    assert result.ok is False
    assert result.attempts == 3
    assert result.status_code == 0
    assert result.state == "Unknown"
    assert calls["post"] == 3
    assert "ReadTimeout" in result.detail


def test_a_409_for_a_system_that_is_not_off_is_a_failure(cfg) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json=_power_body("On"))
        return httpx.Response(409, json={"error": {"message": "conflict"}})

    result = _redfish(cfg, handler).force_off()

    assert result.ok is False
    assert result.status_code == 409
    assert result.state == "On"
    assert result.status != VERIFIED_OFF


# ---------------------------------------------------------------------------
# Modbus: a rejected command is a failed close, even when reads would look OK.
# ---------------------------------------------------------------------------


class _OkResponse:
    def __init__(self, *, bits=(), registers=()) -> None:
        self.bits = list(bits)
        self.registers = list(registers)

    def isError(self) -> bool:
        return False


class _ErrorResponse:
    bits: list = []
    registers: list = []

    def isError(self) -> bool:
        return True


class _RejectedCoilWrite:
    def __init__(self) -> None:
        self.closed = False
        self.reads = 0

    def write_coil(self, *args, **kwargs) -> _ErrorResponse:
        return _ErrorResponse()

    def read_coils(self, *args, **kwargs) -> _OkResponse:
        self.reads += 1
        raise AssertionError("a rejected write must not be followed by polling")

    def read_discrete_inputs(self, *args, **kwargs) -> _OkResponse:
        self.reads += 1
        raise AssertionError("a rejected write must not be followed by polling")

    def close(self) -> None:
        self.closed = True


def test_a_rejected_coil_write_is_a_fault_without_readback_claims(cfg, monkeypatch) -> None:
    stub = _RejectedCoilWrite()
    client = PlcClient(dict(cfg["plc"]))
    monkeypatch.setattr(PlcClient, "_connect", lambda self: stub)

    result = client.close_valve()

    assert result.status is ValveStatus.FAULT
    assert result.status is not ValveStatus.VERIFIED
    assert result.travel_ms == 0.0
    assert "coil write failed" in result.detail
    assert stub.reads == 0
    assert stub.closed is True


class _PhaseOneReadFailure:
    def __init__(self, *, coils_error: bool = False) -> None:
        self.coils_error = coils_error
        self.closed = False

    def write_coil(self, *args, **kwargs) -> _OkResponse:
        return _OkResponse()

    def read_coils(self, *args, **kwargs) -> _OkResponse | _ErrorResponse:
        if self.coils_error:
            return _ErrorResponse()
        return _OkResponse(bits=[True, False, False])

    def read_discrete_inputs(self, *args, **kwargs) -> _OkResponse:
        return _OkResponse(bits=[False, False])

    def close(self) -> None:
        self.closed = True


def test_a_control_path_read_failure_is_a_fault_not_a_close(cfg, monkeypatch) -> None:
    stub = _PhaseOneReadFailure(coils_error=True)
    section = dict(cfg["plc"], valve_fault_ms=120, poll_ms=1)
    client = PlcClient(section)
    monkeypatch.setattr(PlcClient, "_connect", lambda self: stub)

    result = client.close_valve()

    assert result.status is ValveStatus.FAULT
    assert result.status is not ValveStatus.VERIFIED
    assert "read failed while polling the valve" in result.detail
    assert stub.closed is True


# ---------------------------------------------------------------------------
# Action log: one ordered, uniquely numbered record, with honest provenance.
# ---------------------------------------------------------------------------


def test_action_log_sequence_numbers_and_fields_are_complete(cfg) -> None:
    emitted: list[tuple[int, SequenceStep]] = []

    def on_step(step: SequenceStep, seq: int) -> None:
        emitted.append((seq, step))

    orchestrator = _confirmed_orchestrator(cfg, on_step=on_step)
    fsm = orchestrator.run()

    assert fsm.state is IncidentState.RESOLVED_PENDING_REPAIR
    assert [seq for seq, _ in emitted] == list(range(1, len(emitted) + 1))
    assert len({seq for seq, _ in emitted}) == len(emitted)
    assert [step for _, step in emitted] == list(orchestrator.steps)
    assert [step.ts for step in orchestrator.steps] == sorted(
        step.ts for step in orchestrator.steps
    )
    for step in orchestrator.steps:
        assert step.actor
        assert step.command
        assert step.status
        assert step.detail

    statuses = {step.command: step.status for step in orchestrator.steps}
    assert statuses["CLAMP"] == "READBACK_VERIFIED"
    assert statuses["VALVE_CLOSE"] == "LIMIT_SWITCH_CLOSED"
    assert statuses["CHECKPOINT_VERIFY"] == "COMMITTED"
    assert statuses["FORCE_OFF"] == "ACCEPTED"
    assert statuses["REQUEUE"] == "REQUEUED"

    actions = _steps_to_actions(orchestrator.steps)
    assert [action.seq for action in actions] == list(range(1, len(actions) + 1))


def test_action_events_carry_simulated_and_live_provenance() -> None:
    step = SequenceStep(ts=1.0, actor="BMC", command="FORCE_OFF", status="ACCEPTED")

    simulated_bus = EventBus()
    _step_emitter(simulated_bus, "memory")(step, 1)
    live_bus = EventBus()
    _step_emitter(live_bus, "external")(step, 1)

    simulated = simulated_bus.snapshot()[-1]
    live = live_bus.snapshot()[-1]
    assert simulated.type == live.type == "ACTION_EXECUTED"
    assert simulated.data["simulated"] is True
    assert live.data["simulated"] is False
    assert simulated.data["actor"] == live.data["actor"] == "BMC"
    assert simulated.data["command"] == live.data["command"] == "FORCE_OFF"
