"""Vertical slice: clip -> candidates -> fusion -> decision -> safe sequence -> incident.

The loop is asynchronous only because SAM is. Every other stage is a pure
function of the frame and the scenario clock; nothing waits on the model.

Order inside a tick matters:

1. classical tracker update (5-10 Hz, always);
2. drain SAM results and re-anchor the tracker (<=2 Hz, never awaited);
3. offer this frame to SAM (returns immediately);
4. read the candidate set once, so W advances exactly once per tick.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from cryoguard.adapters.loopback import SimEndpoints, loopback_simulators
from cryoguard.adapters.nvml import build_nvml
from cryoguard.adapters.plc_client import PlcClient
from cryoguard.adapters.redfish_client import RedfishClient
from cryoguard.adapters.scheduler import CheckpointAdapter, SchedulerAdapter
from cryoguard.capture.rectify import estimate
from cryoguard.capture.source import VideoSource
from cryoguard.config import config_hash, get
from cryoguard.fusion.features import hydraulic_phi, thermal_phi, visual_phi
from cryoguard.fusion.hysteresis import HysteresisDecision
from cryoguard.models import Decision, Evidence, IncidentState, SystemLabel
from cryoguard.orchestrator.adapters import (
    ScenarioClock,
    SimulatedBmc,
    SimulatedCheckpoint,
    SimulatedNvml,
    SimulatedPlc,
    SimulatedScheduler,
)
from cryoguard.orchestrator.emergency import EmergencySequence
from cryoguard.orchestrator.incident_fsm import ConfirmedLeakOrchestrator, SequenceStep
from cryoguard.orchestrator.watchdog import (
    NvmlJunctionSensor,
    ThermalWatchdog,
    TripRecord,
    WatchdogTrip,
)
from cryoguard.perception.candidates import Candidate, CandidateManager
from cryoguard.perception.health import assess_health
from cryoguard.perception.sam_client import SamClient
from cryoguard.perception.sam_worker import SamStats, SamWorker
from cryoguard.perception.sampling import SamplingPolicy
from cryoguard.report.fallback import build_fallback_report, scenario_clock
from cryoguard.report.muse_client import MuseClient, MuseUnavailable
from cryoguard.report.schema import ActionModel, EvidenceModel, IncidentReport
from cryoguard.report.writer import write_incident
from cryoguard.telemetry.replay import TelemetryReplay, TelemetrySample

ROOT = Path(__file__).resolve().parents[2]
SCENARIOS = {
    "leak": "slow_leak_qd4",
    "condensation": "condensation_bead",
    "glint": "glint_noise",
    "hidden": "hidden_hydraulic",
}

# Fixed per-scenario offsets keep incident ids unique and deterministic without
# reading the clock or counting files: leak 01, condensation 02, glint 03, hidden 04.
SCENARIO_INDEX = {
    "slow_leak_qd4": 1,
    "condensation_bead": 2,
    "glint_noise": 3,
    "hidden_hydraulic": 4,
}
INCIDENT_LABELS = {
    SystemLabel.CONFIRMED_LEAK,
    SystemLabel.EMERGENCY_TRIP,
    SystemLabel.HIDDEN_HYDRAULIC_EVENT,
}


class PipelineError(RuntimeError):
    """Raised when the chain cannot complete; never silently swallowed."""


@dataclass(frozen=True)
class SamSummary:
    mode: str
    submitted: int
    applied: int
    cached: int
    live: int
    fixture: int
    failures: int
    skipped: int
    regions: int

    @classmethod
    def from_stats(cls, mode: str, stats: SamStats) -> SamSummary:
        return cls(
            mode=mode,
            submitted=stats.submitted,
            applied=stats.applied,
            cached=stats.cached,
            live=stats.live,
            fixture=stats.fixture,
            failures=stats.failures,
            skipped=stats.skipped,
            regions=stats.regions,
        )


@dataclass(frozen=True)
class RunResult:
    scenario: str
    decision: Decision | None
    report: IncidentReport | None
    path: Path | None
    frames: int
    ticks: int
    tick_rate_hz: float
    rate_changes: int
    sam: SamSummary | None
    protocol_mode: str = "memory"
    sequence_state: IncidentState | None = None


def resolve_scenario(name: str) -> str:
    return SCENARIOS.get(name, name)


def load_scenario(name: str) -> dict[str, Any]:
    path = ROOT / "configs" / "scenarios" / f"{resolve_scenario(name)}.yaml"
    if not path.is_file():
        raise PipelineError(f"unknown scenario: {name} ({path})")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _incident_time(cfg: dict[str, Any], t_s: float) -> datetime:
    epoch = datetime.fromisoformat(get(cfg, "system.incident_epoch").replace("Z", "+00:00"))
    return scenario_clock(epoch, t_s)


def _action(
    seq: int, t_s: float, actor: str, command: str, status: str, detail: str = ""
) -> ActionModel:
    return ActionModel(
        seq=seq,
        t_s=round(float(t_s), 3),
        actor=actor,
        command=command,
        status=status,
        detail=detail,
    )


def _steps_to_actions(steps: Sequence[SequenceStep]) -> list[ActionModel]:
    """The orchestrator's executed steps are the only source of action records."""

    return [
        ActionModel(
            seq=index,
            t_s=round(float(step.ts), 3),
            actor=step.actor,
            command=step.command,
            status=step.status,
            detail=step.detail,
        )
        for index, step in enumerate(steps, 1)
    ]


def _run_memory_sequence(
    decision_t: float, cfg: dict[str, Any]
) -> tuple[list[ActionModel], IncidentState]:
    """No sockets: the same FSM with in-memory ports (unit tests, offline CI)."""

    clock = ScenarioClock(decision_t)
    orchestrator = ConfirmedLeakOrchestrator(
        clock=clock,
        nvml=SimulatedNvml(get(cfg, "nvml.fake_min_limit_w")),
        plc=SimulatedPlc(get(cfg, "plc.valve_travel_ms")),
        checkpoint=SimulatedCheckpoint(
            clock,
            settle_s=get(cfg, "orchestrator.checkpoint_budget_ms") / 1000.0,
            step=int(get(cfg, "orchestrator.checkpoint_step")),
            reason="in-memory checkpoint (SIMULATED)",
        ),
        bmc=SimulatedBmc(),
        scheduler=SimulatedScheduler(cfg["scheduler"]),
        clamp_latency_ms=get(cfg, "orchestrator.clamp_latency_ms"),
    )
    fsm = orchestrator.run()
    return _steps_to_actions(orchestrator.steps), fsm.state


def _run_live_sequence(
    decision_t: float,
    cfg: dict[str, Any],
    endpoints: SimEndpoints,
    checkpoint_dir: str | Path | None,
) -> tuple[list[ActionModel], IncidentState]:
    """Real Modbus-TCP and Redfish clients; the blocking run happens in a thread."""

    clock = ScenarioClock(decision_t)
    directory = (
        Path(checkpoint_dir)
        if checkpoint_dir is not None
        else ROOT / str(get(cfg, "orchestrator.checkpoint_dir"))
    )
    orchestrator = ConfirmedLeakOrchestrator(
        clock=clock,
        nvml=build_nvml(cfg["nvml"], tolerance_w=float(get(cfg, "orchestrator.clamp_tolerance_w"))),
        plc=PlcClient(cfg["plc"], host=endpoints.plc_host, port=endpoints.plc_port),
        checkpoint=CheckpointAdapter(
            clock,
            directory,
            step=int(get(cfg, "orchestrator.checkpoint_step")),
            scheduler_cfg=cfg["scheduler"],
        ),
        bmc=RedfishClient(cfg["bmc"], base_url=endpoints.bmc_url),
        scheduler=SchedulerAdapter(cfg["scheduler"]),
        clamp_latency_ms=float(get(cfg, "orchestrator.clamp_latency_ms")),
    )
    fsm = orchestrator.run()
    return _steps_to_actions(orchestrator.steps), fsm.state


async def _confirmed_leak_actions(
    decision_t: float,
    cfg: dict[str, Any],
    checkpoint_dir: str | Path | None,
    mode: str,
) -> tuple[list[ActionModel], IncidentState]:
    if mode == "memory":
        return _run_memory_sequence(decision_t, cfg)
    if mode == "loopback":
        # The PLC ticker and the BMC server live on this event loop. The blocking
        # clients must run in a worker thread or the simulation would deadlock.
        async with loopback_simulators(cfg) as endpoints:
            return await asyncio.to_thread(
                _run_live_sequence, decision_t, cfg, endpoints, checkpoint_dir
            )
    if mode == "external":
        return await asyncio.to_thread(
            _run_live_sequence,
            decision_t,
            cfg,
            SimEndpoints.from_config(cfg),
            checkpoint_dir,
        )
    raise PipelineError(
        f"unknown orchestrator.protocol_mode {mode!r} (expected memory, loopback, or external)"
    )


async def _emergency_actions(
    t_s: float, cfg: dict[str, Any], reason: str, mode: str
) -> tuple[list[ActionModel], IncidentState]:
    """R1: valve, cordon, power cut - the same transports as every other path."""

    mode = mode.lower()
    if mode == "memory":
        return await asyncio.to_thread(_run_emergency_memory, t_s, cfg, reason)
    if mode == "loopback":
        async with loopback_simulators(cfg) as endpoints:
            return await asyncio.to_thread(_run_emergency_live, t_s, cfg, endpoints, reason)
    if mode == "external":
        return await asyncio.to_thread(
            _run_emergency_live, t_s, cfg, SimEndpoints.from_config(cfg), reason
        )
    raise PipelineError(f"unknown orchestrator.protocol_mode {mode!r}")


def _run_emergency_memory(
    t_s: float, cfg: dict[str, Any], reason: str
) -> tuple[list[ActionModel], IncidentState]:
    result = EmergencySequence(
        ScenarioClock(t_s),
        SimulatedPlc(get(cfg, "plc.valve_travel_ms")),
        SimulatedBmc(),
        SimulatedScheduler(cfg["scheduler"]),
    ).run(reason)
    return _steps_to_actions(result.steps), result.state


def _run_emergency_live(
    t_s: float,
    cfg: dict[str, Any],
    endpoints: SimEndpoints,
    reason: str,
) -> tuple[list[ActionModel], IncidentState]:
    result = EmergencySequence(
        ScenarioClock(t_s),
        PlcClient(cfg["plc"], host=endpoints.plc_host, port=endpoints.plc_port),
        RedfishClient(cfg["bmc"], base_url=endpoints.bmc_url),
        SchedulerAdapter(cfg["scheduler"]),
    ).run(reason)
    return _steps_to_actions(result.steps), result.state


async def _actions_for(
    decision: Decision,
    t_s: float,
    cfg: dict[str, Any],
    checkpoint_dir: str | Path | None,
    protocol: str,
) -> tuple[list[ActionModel], IncidentState | None]:
    if decision.label is SystemLabel.CONFIRMED_LEAK:
        return await _confirmed_leak_actions(t_s, cfg, checkpoint_dir, protocol)

    if decision.label is SystemLabel.EMERGENCY_TRIP:
        # R1: no checkpoint wait and no requeue. The hardware condition is the trigger.
        return await _emergency_actions(t_s, cfg, f"{decision.rule_id}: emergency trip", protocol)

    if decision.label is SystemLabel.HIDDEN_HYDRAULIC_EVENT:
        drained = SimulatedScheduler(cfg["scheduler"]).drain()
        return (
            [
                _action(1, t_s, "CONSOLE", "ALERT", "RAISED", "no visual evidence"),
                _action(2, t_s, "SLURM", "SCHEDULER_DRAIN", drained, "soft cordon; no valve"),
            ],
            None,
        )

    return [], None


def _run_index(name: str, cfg: dict[str, Any]) -> int:
    base = int(get(cfg, "demo.run_index"))
    return base + SCENARIO_INDEX.get(resolve_scenario(name), 1) - 1


def _build_sam_worker(
    mode: str,
    cache_dir: str | Path | None,
    cfg: dict[str, Any],
    topology: dict[str, Any],
    px_per_mm: float,
    canvas: tuple[int, int],
    fps: float,
) -> tuple[SamWorker | None, SamClient | None]:
    if mode == "off":
        return None, None
    client = SamClient(cfg["sam"], mode=mode, cache_dir=cache_dir)
    worker = SamWorker(client, topology, cfg["sam"], px_per_mm, canvas, fps)
    return worker, client


def _junction_of(watchdog: ThermalWatchdog) -> float | None:
    """Best-effort display value; the trip itself is already recorded."""

    try:
        return float(watchdog.sensor.read().junction_c)
    except Exception:
        return None


async def run_scenario(
    name: str,
    cfg: dict[str, Any],
    topology: dict[str, Any],
    out_dir: str | Path | None = None,
    max_seconds: float | None = None,
    sam_mode: str | None = None,
    sam_cache_dir: str | Path | None = None,
    protocol_mode: str | None = None,
    checkpoint_dir: str | Path | None = None,
) -> RunResult:
    scenario = load_scenario(name)
    clip = ROOT / scenario["clip"]
    if not clip.is_file():
        raise PipelineError(f"clip not found: {clip}")

    px_per_mm = float(get(cfg, "camera.px_per_mm"))
    canvas = (
        int(round(float(topology["chassis"]["width_mm"]) * px_per_mm)),
        int(round(float(topology["chassis"]["height_mm"]) * px_per_mm)),
    )

    mode = str(sam_mode or cfg["sam"].get("mode", "off"))
    protocol = str(protocol_mode or cfg["orchestrator"]["protocol_mode"]).lower()
    sampler = CandidateManager(
        cfg["perception"],
        topology,
        px_per_mm,
        canvas,
        anchor_ttl_s=float(cfg["sam"].get("anchor_ttl_s", 1.2)),
    )
    telemetry = TelemetryReplay(scenario)
    engine = HysteresisDecision(cfg)
    policy = SamplingPolicy(cfg["perception"])
    worker: SamWorker | None = None
    last_decision: Decision | None = None
    last_frame = 0
    ticks = 0
    next_tick_t = 0.0
    hydraulic_since: float | None = None
    last_t_s = 0.0

    watchdog: ThermalWatchdog | None = None
    watchdog_task: asyncio.Task[TripRecord | None] | None = None
    watchdog_stop = asyncio.Event()
    emergency_actions: list[ActionModel] = []
    emergency_state: IncidentState | None = None
    emergency_reason: str | None = None
    last_evidence: Evidence | None = None
    last_sample: TelemetrySample | None = None
    last_candidate: Candidate | None = None
    last_index = 0

    if bool(get(cfg, "watchdog.enabled")):
        sensor = NvmlJunctionSensor(
            build_nvml(
                cfg["nvml"],
                tolerance_w=float(get(cfg, "orchestrator.clamp_tolerance_w")),
            )
        )

        async def actuate(reason: WatchdogTrip, detail: str) -> None:
            nonlocal emergency_actions, emergency_state, emergency_reason
            emergency_reason = reason.value
            emergency_actions, emergency_state = await _emergency_actions(
                last_t_s, cfg, detail, protocol
            )

        watchdog = ThermalWatchdog(
            sensor,
            limit_c=float(get(cfg, "thermal_model.watchdog_limit_c")),
            period_s=float(get(cfg, "watchdog.period_ms")) / 1000.0,
            actuate=actuate,
        )
        watchdog_task = asyncio.create_task(watchdog.run(watchdog_stop))

    with VideoSource(clip) as source:
        first = source.read(0)
        if first is None:
            raise PipelineError(f"clip has no frames: {clip}")
        rect = estimate(first.image, topology["chassis"]["aruco"])
        worker, _ = _build_sam_worker(
            mode, sam_cache_dir, cfg, topology, px_per_mm, canvas, source.fps
        )

        try:
            for frame in source.frames():
                last_frame = frame.index
                if max_seconds is not None and frame.t_s > max_seconds:
                    break
                if frame.t_s + 1e-9 < next_tick_t:
                    continue

                health = assess_health(frame.image, cfg["camera"])
                sampler.update(frame.image, frame.t_s, rect)

                if worker is not None:
                    for sam_sample in worker.poll():
                        sampler.reanchor(sam_sample.mask_full, sam_sample.t_s)
                    if sampler.last_warped is None:
                        raise PipelineError("tracker did not produce a rectified frame")
                    worker.submit(sampler.last_warped, frame.index, frame.t_s)

                candidates = sampler.current(frame.t_s)
                chosen = CandidateManager.select(candidates)
                sample = telemetry.sample(frame.t_s)

                if chosen is None:
                    phi_v = 0.0
                    wet = 0.0
                    zone = None
                    quality = 0.0
                    growth = 0.0
                else:
                    phi_v = visual_phi(
                        chosen.growth_mm2_per_s,
                        chosen.mean_intensity,
                        chosen.intensity_var,
                        cfg["fusion"]["visual"],
                    )
                    wet = chosen.wetness
                    zone = chosen.zone
                    quality = chosen.quality
                    growth = chosen.growth_mm2_per_s

                phi_h = hydraulic_phi(
                    max(0.0, sample.pressure_rate_psi_per_s),
                    max(0.0, sample.flow_imbalance_lpm),
                    cfg["fusion"]["hydraulic"],
                )
                phi_t = thermal_phi(sample.dew_margin_c, cfg["fusion"]["thermal"])

                if phi_h >= float(get(cfg, "decision.hydraulic_confirm_phi")):
                    hydraulic_since = frame.t_s if hydraulic_since is None else hydraulic_since
                    hydraulic_low_ms = int(round((frame.t_s - hydraulic_since) * 1000.0))
                else:
                    hydraulic_since = None
                    hydraulic_low_ms = 0

                evidence = Evidence(
                    zone=zone if zone is not None else -1,
                    phi_v=phi_v,
                    phi_h=phi_h,
                    phi_t=phi_t,
                    W=wet,
                    dA_dt=growth,
                    q=quality,
                    camera_healthy=health.ok,
                    temp_c=sample.surface_c,
                    ticks=1,
                    hydraulic_low_ms=hydraulic_low_ms,
                    maintenance=bool(get(cfg, "system.maintenance_mode")),
                )
                decision = engine.update(evidence, frame.t_s)
                last_decision = decision
                last_t_s = frame.t_s
                last_evidence = evidence
                last_sample = sample
                last_candidate = chosen
                last_index = frame.index
                ticks += 1

                hz = policy.update(decision.label, chosen is not None)
                next_tick_t = frame.t_s + 1.0 / hz

                # Hand the loop to the watchdog task. It is not fed by this loop:
                # it reads the GPU itself and can trip while perception is stalled.
                await asyncio.sleep(0)
                if watchdog is not None and watchdog.actuated:
                    assert watchdog.trip is not None
                    return await _finish(
                        name=name,
                        scenario=scenario,
                        cfg=cfg,
                        decision=decision,
                        label=SystemLabel.EMERGENCY_TRIP,
                        evidence=evidence,
                        sample=sample,
                        candidate=chosen,
                        actions=emergency_actions,
                        sequence_state=emergency_state,
                        trip_reason=emergency_reason,
                        junction_c=_junction_of(watchdog),
                        sam_mode=mode,
                        sam_stats=worker.stats if worker is not None else None,
                        ticks=ticks,
                        frame_range=(0, frame.index),
                        frames=frame.index + 1,
                        out_dir=out_dir,
                        tick_rate_hz=policy.hz,
                        rate_changes=policy.changes,
                        protocol_mode=protocol,
                    )

                if decision.label in INCIDENT_LABELS:
                    actions, sequence_state = await _actions_for(
                        decision, frame.t_s, cfg, checkpoint_dir, protocol
                    )
                    return await _finish(
                        name=name,
                        scenario=scenario,
                        cfg=cfg,
                        decision=decision,
                        label=decision.label,
                        evidence=evidence,
                        sample=sample,
                        candidate=chosen,
                        actions=actions,
                        sequence_state=sequence_state,
                        trip_reason=None,
                        junction_c=None,
                        sam_mode=mode,
                        sam_stats=worker.stats if worker is not None else None,
                        ticks=ticks,
                        frame_range=(0, frame.index),
                        frames=frame.index + 1,
                        out_dir=out_dir,
                        tick_rate_hz=policy.hz,
                        rate_changes=policy.changes,
                        protocol_mode=protocol,
                    )
        finally:
            watchdog_stop.set()
            if watchdog_task is not None:
                try:
                    await asyncio.wait_for(watchdog_task, timeout=5.0)
                except TimeoutError:
                    watchdog_task.cancel()
            if worker is not None:
                await worker.close()

    # The clip can end before a real-socket emergency sequence finishes. The
    # actions already ran; finish the filing so the trip is never lost.
    if (
        watchdog is not None
        and watchdog.actuated
        and last_evidence is not None
        and last_sample is not None
        and last_decision is not None
    ):
        assert watchdog.trip is not None
        return await _finish(
            name=name,
            scenario=scenario,
            cfg=cfg,
            decision=last_decision,
            label=SystemLabel.EMERGENCY_TRIP,
            evidence=last_evidence,
            sample=last_sample,
            candidate=last_candidate,
            actions=emergency_actions,
            sequence_state=emergency_state,
            trip_reason=emergency_reason,
            junction_c=_junction_of(watchdog),
            sam_mode=mode,
            sam_stats=worker.stats if worker is not None else None,
            ticks=ticks,
            frame_range=(0, last_index),
            frames=last_index + 1,
            out_dir=out_dir,
            tick_rate_hz=policy.hz,
            rate_changes=policy.changes,
            protocol_mode=protocol,
        )

    return RunResult(
        scenario=resolve_scenario(name),
        decision=last_decision,
        report=None,
        path=None,
        frames=last_frame + 1,
        ticks=ticks,
        tick_rate_hz=policy.hz,
        rate_changes=policy.changes,
        sam=(SamSummary.from_stats(mode, worker.stats) if worker is not None else None),
        protocol_mode=protocol,
        sequence_state=None,
    )


async def _explained_report(
    *,
    report: IncidentReport,
    cfg: dict[str, Any],
    evidence: EvidenceModel,
    actions: list[ActionModel],
    label: str,
    fixture_source: str,
    trip_reason: str | None,
) -> IncidentReport:
    """Ask Muse after the incident is sealed. Any failure keeps the fallback.

    NFR-SAF-01: the model only fills ``root_cause_explanation`` and
    ``llm_uncertainty``. It cannot change the actions, the evidence, or the
    recommended operator action, and it cannot raise into the caller.
    """

    section = cfg.get("report", {})
    if not bool(section.get("enabled", True)):
        return report

    client = MuseClient(section)
    if not client.available:
        return report

    try:
        result = await client.explain(
            incident_id=report.incident_id,
            label=label,
            evidence=evidence,
            actions=actions,
            fixture_source=fixture_source,
            trip_reason=trip_reason,
        )
    except MuseUnavailable:
        return report
    except Exception:
        # A report-service bug must not turn an incident into a crash.
        return report

    return report.model_copy(
        update={
            "report_source": "llm",
            "root_cause_explanation": result.body.root_cause,
            "llm_uncertainty": result.body.uncertainty,
            "llm_model": result.model,
        }
    )


async def _finish(
    *,
    name: str,
    scenario: dict[str, Any],
    cfg: dict[str, Any],
    decision: Decision,
    label: SystemLabel,
    evidence: Evidence,
    sample: TelemetrySample,
    candidate: Candidate | None,
    actions: list[ActionModel],
    sequence_state: IncidentState | None,
    trip_reason: str | None,
    junction_c: float | None,
    sam_mode: str,
    sam_stats: SamStats | None,
    ticks: int,
    frame_range: tuple[int, int],
    frames: int,
    out_dir: str | Path | None,
    tick_rate_hz: float,
    rate_changes: int,
    protocol_mode: str,
) -> RunResult:
    fixture_source = scenario.get("provenance", "synthetic")
    if fixture_source not in ("synthetic", "capture"):
        raise PipelineError(f"scenario provenance must be synthetic or capture: {fixture_source}")

    incident_time = _incident_time(cfg, sample.t_s)

    evidence_model = EvidenceModel(
        label=label.value,
        zone=candidate.zone if candidate is not None else None,
        qd_origin=str(scenario["expected"]["qd"]),
        p_leak=decision.p,
        phi_visual=evidence.phi_v,
        phi_hydraulic=evidence.phi_h,
        phi_thermal=evidence.phi_t,
        wetness=evidence.W,
        area_mm2=candidate.area_mm2 if candidate is not None else 0.0,
        growth_mm2_per_s=candidate.growth_mm2_per_s if candidate is not None else 0.0,
        mask_quality=evidence.q,
        pressure_psi=sample.pressure_psi,
        pressure_rate_psi_per_s=sample.pressure_rate_psi_per_s,
        flow_imbalance_lpm=sample.flow_imbalance_lpm,
        surface_c=sample.surface_c,
        dew_point_c=sample.dew_point_c,
        junction_c=junction_c,
    )

    report = build_fallback_report(
        evidence=evidence_model,
        actions=actions,
        incident_time=incident_time,
        rack_id=str(get(cfg, "system.rack_id")),
        run_index=_run_index(name, cfg),
        config_hash=config_hash(cfg),
        clip_id=resolve_scenario(name),
        frame_range=frame_range,
        fixture_source=fixture_source,
        trip_reason=trip_reason,
    )
    report = await _explained_report(
        report=report,
        cfg=cfg,
        evidence=evidence_model,
        actions=actions,
        label=label.value,
        fixture_source=fixture_source,
        trip_reason=trip_reason,
    )

    directory = Path(out_dir) if out_dir is not None else ROOT / str(get(cfg, "demo.out_dir"))
    path = write_incident(report, directory)
    return RunResult(
        scenario=resolve_scenario(name),
        decision=decision,
        report=report,
        path=path,
        frames=frames,
        ticks=ticks,
        tick_rate_hz=tick_rate_hz,
        rate_changes=rate_changes,
        sam=SamSummary.from_stats(sam_mode, sam_stats) if sam_stats is not None else None,
        protocol_mode=protocol_mode,
        sequence_state=sequence_state,
    )
