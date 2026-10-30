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

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from cryoguard.capture.rectify import estimate
from cryoguard.capture.source import VideoSource
from cryoguard.config import config_hash, get
from cryoguard.fusion.features import hydraulic_phi, thermal_phi, visual_phi
from cryoguard.fusion.hysteresis import HysteresisDecision
from cryoguard.models import Decision, Evidence, IncidentState, SystemLabel
from cryoguard.orchestrator.adapters import (
    ScenarioClock,
    SimulatedCheckpoint,
    SimulatedNvml,
    SimulatedPlc,
)
from cryoguard.orchestrator.incident_fsm import ConfirmedLeakOrchestrator
from cryoguard.perception.candidates import Candidate, CandidateManager
from cryoguard.perception.health import assess_health
from cryoguard.perception.sam_client import SamClient
from cryoguard.perception.sam_worker import SamStats, SamWorker
from cryoguard.perception.sampling import SamplingPolicy
from cryoguard.report.fallback import build_fallback_report, scenario_clock
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


def _confirmed_leak_actions(decision_t: float, cfg: dict[str, Any]) -> list[ActionModel]:
    clock = ScenarioClock(decision_t)
    nvml = SimulatedNvml(get(cfg, "nvml.fake_min_limit_w"))
    plc = SimulatedPlc(get(cfg, "plc.valve_travel_ms"))
    checkpoint = SimulatedCheckpoint(
        clock,
        settle_s=get(cfg, "orchestrator.checkpoint_budget_ms") / 1000.0,
        step=4200,
    )
    orchestrator = ConfirmedLeakOrchestrator(
        clock=clock,
        nvml=nvml,
        plc=plc,
        checkpoint=checkpoint,
        clamp_latency_ms=get(cfg, "orchestrator.clamp_latency_ms"),
    )
    fsm = orchestrator.run()

    if fsm.state is not IncidentState.RESOLVED_PENDING_REPAIR:
        raise PipelineError(f"safe sequence ended in {fsm.state.value}, not a resolved incident")

    return [
        _action(
            1,
            fsm.timestamp("clamp_verified"),
            "NVML",
            "CLAMP",
            "READBACK_VERIFIED",
            f"{nvml.clamp_w:.0f} W (simulated)",
        ),
        _action(
            2,
            fsm.timestamp("valve_closed"),
            "PLC",
            "VALVE_CLOSE",
            "LIMIT_SWITCH_CLOSED",
            f"travel {plc.travel_ms:.0f} ms (simulated)",
        ),
        _action(
            3, fsm.timestamp("checkpoint_settled"), "STORAGE", "CHECKPOINT_VERIFY", "COMMITTED"
        ),
        _action(4, fsm.timestamp("power_off"), "BMC", "FORCE_OFF", "ACCEPTED"),
        _action(5, fsm.timestamp("requeued"), "SLURM", "SCHEDULER_DRAIN", "DRAINED"),
        _action(6, fsm.timestamp("requeued"), "SLURM", "REQUEUE", "REQUEUED"),
    ]


def _actions_for(decision: Decision, t_s: float, cfg: dict[str, Any]) -> list[ActionModel]:
    if decision.label is SystemLabel.CONFIRMED_LEAK:
        return _confirmed_leak_actions(t_s, cfg)

    if decision.label is SystemLabel.EMERGENCY_TRIP:
        return [
            _action(1, t_s, "PLC", "VALVE_CLOSE", "COMMANDED", "spring-return close, no wait"),
            _action(2, t_s, "BMC", "FORCE_OFF", "COMMANDED", "checkpoint wait aborted"),
            _action(3, t_s, "SLURM", "SCHEDULER_DRAIN", "COMMANDED", ""),
        ]

    if decision.label is SystemLabel.HIDDEN_HYDRAULIC_EVENT:
        return [
            _action(1, t_s, "CONSOLE", "ALERT", "RAISED", "no visual evidence"),
            _action(2, t_s, "SLURM", "SCHEDULER_DRAIN", "DRAINED", "soft cordon; no valve"),
        ]

    return []


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


async def run_scenario(
    name: str,
    cfg: dict[str, Any],
    topology: dict[str, Any],
    out_dir: str | Path | None = None,
    max_seconds: float | None = None,
    sam_mode: str | None = None,
    sam_cache_dir: str | Path | None = None,
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
                ticks += 1

                hz = policy.update(decision.label, chosen is not None)
                next_tick_t = frame.t_s + 1.0 / hz

                if decision.label in INCIDENT_LABELS:
                    return _finish(
                        name=name,
                        scenario=scenario,
                        cfg=cfg,
                        decision=decision,
                        evidence=evidence,
                        sample=sample,
                        candidate=chosen,
                        sam_mode=mode,
                        sam_stats=worker.stats if worker is not None else None,
                        ticks=ticks,
                        frame_range=(0, frame.index),
                        frames=frame.index + 1,
                        out_dir=out_dir,
                        tick_rate_hz=policy.hz,
                        rate_changes=policy.changes,
                    )
        finally:
            if worker is not None:
                await worker.close()

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
    )


def _finish(
    *,
    name: str,
    scenario: dict[str, Any],
    cfg: dict[str, Any],
    decision: Decision,
    evidence: Evidence,
    sample: TelemetrySample,
    candidate: Candidate | None,
    sam_mode: str,
    sam_stats: SamStats | None,
    ticks: int,
    frame_range: tuple[int, int],
    frames: int,
    out_dir: str | Path | None,
    tick_rate_hz: float,
    rate_changes: int,
) -> RunResult:
    fixture_source = scenario.get("provenance", "synthetic")
    if fixture_source not in ("synthetic", "capture"):
        raise PipelineError(f"scenario provenance must be synthetic or capture: {fixture_source}")

    actions = _actions_for(decision, sample.t_s, cfg)
    incident_time = _incident_time(cfg, sample.t_s)

    evidence_model = EvidenceModel(
        label=decision.label.value,
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
    )
