"""Vertical slice: clip -> tracker -> fusion -> decision -> safe sequence -> incident.

This is the M1 chain. It is deliberately plain: one loop, one decision engine, one
incident file. The dashboard, the LLM and the real PLC attach to this later.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from cryoguard.capture.rectify import estimate
from cryoguard.capture.source import VideoSource
from cryoguard.capture.zones import ZoneMap
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
from cryoguard.orchestrator.incident import write_incident
from cryoguard.orchestrator.incident_fsm import ConfirmedLeakOrchestrator
from cryoguard.perception.fast_tracker import FastTracker, Track
from cryoguard.perception.quality import mask_quality
from cryoguard.perception.wetness import WetnessTracker
from cryoguard.report.fallback import build_fallback_report, scenario_clock
from cryoguard.report.schema import ActionModel, EvidenceModel, IncidentReport
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
class RunResult:
    scenario: str
    decision: Decision | None
    report: IncidentReport | None
    path: Path | None
    frames: int
    ticks: int


def resolve_scenario(name: str) -> str:
    return SCENARIOS.get(name, name)


def load_scenario(name: str) -> dict[str, Any]:
    path = ROOT / "configs" / "scenarios" / f"{resolve_scenario(name)}.yaml"
    if not path.is_file():
        raise PipelineError(f"unknown scenario: {name} ({path})")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _select(tracks: list[Track]) -> Track | None:
    if not tracks:
        return None
    # Largest area wins; ties break on the lower id so the choice is stable.
    return max(tracks, key=lambda track: (track.area_mm2, -track.track_id))


def _incident_time(cfg: dict[str, Any], t_s: float) -> datetime:
    epoch = datetime.fromisoformat(get(cfg, "system.incident_epoch").replace("Z", "+00:00"))
    return scenario_clock(epoch, t_s)


def _run_index(name: str, cfg: dict[str, Any]) -> int:
    base = int(get(cfg, "demo.run_index"))
    return base + SCENARIO_INDEX.get(resolve_scenario(name), 1) - 1


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


def run_scenario(
    name: str,
    cfg: dict[str, Any],
    topology: dict[str, Any],
    out_dir: str | Path | None = None,
    max_seconds: float | None = None,
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
    zones = ZoneMap.build(topology, px_per_mm)
    tracker = FastTracker(cfg["perception"])
    wetness = WetnessTracker(topology, px_per_mm, cfg["perception"], zones)
    telemetry = TelemetryReplay(scenario)
    engine = HysteresisDecision(cfg)

    with VideoSource(clip) as source:
        first = source.read(0)
        if first is None:
            raise PipelineError(f"clip has no frames: {clip}")
        rect = estimate(first.image, topology["chassis"]["aruco"])

        frame_step = max(1, round(source.fps / float(get(cfg, "perception.track_hz"))))
        ticks = 0
        hydraulic_since: float | None = None
        last_frame = 0
        last_decision: Decision | None = None

        for frame in source.frames():
            last_frame = frame.index
            if max_seconds is not None and frame.t_s > max_seconds:
                break
            if frame.index % frame_step:
                continue

            tracks = tracker.update(frame.image, frame.t_s, rect, canvas, px_per_mm)
            chosen = _select(tracks)
            wetness.prune({track.track_id for track in tracks})
            sample = telemetry.sample(frame.t_s)

            if chosen is None:
                phi_v = 0.0
                wet = 0.0
                zone = None
                quality = 0.0
                area = 0.0
                growth = 0.0
            else:
                phi_v = visual_phi(
                    chosen.dA_dt_mm2_per_s,
                    chosen.mean_intensity,
                    chosen.intensity_var,
                    cfg["fusion"]["visual"],
                )
                wet = 1.0 if wetness.update(chosen, tracker.last_mask) else 0.0
                zone = zones.zone_for_cropped(chosen.mask, chosen.offset_px)
                quality = mask_quality(chosen, cfg["perception"])
                area = chosen.area_mm2
                growth = chosen.dA_dt_mm2_per_s

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
                camera_healthy=True,
                temp_c=sample.surface_c,
                ticks=1,
                hydraulic_low_ms=hydraulic_low_ms,
            )
            decision = engine.update(evidence, frame.t_s)
            last_decision = decision
            ticks += 1

            if decision.label in INCIDENT_LABELS:
                return _finish(
                    name=name,
                    scenario=scenario,
                    cfg=cfg,
                    decision=decision,
                    evidence=evidence,
                    sample=sample,
                    zone=zone,
                    area=area,
                    growth=growth,
                    quality=quality,
                    wet=wet,
                    t_s=frame.t_s,
                    frame_range=(0, frame.index),
                    frames=frame.index + 1,
                    ticks=ticks,
                    out_dir=out_dir,
                )

    return RunResult(
        scenario=resolve_scenario(name),
        decision=last_decision,
        report=None,
        path=None,
        frames=last_frame + 1,
        ticks=ticks,
    )


def _finish(
    *,
    name: str,
    scenario: dict[str, Any],
    cfg: dict[str, Any],
    decision: Decision,
    evidence: Evidence,
    sample: TelemetrySample,
    zone: int | None,
    area: float,
    growth: float,
    quality: float,
    wet: float,
    t_s: float,
    frame_range: tuple[int, int],
    frames: int,
    ticks: int,
    out_dir: str | Path | None,
) -> RunResult:
    fixture_source = scenario.get("provenance", "synthetic")
    if fixture_source not in ("synthetic", "capture"):
        raise PipelineError(f"scenario provenance must be synthetic or capture: {fixture_source}")

    actions = _actions_for(decision, t_s, cfg)
    incident_time = _incident_time(cfg, t_s)

    evidence_model = EvidenceModel(
        label=decision.label.value,
        zone=zone,
        qd_origin=str(scenario["expected"]["qd"]),
        p_leak=decision.p,
        phi_visual=evidence.phi_v,
        phi_hydraulic=evidence.phi_h,
        phi_thermal=evidence.phi_t,
        wetness=wet,
        area_mm2=area,
        growth_mm2_per_s=growth,
        mask_quality=quality,
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
    )
