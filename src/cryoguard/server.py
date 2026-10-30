"""FastAPI dashboard server (P9-T02).

The server is an observer, never a controller. It starts a scenario, forwards
the pipeline's typed events to browsers, and serves the sealed incident files.
Killing the server or every browser tab cannot change an actuation: the
pipeline does not await a subscriber and ``EventBus.publish`` never raises.

Endpoints
---------
``GET  /api/state``              topology, thresholds, manifest, run, fold and the
                                 current run's event history (for the rewind scrubber)
``GET  /api/manifest``           fixture manifest (every clip carries its source)
``GET  /api/incidents/{id}``     sealed incident JSON, read-only
``POST /api/run``                start one scenario; 409 while another is running
``WS   /ws``                     replay of the current run, then live events

The UI is plain HTML/CSS/JS under ``ui/``; there is no build step and no CDN.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import uvicorn
import yaml
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from cryoguard.config import config_hash, get, load
from cryoguard.env import load_env
from cryoguard.events import EventBus, PipelineEvent, Subscription
from cryoguard.pipeline import SCENARIOS, resolve_scenario, run_scenario

ROOT = Path(__file__).resolve().parents[2]
INCIDENT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
ALLOWED_SCENARIOS = set(SCENARIOS) | set(SCENARIOS.values())


class DashboardError(RuntimeError):
    """The dashboard cannot serve an honest response."""


@dataclass
class RunRecord:
    id: str
    status: Literal["idle", "running", "finished", "failed"]
    scenario: str | None = None
    scenario_key: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None
    incident_id: str | None = None
    incident_path: str | None = None
    report_source: str | None = None
    sequence_state: str | None = None
    ticks: int = 0
    frames: int = 0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class RunRequest(BaseModel):
    scenario: str = Field(min_length=1, max_length=64)
    sam: Literal["off", "cache", "live"] | None = None
    protocols: Literal["memory", "loopback", "external"] | None = None


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def load_settings(path: str | Path | None = None) -> dict[str, Any]:
    candidate = Path(path) if path is not None else ROOT / "configs" / "dashboard.yaml"
    if not candidate.is_file():
        raise DashboardError(f"dashboard config not found: {candidate}")
    data = yaml.safe_load(candidate.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("server"), dict):
        raise DashboardError(f"{candidate} must contain a server mapping")
    run = data.get("run")
    if not isinstance(run, dict):
        raise DashboardError(f"{candidate} must contain a run mapping")
    for key in ("host", "port", "ui_dir", "out_dir", "max_history", "event_queue"):
        if key not in data["server"]:
            raise DashboardError(f"{candidate} is missing server.{key}")
    for key in ("default_scenario", "sam_mode", "protocol_mode"):
        if key not in run:
            raise DashboardError(f"{candidate} is missing run.{key}")
    return data


def _load_fusion_hash() -> str | None:
    path = ROOT / "reports" / "fusion-freeze.json"
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8")).get("fusion_hash")
    except (OSError, ValueError):
        return None
    return str(value) if value else None


def _fold(events: tuple[PipelineEvent, ...]) -> dict[str, Any]:
    """Reduce the current run to the latest value of each event family."""

    frame: dict[str, Any] | None = None
    evidence: dict[str, Any] | None = None
    decision: dict[str, Any] | None = None
    incident: dict[str, Any] | None = None
    actions: list[dict[str, Any]] = []
    started: dict[str, Any] | None = None
    finished: dict[str, Any] | None = None

    for event in events:
        payload = {"t": round(float(event.t_s), 3), **event.data}
        if event.type == "RUN_STARTED":
            started = payload
        elif event.type == "FRAME_TICK":
            frame = payload
        elif event.type == "EVIDENCE_UPDATE":
            evidence = payload
        elif event.type == "DECISION_CHANGE":
            decision = payload
        elif event.type == "ACTION_EXECUTED":
            actions.append(payload)
        elif event.type == "INCIDENT_CLOSED":
            incident = payload
        elif event.type == "RUN_FINISHED":
            finished = payload

    return {
        "started": started,
        "frame": frame,
        "evidence": evidence,
        "decision": decision,
        "actions": actions,
        "incident": incident,
        "finished": finished,
    }


class Dashboard:
    def __init__(
        self,
        *,
        cfg: dict[str, Any],
        topology: dict[str, Any],
        settings: dict[str, Any],
        incident_root: str | Path | None = None,
        allow_llm: bool | None = None,
    ) -> None:
        self.cfg = cfg
        self.topology = topology
        self.settings = settings
        server = settings["server"]
        self.ui_dir = ROOT / str(server["ui_dir"])
        if not (self.ui_dir / "index.html").is_file():
            raise DashboardError(f"UI entry point is missing: {self.ui_dir / 'index.html'}")
        self.incident_root = (
            Path(incident_root) if incident_root is not None else ROOT / str(server["out_dir"])
        )
        self.incident_root.mkdir(parents=True, exist_ok=True)
        self.allow_llm = (
            bool(settings["run"].get("allow_llm", True)) if allow_llm is None else allow_llm
        )

        self.bus = EventBus(
            history=int(server["max_history"]), queue_size=int(server["event_queue"])
        )
        self.config_hash = config_hash(cfg)
        self.fusion_hash = _load_fusion_hash()
        self.manifest = self._load_manifest()
        self.run = RunRecord(id="", status="idle")
        self._task: asyncio.Task[None] | None = None
        self._lock = asyncio.Lock()
        self._heartbeat_s = float(server.get("heartbeat_s", 20))

    def _load_manifest(self) -> dict[str, Any]:
        path = ROOT / str(get(self.cfg, "bench.manifest"))
        if not path.is_file():
            raise DashboardError(f"fixture manifest is missing: {path}")
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not isinstance(data.get("clips"), list):
            raise DashboardError(f"{path} has no clips list")
        return data

    @property
    def simulated(self) -> bool:
        return all(clip.get("source") == "synthetic" for clip in self.manifest["clips"])

    def scenarios(self) -> list[dict[str, Any]]:
        by_clip = {str(clip.get("scenario")): clip for clip in self.manifest["clips"]}
        result = []
        for key, clip_id in SCENARIOS.items():
            clip = by_clip.get(clip_id, {})
            result.append(
                {
                    "key": key,
                    "clip": clip_id,
                    "label": clip.get("ground_truth_label"),
                    "source": clip.get("source", "synthetic"),
                    "expectedZone": clip.get("expected_zone"),
                    "notes": clip.get("notes", ""),
                }
            )
        return result

    def thresholds(self) -> dict[str, float]:
        section = self.cfg["fusion"]["thresholds"]
        return {
            "nominal": float(section["nominal_cutoff"]),
            "suspect": float(section["suspect_cutoff"]),
            "escalation": float(section["escalation_cutoff"]),
            "confirmedLeak": float(section["confirmed_leak"]),
            "zone2Trip": float(section["zone2_trip"]),
            "deescalationMargin": float(section["deescalation_margin"]),
        }

    def snapshot(self) -> dict[str, Any]:
        canvas = {
            "widthPx": int(
                round(
                    float(self.topology["chassis"]["width_mm"])
                    * float(get(self.cfg, "camera.px_per_mm"))
                )
            ),
            "heightPx": int(
                round(
                    float(self.topology["chassis"]["height_mm"])
                    * float(get(self.cfg, "camera.px_per_mm"))
                )
            ),
            "pxPerMm": float(get(self.cfg, "camera.px_per_mm")),
        }
        events = self.bus.snapshot()
        latest = _fold(events)
        run = self.run.as_dict()

        # The event stream is the source of truth for a terminal run. The
        # background task updates ``self.run`` only after run_scenario returns,
        # which is *after* the pipeline publishes RUN_FINISHED; without this fold
        # /api/state could report "running" one frame after the WS said closed.
        incident = latest["incident"]
        finished = latest["finished"]
        if finished is not None:
            run["status"] = "failed" if finished.get("outcome") == "error" else "finished"
            run["ticks"] = finished.get("ticks", run["ticks"])
            run["frames"] = finished.get("frames", run["frames"])
            run["error"] = finished.get("error") or run["error"]
            run["sequence_state"] = finished.get("sequenceState") or run["sequence_state"]
            run["incident_id"] = finished.get("incidentId") or run["incident_id"]
        if incident is not None:
            run["incident_id"] = incident.get("incidentId") or run["incident_id"]
            run["incident_path"] = incident.get("path") or run["incident_path"]
            run["report_source"] = incident.get("reportSource") or run["report_source"]
            run["sequence_state"] = incident.get("sequenceState") or run["sequence_state"]

        return {
            "schemaVersion": "1.0",
            "simulated": self.simulated,
            "rackId": str(get(self.cfg, "system.rack_id")),
            "branchId": str(get(self.cfg, "system.branch_id")),
            "configHash": self.config_hash,
            "fusionHash": self.fusion_hash,
            "server": {
                "protocolMode": str(self.settings["run"]["protocol_mode"]),
                "samMode": str(self.settings["run"]["sam_mode"]),
                "allowLlm": self.allow_llm,
            },
            "canvas": canvas,
            "thresholds": self.thresholds(),
            "topology": self.topology,
            "manifest": self.manifest,
            "scenarios": self.scenarios(),
            "run": run,
            "latest": latest,
            "history": [event.to_json() for event in events],
            "notes": [
                "All clips are synthetic unless a manifest row says source: capture.",
                "PLC, BMC, NVML, Slurm and telemetry are simulators in the default demo.",
                "The LLM explains a sealed incident; it never decides or actuates.",
                "The dashboard only observes the event stream; it has no command path.",
            ],
        }

    async def start(
        self,
        *,
        scenario: str,
        sam_mode: str | None = None,
        protocol_mode: str | None = None,
    ) -> RunRecord:
        if scenario not in ALLOWED_SCENARIOS:
            raise DashboardError(f"unknown scenario: {scenario!r}")
        async with self._lock:
            if self.run.status == "running":
                raise RuntimeError("a scenario is already running")
            self.bus.reset()
            run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid.uuid4().hex[:6]
            self.run = RunRecord(
                id=run_id,
                status="running",
                scenario=resolve_scenario(scenario),
                scenario_key=scenario,
                started_at=_now(),
            )
            self._task = asyncio.create_task(
                self._execute(
                    run_id=run_id,
                    scenario=scenario,
                    sam_mode=str(sam_mode or self.settings["run"]["sam_mode"]),
                    protocol_mode=str(protocol_mode or self.settings["run"]["protocol_mode"]),
                ),
                name=f"cryoguard-run-{run_id}",
            )
            return self.run

    async def _execute(
        self, *, run_id: str, scenario: str, sam_mode: str, protocol_mode: str
    ) -> None:
        try:
            result = await run_scenario(
                scenario,
                self.cfg,
                self.topology,
                out_dir=self.incident_root / run_id,
                sam_mode=sam_mode,
                protocol_mode=protocol_mode,
                allow_llm=self.allow_llm,
                bus=self.bus,
            )
        except asyncio.CancelledError:
            self.run.status = "failed"
            self.run.error = "cancelled"
            self.run.finished_at = _now()
            raise
        except Exception as exc:  # the dashboard reports the failure; it cannot hide it
            self.run.status = "failed"
            self.run.error = f"{type(exc).__name__}: {exc}"
            self.run.finished_at = _now()
            self.bus.publish(
                "RUN_FINISHED",
                0.0,
                outcome="error",
                ticks=self.run.ticks,
                frames=self.run.frames,
                durationS=0.0,
                incidentId=None,
                sequenceState=None,
                error=self.run.error,
            )
            return

        self.run.status = "finished"
        self.run.finished_at = _now()
        self.run.ticks = result.ticks
        self.run.frames = result.frames
        if result.report is not None and result.path is not None:
            self.run.incident_id = result.report.incident_id
            self.run.incident_path = str(result.path)
            self.run.report_source = result.report.report_source
        if result.sequence_state is not None:
            self.run.sequence_state = result.sequence_state.value

    async def close(self) -> None:
        if self._task is not None and not self._task.done():
            self._task.cancel()
            with suppress(asyncio.CancelledError):
                await self._task

    def incident_file(self, incident_id: str) -> Path | None:
        """Resolve an incident id without ever accepting a path separator."""

        if not INCIDENT_ID.fullmatch(incident_id):
            return None
        if self.run.incident_id == incident_id and self.run.incident_path:
            candidate = Path(self.run.incident_path)
            if candidate.is_file():
                return candidate

        roots = [self.incident_root]
        configured = str(self.settings["server"].get("incident_dir", "")).strip()
        if configured:
            roots.append(ROOT / configured)
        seen: set[Path] = set()
        matches: list[Path] = []
        for root in roots:
            if not root.is_dir():
                continue
            for candidate in root.glob(f"**/{incident_id}.json"):
                resolved = candidate.resolve()
                if resolved in seen or not resolved.is_relative_to(root.resolve()):
                    continue
                seen.add(resolved)
                matches.append(resolved)
        if not matches:
            return None
        return max(matches, key=lambda path: path.stat().st_mtime)


def create_app(
    *,
    cfg: dict[str, Any] | None = None,
    topology: dict[str, Any] | None = None,
    settings: dict[str, Any] | None = None,
    incident_root: str | Path | None = None,
    allow_llm: bool | None = None,
    startup_scenario: str | None = None,
) -> FastAPI:
    active_settings = settings or load_settings()
    active_cfg = cfg or load(ROOT / "configs" / "default.yaml")
    active_topology = topology or yaml.safe_load(
        (ROOT / "configs" / "topology.yaml").read_text(encoding="utf-8")
    )
    dashboard = Dashboard(
        cfg=active_cfg,
        topology=active_topology,
        settings=active_settings,
        incident_root=incident_root,
        allow_llm=allow_llm,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        startup_task: asyncio.Task[None] | None = None
        if startup_scenario:

            async def start_later() -> None:
                await asyncio.sleep(0.75)
                try:
                    await dashboard.start(scenario=str(startup_scenario))
                except Exception as exc:
                    print(f"startup run failed: {exc}", file=sys.stderr)

            startup_task = asyncio.create_task(start_later())
        yield
        if startup_task is not None and not startup_task.done():
            startup_task.cancel()
            with suppress(asyncio.CancelledError):
                await startup_task
        await dashboard.close()

    app = FastAPI(
        title="CryoGuard AI dashboard",
        version="1.0.0",
        description="Observer-only console for the CryoGuard deterministic safety pipeline.",
        lifespan=lifespan,
    )
    app.state.dashboard = dashboard

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        return {
            "ok": True,
            "simulated": dashboard.simulated,
            "run": dashboard.run.status,
            "configHash": dashboard.config_hash,
        }

    @app.get("/api/state")
    async def api_state() -> JSONResponse:
        return JSONResponse(dashboard.snapshot(), headers={"Cache-Control": "no-store"})

    @app.get("/api/manifest")
    async def api_manifest() -> JSONResponse:
        payload = {
            **dashboard.manifest,
            "simulated": dashboard.simulated,
            "dashboardNote": (
                "source is per clip; the dashboard badges synthetic rows as SIMULATED "
                "and never presents them as captured hardware."
            ),
        }
        return JSONResponse(payload, headers={"Cache-Control": "no-store"})

    @app.get("/api/incidents/{incident_id}")
    async def api_incident(incident_id: str) -> FileResponse:
        path = dashboard.incident_file(incident_id)
        if path is None:
            raise HTTPException(status_code=404, detail="incident not found")
        return FileResponse(
            path, media_type="application/json", headers={"Cache-Control": "no-store"}
        )

    @app.post("/api/run", status_code=202)
    async def api_run(request: RunRequest) -> JSONResponse:
        try:
            run = await dashboard.start(
                scenario=request.scenario,
                sam_mode=request.sam,
                protocol_mode=request.protocols,
            )
        except DashboardError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return JSONResponse({"run": run.as_dict()}, status_code=202)

    @app.websocket("/ws")
    async def websocket(socket: WebSocket) -> None:
        await socket.accept()
        subscription: Subscription = dashboard.bus.subscribe(replay=True)
        try:
            while True:
                try:
                    event = await asyncio.wait_for(
                        subscription.queue.get(), timeout=dashboard._heartbeat_s
                    )
                except TimeoutError:
                    await socket.send_json(
                        {"type": "HEARTBEAT", "seq": dashboard.bus.seq, "tS": 0.0, "data": {}}
                    )
                    continue
                await socket.send_json(event.to_json())
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            dashboard.bus.unsubscribe(subscription)

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(dashboard.ui_dir / "index.html", media_type="text/html")

    app.mount("/ui", StaticFiles(directory=dashboard.ui_dir), name="ui")
    return app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CryoGuard AI dashboard server")
    parser.add_argument("--host", default=None, help="bind address (default: dashboard.yaml)")
    parser.add_argument("--port", type=int, default=None, help="port (default: dashboard.yaml)")
    parser.add_argument("--scenario", default=None, help="optional scenario to run on startup")
    parser.add_argument(
        "--protocols",
        choices=("memory", "loopback", "external"),
        default=None,
        help="override run.protocol_mode",
    )
    parser.add_argument("--sam", choices=("off", "cache", "live"), default=None)
    args = parser.parse_args(argv)
    load_env()

    settings = load_settings()
    if args.protocols:
        settings["run"]["protocol_mode"] = args.protocols
    if args.sam:
        settings["run"]["sam_mode"] = args.sam
    host = args.host or str(settings["server"]["host"])
    port = args.port or int(settings["server"]["port"])
    if host not in ("127.0.0.1", "localhost", "::1"):
        print(
            "WARNING: this server has no authentication; binding a non-loopback "
            "address exposes /api/run to the network.",
            file=sys.stderr,
        )

    app = create_app(settings=settings, startup_scenario=args.scenario)
    uvicorn.run(app, host=host, port=port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
