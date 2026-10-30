# CryoGuard AI dashboard

Observer console for the deterministic safety pipeline. Plain HTML, CSS, and one
ES module: no npm, no bundler, no CDN, no runtime asset request. Both typefaces
are bundled under `ui/fonts/` (SIL OFL 1.1), so the browser can be offline except
for the local server.

```bash
make serve                     # http://127.0.0.1:8080, auto-runs the leak clip
make serve SCENARIO=glint      # auto-run a different scenario
make serve SCENARIO=           # start idle; choose a scenario in the UI
```

`make serve` starts `src/cryoguard/server.py` with the settings in
`configs/dashboard.yaml` (loopback bind, port 8080). It loads `.env` if present,
so a live SAM or Muse key can be supplied there without exporting it; `make
env-check` reports presence without printing values. The default run uses
`protocol_mode: loopback` (real Modbus-TCP and Redfish clients against the
SIMULATED servers) and SAM `cache` (the committed fixture cache, no network).

## What it shows

| Region | Source | Contract |
|---|---|---|
| Chassis stage | `topology.yaml` + `FRAME_TICK` | Zones 0/1/2, QD-04, printed ArUco targets, candidate contours, camera health, the trace of the leak |
| Decision plate | `DECISION_CHANGE` only | P(leak), label, rule, streak, time in band |
| Trace | `DECISION_CHANGE` only | P(leak) over the run, threshold lines, drag-to-rewind |
| Decision bands | `configs/default.yaml` | The four configured cutoffs, the reached band marked |
| Actuation sequence | `ACTION_EXECUTED` | Append-only, execution order, readback status per step |
| Telemetry / report band | `EVIDENCE_UPDATE`, `INCIDENT_CLOSED` | Fusion inputs and hydraulic readouts at the cursor; the sealed report in three columns |

Three things are worth knowing before reading the code.

**The console is a rewind.** `/api/state` carries the whole event history of the
current run, and `/ws` replays the same events. Dragging the trace (or arrow keys
on it) folds that history at a cursor: the chassis overlay, the decision, the
telemetry and the actuation ladder all re-render at that tick. Nothing is
recorded from the screen, so a rewind cannot show a frame the pipeline never
produced. `LIVE` in the trace footer means the cursor is at the newest event.

**There is no probability literal in `app.js`.** A test
(`tests/test_ui_contract.py`) fails if a decimal literal in `[0, 1)` appears in
the script; every number on screen comes from an event or from `/api/state`. The
only REST calls are `GET /api/state` for geometry, configuration and history, and
`POST /api/run` to start a scenario.

**The dashboard observes; it cannot actuate.** There is no control path to the
PLC, BMC or scheduler. The only mutating request is starting a scenario, and the
pipeline never awaits a subscriber: a dead browser tab cannot stop a valve close
(NFR-SAF-01).

## Design

The console follows the `data-terminal` direction dealt by the design library
(seed 1935754261): cards darker than the ground, figures in a monospace, signs
instead of colour, and every selection marked by the same 1px accent line. The
tokens live at the top of `styles.css`.

- **Ground and surfaces.** `#171719` page, `#131217` instrument wells, `#212023`
  raised chips. Elevation is a lightness step; no resting card carries a shadow.
- **Faces.** Public Sans for prose, IBM Plex Mono for every figure, ID, status
  code and path. Both are latin-subset woff2, 76 KB total, self-hosted.
- **Colour is state.** Steel blue `#4a8ad6` marks a selection or the rewind
  cursor, green `#44a282` is nominal, amber `#d9a441` is caution, red `#e5566f`
  is an incident. The fluorescent yellow-green `#d8f26a` appears in exactly one
  place: the tracer mask inside the chassis. Nothing decorative is coloured.
- **Motion is feedback, never an entrance.** The scan sweep advances one eighth
  of the plane per pipeline tick and stops when the loop stops; the trace is
  drawn as events arrive; a new actuation row slides in; the report band
  animates open when the incident is sealed. Under
  `prefers-reduced-motion: reduce` the sweep, the pulse and every transition are
  removed and the content is unchanged.
- **The chassis is letterboxed on purpose.** The plane is 3:2 and the stage is
  wider, so the margin around it is the inspection field: a 50 mm grid, the
  400 mm and 600 mm dimension lines, the scale bar and the corner HUD. Nothing is
  cropped and nothing is stretched.

## Endpoints

```text
GET  /api/state              topology, thresholds, manifest, run record, latest
                             fold, and the current run's event history
GET  /api/manifest           fixture manifest, source per clip
GET  /api/incidents/{id}     sealed incident JSON (read-only)
POST /api/run                {"scenario": "leak"}; 202 accepted, 409 while running
WS   /ws                     history replay, then live events; 20 s heartbeat
GET  /healthz                liveness and current run state
```

## Event stream

Every event is `{"type", "seq", "tS", "data"}` with a monotonically increasing
`seq` per run. `seq` restarts at 1 on a new run, and the UI drops any event at or
below the highest `seq` it has applied, so the `/api/state` snapshot and the
socket replay can both arrive without duplicating a tick.

```text
RUN_STARTED       scenario, clip, provenance, simulated, configHash, samMode,
                  protocolMode, tickRateHz, canvas
FRAME_TICK        frame, contours, candidate, camera
EVIDENCE_UPDATE   phiV, phiH, phiT, W, pressure, pressureRate, dew margin, ...
DECISION_CHANGE   ruleId, label, p, changed, previousLabel, streak, actuates
ACTION_EXECUTED   seq, actor, command, status, detail, simulated
INCIDENT_CLOSED   incidentId, path, reportSource, sequenceState, report
RUN_FINISHED      outcome, ticks, frames, durationS, incidentId, sequenceState
```

`ACTION_EXECUTED` is emitted from `ActionLog.append` as the step is recorded, so
the sequence reflects execution order even when the orchestration runs on a
worker thread. `EventBus.publish` never raises and never blocks the pipeline.

## Honesty rules

- Every synthetic clip is badged `SIMULATED` at the camera, telemetry, PLC, and
  BMC surfaces. The banner says the frames are not captured hardware.
- A `capture` manifest row switches the camera badge to `CAPTURE`; the transport
  badges follow `protocolMode` (`external` is not called simulated).
- The report names its source. The fallback is not dressed up as a model answer,
  and the model id is shown when one was used.
- The canvas is drawn from the configured topology, not a screenshot. The
  dimension lines and the scale bar are the configured millimetres and
  `px_per_mm`, not a measured calibration.
- The HUD's frame counter reads `0228/600` only when the fixture manifest gives
  the clip length; otherwise it prints the frame index alone.
- A readout with no tick yet prints an em dash, never a zero.

## Verification

```bash
make check                                   # lint, types, full test suite
PYTHONPATH=src .venv/bin/python -m pytest tests/test_ui_contract.py -q
PYTHONPATH=src .venv/bin/python -m pytest tests/test_server.py -q
```

`tests/test_pipeline_events.py` runs the same clip with and without a bus and
compares the incident files byte for byte: observing a run cannot change it.
`tests/test_server.py` asserts that `/api/state` history is the same sequence the
socket streamed, because the rewind depends on it.
