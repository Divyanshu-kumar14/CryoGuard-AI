# CryoGuard AI dashboard

Observer console for the deterministic safety pipeline. Plain HTML, CSS, and
one ES module: no npm, no bundler, no CDN, no external font. The browser can be
offline except for the local server.

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
| Chassis canvas | `topology.yaml` + `FRAME_TICK` | Zones 0/1/2, QD-04, ArUco fiducials, candidate contours, camera health |
| Probability strip | `DECISION_CHANGE` only | P(leak), label, rule, streak, threshold lines from config |
| Fusion meters | `EVIDENCE_UPDATE` | φv, φh, φt (signed), W, pressure rate, dew margin, flow imbalance |
| Action timeline | `ACTION_EXECUTED` | Append-only, execution order, readback status per step |
| Forensic report | `INCIDENT_CLOSED` | `[REPORT: MUSE SPARK OPEN]` or `[FALLBACK: DETERMINISTIC]` |

The decision strip has no probability literal in `app.js`. A test
(`tests/test_ui_contract.py`) fails if one appears; every number on the strip is
rendered from a WebSocket event. The only REST calls are `GET /api/state` for
geometry/configuration and `POST /api/run` to start a scenario.

## Endpoints

```text
GET  /api/state              topology, thresholds, manifest, run record, latest event fold
GET  /api/manifest           fixture manifest, source per clip
GET  /api/incidents/{id}     sealed incident JSON (read-only)
POST /api/run                {"scenario": "leak"}; 202 accepted, 409 while running
WS   /ws                     history replay, then live events; 20 s heartbeat
GET  /healthz                liveness and current run state
```

## Event stream

Every event is `{"type", "seq", "tS", "data"}` with a monotonically increasing
`seq` per run.

```text
RUN_STARTED       scenario, clip, provenance, simulated, configHash, samMode,
                  protocolMode, canvas, chassis
FRAME_TICK        frame, contours, candidate, camera
EVIDENCE_UPDATE   phiV, phiH, phiT, W, pressure, pressureRate, dew margin, ...
DECISION_CHANGE   ruleId, label, p, changed, previousLabel, streak, actuates
ACTION_EXECUTED   seq, actor, command, status, detail, simulated
INCIDENT_CLOSED   incidentId, path, reportSource, sequenceState, report
RUN_FINISHED      outcome, ticks, frames, durationS, incidentId, sequenceState
```

`ACTION_EXECUTED` is emitted from `ActionLog.append` as the step is recorded, so
the timeline reflects execution order even when the sequence runs on a worker
thread. `EventBus.publish` never raises and never blocks the pipeline; a dead
browser tab cannot stop a valve close (NFR-SAF-01).

## Honesty rules

- Every synthetic clip is badged `SIMULATED` at the camera, telemetry, PLC, and
  BMC surfaces. The banner says the frames are not captured hardware.
- A `capture` manifest row switches the camera badge to `CAPTURE`; the transport
  badges follow `protocolMode` (`external` is not called simulated).
- The report pane names its source. The fallback is not dressed up as a model
  answer, and the model id is shown when one was used.
- The canvas is drawn from the configured topology, not a screenshot. The scale
  label is the configured `px_per_mm`, not a measured calibration.

## Verification

```bash
make check                                   # lint, types, full test suite
PYTHONPATH=src .venv/bin/python -m pytest tests/test_ui_contract.py -q
PYTHONPATH=src .venv/bin/python -m pytest tests/test_server.py -q
```

`tests/test_pipeline_events.py` runs the same clip with and without a bus and
compares the incident files byte for byte: observing a run cannot change it.
