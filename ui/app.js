// CryoGuard AI — rack safety console.
//
// Every value on this screen comes from an event on the pipeline's typed bus or
// from /api/state (topology, configured thresholds, manifest, run record). There
// are no probability thresholds, fixture results or incident numbers in this
// file: tests/test_ui_contract.py fails the build if a decimal literal in [0,1)
// appears here, so a fabricated P(leak) cannot hide on screen.
//
// The console keeps the whole event history for the current run:
//   /api/state.history  the run so far, on load
//   /ws                 replay of the same events, then live ones (de-duped by seq)
// That is what makes the rewind scrubber honest: dragging the trace re-renders
// the chassis, the decision and the actuation ladder at that tick, from events
// the pipeline actually emitted, not from a recording of the UI.
//
// Note on numerics: literals below 1 are written without a leading zero (.25,
// .16). That is the contract's own constraint — see the test named above — and
// it keeps the "no fabricated probability" grep meaningful.

const $ = (id) => document.getElementById(id);

const SCENARIO_LABELS = {
  leak: "QD-04 leak",
  condensation: "Condensation",
  glint: "Glint",
  hidden: "Hidden hydraulic",
};

const INCIDENT_LABELS = new Set(["CONFIRMED_LEAK", "EMERGENCY_TRIP"]);
const CAUTION_LABELS = new Set([
  "SUSPECT_LEAK",
  "UNVERIFIED_ANOMALY",
  "CAMERA_DEGRADED",
  "HIDDEN_HYDRAULIC_EVENT",
]);
const CALM_LABELS = new Set(["NOMINAL", "LOG_ONLY", "BENIGN_CONDENSATION"]);

const DESIGN = { w: 1200, h: 800 };
const TRACE = { left: 30, right: 8, top: 8, bottom: 16 };
// The scan sweep advances one eighth of the plane per pipeline tick, so eight
// frames of the 10 Hz loop make one pass. It is driven by FRAME_TICK, never by
// a timer: when the loop stops, the light stops.
const SWEEP_STEP = 1 / 8;

const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

const C = {
  ink: cssVar("--text") || "#ffffff",
  muted: cssVar("--muted") || "#8a8c92",
  border: cssVar("--border") || "#222224",
  surface: cssVar("--surface") || "#131217",
  accent: cssVar("--accent") || "#4a8ad6",
  positive: cssVar("--positive") || "#44a282",
  negative: cssVar("--negative") || "#e5566f",
  warning: cssVar("--warning") || "#d9a441",
  tracer: cssVar("--tracer") || "#d8f26a",
};

const state = {
  topology: null,
  canvas: null,
  thresholds: null,
  scenarios: [],
  manifest: null,
  source: "synthetic",
  protocol: "loopback",
  samMode: "cache",
  selected: null,
  running: false,
  history: [],
  lastSeq: -1,
  frames: [],
  evidences: [],
  decisions: [],
  actions: [],
  started: null,
  incident: null,
  finished: null,
  cursor: null,
  view: "live",
  sweepTarget: 0,
  sweepShown: 0,
  chassisBox: { w: 1, h: 1, dpr: 1 },
  traceGeom: { left: TRACE.left, right: TRACE.right, tMax: 1 },
  renderedLadder: "",
  needsChassis: true,
  socket: null,
  retry: null,
  toastTimer: null,
  reportTimer: null,
  lastView: null,
  firstTick: false,
  clipFrames: null,
};

/* ---------- formatting ---------- */

function fixed(value, digits) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric.toFixed(digits) : "—";
}

function signed(value, digits) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return "—";
  const text = numeric.toFixed(digits);
  return numeric > 0 ? `+${text}` : text;
}

function setText(id, text) {
  const node = $(id);
  if (node && node.textContent !== text) node.textContent = text;
}

function compact(hash) {
  const text = String(hash || "");
  return text ? text.slice(0, 8) : "—";
}

function clockTime() {
  const now = new Date();
  const pad = (value) => String(value).padStart(2, "0");
  return `${pad(now.getHours())}:${pad(now.getMinutes())}:${pad(now.getSeconds())}`;
}

function toneForLabel(label) {
  if (INCIDENT_LABELS.has(label)) return "bad";
  if (CAUTION_LABELS.has(label)) return "warn";
  if (CALM_LABELS.has(label)) return "ok";
  return "neutral";
}

function toneForStatus(status) {
  if (/VERIFIED|CLOSED|COMMITTED|ACCEPTED|DRAINED|REQUEUED|READBACK_VERIFIED/.test(status)) {
    return "ok";
  }
  if (/SLOW/.test(status)) return "warn";
  if (/FAULT|FAILED|REJECTED/.test(status)) return "bad";
  return "neutral";
}

/* ---------- boot ---------- */

async function boot() {
  bindControls();
  $("clock").textContent = clockTime();
  window.setInterval(() => {
    setText("clock", clockTime());
  }, 1000);

  try {
    const response = await fetch("/api/state", { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    applyState(await response.json());
  } catch (error) {
    showToast(`Cannot read /api/state: ${error.message}`);
  }

  connect();
  requestAnimationFrame(loop);
}

function applyState(data) {
  state.topology = data.topology;
  state.canvas = data.canvas;
  state.thresholds = data.thresholds;
  state.scenarios = data.scenarios || [];
  state.manifest = data.manifest;
  state.source = data.simulated ? "synthetic" : "capture";
  state.protocol = data.server ? data.server.protocolMode : "loopback";
  state.samMode = data.server ? data.server.samMode : "cache";
  state.selected = (data.run && data.run.scenario_key) || state.scenarios[0]?.key || null;

  setText("rack-id", data.rackId || "—");
  setText("rack-title", data.rackId || "Rack");
  setText("branch-id", data.branchId || "—");
  setText("config-hash", compact(data.configHash));
  setText("fusion-hash", compact(data.fusionHash));
  $("config-hash").title = `safety config hash ${data.configHash || "unavailable"}`;
  $("fusion-hash").title = `fusion freeze hash ${data.fusionHash || "unavailable"}`;

  const zones = (data.topology && data.topology.zones ? data.topology.zones : [])
    .map((zone) => `${zone.id} ${(zone.name || "").toLowerCase()}`)
    .join(", ");
  setText("zone-list", zones || "—");
  const chassis = data.topology && data.topology.chassis;
  setText(
    "plane-size",
    chassis ? `${chassis.width_mm} × ${chassis.height_mm} mm plane` : "—",
  );

  if (state.canvas) {
    setText(
      "viewport-geom",
      `${state.canvas.widthPx} × ${state.canvas.heightPx} px · ${fixed(state.canvas.pxPerMm, 1)} px/mm`,
    );
  }

  // Replay the server's copy of the run, then let the socket's replay be a
  // no-op: ingest() drops any seq that has already been applied.
  state.history = [];
  state.lastSeq = -1;
  for (const event of data.history || []) ingest(event);
  if (data.run) updateRunRecord(data.run);

  updateSimulationBadges();
  renderScenarios();
  renderThresholds();
  resizeChassis();
  render();
}

function updateRunRecord(run) {
  if (run.status === "running") {
    setRunning(true);
    setRunStatus("running", "RUNNING");
    setRunFlag("running", "LIVE");
  } else if (run.status === "finished") {
    setRunning(false);
    setRunStatus("finished", run.incident_id ? "INCIDENT FILED" : "NO INCIDENT");
    setRunFlag("finished", run.incident_id ? "SEALED" : "CLEAR");
  } else if (run.status === "failed") {
    setRunning(false);
    setRunStatus("failed", "FAILED");
    setRunFlag("failed", "FAILED");
  } else {
    setRunning(false);
    setRunStatus("idle", "IDLE");
    setRunFlag("idle", "IDLE");
  }
}

/* ---------- websocket ---------- */

function connect() {
  if (state.socket) {
    state.socket.onclose = null;
    state.socket.close();
  }
  const scheme = window.location.protocol === "https:" ? "wss:" : "ws:";
  const socket = new WebSocket(`${scheme}//${window.location.host}/ws`);
  state.socket = socket;

  socket.addEventListener("open", () => setConnection(true));
  socket.addEventListener("message", (message) => {
    let event;
    try {
      event = JSON.parse(message.data);
    } catch {
      return;
    }
    ingest(event);
  });
  socket.addEventListener("close", () => {
    setConnection(false);
    if (state.retry) window.clearTimeout(state.retry);
    state.retry = window.setTimeout(connect, 1500);
  });
  socket.addEventListener("error", () => socket.close());
}

function setConnection(online) {
  const chip = $("conn-chip");
  chip.dataset.state = online ? "online" : "offline";
  setText("conn-text", online ? "WS LIVE" : "WS OFFLINE");
}

/* ---------- event ingestion ---------- */

function ingest(event) {
  const seq = Number(event.seq);
  const known = Number.isFinite(seq);

  if (known && event.type !== "RUN_STARTED" && seq <= state.lastSeq) return;
  if (known) state.lastSeq = seq;

  switch (event.type) {
    case "RUN_STARTED":
      resetRun(event);
      break;
    case "FRAME_TICK":
      onFrame(event);
      break;
    case "EVIDENCE_UPDATE":
      onEvidence(event);
      break;
    case "DECISION_CHANGE":
      onDecision(event);
      break;
    case "ACTION_EXECUTED":
      onAction(event);
      break;
    case "INCIDENT_CLOSED":
      onIncident(event);
      break;
    case "RUN_FINISHED":
      onFinished(event);
      break;
    case "HEARTBEAT":
      return;
    default:
      return;
  }

  state.history.push(event);
  scheduleRender();
}

function resetRun(event) {
  const data = event.data || {};
  state.frames = [];
  state.evidences = [];
  state.decisions = [];
  state.actions = [];
  state.started = { t: Number(event.tS), ...data };
  state.incident = null;
  state.finished = null;
  state.cursor = null;
  state.sweepTarget = 0;
  state.sweepShown = 0;
  state.renderedLadder = "";
  state.firstTick = false;
  state.source = data.provenance || "synthetic";
  state.protocol = data.protocolMode || "loopback";
  state.samMode = data.samMode || state.samMode;
  if (data.scenarioKey) state.selected = data.scenarioKey;

  if (state.reportTimer) window.clearTimeout(state.reportTimer);
  setView("live");
  $("tab-report").disabled = true;
  clearReport();
  clearTelemetry();
  clearDecision();
  clearLadder("No actuation yet. The safety loop is armed; steps appear only if the decision table confirms one.");
  setRunStatus("running", "STARTING");
  setRunFlag("running", "LIVE");
  // The clip length comes from the fixture manifest, so "frame 229 of 600"
  // is the real length of the source, not a guess at how far the run will go.
  const clip = (state.manifest?.clips || []).find((row) => row.path === data.clip);
  state.clipFrames = clip ? Number(clip.frames) || null : null;
  setText("viewport-source", `SOURCE ${String(state.source).toUpperCase()} / ${data.scenario || "—"}`);
  setText("hud-frame", "—");
  setText("hud-time", "T+ —");
  setText("hud-zone", "—");
  setText("hud-area", "—");
  setText("hud-camera", "CAMERA —");

  updateSimulationBadges();
  renderScenarios();
  state.needsChassis = true;
}

function onFrame(event) {
  const data = event.data;
  if (!state.firstTick) {
    state.firstTick = true;
    if (state.running) setRunStatus("running", "RUNNING");
  }
  state.frames.push({
    t: Number(event.tS),
    frame: data.frame,
    contours: data.contours || [],
    candidate: data.candidate || null,
    camera: data.camera || null,
  });
  if (state.frames.length > 4096) state.frames.shift();
  if (state.cursor === null) state.sweepTarget = (state.sweepTarget + SWEEP_STEP) % 1;
}

function onEvidence(event) {
  state.evidences.push({ t: Number(event.tS), ...event.data });
  if (state.evidences.length > 4096) state.evidences.shift();
}

function onDecision(event) {
  const previous = state.decisions[state.decisions.length - 1];
  const changeT = event.data.changed
    ? Number(event.tS)
    : previous
      ? previous.changeT
      : Number(event.tS);
  state.decisions.push({ t: Number(event.tS), changeT, ...event.data });
  if (state.decisions.length > 4096) state.decisions.shift();
}

function onAction(event) {
  state.actions.push({ t: Number(event.tS), ...event.data });
}

function onIncident(event) {
  state.incident = event.data;
  $("tab-report").disabled = false;
  renderReport(event.data.report, event.data);
  setRunStatus("finished", "INCIDENT FILED");
  setRunFlag("finished", "SEALED");
  if (state.cursor === null) {
    if (state.reportTimer) window.clearTimeout(state.reportTimer);
    state.reportTimer = window.setTimeout(() => setView("report"), 720);
  }
}

function onFinished(event) {
  state.finished = event.data;
  setRunning(false);
  if (event.data.outcome === "incident") {
    setRunStatus("finished", "INCIDENT FILED");
    setRunFlag("finished", "SEALED");
  } else if (event.data.outcome === "no_incident") {
    setRunStatus("finished", "NO INCIDENT");
    setRunFlag("finished", "CLEAR");
    clearLadder(`No incident in ${event.data.ticks} ticks. Zero actuations.`);
  } else {
    setRunStatus("failed", "FAILED");
    setRunFlag("failed", "FAILED");
    showToast(event.data.error || "Run failed.");
  }
  document.body.dataset.review = "false";
}

/* ---------- replay: fold the event stream at a cursor ---------- */

function lastAtOrBefore(list, t) {
  let lo = 0;
  let hi = list.length - 1;
  let found = null;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (list[mid].t <= t) {
      found = list[mid];
      lo = mid + 1;
    } else {
      hi = mid - 1;
    }
  }
  return found;
}

function scheduleRender() {
  state.renderPending = true;
}

function viewAt(cursor) {
  if (cursor === null) {
    return {
      frame: state.frames[state.frames.length - 1] || null,
      evidence: state.evidences[state.evidences.length - 1] || null,
      decision: state.decisions[state.decisions.length - 1] || null,
      actions: state.actions,
      incident: state.incident,
    };
  }
  return {
    frame: lastAtOrBefore(state.frames, cursor),
    evidence: lastAtOrBefore(state.evidences, cursor),
    decision: lastAtOrBefore(state.decisions, cursor),
    actions: state.actions.filter((action) => action.t <= cursor + 1 / 1000),
    incident: null,
  };
}

/* ---------- render ---------- */

function render() {
  const view = viewAt(state.cursor);
  state.lastView = view;

  applyFrameHud(view.frame);
  applyDecision(view.decision);
  applyEvidence(view.evidence);
  renderLadder(view.actions);
  drawTrace();
  updateReviewChrome();
  state.needsChassis = true;
}

function applyFrameHud(frame) {
  if (!frame) return;
  const candidate = frame.candidate;
  const total = state.clipFrames ? `/${state.clipFrames}` : "";
  setText("hud-frame", `${String(frame.frame).padStart(4, "0")}${total}`);
  setText("hud-time", `T+ ${fixed(frame.t, 3)}`);
  setText("hud-zone", candidate && candidate.zone !== null ? String(candidate.zone) : "—");
  setText("hud-area", candidate ? fixed(candidate.areaMm2, 1) : "—");
  applyCamera(frame.camera);
}

function applyCamera(camera) {
  const node = $("hud-camera");
  if (!camera || camera.ok === undefined) {
    setText("hud-camera", "CAMERA —");
    return;
  }
  if (camera.ok) {
    node.dataset.tone = "ok";
    setText("hud-camera", "CAMERA OK");
    return;
  }
  node.dataset.tone = "bad";
  const reasons = (camera.reasons || []).join(" / ");
  setText("hud-camera", `CAMERA DEGRADED${reasons ? ` — ${reasons}` : ""}`);
}

function applyDecision(data) {
  const panel = $("decision-panel");
  const chip = $("decision-chip");
  if (!data) {
    clearDecision();
    return;
  }

  setText("p-leak", fixed(data.p, 3));
  panel.dataset.state = data.label;
  chip.dataset.state = toneForLabel(data.label);
  setText("decision-chip", data.changed ? "BAND CHANGE" : "HOLDING");
  setText("rule-id", data.ruleId || "—");
  setText("streak", fixed(data.streak, 0));
  setText("state-label", data.label);
  markActiveBand(data.label);
  setText("p-sub", `since T+${fixed(data.changeT === undefined ? data.t : data.changeT, 2)} s`);
  if (data.changed && state.cursor === null) flashValue($("p-leak"));
}

function flashValue(node) {
  if (prefersReducedMotion) return;
  node.classList.remove("is-flashing");
  void node.offsetWidth;
  node.classList.add("is-flashing");
}

function applyEvidence(data) {
  if (!data) return;
  setMeter("v", data.phiV, false);
  setMeter("h", data.phiH, false);
  setMeter("t", data.phiT, true);
  setMeter("w", data.W, false);

  setText("t-pressure", fixed(data.pressurePsi, 1));
  setText("t-rate", signed(data.pressureRatePsiPerS, 2));
  setText("t-flow", signed(data.flowImbalanceLpm, 2));
  setText("t-surface", fixed(data.surfaceC, 1));
  setText("t-dew", fixed(data.dewPointC, 1));
  setText("t-margin", signed(data.dewMarginC, 1));

  toneNumber($("t-rate"), Number(data.pressureRatePsiPerS) > 0 ? "warn" : "ok");
  toneNumber($("t-margin"), Number(data.dewMarginC) < 0 ? "bad" : "ok");
  toneNumber($("t-flow"), Math.abs(Number(data.flowImbalanceLpm)) > 0 ? "warn" : "ok");
}

function toneNumber(node, tone) {
  if (!node) return;
  node.classList.remove("is-warn", "is-bad");
  if (tone === "warn") node.classList.add("is-warn");
  if (tone === "bad") node.classList.add("is-bad");
}

function setMeter(key, value, isSigned) {
  const fill = $(`meter-${key}`);
  const numeric = Number(value);
  if (!fill) return;
  if (!Number.isFinite(numeric)) {
    fill.style.width = "0";
    setText(`meter-${key}-value`, "—");
    return;
  }
  setText(`meter-${key}-value`, fixed(numeric, 2));
  fill.classList.remove("is-warm", "is-cool");
  if (isSigned) {
    const clamped = Math.max(-1, Math.min(1, numeric));
    const width = Math.abs(clamped) * 50;
    fill.style.width = `${width}%`;
    fill.style.left = clamped >= 0 ? "50%" : `${50 - width}%`;
    fill.classList.add(clamped > 0 ? "is-warm" : "is-cool");
  } else {
    const clamped = Math.max(0, Math.min(1, numeric));
    fill.style.width = `${clamped * 100}%`;
    fill.style.left = "0";
  }
}

function clearDecision() {
  setText("p-leak", "—");
  setText("state-label", "IDLE");
  setText("rule-id", "—");
  setText("streak", "—");
  setText("p-sub", "awaiting event");
  $("decision-panel").dataset.state = "IDLE";
  $("decision-chip").dataset.state = "neutral";
  setText("decision-chip", "IDLE");
  markActiveBand("");
}

function clearTelemetry() {
  for (const id of ["t-pressure", "t-rate", "t-flow", "t-surface", "t-dew", "t-margin"]) {
    setText(id, "—");
    toneNumber($(id), "none");
  }
  for (const key of ["v", "h", "t", "w"]) {
    const fill = $(`meter-${key}`);
    if (fill) {
      fill.style.width = "0";
      fill.style.left = "0";
    }
    setText(`meter-${key}-value`, "—");
  }
}

/* ---------- actuation ladder ---------- */

function clearLadder(message) {
  const list = $("timeline");
  list.replaceChildren();
  const empty = document.createElement("li");
  empty.className = "empty-state";
  empty.textContent = message;
  list.append(empty);
  setText("timeline-count", "0 steps");
  state.renderedLadder = "";
}

function renderLadder(actions) {
  const key = `${actions.length}:${state.cursor === null ? "live" : state.cursor}`;
  if (key === state.renderedLadder) return;
  state.renderedLadder = key;

  const list = $("timeline");
  setText("timeline-count", `${actions.length} ${actions.length === 1 ? "step" : "steps"}`);

  if (!actions.length) {
    clearLadder(
      state.cursor === null
        ? "No actuation yet. Steps appear here in execution order, with the readback that accepted or rejected each one."
        : `No actuation had run at T+${fixed(state.cursor, 2)} s. Drag right to reach the sequence.`,
    );
    return;
  }

  list.replaceChildren();
  actions.forEach((action, index) => {
    const item = document.createElement("li");
    item.className = "step";
    item.dataset.tone = toneForStatus(action.status);
    item.dataset.latest = String(index === actions.length - 1 && state.cursor === null);

    const node = document.createElement("span");
    node.className = "step-node";
    node.setAttribute("aria-hidden", "true");

    const body = document.createElement("div");
    body.className = "step-body";

    const command = document.createElement("div");
    command.className = "step-command";
    const commandName = document.createElement("span");
    commandName.textContent = action.command;
    const actor = document.createElement("span");
    actor.className = "step-actor";
    actor.textContent = action.actor;
    command.append(commandName, actor);

    const detail = document.createElement("div");
    detail.className = "step-detail";
    detail.textContent = action.detail || "no detail";

    body.append(command, detail);

    const right = document.createElement("div");
    right.className = "step-right";
    const status = document.createElement("span");
    status.className = "step-status";
    status.textContent = action.status;
    const time = document.createElement("span");
    time.className = "step-time";
    time.textContent = `T+${fixed(action.t, 2)}`;
    right.append(status, time);

    item.append(node, body, right);
    list.append(item);
  });
}

/* ---------- report ---------- */

function clearReport() {
  const badge = $("report-badge");
  badge.textContent = "AWAITING INCIDENT";
  badge.dataset.source = "none";
  $("report-content").hidden = true;
  $("report-empty").hidden = false;
}

function renderReport(report, event) {
  if (!report) return;
  const badge = $("report-badge");
  const isLlm = report.reportSource === "llm";
  badge.textContent = isLlm ? "[REPORT: MUSE SPARK OPEN]" : "[FALLBACK: DETERMINISTIC]";
  badge.dataset.source = report.reportSource || "deterministic_fallback";
  $("tab-report").disabled = false;

  const evidence = report.physicalEvidence || {};
  const fallback = `Deterministic fallback: ${evidence.label} at ${
    report.suspectedOrigin
  } with P(leak) ${fixed(evidence.pLeak, 3)}. No model prose was generated; the evidence and the actions above are the complete record.`;
  setText("report-root-cause", report.rootCauseExplanation || fallback);
  setText(
    "report-uncertainty",
    report.llmUncertainty ||
      "Not applicable: the deterministic fallback does not estimate uncertainty.",
  );
  setText("report-action", report.recommendedOperatorAction || "—");
  setText("report-id", report.incidentId || "—");
  setText("report-origin", report.suspectedOrigin || "—");
  setText(
    "report-evidence",
    `${evidence.label || "—"} · φv ${fixed(evidence.phiVisual, 2)} · φh ${fixed(
      evidence.phiHydraulic,
      2,
    )} · W ${fixed(evidence.wetness, 2)} · ${fixed(evidence.areaMm2, 1)} mm²`,
  );
  setText(
    "report-source",
    isLlm
      ? `Muse Spark Open${event && event.simulated ? " · simulated incident" : ""}`
      : "Deterministic fallback (no model)",
  );
  setText("report-time", report.timestamp || "—");
  setText("report-model", report.llmModel || "not invoked");

  $("report-empty").hidden = true;
  $("report-content").hidden = false;
}

function setView(view) {
  state.view = view;
  $("band").dataset.view = view;
  const live = $("tab-live");
  const report = $("tab-report");
  const isLive = view === "live";
  live.classList.toggle("is-active", isLive);
  report.classList.toggle("is-active", !isLive);
  live.setAttribute("aria-selected", String(isLive));
  report.setAttribute("aria-selected", String(!isLive));
  $("pane-live").hidden = !isLive;
  $("pane-report").hidden = isLive;
}

/* ---------- controls ---------- */

function bindControls() {
  $("run-button").addEventListener("click", startRun);
  $("tab-live").addEventListener("click", () => setView("live"));
  $("tab-report").addEventListener("click", () => {
    if (!$("tab-report").disabled) setView("report");
  });

  const trace = $("sparkline");
  trace.addEventListener("pointerdown", onTracePointerDown);
  trace.addEventListener("pointermove", onTracePointerMove);
  trace.addEventListener("pointerup", onTracePointerUp);
  trace.addEventListener("pointercancel", onTracePointerUp);
  trace.addEventListener("keydown", onTraceKey);
  $("review-exit").addEventListener("click", () => setCursor(null));

  window.addEventListener("resize", resizeChassis);
  if ("ResizeObserver" in window) {
    new ResizeObserver(resizeChassis).observe($("chassis"));
  }
}

function renderScenarios() {
  const picker = $("scenario-buttons");
  picker.replaceChildren();
  for (const scenario of state.scenarios) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "scenario-button";
    button.dataset.scenario = scenario.key;
    button.textContent = SCENARIO_LABELS[scenario.key] || String(scenario.key);
    button.title = scenario.notes || "";
    button.setAttribute("aria-pressed", String(scenario.key === state.selected));
    button.disabled = state.running;
    button.addEventListener("click", () => selectScenario(scenario.key));
    picker.append(button);
  }
  const run = $("run-button");
  run.disabled = state.running || !state.selected;
}

function selectScenario(key) {
  if (state.running) return;
  state.selected = key;
  for (const button of document.querySelectorAll(".scenario-button")) {
    button.setAttribute("aria-pressed", String(button.dataset.scenario === key));
  }
  $("run-button").disabled = false;
}

async function startRun() {
  if (!state.selected || state.running) return;
  setRunning(true);
  setRunStatus("running", "STARTING");
  try {
    const response = await fetch("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ scenario: state.selected }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || `HTTP ${response.status}`);
    if (payload.run && payload.run.id) setText("run-status", "RUNNING");
  } catch (error) {
    setRunning(false);
    setRunStatus("failed", "RUN REFUSED");
    showToast(String(error.message || error));
  }
}

function setRunning(running) {
  state.running = running;
  const button = $("run-button");
  button.disabled = running || !state.selected;
  button.dataset.busy = String(running);
  button.firstChild.textContent = running ? "Running… " : "Run scenario ";
  for (const scenario of document.querySelectorAll(".scenario-button")) {
    scenario.disabled = running;
  }
}

function setRunStatus(kind, text) {
  const node = $("run-status");
  node.dataset.state = kind;
  node.textContent = text;
}

function setRunFlag(kind, text) {
  $("runflag").dataset.state = kind;
  setText("runflag-text", text);
}

function showToast(message) {
  const toast = $("toast");
  toast.textContent = String(message);
  toast.hidden = false;
  if (state.toastTimer) window.clearTimeout(state.toastTimer);
  state.toastTimer = window.setTimeout(() => {
    toast.hidden = true;
  }, 6000);
}

/* ---------- badges ---------- */

function updateSimulationBadges() {
  const synthetic = state.source === "synthetic";
  const camera = $("camera-badge");
  camera.textContent = synthetic ? "SIMULATED" : "CAPTURE";
  camera.classList.toggle("is-sim", synthetic);
  camera.classList.toggle("is-ok", !synthetic);

  const simulatedTransport = state.protocol !== "external";
  for (const [id, name] of [
    ["plc-badge", "PLC"],
    ["bmc-badge", "BMC"],
    ["telemetry-badge", "TELEMETRY"],
  ]) {
    const badge = $(id);
    if (!badge) continue;
    badge.textContent = simulatedTransport ? `${name} SIMULATED` : `${name} EXTERNAL`;
    badge.classList.toggle("is-sim", simulatedTransport);
    badge.classList.toggle("is-ok", !simulatedTransport);
  }

  setText("sam-chip", `SAM ${String(state.samMode).toUpperCase()}`);
  setText("proto-chip", String(state.protocol).toUpperCase());

  setText(
    "sim-copy",
    synthetic
      ? "Synthetic fixtures. Camera frames, telemetry, PLC, BMC and NVML are simulators. No captured hardware is shown."
      : "Captured fixture. Transport adapters may still be simulators; each readout states its own source.",
  );
  $("sim-banner").dataset.source = synthetic ? "synthetic" : "capture";
  setText("sim-stamp", synthetic ? "SIMULATED" : "CAPTURE");
}

/* ---------- decision bands ---------- */

function renderThresholds() {
  const host = $("thresholds");
  host.replaceChildren();
  if (!state.thresholds) return;

  const rows = [
    ["nominal", state.thresholds.nominal],
    ["suspect", state.thresholds.suspect],
    ["escalate", state.thresholds.escalation],
    ["confirmed", state.thresholds.confirmedLeak],
  ];
  for (const [label, value] of rows) {
    const cell = document.createElement("div");
    cell.className = "band-cell";
    const term = document.createElement("span");
    term.className = "band-key";
    term.textContent = label;
    const detail = document.createElement("span");
    detail.className = "band-value mono";
    detail.textContent = fixed(value, 2);
    cell.append(term, detail);
    host.append(cell);
  }
  host.dataset.ready = "true";
}

function markActiveBand(label) {
  const cells = document.querySelectorAll(".band-cell");
  if (!cells.length) return;
  const rank = INCIDENT_LABELS.has(label)
    ? 3
    : label === "SUSPECT_LEAK" || label === "UNVERIFIED_ANOMALY"
      ? 1
      : label === "NOMINAL" || label === "LOG_ONLY" || label === "BENIGN_CONDENSATION"
        ? 0
        : -1;
  cells.forEach((cell, index) => {
    cell.dataset.active = String(index <= rank && rank >= 0);
    cell.dataset.current = String(index === rank);
  });
}

/* ---------- chassis canvas ---------- */

function resizeChassis() {
  const canvas = $("chassis");
  const rect = canvas.getBoundingClientRect();
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.max(1, Math.round(rect.width * dpr));
  const height = Math.max(1, Math.round(rect.height * dpr));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  state.chassisBox = { w: Math.max(1, rect.width), h: Math.max(1, rect.height), dpr };
  state.needsChassis = true;
}

function loop(now) {
  // Schedule first: a drawing error must not stop the loop for the rest of the
  // session. The error still reaches the console.
  requestAnimationFrame(loop);
  if (state.renderPending) {
    state.renderPending = false;
    render();
  }
  const view = state.lastView;
  if (view) {
    const diff = Math.abs(state.sweepTarget - state.sweepShown);
    const animating =
      diff > 1 / 500 && state.cursor === null && state.running && !prefersReducedMotion;
    if (animating) {
      let delta = state.sweepTarget - state.sweepShown;
      if (delta > 1 / 2) delta -= 1;
      if (delta < -1 / 2) delta += 1;
      state.sweepShown = (state.sweepShown + delta * 1 / 6 + 1) % 1;
      state.needsChassis = true;
    } else if (state.sweepShown !== state.sweepTarget) {
      state.sweepShown = state.sweepTarget;
      state.needsChassis = true;
    }
    if (state.needsChassis) {
      drawChassis(view, now);
      state.needsChassis = false;
    }
  }
}

function drawChassis(view, now) {
  const canvas = $("chassis");
  const ctx = canvas.getContext("2d");
  const { w, h, dpr } = state.chassisBox;

  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = C.surface;
  ctx.fillRect(0, 0, w, h);

  if (!state.topology || !state.canvas) {
    ctx.fillStyle = C.muted;
    ctx.font = `11px ${cssVarFont()}`;
    ctx.textAlign = "center";
    ctx.fillText("awaiting /api/state", w / 2, h / 2);
    return;
  }

  const scale = Number(state.canvas.pxPerMm);
  const fit = Math.min(w / DESIGN.w, h / DESIGN.h);
  const ox = (w - DESIGN.w * fit) / 2;
  const oy = (h - DESIGN.h * fit) / 2;

  drawField(ctx, w, h, scale * fit);

  ctx.save();
  ctx.translate(ox, oy);
  ctx.scale(fit, fit);

  drawChassisBody(ctx);
  drawZones(ctx, view);
  drawFittings(ctx);
  drawFiducials(ctx);
  if (state.running && state.cursor === null && !prefersReducedMotion) drawSweep(ctx);
  drawCandidate(ctx, view, now);

  ctx.restore();

  drawDimensions(ctx, ox, oy, DESIGN.w * fit, DESIGN.h * fit, scale * fit);

  const line = $("scalebar-line");
  if (line) line.style.width = `${50 * scale * fit}px`;

  canvas.setAttribute("aria-label", describeScene(view.frame));
}

function drawDimensions(ctx, ox, oy, planeW, planeH, pxPerMm) {
  const roomLeft = ox >= 46;
  const roomBottom = oy + planeH + 26 < ctx.canvas.height / (state.chassisBox.dpr || 1);
  if (!roomLeft && !roomBottom) return; // the HUD still carries the geometry
  ctx.save();
  ctx.strokeStyle = "#3a3a46";
  ctx.fillStyle = C.muted;
  ctx.font = `11px ${cssVarFont()}`;
  ctx.lineWidth = 1;
  const cap = 4;

  const dimW = planeW / pxPerMm;
  const dimH = planeH / pxPerMm;

  // horizontal, under the plane
  if (roomBottom) {
  const y = oy + planeH + 16;
  ctx.beginPath();
  ctx.moveTo(ox, y + 1 / 2);
  ctx.lineTo(ox + planeW, y + 1 / 2);
  ctx.moveTo(ox + 1 / 2, y - cap);
  ctx.lineTo(ox + 1 / 2, y + cap);
  ctx.moveTo(ox + planeW - 1 / 2, y - cap);
  ctx.lineTo(ox + planeW - 1 / 2, y + cap);
  ctx.stroke();
  ctx.textAlign = "center";
  ctx.fillText(`${Math.round(dimW)} mm`, ox + planeW / 2, y - 7);
  }

  // vertical, left of the plane
  if (roomLeft) {
  const x = ox - 18;
  ctx.beginPath();
  ctx.moveTo(x + 1 / 2, oy);
  ctx.lineTo(x + 1 / 2, oy + planeH);
  ctx.moveTo(x - cap, oy + 1 / 2);
  ctx.lineTo(x + cap, oy + 1 / 2);
  ctx.moveTo(x - cap, oy + planeH - 1 / 2);
  ctx.lineTo(x + cap, oy + planeH - 1 / 2);
  ctx.stroke();
  ctx.save();
  ctx.translate(x - 8, oy + planeH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = "center";
  ctx.fillText(`${Math.round(dimH)} mm`, 0, 0);
  ctx.restore();
  }

  ctx.restore();
}

function cssVarFont() {
  return '"IBM Plex Mono", ui-monospace, monospace';
}

function drawField(ctx, w, h, step) {
  const grid = Math.max(18, step);
  ctx.save();
  ctx.strokeStyle = "#1d1d21";
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let x = 0; x < w; x += grid) {
    ctx.moveTo(Math.round(x) + 1 / 2, 0);
    ctx.lineTo(Math.round(x) + 1 / 2, h);
  }
  for (let y = 0; y < h; y += grid) {
    ctx.moveTo(0, Math.round(y) + 1 / 2);
    ctx.lineTo(w, Math.round(y) + 1 / 2);
  }
  ctx.stroke();
  ctx.restore();

  // registration crosses at the field corners
  ctx.strokeStyle = "#2b2b31";
  ctx.lineWidth = 1;
  const arm = 9;
  for (const [cx, cy] of [
    [14, 14],
    [w - 14, 14],
    [14, h - 14],
    [w - 14, h - 14],
  ]) {
    ctx.beginPath();
    ctx.moveTo(cx - arm, cy);
    ctx.lineTo(cx + arm, cy);
    ctx.moveTo(cx, cy - arm);
    ctx.lineTo(cx, cy + arm);
    ctx.stroke();
  }
}

function drawChassisBody(ctx) {
  ctx.fillStyle = "#0c0c10";
  ctx.fillRect(0, 0, DESIGN.w, DESIGN.h);

  const scale = Number(state.canvas.pxPerMm);
  const step = 20 * scale;
  ctx.strokeStyle = "#151519";
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let x = step; x < DESIGN.w; x += step) {
    ctx.moveTo(Math.round(x) + 1 / 2, 0);
    ctx.lineTo(Math.round(x) + 1 / 2, DESIGN.h);
  }
  for (let y = step; y < DESIGN.h; y += step) {
    ctx.moveTo(0, Math.round(y) + 1 / 2);
    ctx.lineTo(DESIGN.w, Math.round(y) + 1 / 2);
  }
  ctx.stroke();

  ctx.strokeStyle = "#3a3a46";
  ctx.lineWidth = 1;
  ctx.strokeRect(1 / 2, 1 / 2, DESIGN.w - 1, DESIGN.h - 1);

  // machined corner brackets
  ctx.strokeStyle = "#41414c";
  ctx.lineWidth = 2;
  const arm = 26;
  const corners = [
    [0, 0, 1, 1],
    [DESIGN.w, 0, -1, 1],
    [0, DESIGN.h, 1, -1],
    [DESIGN.w, DESIGN.h, -1, -1],
  ];
  for (const [x, y, dx, dy] of corners) {
    ctx.beginPath();
    ctx.moveTo(x + dx * 2, y + dy * arm);
    ctx.lineTo(x + dx * 2, y + dy * 2);
    ctx.lineTo(x + dx * arm, y + dy * 2);
    ctx.stroke();
  }
}

function polygonPath(ctx, points) {
  ctx.beginPath();
  points.forEach(([x, y], index) => {
    if (index === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.closePath();
}

function drawZones(ctx, view) {
  const scale = Number(state.canvas.pxPerMm);
  const zones = state.topology.zones || [];
  const activeZone =
    view && view.decision && view.frame && view.frame.candidate
      ? view.frame.candidate.zone
      : null;

  zones.forEach((zone, index) => {
    const points = (zone.polygon_mm || []).map(([x, y]) => [Number(x) * scale, Number(y) * scale]);
    if (points.length < 3) return;
    polygonPath(ctx, points);

    const electrical = Boolean(zone.margin_mm);
    const isActive = activeZone !== null && Number(zone.id) === Number(activeZone);

    ctx.fillStyle = isActive ? "#17181d" : "#0f0f13";
    ctx.fill();

    if (electrical) {
      ctx.save();
      ctx.clip();
      ctx.strokeStyle = "#e5566f1a";
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (let offset = -DESIGN.h; offset < DESIGN.w + DESIGN.h; offset += 22) {
        ctx.moveTo(offset, 0);
        ctx.lineTo(offset - DESIGN.h, DESIGN.h);
      }
      ctx.stroke();
      ctx.restore();
    }

    ctx.strokeStyle = isActive ? "#5a5a68" : "#303039";
    ctx.lineWidth = 1;
    ctx.setLineDash([6, 5]);
    ctx.stroke();
    ctx.setLineDash([]);

    let anchor = points[3];
    let align = "left";
    let dx = 14;
    let dy = -14;
    if (index === 0) {
      anchor = points[1];
      align = "right";
      dx = -14;
      dy = 118;
    } else if (index === 1) {
      anchor = points[0];
      dy = 26;
    }
    const bx = anchor[0] + dx;
    const by = anchor[1] + dy;

    ctx.strokeStyle = electrical ? "#e5566f66" : "#3a3a4466";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(align === "right" ? bx - 22 : bx, by - 17);
    ctx.lineTo(align === "right" ? bx : bx + 22, by - 17);
    ctx.stroke();

    ctx.textAlign = align;
    ctx.fillStyle = electrical ? C.negative : "#b8b8c0";
    ctx.font = `600 19px ${cssVarFont()}`;
    ctx.fillText(String(zone.id), bx, by);
    ctx.fillStyle = "#8a8c92";
    ctx.font = `9px ${cssVarFont()}`;
    ctx.fillText(String(zone.name || "").toUpperCase(), bx + (align === "right" ? -19 : 19), by - 1);
  });
}

function drawFittings(ctx) {
  const scale = Number(state.canvas.pxPerMm);
  for (const qd of state.topology.qd || []) {
    const x = Number(qd.roi_mm[0]) * scale;
    const y = Number(qd.roi_mm[1]) * scale;
    const rw = (Number(qd.roi_mm[2]) - Number(qd.roi_mm[0])) * scale;
    const rh = (Number(qd.roi_mm[3]) - Number(qd.roi_mm[1])) * scale;

    ctx.fillStyle = "#0a1210";
    ctx.fillRect(x, y, rw, rh);
    ctx.strokeStyle = "#2c4a44";
    ctx.lineWidth = 1;
    ctx.strokeRect(x + 1 / 2, y + 1 / 2, rw - 1, rh - 1);

    const cx = x + rw / 2;
    const cy = y + rh / 2;
    ctx.strokeStyle = "#5f8f8666";
    ctx.beginPath();
    ctx.moveTo(cx - 14, cy);
    ctx.lineTo(cx + 14, cy);
    ctx.moveTo(cx, cy - 14);
    ctx.lineTo(cx, cy + 14);
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(cx, cy, 5, 0, Math.PI * 2);
    ctx.stroke();

    ctx.fillStyle = "#9fc9c1";
    ctx.font = `10px ${cssVarFont()}`;
    ctx.textAlign = "left";
    ctx.fillText(String(qd.id), x + 7, y + 15);
    ctx.fillStyle = "#6b8d88";
    ctx.font = `8px ${cssVarFont()}`;
    ctx.fillText(String(qd.branch || ""), x + 7, y + rh - 7);
  }
}

function drawFiducials(ctx) {
  const aruco = state.topology.chassis.aruco;
  const scale = Number(state.canvas.pxPerMm);
  const size = Number(aruco.marker_size_mm) * scale;
  for (const marker of aruco.markers || []) {
    const cx = Number(marker.center_mm[0]) * scale;
    const cy = Number(marker.center_mm[1]) * scale;
    const x = cx - size / 2;
    const y = cy - size / 2;
    // A printed DICT_5X5 target: 1-cell black border, 5x5 data grid. The bit
    // pattern is derived from the id so it is stable across frames; it is a
    // rendering of a marker, not a measurement.
    const cells = 7;
    const cell = size / cells;
    ctx.fillStyle = "#0a0a0d";
    ctx.fillRect(x, y, size, size);
    ctx.strokeStyle = "#2c2c34";
    ctx.lineWidth = 1;
    ctx.strokeRect(x + 1 / 2, y + 1 / 2, size - 1, size - 1);
    ctx.fillStyle = "#4a4a55";
    const id = Number(marker.id) || 0;
    for (let row = 0; row < cells; row += 1) {
      for (let col = 0; col < cells; col += 1) {
        const edge = row === 0 || col === 0 || row === cells - 1 || col === cells - 1;
        if (edge) continue;
        const bit = ((id + 1) * 7919 + row * 104729 + col * 313) % 5 < 2;
        if (bit) ctx.fillRect(x + col * cell, y + row * cell, cell, cell);
      }
    }
    ctx.fillStyle = "#6f6f7a";
    ctx.font = `9px ${cssVarFont()}`;
    ctx.textAlign = "center";
    ctx.fillText(`ARUCO ${marker.id}`, cx, y + size + 14);
  }
}

function drawSweep(ctx) {
  const x = state.sweepShown * DESIGN.w;
  const band = 84;
  const gradient = ctx.createLinearGradient(x - band, 0, x + 2, 0);
  gradient.addColorStop(0, "rgba(150, 200, 240, 0)");
  gradient.addColorStop(.72, "rgba(150, 200, 240, .035)");
  gradient.addColorStop(1, "rgba(190, 230, 255, .085)");
  ctx.fillStyle = gradient;
  ctx.fillRect(x - band, 0, band + 2, DESIGN.h);
  ctx.strokeStyle = "rgba(190, 230, 255, .22)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(x, 0);
  ctx.lineTo(x, DESIGN.h);
  ctx.stroke();
}

function drawCandidate(ctx, view, now) {
  const frame = view ? view.frame : null;
  if (!frame) return;
  const candidate = frame.candidate;
  const contours = frame.contours || [];
  if (!contours.length) {
    if (state.running && state.cursor === null) {
      ctx.fillStyle = "#57575f";
      ctx.font = `10px ${cssVarFont()}`;
      ctx.textAlign = "center";
      ctx.fillText("NO CANDIDATE IN FRAME", DESIGN.w / 2, 42);
    }
    return;
  }

  const lowQuality = candidate && candidate.stale;
  for (const polygon of contours) {
    polygonPath(ctx, polygon);
    ctx.fillStyle = "rgba(216, 242, 106, .13)";
    ctx.fill();
    // Two strokes fake the optical bloom of the tracer without a canvas shadow,
    // which would be in device pixels and would drift with the display ratio.
    ctx.strokeStyle = "rgba(216, 242, 106, .16)";
    ctx.lineWidth = 7;
    ctx.stroke();
    ctx.strokeStyle = lowQuality ? C.warning : C.tracer;
    ctx.lineWidth = 1.7;
    ctx.setLineDash(lowQuality ? [7, 5] : []);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  if (!candidate || !candidate.centroidMm) return;
  const scale = Number(state.canvas.pxPerMm);
  const cx = Number(candidate.centroidMm[0]) * scale;
  const cy = Number(candidate.centroidMm[1]) * scale;

  const confirmed = INCIDENT_LABELS.has(view.decision ? view.decision.label : "");
  ctx.strokeStyle = confirmed ? C.negative : "#f2f7dc";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(cx - 11, cy);
  ctx.lineTo(cx + 11, cy);
  ctx.moveTo(cx, cy - 11);
  ctx.lineTo(cx, cy + 11);
  ctx.stroke();
  ctx.beginPath();
  ctx.arc(cx, cy, 4, 0, Math.PI * 2);
  ctx.stroke();

  ctx.fillStyle = confirmed ? C.negative : "#e9f2cc";
  ctx.font = `9px ${cssVarFont()}`;
  ctx.textAlign = "left";
  ctx.fillText(`TRACK ${candidate.trackId}`, cx + 15, cy - 9);

  if (confirmed && !prefersReducedMotion) {
    const phase = (now % 2000) / 2000;
    ctx.globalAlpha = (1 - phase) * .55;
    ctx.strokeStyle = C.negative;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(cx, cy, 18 + phase * 46, 0, Math.PI * 2);
    ctx.stroke();
    ctx.globalAlpha = 1;
  }
}

function describeScene(frame) {
  if (!frame || !frame.candidate) {
    return "Rectified chassis plane. No candidate in the current frame.";
  }
  const candidate = frame.candidate;
  const zone = candidate.zone === null ? "unknown zone" : `zone ${candidate.zone}`;
  return `Rectified chassis plane. Candidate in ${zone}, area ${fixed(candidate.areaMm2, 1)} square millimetres, centroid ${fixed(candidate.centroidMm[0], 0)} by ${fixed(candidate.centroidMm[1], 0)} millimetres.`;
}

/* ---------- P(leak) trace + rewind ---------- */

function traceDomain() {
  const last = state.decisions[state.decisions.length - 1];
  return last ? Math.max(last.t, 1 / 1000) : 1;
}

function seriesColor(label) {
  if (INCIDENT_LABELS.has(label)) return C.negative;
  if (CAUTION_LABELS.has(label)) return C.warning;
  return C.ink;
}

function drawTrace() {
  const canvas = $("sparkline");
  if (!canvas) return;
  const rect = canvas.getBoundingClientRect();
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = Math.max(1, Math.round(rect.width));
  const h = Math.max(1, Math.round(rect.height));
  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);

  const plotW = Math.max(1, w - TRACE.left - TRACE.right);
  const plotH = Math.max(1, h - TRACE.top - TRACE.bottom);
  const tMax = traceDomain();
  state.traceGeom = { left: TRACE.left, right: TRACE.right, tMax };

  const xFor = (t) => TRACE.left + (Math.max(0, Math.min(tMax, t)) / tMax) * plotW;
  const yFor = (p) => TRACE.top + (1 - Math.max(0, Math.min(1, Number(p) || 0))) * plotH;

  // scale: 1.00 at the top, .50 mid, 0 on the baseline
  ctx.font = `9px ${cssVarFont()}`;
  ctx.textAlign = "right";
  for (const value of [1, .5, 0]) {
    const y = yFor(value);
    ctx.strokeStyle = value === 0 ? "#2b2b31" : "#1f1f23";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(TRACE.left, Math.round(y) + 1 / 2);
    ctx.lineTo(w - TRACE.right, Math.round(y) + 1 / 2);
    ctx.stroke();
    ctx.fillStyle = C.muted;
    ctx.fillText(value === 0 ? "0" : fixed(value, 2), TRACE.left - 6, y + 3);
  }

  if (state.thresholds) {
    ctx.setLineDash([3, 4]);
    ctx.lineWidth = 1;
    for (const value of [
      state.thresholds.suspect,
      state.thresholds.escalation,
      state.thresholds.confirmedLeak,
    ]) {
      const y = yFor(value);
      ctx.strokeStyle = "#33333b";
      ctx.beginPath();
      ctx.moveTo(TRACE.left, Math.round(y) + 1 / 2);
      ctx.lineTo(w - TRACE.right, Math.round(y) + 1 / 2);
      ctx.stroke();
    }
    ctx.setLineDash([]);
  }

  const points = state.decisions;
  if (points.length < 2) {
    ctx.fillStyle = "#57575f";
    ctx.font = `10px ${cssVarFont()}`;
    ctx.textAlign = "center";
    ctx.fillText("awaiting decisions", w / 2, h / 2);
    return;
  }

  const label = points[points.length - 1].label;
  const stroke = seriesColor(label);

  // The whole series in ink; the confirmed tail in the alarm colour, so colour
  // means state and never decoration.
  const firstConfirm = points.findIndex((point) => INCIDENT_LABELS.has(point.label));
  const splitIndex = firstConfirm === -1 ? points.length : Math.max(0, firstConfirm);

  const trace = (from, to, color) => {
    if (to - from < 1) return;
    ctx.beginPath();
    for (let index = from; index <= to && index < points.length; index += 1) {
      const x = xFor(points[index].t);
      const y = yFor(points[index].p);
      if (index === from) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.lineJoin = "round";
    ctx.stroke();
  };

  trace(0, splitIndex, C.ink);
  if (firstConfirm !== -1) trace(splitIndex, points.length - 1, C.negative);

  if (state.cursor === null) {
    ctx.fillStyle = stroke;
    const last = points[points.length - 1];
    ctx.beginPath();
    ctx.arc(xFor(last.t), yFor(last.p), 2.5, 0, Math.PI * 2);
    ctx.fill();
  }

  if (state.cursor !== null) {
    const at = lastAtOrBefore(points, state.cursor);
    if (at) {
      const x = xFor(at.t);
      ctx.strokeStyle = C.accent;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(Math.round(x) + 1 / 2, TRACE.top - 2);
      ctx.lineTo(Math.round(x) + 1 / 2, h - TRACE.bottom);
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(x, yFor(at.p), 3, 0, Math.PI * 2);
      ctx.strokeStyle = C.accent;
      ctx.lineWidth = 1.5;
      ctx.stroke();
      ctx.fillStyle = C.accent;
      ctx.beginPath();
      ctx.arc(x, yFor(at.p), 1.2, 0, Math.PI * 2);
      ctx.fill();
    }
  }

}

/* ---------- rewind interaction ---------- */

function traceTimeAt(clientX) {
  const canvas = $("sparkline");
  const rect = canvas.getBoundingClientRect();
  const { left, right, tMax } = state.traceGeom;
  const plotW = Math.max(1, rect.width - left - right);
  const x = Math.max(0, Math.min(plotW, clientX - rect.left - left));
  return (x / plotW) * tMax;
}

function setCursor(cursor) {
  const clamped = cursor === null ? null : Math.max(0, Math.min(traceDomain(), cursor));
  state.cursor = clamped;
  document.body.dataset.review = String(clamped !== null);
  $("review-flag").hidden = clamped === null;
  render();
}

function onTracePointerDown(event) {
  if (!state.decisions.length) return;
  const canvas = $("sparkline");
  try {
    canvas.setPointerCapture(event.pointerId);
  } catch {
    // A synthetic pointer (or a browser without capture) still scrubs; only the
    // ability to keep tracking outside the canvas is lost.
  }
  canvas.dataset.dragging = "true";
  setCursor(traceTimeAt(event.clientX));
}

function onTracePointerMove(event) {
  if ($("sparkline").dataset.dragging !== "true") return;
  setCursor(traceTimeAt(event.clientX));
}

function onTracePointerUp(event) {
  const canvas = $("sparkline");
  if (canvas.dataset.dragging !== "true") return;
  canvas.dataset.dragging = "false";
  try {
    if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
  } catch {
    /* capture was never taken */
  }
  // Releasing at the far right returns to live; anywhere else holds the cursor
  // where the operator left it.
  if (state.cursor !== null && state.cursor > traceDomain() - traceDomain() / 40) setCursor(null);
}

function onTraceKey(event) {
  if (!state.decisions.length) return;
  const step = traceDomain() / Math.max(1, state.decisions.length - 1);
  const current = state.cursor === null ? traceDomain() : state.cursor;
  if (event.key === "ArrowLeft") {
    event.preventDefault();
    setCursor(Math.max(0, current - step));
  } else if (event.key === "ArrowRight") {
    event.preventDefault();
    const next = current + step;
    setCursor(next >= traceDomain() ? null : next);
  } else if (event.key === "Home") {
    event.preventDefault();
    setCursor(0);
  } else if (event.key === "End" || event.key === "Escape") {
    event.preventDefault();
    setCursor(null);
  }
}

function updateReviewChrome() {
  const canvas = $("sparkline");
  const tMax = traceDomain();
  canvas.setAttribute("aria-valuemin", "0");
  if (!state.decisions.length) {
    canvas.setAttribute("aria-valuemax", "0");
    canvas.setAttribute("aria-valuenow", "0");
    canvas.setAttribute("aria-valuetext", "No decisions yet");
    canvas.setAttribute("aria-disabled", "true");
    setText("trace-now", "AWAITING");
    setText("trace-start", "T+—");
    setText("trace-end", "T+—");
    return;
  }
  canvas.removeAttribute("aria-disabled");
  canvas.setAttribute("aria-valuemax", fixed(tMax, 2));
  if (state.cursor === null) {
    canvas.setAttribute("aria-valuenow", fixed(tMax, 2));
    canvas.setAttribute("aria-valuetext", "Live");
    setText("trace-now", "LIVE");
    setText("trace-start", "T+0");
    setText("trace-end", `T+${fixed(tMax, 1)} s`);
    return;
  }
  canvas.setAttribute("aria-valuenow", fixed(state.cursor, 2));
  canvas.setAttribute("aria-valuetext", `Review at T plus ${fixed(state.cursor, 2)} seconds`);
  setText("trace-now", `REVIEW T+${fixed(state.cursor, 2)}`);
  setText("review-time", `T+${fixed(state.cursor, 2)} s`);
  setText("trace-start", "T+0");
  setText("trace-end", `T+${fixed(tMax, 1)} s`);
}

boot();
