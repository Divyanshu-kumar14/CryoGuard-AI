// CryoGuard AI — observer console.
//
// The decision strip is fed exclusively by the WebSocket event stream. There
// are no probability thresholds, fixture results, or incident numbers in this
// file: every value is rendered from a pipeline event or from /api/state
// (topology, configured thresholds, manifest). The grep gate in
// tests/test_ui_contract.py enforces the no-literal rule.

const $ = (id) => document.getElementById(id);

const SCENARIO_LABELS = {
  leak: "QD-04 LEAK",
  condensation: "CONDENSATION",
  glint: "GLINT",
  hidden: "HIDDEN HYDRAULIC",
};

const INCIDENT_LABELS = new Set(["CONFIRMED_LEAK", "EMERGENCY_TRIP"]);
const CAUTION_LABELS = new Set([
  "SUSPECT_LEAK",
  "UNVERIFIED_ANOMALY",
  "CAMERA_DEGRADED",
  "HIDDEN_HYDRAULIC_EVENT",
]);
const CALM_LABELS = new Set(["NOMINAL", "LOG_ONLY", "BENIGN_CONDENSATION"]);

const state = {
  topology: null,
  canvas: null,
  thresholds: null,
  scenarios: [],
  manifest: null,
  source: "synthetic",
  protocol: "loopback",
  selected: null,
  running: false,
  frame: null,
  camera: null,
  evidence: null,
  decision: null,
  actions: [],
  incident: null,
  started: null,
  finished: null,
  pHistory: [],
  needsDraw: true,
  socket: null,
  retry: null,
  toastTimer: null,
  reducedMotion: window.matchMedia("(prefers-reduced-motion: reduce)").matches,
};

const MONO = 'ui-monospace, "SFMono-Regular", "Cascadia Mono", Menlo, monospace';
const SANS = 'system-ui, -apple-system, "Segoe UI", Roboto, sans-serif';

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
  if (node) node.textContent = text;
}

/* ---------- boot ---------- */

async function boot() {
  bindControls();
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
  state.selected = (data.run && data.run.scenario_key) || state.scenarios[0]?.key || null;

  setText("rack-id", data.rackId || "—");
  setText("branch-id", data.branchId || "—");
  setText("config-hash", data.configHash || "—");
  setText("fusion-hash", data.fusionHash || "—");
  const zones = (data.topology && data.topology.zones ? data.topology.zones : [])
    .map((zone) => `${zone.id} ${zone.name}`)
    .join(" · ");
  setText("zone-list", zones || "—");

  updateSimulationBadges();
  renderScenarios();
  renderThresholds();
  resizeCanvas();
  if (data.run) updateRunRecord(data.run);
  state.needsDraw = true;
}

function updateRunRecord(run) {
  if (run.status === "running") {
    setRunStatus("running", "RUNNING");
    setRunning(true);
  } else if (run.status === "finished") {
    setRunStatus("finished", run.incident_id ? "INCIDENT FILED" : "NO INCIDENT");
    setRunning(false);
  } else if (run.status === "failed") {
    setRunStatus("failed", "FAILED");
    setRunning(false);
  } else {
    setRunStatus("idle", "IDLE");
    setRunning(false);
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
    handleEvent(event);
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

/* ---------- event routing ---------- */

function handleEvent(event) {
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
      break;
    default:
      break;
  }
}

function resetRun(event) {
  const data = event.data || {};
  state.frame = null;
  state.camera = null;
  state.evidence = null;
  state.decision = null;
  state.actions = [];
  state.incident = null;
  state.started = data;
  state.finished = null;
  state.pHistory = [];
  state.source = data.provenance || "synthetic";
  state.protocol = data.protocolMode || "loopback";
  if (data.scenarioKey) state.selected = data.scenarioKey;

  updateSimulationBadges();
  renderScenarios();
  clearDecision();
  clearTelemetry();
  clearTimeline("No actuation yet. The safety loop is armed; steps appear only if the decision table confirms one.");
  clearReport();
  setRunStatus("running", "RUNNING");
  setRunning(true);

  setText("viewport-source", `SOURCE ${state.source.toUpperCase()} / ${data.scenario || "—"}`);
  setText("hud-frame", "—");
  setText("hud-time", "—");
  setText("hud-zone", "—");
  setText("hud-area", "—");
  setText("camera-health", "CAMERA —");
  const health = $("camera-health");
  health.classList.remove("is-ok", "is-warn", "is-bad");
  state.needsDraw = true;
}

function onFrame(event) {
  state.frame = event.data;
  state.camera = event.data.camera;
  const candidate = event.data.candidate;
  setText("hud-frame", String(event.data.frame).padStart(4, "0"));
  setText("hud-time", fixed(event.tS, 2));
  setText("hud-zone", candidate && candidate.zone !== null ? String(candidate.zone) : "—");
  setText("hud-area", candidate ? fixed(candidate.areaMm2, 1) : "—");
  updateCameraBadge(event.data.camera);
  state.needsDraw = true;
}

function updateCameraBadge(camera) {
  const badge = $("camera-health");
  badge.classList.remove("is-ok", "is-warn", "is-bad");
  if (!camera || camera.ok === undefined) {
    badge.textContent = "CAMERA —";
    return;
  }
  if (camera.ok) {
    badge.textContent = "CAMERA OK";
    badge.classList.add("is-ok");
    return;
  }
  const reasons = (camera.reasons || []).join(" / ");
  badge.textContent = `CAMERA DEGRADED${reasons ? `: ${reasons}` : ""}`;
  badge.classList.add("is-bad");
}

function onEvidence(event) {
  const data = event.data;
  state.evidence = data;
  setMeter("v", data.phiV, false);
  setMeter("h", data.phiH, false);
  setMeter("t", data.phiT, true);
  setMeter("w", data.W, false);

  setText("t-rate", signed(data.pressureRatePsiPerS, 2));
  setText("t-margin", signed(data.dewMarginC, 1));
  setText("t-pressure", fixed(data.pressurePsi, 1));
  setText("t-flow", signed(data.flowImbalanceLpm, 2));
  setText("t-surface", fixed(data.surfaceC, 1));
  setText("t-dew", fixed(data.dewPointC, 1));

  colorTone($("t-rate"), Number(data.pressureRatePsiPerS) > 0 ? "warn" : "ok");
  colorTone($("t-margin"), Number(data.dewMarginC) < 0 ? "warn" : "ok");
}

function onDecision(event) {
  const data = event.data;
  state.decision = data;
  state.pHistory.push({ t: Number(event.tS), p: Number(data.p), label: data.label });
  if (state.pHistory.length > 400) state.pHistory.shift();

  setText("p-leak", fixed(data.p, 3));
  setText("rule-id", data.ruleId || "—");
  setText("streak", fixed(data.streak, 0));
  const panel = $("decision-panel");
  panel.dataset.state = data.label;
  const chip = $("decision-chip");
  chip.textContent = data.changed ? "BAND CHANGE" : "HOLDING";
  colorTone(chip, toneForLabel(data.label));

  if (data.changed) {
    setText("state-label", data.label);
    setText("p-sub", `since T+${fixed(event.tS, 2)} s`);
  }
  drawSparkline();
}

function onAction(event) {
  state.actions.push({ ...event.data, t: event.tS });
  renderTimeline();
}

function onIncident(event) {
  state.incident = event.data;
  renderReport(event.data.report, event.data);
  setRunStatus("finished", "INCIDENT FILED");
}

function onFinished(event) {
  state.finished = event.data;
  setRunning(false);
  if (event.data.outcome === "incident") {
    setRunStatus("finished", "INCIDENT FILED");
  } else if (event.data.outcome === "no_incident") {
    setRunStatus("finished", "NO INCIDENT");
    clearTimeline(
      `No incident in ${event.data.ticks} ticks. Zero actuations.`,
    );
  } else {
    setRunStatus("failed", "FAILED");
    showToast(event.data.error || "Run failed.");
  }
}

/* ---------- controls ---------- */

function bindControls() {
  $("run-button").addEventListener("click", startRun);
}

function renderScenarios() {
  const picker = $("scenario-buttons");
  picker.replaceChildren();
  for (const scenario of state.scenarios) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "scenario-button";
    button.dataset.scenario = scenario.key;
    button.textContent = SCENARIO_LABELS[scenario.key] || String(scenario.key).toUpperCase();
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
  button.textContent = running ? "RUNNING…" : "RUN SIMULATION";
  for (const scenario of document.querySelectorAll(".scenario-button")) {
    scenario.disabled = running;
  }
}

function setRunStatus(kind, text) {
  const node = $("run-status");
  node.dataset.state = kind;
  node.textContent = text;
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

/* ---------- badges and panes ---------- */

function updateSimulationBadges() {
  const synthetic = state.source === "synthetic";
  const camera = $("camera-badge");
  camera.textContent = synthetic ? "SIMULATED" : "CAPTURE";
  camera.classList.toggle("badge-sim", synthetic);
  camera.classList.toggle("badge-quiet", !synthetic);

  const simulatedTransport = state.protocol !== "external";
  for (const [id, name] of [
    ["plc-badge", "PLC"],
    ["bmc-badge", "BMC"],
    ["telemetry-badge", "TELEMETRY"],
  ]) {
    const badge = $(id);
    if (!badge) continue;
    badge.textContent = simulatedTransport ? `${name} SIMULATED` : `${name} EXTERNAL`;
    badge.classList.toggle("badge-sim", simulatedTransport);
    badge.classList.toggle("badge-quiet", !simulatedTransport);
  }

  setText(
    "sim-copy",
    synthetic
      ? "Synthetic fixtures. Camera frames, telemetry, PLC, BMC and NVML are simulators. No captured hardware is shown. The dashboard observes; it cannot actuate."
      : "Captured fixture. Transport adapters may still be simulators; each widget states its own source. The dashboard observes; it cannot actuate.",
  );
  const banner = $("sim-banner");
  banner.dataset.source = synthetic ? "synthetic" : "capture";
  setText("sim-stamp", synthetic ? "SIMULATED" : "CAPTURE");
}

function renderThresholds() {
  const host = $("thresholds");
  host.replaceChildren();
  if (!state.thresholds) return;
  const rows = [
    ["CONFIRMED", state.thresholds.confirmedLeak],
    ["ESCALATE", state.thresholds.escalation],
    ["SUSPECT", state.thresholds.suspect],
    ["NOMINAL", state.thresholds.nominal],
  ];
  for (const [label, value] of rows) {
    const cell = document.createElement("div");
    const term = document.createElement("dt");
    term.textContent = label;
    const detail = document.createElement("dd");
    detail.textContent = fixed(value, 2);
    cell.append(term, detail);
    host.append(cell);
  }
}

function clearDecision() {
  setText("p-leak", "—");
  setText("state-label", "IDLE");
  setText("rule-id", "—");
  setText("streak", "—");
  setText("p-sub", "awaiting event");
  setText("decision-chip", "IDLE");
  $("decision-panel").dataset.state = "IDLE";
  colorTone($("decision-chip"), "neutral");
  state.pHistory = [];
  drawSparkline();
}

function clearTelemetry() {
  for (const id of ["t-rate", "t-margin", "t-pressure", "t-flow", "t-surface", "t-dew"]) {
    setText(id, "—");
    colorTone($(id), "neutral");
  }
  for (const key of ["v", "h", "t", "w"]) {
    const fill = $(`meter-${key}`);
    fill.style.width = "0%";
    setText(`meter-${key}-value`, "—");
  }
}

function setMeter(key, value, isSigned) {
  const fill = $(`meter-${key}`);
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    fill.style.width = "0%";
    setText(`meter-${key}-value`, "—");
    return;
  }
  setText(`meter-${key}-value`, fixed(numeric, 2));
  if (isSigned) {
    const clamped = Math.max(-1, Math.min(1, numeric));
    const width = Math.abs(clamped) * 50;
    fill.style.width = `${width}%`;
    fill.style.left = clamped >= 0 ? "50%" : `${50 - width}%`;
    fill.style.background = clamped < 0 ? "var(--amber)" : "var(--blue)";
  } else {
    const clamped = Math.max(0, Math.min(1, numeric));
    fill.style.width = `${clamped * 100}%`;
    fill.style.left = "0";
    fill.style.background = "var(--aqua)";
  }
}

function colorTone(node, tone) {
  node.classList.remove("is-ok", "is-warn", "is-bad");
  if (tone === "ok") node.classList.add("is-ok");
  if (tone === "warn") node.classList.add("is-warn");
  if (tone === "bad") node.classList.add("is-bad");
}

function toneForLabel(label) {
  if (INCIDENT_LABELS.has(label)) return "bad";
  if (CAUTION_LABELS.has(label)) return "warn";
  if (CALM_LABELS.has(label)) return "ok";
  return "neutral";
}

function toneForStatus(status) {
  if (/VERIFIED|CLOSED|COMMITTED|ACCEPTED|DRAINED|REQUEUED/.test(status)) return "ok";
  if (/SLOW/.test(status)) return "warn";
  if (/FAULT|FAILED|REJECTED/.test(status)) return "bad";
  return "neutral";
}

/* ---------- timeline ---------- */

function clearTimeline(message) {
  const list = $("timeline");
  list.replaceChildren();
  const empty = document.createElement("li");
  empty.className = "empty-state";
  empty.textContent = message;
  list.append(empty);
  setText("timeline-count", "0 STEPS");
}

function renderTimeline() {
  const list = $("timeline");
  list.replaceChildren();
  setText(
    "timeline-count",
    `${state.actions.length} ${state.actions.length === 1 ? "STEP" : "STEPS"}`,
  );
  if (!state.actions.length) {
    clearTimeline("No actuation yet. The safety loop is armed.");
    return;
  }
  for (const action of state.actions) {
    const item = document.createElement("li");
    item.className = "step";

    const seq = document.createElement("span");
    seq.className = "step-seq";
    seq.textContent = String(action.seq).padStart(2, "0");

    const body = document.createElement("div");
    const command = document.createElement("div");
    command.className = "step-command";
    command.textContent = `${action.actor} / ${action.command}`;
    const meta = document.createElement("div");
    meta.className = "step-meta";
    meta.textContent = `T+${fixed(action.t, 3)} s · ${action.detail || "no detail"} · ${
      action.simulated ? "SIMULATED" : "EXTERNAL"
    }`;
    body.append(command, meta);

    const status = document.createElement("span");
    status.className = "step-status";
    status.dataset.tone = toneForStatus(action.status);
    status.textContent = action.status;

    item.append(seq, body, status);
    list.append(item);
  }
}

/* ---------- report ---------- */

function clearReport() {
  const badge = $("report-badge");
  badge.textContent = "AWAITING INCIDENT";
  badge.dataset.source = "none";
  badge.classList.remove("is-ok", "is-warn");
  $("report-content").hidden = true;
  $("report-empty").hidden = false;
}

function renderReport(report, event) {
  if (!report) return;
  const badge = $("report-badge");
  const isLlm = report.reportSource === "llm";
  badge.textContent = isLlm ? "[REPORT: MUSE SPARK OPEN]" : "[FALLBACK: DETERMINISTIC]";
  badge.dataset.source = report.reportSource || "deterministic_fallback";
  badge.classList.toggle("is-ok", isLlm);
  badge.classList.toggle("is-warn", !isLlm);

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
      ? `Muse Spark Open (${report.llmModel || "model id not recorded"})`
      : "Deterministic fallback (no model)",
  );
  setText("report-time", report.timestamp || "—");
  setText("report-config", report.configHash || "—");

  $("report-empty").hidden = true;
  $("report-content").hidden = false;
  if (event && event.simulated) {
    setText("report-config", `${report.configHash || "—"} · SIMULATED`);
  }
}

/* ---------- canvas ---------- */

function resizeCanvas() {
  const canvas = $("chassis");
  if (state.canvas && state.canvas.widthPx && state.canvas.heightPx) {
    canvas.width = state.canvas.widthPx;
    canvas.height = state.canvas.heightPx;
  }
}

function polygonFromMm(points) {
  const scale = state.canvas ? Number(state.canvas.pxPerMm) : 1;
  return points.map(([x, y]) => [Number(x) * scale, Number(y) * scale]);
}

function loop(now) {
  if (state.needsDraw || pulseActive()) {
    draw(now);
    state.needsDraw = false;
  }
  requestAnimationFrame(loop);
}

function pulseActive() {
  if (state.reducedMotion || !state.frame) return false;
  if (!state.decision) return false;
  return INCIDENT_LABELS.has(state.decision.label) && Boolean(state.frame.candidate);
}

function draw(now) {
  const canvas = $("chassis");
  const ctx = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;

  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#080d10";
  ctx.fillRect(0, 0, width, height);

  if (!state.topology || !state.canvas) {
    ctx.fillStyle = "#5f747d";
    ctx.font = `11px ${MONO}`;
    ctx.textAlign = "center";
    ctx.fillText("AWAITING /api/state", width / 2, height / 2);
    return;
  }

  drawGrid(ctx, width, height);
  drawZones(ctx);
  drawFittings(ctx);
  drawFiducials(ctx);
  drawContours(ctx, now);
  drawLabels(ctx, width, height);
  drawCameraState(ctx, width);
  canvas.setAttribute("aria-label", describeScene());
}

function drawGrid(ctx, width, height) {
  const scale = Number(state.canvas.pxPerMm);
  const step = scale * 50;
  ctx.strokeStyle = "#131f24";
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let x = step; x < width; x += step) {
    ctx.moveTo(x, 0);
    ctx.lineTo(x, height);
  }
  for (let y = step; y < height; y += step) {
    ctx.moveTo(0, y);
    ctx.lineTo(width, y);
  }
  ctx.stroke();

  ctx.strokeStyle = "#2b3d44";
  ctx.strokeRect(1, 1, width - 2, height - 2);

  const tick = 10;
  ctx.strokeStyle = "#3a5058";
  ctx.beginPath();
  ctx.moveTo(tick, 0);
  ctx.lineTo(0, 0);
  ctx.lineTo(0, tick);
  ctx.moveTo(width - tick, 0);
  ctx.lineTo(width, 0);
  ctx.lineTo(width, tick);
  ctx.moveTo(width - tick, height);
  ctx.lineTo(width, height);
  ctx.lineTo(width, height - tick);
  ctx.moveTo(tick, height);
  ctx.lineTo(0, height);
  ctx.lineTo(0, height - tick);
  ctx.stroke();
}

function drawZones(ctx) {
  const zones = state.topology.zones || [];
  zones.forEach((zone, index) => {
    const points = polygonFromMm(zone.polygon_mm);
    ctx.beginPath();
    points.forEach(([x, y], index) => {
      if (index === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.closePath();

    const electrical = Boolean(zone.margin_mm);
    ctx.fillStyle = electrical ? "#190f10" : "#0d1a1d";
    ctx.fill();
    ctx.setLineDash([5, 5]);
    ctx.strokeStyle = electrical ? "#5a2a26" : "#2b3d44";
    ctx.lineWidth = 1;
    ctx.stroke();
    ctx.setLineDash([]);

    if (electrical) {
      ctx.save();
      ctx.clip();
      ctx.strokeStyle = "#ff625914";
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (let offset = -800; offset < 1400; offset += 26) {
        ctx.moveTo(offset, 0);
        ctx.lineTo(offset - 800, 800);
      }
      ctx.stroke();
      ctx.restore();
    }

    // Anchor per zone so the label never lands under the QD ROI or a fiducial:
    // zone 0 top-right, zone 1 top-left, zone 2 bottom-left.
    let anchor = points[3];
    let align = "left";
    let dx = 14;
    let dy = -14;
    if (index === 0) {
      anchor = points[1];
      align = "right";
      dx = -14;
      dy = 120;
    } else if (index === 1) {
      anchor = points[0];
      dy = 24;
    }
    const bx = anchor[0] + dx;
    const by = anchor[1] + dy;
    ctx.strokeStyle = electrical ? "#ff625966" : "#3a505866";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(align === "right" ? bx - 24 : bx, by - 16);
    ctx.lineTo(align === "right" ? bx : bx + 24, by - 16);
    ctx.stroke();
    ctx.textAlign = align;
    ctx.fillStyle = electrical ? "#ff6259" : "#9db4bc";
    ctx.font = `600 18px ${MONO}`;
    ctx.fillText(String(zone.id), bx, by);
    ctx.fillStyle = "#7d939b";
    ctx.font = `9px ${MONO}`;
    ctx.fillText(String(zone.name || "").toUpperCase(), bx + (align === "right" ? -18 : 18), by - 1);
  });
}

function drawFittings(ctx) {
  for (const qd of state.topology.qd || []) {
    const [x0, y0, x1, y1] = polygonFromMm([
      [qd.roi_mm[0], qd.roi_mm[1]],
      [qd.roi_mm[2], qd.roi_mm[3]],
    ]).flat();
    const rect = { x: x0, y: y0, w: x1 - x0, h: y1 - y0 };
    ctx.fillStyle = "#0b1917";
    ctx.fillRect(rect.x, rect.y, rect.w, rect.h);
    ctx.strokeStyle = "#2d5a54";
    ctx.lineWidth = 1;
    ctx.strokeRect(rect.x, rect.y, rect.w, rect.h);

    const cx = rect.x + rect.w / 2;
    const cy = rect.y + rect.h / 2;
    ctx.strokeStyle = "#37e6c566";
    ctx.beginPath();
    ctx.moveTo(cx - 16, cy);
    ctx.lineTo(cx + 16, cy);
    ctx.moveTo(cx, cy - 16);
    ctx.lineTo(cx, cy + 16);
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(cx, cy, 5, 0, Math.PI * 2);
    ctx.stroke();

    ctx.fillStyle = "#9fc9c1";
    ctx.font = `10px ${MONO}`;
    ctx.textAlign = "left";
    ctx.fillText(String(qd.id), rect.x + 6, rect.y + 14);
    ctx.fillStyle = "#5f747d";
    ctx.font = `8px ${MONO}`;
    ctx.fillText(String(qd.branch || ""), rect.x + 6, rect.y + rect.h - 6);
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
    ctx.strokeStyle = "#6f858d";
    ctx.lineWidth = 2;
    ctx.strokeRect(x, y, size, size);
    ctx.strokeStyle = "#43565d";
    ctx.lineWidth = 1;
    ctx.strokeRect(x + 8, y + 8, size - 16, size - 16);
    ctx.fillStyle = "#43565d";
    const cell = (size - 16) / 4;
    for (const [ax, ay] of [
      [1, 1],
      [2, 1],
      [1, 2],
    ]) {
      ctx.fillRect(x + 8 + ax * cell, y + 8 + ay * cell, cell, cell);
    }
    ctx.fillStyle = "#7d939b";
    ctx.font = `9px ${MONO}`;
    ctx.textAlign = "center";
    ctx.fillText(`ARUCO ${marker.id}`, cx, y + size + 13);
  }
}

function drawContours(ctx, now) {
  const candidate = state.frame ? state.frame.candidate : null;
  const contours = state.frame ? state.frame.contours || [] : [];
  if (!contours.length) {
    if (state.running) {
      ctx.fillStyle = "#3d5058";
      ctx.font = `10px ${MONO}`;
      ctx.textAlign = "center";
      ctx.fillText("NO CANDIDATE IN FRAME", state.canvas.widthPx / 2, 52);
    }
    return;
  }

  const lowQuality = candidate && candidate.stale;
  for (const polygon of contours) {
    ctx.beginPath();
    polygon.forEach(([x, y], index) => {
      if (index === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.closePath();
    ctx.fillStyle = "#ff62591f";
    ctx.fill();
    ctx.strokeStyle = lowQuality ? "#f3b64b" : "#ff7a72";
    ctx.lineWidth = 2;
    ctx.setLineDash(lowQuality ? [6, 4] : []);
    ctx.shadowColor = "#ff6259";
    ctx.shadowBlur = 14;
    ctx.stroke();
    ctx.shadowBlur = 0;
    ctx.setLineDash([]);
  }

  if (candidate && candidate.centroidMm) {
    const scale = Number(state.canvas.pxPerMm);
    const cx = Number(candidate.centroidMm[0]) * scale;
    const cy = Number(candidate.centroidMm[1]) * scale;
    ctx.strokeStyle = "#ffd9d6";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(cx - 10, cy);
    ctx.lineTo(cx + 10, cy);
    ctx.moveTo(cx, cy - 10);
    ctx.lineTo(cx, cy + 10);
    ctx.stroke();
    ctx.fillStyle = "#ffd9d6";
    ctx.font = `9px ${MONO}`;
    ctx.textAlign = "left";
    ctx.fillText(`TRACK ${candidate.trackId}`, cx + 14, cy - 8);

    if (pulseActive()) {
      const phase = (now % 1800) / 1800;
      const radius = 16 + phase * 44;
      ctx.globalAlpha = (1 - phase) * 60 / 100;
      ctx.strokeStyle = "#ff6259";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(cx, cy, radius, 0, Math.PI * 2);
      ctx.stroke();
      ctx.globalAlpha = 1;
    }
  }
}

function drawLabels(ctx, width, height) {
  ctx.fillStyle = "#5f747d";
  ctx.font = `9px ${MONO}`;
  ctx.textAlign = "left";
  ctx.fillText(
    `${state.topology.chassis.width_mm} × ${state.topology.chassis.height_mm} mm · GRID 50 mm`,
    12,
    height - 12,
  );
  ctx.textAlign = "right";
  ctx.fillText("RECTIFIED / TOP-DOWN", width - 12, 20);
}

function drawCameraState(ctx, width) {
  if (!state.camera || state.camera.ok !== false) return;
  const reasons = (state.camera.reasons || []).join(" / ");
  ctx.fillStyle = "#3a1512d9";
  ctx.fillRect(0, 0, width, 34);
  ctx.fillStyle = "#ffb3ad";
  ctx.font = `10px ${MONO}`;
  ctx.textAlign = "left";
  ctx.fillText(`CAMERA DEGRADED${reasons ? ` — ${reasons}` : ""}`, 12, 22);
}

function describeScene() {
  const candidate = state.frame ? state.frame.candidate : null;
  if (!candidate) return "Rectified chassis plane. No candidate in the current frame.";
  const zone = candidate.zone === null ? "unknown zone" : `zone ${candidate.zone}`;
  return `Rectified chassis plane. Candidate in ${zone}, area ${fixed(
    candidate.areaMm2,
    1,
  )} square millimetres, centroid ${fixed(candidate.centroidMm[0], 0)} by ${fixed(
    candidate.centroidMm[1],
    0,
  )} millimetres.`;
}

/* ---------- sparkline ---------- */

function drawSparkline() {
  const canvas = $("sparkline");
  const ctx = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  const left = 28;
  const right = 8;
  const top = 10;
  const bottom = 18;
  const plotWidth = width - left - right;
  const plotHeight = height - top - bottom;

  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#0b1114";
  ctx.fillRect(0, 0, width, height);

  const yFor = (p) => top + (1 - Math.max(0, Math.min(1, Number(p)))) * plotHeight;

  if (state.thresholds) {
    ctx.font = `8px ${MONO}`;
    ctx.textAlign = "left";
    for (const [label, value] of [
      ["confirmed", state.thresholds.confirmedLeak],
      ["escalate", state.thresholds.escalation],
      ["suspect", state.thresholds.suspect],
      ["nominal", state.thresholds.nominal],
    ]) {
      const y = yFor(value);
      ctx.strokeStyle = "#25353b";
      ctx.setLineDash([3, 4]);
      ctx.beginPath();
      ctx.moveTo(left, y);
      ctx.lineTo(width - right, y);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = "#4d626b";
      ctx.fillText(fixed(value, 2), 2, y + 3);
    }
  }

  const history = state.pHistory;
  if (history.length < 2) {
    ctx.fillStyle = "#4d626b";
    ctx.font = `10px ${MONO}`;
    ctx.textAlign = "center";
    ctx.fillText("P(leak) history", width / 2, height / 2);
    return;
  }

  const t0 = history[0].t;
  const t1 = history[history.length - 1].t;
  const span = Math.max(1, t1 - t0);
  const xFor = (t) => left + ((t - t0) / span) * plotWidth;

  ctx.beginPath();
  history.forEach((point, index) => {
    const x = xFor(point.t);
    const y = yFor(point.p);
    if (index === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  const label = state.decision ? state.decision.label : "NOMINAL";
  const stroke = INCIDENT_LABELS.has(label)
    ? "#ff6259"
    : CAUTION_LABELS.has(label)
      ? "#f3b64b"
      : "#37e6c5";
  ctx.strokeStyle = stroke;
  ctx.lineWidth = 2;
  ctx.stroke();

  ctx.lineTo(width - right, height - bottom);
  ctx.lineTo(left, height - bottom);
  ctx.closePath();
  ctx.fillStyle = INCIDENT_LABELS.has(label) ? "#ff625918" : "#37e6c514";
  ctx.fill();

  const last = history[history.length - 1];
  ctx.fillStyle = stroke;
  ctx.beginPath();
  ctx.arc(xFor(last.t), yFor(last.p), 3, 0, Math.PI * 2);
  ctx.fill();

  ctx.fillStyle = "#4d626b";
  ctx.font = `8px ${MONO}`;
  ctx.textAlign = "left";
  ctx.fillText(`T+${fixed(history[0].t, 1)}`, left, height - 4);
  ctx.textAlign = "right";
  ctx.fillText(`T+${fixed(last.t, 1)} s`, width - right, height - 4);
}

boot();
