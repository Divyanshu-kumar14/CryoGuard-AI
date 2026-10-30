"""UI contract (P9-T03..T07): no build step, no literal probabilities, honest badges."""

from __future__ import annotations

import re
from pathlib import Path

from cryoguard.server import load_settings

ROOT = Path(__file__).resolve().parents[1]
UI = ROOT / "ui"
HTML = (UI / "index.html").read_text(encoding="utf-8")
CSS = (UI / "styles.css").read_text(encoding="utf-8")
JS = (UI / "app.js").read_text(encoding="utf-8")

# A probability is a decimal literal in [0, 1). The decision strip must be fed by
# the WebSocket; a literal here would be a fabricated number on screen.
PROBABILITY_LITERAL = re.compile(r"(?<![\w.])0\.\d+")


def test_the_decision_strip_has_no_probability_literal() -> None:
    assert PROBABILITY_LITERAL.search(JS) is None
    assert 'setText("p-leak", fixed(data.p, 3))' in JS


def test_every_simulated_surface_is_badged_at_the_point_of_display() -> None:
    assert 'id="camera-badge"' in HTML
    assert 'id="plc-badge"' in HTML
    assert 'id="bmc-badge"' in HTML
    assert 'id="telemetry-badge"' in HTML
    assert 'id="sim-stamp"' in HTML
    assert "SIMULATED" in HTML
    assert "PLC SIMULATED" in HTML
    assert "BMC SIMULATED" in HTML


def test_the_report_pane_names_both_possible_sources() -> None:
    assert "[REPORT: MUSE SPARK OPEN]" in JS
    assert "[FALLBACK: DETERMINISTIC]" in JS
    assert 'id="report-badge"' in HTML


def test_every_pipeline_event_type_is_handled() -> None:
    for event_type in (
        "RUN_STARTED",
        "FRAME_TICK",
        "EVIDENCE_UPDATE",
        "DECISION_CHANGE",
        "ACTION_EXECUTED",
        "INCIDENT_CLOSED",
        "RUN_FINISHED",
    ):
        assert f'case "{event_type}"' in JS


def test_the_ui_makes_no_external_request_and_has_no_build_step() -> None:
    combined = HTML + CSS + JS
    assert re.search(r"https?://", combined) is None
    assert "@import" not in CSS
    assert "fonts.googleapis" not in combined

    assert not (UI / "package.json").exists()
    assert not (UI / "node_modules").exists()
    assert {path.suffix for path in UI.iterdir() if path.is_file()} <= {
        ".html",
        ".css",
        ".js",
        ".md",
    }


def test_the_canvas_is_accessible_and_motion_is_bounded() -> None:
    assert 'role="img"' in HTML
    assert "aria-label" in HTML
    assert 'aria-live="polite"' in HTML
    assert "prefers-reduced-motion" in CSS
    assert 'id="state-label" aria-live="polite"' in HTML
    assert 'id="run-status" data-state="idle" aria-live="polite"' in HTML


def test_the_only_mutating_request_is_a_scenario_start() -> None:
    assert JS.count("fetch(") == 2
    assert 'fetch("/api/state"' in JS
    assert 'fetch("/api/run"' in JS
    assert "modbus" not in JS.lower()
    assert "redfish" not in JS.lower()


def test_the_dashboard_config_keeps_the_server_on_loopback() -> None:
    settings = load_settings()

    assert settings["server"]["host"] in {"127.0.0.1", "localhost", "::1"}
    assert isinstance(settings["server"]["port"], int)
    assert settings["run"]["protocol_mode"] in {"memory", "loopback", "external"}
    assert settings["run"]["sam_mode"] in {"off", "cache", "live"}
