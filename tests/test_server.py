"""FastAPI dashboard (P9-T02): state, manifest, incident files, WS stream."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from cryoguard.config import config_hash, load
from cryoguard.server import create_app, load_settings

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def settings() -> dict:
    data = load_settings()
    data["run"]["protocol_mode"] = "memory"
    data["run"]["sam_mode"] = "off"
    return data


@pytest.fixture()
def app(settings: dict, tmp_path: Path):  # type: ignore[no-untyped-def]
    return create_app(settings=settings, incident_root=tmp_path / "incidents", allow_llm=False)


def test_the_state_endpoint_describes_the_same_config_and_fixtures(app, settings: dict) -> None:
    cfg = load(ROOT / "configs" / "default.yaml")

    with TestClient(app) as client:
        response = client.get("/api/state")

    assert response.status_code == 200
    state = response.json()
    assert state["configHash"] == config_hash(cfg)
    assert state["fusionHash"] == "164849b04a9afb87"
    assert state["rackId"] == "RACK-08B"
    assert state["simulated"] is True
    assert state["canvas"] == {"widthPx": 1200, "heightPx": 800, "pxPerMm": 2.0}
    assert state["thresholds"]["confirmedLeak"] == cfg["fusion"]["thresholds"]["confirmed_leak"]
    assert state["thresholds"]["nominal"] == cfg["fusion"]["thresholds"]["nominal_cutoff"]
    assert [scenario["key"] for scenario in state["scenarios"]] == [
        "leak",
        "condensation",
        "glint",
        "hidden",
    ]
    assert state["run"]["status"] == "idle"
    assert state["latest"]["decision"] is None


def test_the_manifest_and_the_zero_build_ui_are_served(app) -> None:
    with TestClient(app) as client:
        manifest = client.get("/api/manifest")
        index = client.get("/")
        javascript = client.get("/ui/app.js")
        stylesheet = client.get("/ui/styles.css")

    assert manifest.status_code == 200
    payload = manifest.json()
    assert payload["simulated"] is True
    assert len(payload["clips"]) == 4
    assert {clip["source"] for clip in payload["clips"]} == {"synthetic"}
    assert index.status_code == 200
    assert "SIMULATED" in index.text
    assert "/ui/app.js" in index.text
    assert javascript.status_code == 200
    assert "RUN_STARTED" in javascript.text
    assert stylesheet.status_code == 200
    assert "--aqua" in stylesheet.text


def test_a_run_streams_in_order_and_files_a_readable_incident(app) -> None:
    with TestClient(app) as client:
        with client.websocket_connect("/ws") as socket:
            response = client.post(
                "/api/run",
                json={"scenario": "leak", "protocols": "memory", "sam": "off"},
            )
            assert response.status_code == 202
            events = []
            for _ in range(600):
                event = socket.receive_json()
                events.append(event)
                if event["type"] == "RUN_FINISHED":
                    break
            else:  # pragma: no cover - the run is deterministic
                pytest.fail("the run never finished")

        types = [event["type"] for event in events]
        assert types[0] == "RUN_STARTED"
        assert types[-1] == "RUN_FINISHED"
        assert types.count("INCIDENT_CLOSED") == 1
        assert types.count("ACTION_EXECUTED") == 6
        assert types.index("INCIDENT_CLOSED") < types.index("RUN_FINISHED")
        assert [event["seq"] for event in events] == list(range(1, len(events) + 1))

        state = client.get("/api/state").json()
        assert state["run"]["status"] == "finished"
        assert state["run"]["incident_id"] == "INC-RACK-08B-20261030-01"
        assert state["run"]["report_source"] == "deterministic_fallback"
        assert state["latest"]["decision"]["label"] == "CONFIRMED_LEAK"

        incident = client.get(f"/api/incidents/{state['run']['incident_id']}")
        assert incident.status_code == 200
        assert incident.json()["incidentId"] == state["run"]["incident_id"]
        assert incident.json()["reportSource"] == "deterministic_fallback"


def test_a_second_run_is_refused_while_one_is_running(app) -> None:
    with TestClient(app) as client:
        first = client.post("/api/run", json={"scenario": "leak", "protocols": "memory"})
        second = client.post("/api/run", json={"scenario": "glint", "protocols": "memory"})

    assert first.status_code == 202
    assert second.status_code == 409


def test_unknown_scenarios_and_bad_incident_ids_are_refused(app) -> None:
    with TestClient(app) as client:
        unknown = client.post("/api/run", json={"scenario": "../configs/default"})
        traversal = client.get("/api/incidents/..%2F..%2Fetc%2Fpasswd")
        injection = client.get("/api/incidents/INC-OK%3Brm%20-rf")
        missing = client.get("/api/incidents/INC-RACK-08B-20990101-99")

    assert unknown.status_code == 422
    assert traversal.status_code == 404
    assert injection.status_code == 404
    assert missing.status_code == 404
