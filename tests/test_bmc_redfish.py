"""BMC mock: ForceOff only when powered, and every response says SIMULATED."""

from __future__ import annotations

from fastapi.testclient import TestClient

from cryoguard.sims.bmc_server import BmcState, create_app

SYSTEM = "/redfish/v1/Systems/gpu-node-08b"
RESET = f"{SYSTEM}/Actions/ComputerSystem.Reset"


def client(power: str = "On") -> TestClient:
    return TestClient(create_app(BmcState(power=power)))  # type: ignore[arg-type]


def test_service_root_and_system_are_redfish_shaped() -> None:
    api = client()
    root = api.get("/redfish/v1")
    assert root.status_code == 200
    assert root.json()["Systems"]["@odata.id"] == "/redfish/v1/Systems"

    system = api.get(SYSTEM)
    assert system.status_code == 200
    body = system.json()
    assert body["PowerState"] == "On"
    assert body["Oem"]["CryoGuard"]["simulated"] is True
    assert body["Actions"]["#ComputerSystem.Reset"]["target"] == RESET


def test_force_off_returns_202_and_the_next_get_shows_off() -> None:
    api = client("On")
    response = api.post(RESET, json={"ResetType": "ForceOff"})
    assert response.status_code == 202
    assert response.json()["PowerState"] == "Off"

    assert api.get(SYSTEM).json()["PowerState"] == "Off"


def test_force_off_on_an_off_system_is_a_conflict() -> None:
    api = client("Off")
    response = api.post(RESET, json={"ResetType": "ForceOff"})
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "Base.1.0.ActionNotAvailable"


def test_on_powers_the_node_back_up() -> None:
    api = client("Off")
    assert api.post(RESET, json={"ResetType": "On"}).status_code == 202
    assert api.get(SYSTEM).json()["PowerState"] == "On"


def test_an_unknown_reset_type_is_rejected() -> None:
    api = client()
    response = api.post(RESET, json={"ResetType": "Explode"})
    assert response.status_code == 400


def test_a_missing_reset_type_is_rejected() -> None:
    api = client()
    assert api.post(RESET, json={}).status_code == 400


def test_an_unknown_system_is_404() -> None:
    api = client()
    assert api.get("/redfish/v1/Systems/gpu-node-99z").status_code == 404
    assert (
        api.post(
            "/redfish/v1/Systems/gpu-node-99z/Actions/ComputerSystem.Reset",
            json={"ResetType": "ForceOff"},
        ).status_code
        == 404
    )
