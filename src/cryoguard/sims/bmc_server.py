"""Redfish-shaped BMC mock (P4-T05).

Only the surface the safety sequence needs: one system resource and its
``ComputerSystem.Reset`` action. ``ForceOff`` is honoured only when the node is
powered, and a second ForceOff returns 409 rather than pretending to succeed.

This is a mock. It has no TLS, no auth, and no IPMI underneath. Every response
carries ``Oem.CryoGuard.simulated = true`` so a captured log cannot be mistaken
for a real BMC.
"""

from __future__ import annotations

from typing import Any, Literal

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

PowerState = Literal["On", "Off"]

RESET_TYPES = frozenset({"On", "ForceOff", "GracefulShutdown", "GracefulRestart", "ForceRestart"})


class BmcState:
    def __init__(self, *, node_id: str = "gpu-node-08b", power: PowerState = "On") -> None:
        self.node_id = node_id
        self.power: PowerState = power
        self.resets = 0

    def reset(self, reset_type: str) -> tuple[int, dict[str, Any]]:
        if reset_type not in RESET_TYPES:
            return 400, _error(
                "Base.1.0.GeneralError", f"ResetType {reset_type!r} is not supported"
            )
        if reset_type == "ForceOff":
            if self.power == "Off":
                return 409, _error("Base.1.0.ActionNotAvailable", "the system is already off")
            self.power = "Off"
        elif reset_type == "On":
            if self.power == "On":
                return 409, _error("Base.1.0.ActionNotAvailable", "the system is already on")
            self.power = "On"
        # Restart types leave the system powered on.
        self.resets += 1
        return 202, _system(self)


def _error(code: str, message: str) -> dict[str, Any]:
    return {"error": {"code": code, "message": message}}


def _system(state: BmcState) -> dict[str, Any]:
    system_id = f"/redfish/v1/Systems/{state.node_id}"
    return {
        "@odata.id": system_id,
        "@odata.type": "#ComputerSystem.v1_20_0.ComputerSystem",
        "Id": state.node_id,
        "Name": state.node_id,
        "PowerState": state.power,
        "Status": {"State": "Enabled", "Health": "OK"},
        "Actions": {
            "#ComputerSystem.Reset": {
                "target": f"{system_id}/Actions/ComputerSystem.Reset",
                "ResetType@Redfish.AllowableValues": sorted(RESET_TYPES),
            }
        },
        "Oem": {"CryoGuard": {"simulated": True, "adapter": "bmc-mock"}},
    }


def system_resource(state: BmcState) -> dict[str, Any]:
    """Public helper for the adapter and tests."""

    return _system(state)


def create_app(state: BmcState | None = None) -> FastAPI:
    bmc = state or BmcState()
    app = FastAPI(title="CryoGuard BMC mock", version="1.0.0")

    @app.get("/redfish/v1")
    def root() -> dict[str, Any]:
        return {
            "@odata.id": "/redfish/v1",
            "@odata.type": "#ServiceRoot.v1_15_0.ServiceRoot",
            "Id": "RootService",
            "Name": "CryoGuard BMC mock",
            "Systems": {"@odata.id": "/redfish/v1/Systems"},
            "Oem": {"CryoGuard": {"simulated": True}},
        }

    @app.get("/redfish/v1/Systems")
    def systems() -> dict[str, Any]:
        return {
            "@odata.id": "/redfish/v1/Systems",
            "Members": [{"@odata.id": f"/redfish/v1/Systems/{bmc.node_id}"}],
            "Members@odata.count": 1,
        }

    @app.get("/redfish/v1/Systems/{node_id}")
    def system(node_id: str) -> Any:
        if node_id != bmc.node_id:
            return JSONResponse(
                status_code=404,
                content=_error("Base.1.0.ResourceMissingAtURI", f"unknown system {node_id}"),
            )
        return _system(bmc)

    @app.post("/redfish/v1/Systems/{node_id}/Actions/ComputerSystem.Reset")
    async def reset(node_id: str, request: Request) -> Any:
        if node_id != bmc.node_id:
            return JSONResponse(
                status_code=404,
                content=_error("Base.1.0.ResourceMissingAtURI", f"unknown system {node_id}"),
            )
        payload = await request.json()
        if not isinstance(payload, dict) or "ResetType" not in payload:
            return JSONResponse(
                status_code=400,
                content=_error("Base.1.0.PropertyMissing", "ResetType is required"),
            )
        status_code, body = bmc.reset(str(payload["ResetType"]))
        return JSONResponse(status_code=status_code, content=body)

    return app
