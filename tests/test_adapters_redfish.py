"""Redfish client: retries, idempotent ForceOff, and the SIMULATED flag."""

from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import pytest

from cryoguard.adapters.loopback import loopback_simulators
from cryoguard.adapters.redfish_client import RedfishClient
from cryoguard.config import load
from cryoguard.orchestrator.incident_fsm import VERIFICATION_TIMEOUT, VERIFIED_OFF

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def cfg() -> dict:
    return load(ROOT / "configs" / "default.yaml")


def _body(power: str = "Off") -> dict:
    return {
        "PowerState": power,
        "Oem": {"CryoGuard": {"simulated": True}},
    }


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_the_bmc_token_is_read_as_x_auth_token(cfg, monkeypatch) -> None:
    monkeypatch.setenv("CRYOGUARD_BMC_TOKEN", "session-token-for-test")

    client = RedfishClient(cfg["bmc"])

    assert client.headers["X-Auth-Token"] == "session-token-for-test"


def test_a_missing_bmc_token_sends_no_auth_header(cfg, monkeypatch) -> None:
    monkeypatch.delenv("CRYOGUARD_BMC_TOKEN", raising=False)

    client = RedfishClient(cfg["bmc"])

    assert "X-Auth-Token" not in client.headers


def test_a_503_is_retried_and_the_second_attempt_wins(cfg) -> None:
    calls = {"post": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json=_body("Off"))
        calls["post"] += 1
        if calls["post"] == 1:
            return httpx.Response(503, json={"error": {"message": "busy"}})
        return httpx.Response(202, json=_body("Off"))

    section = dict(cfg["bmc"], retry_backoff_ms=0)
    result = RedfishClient(section, client=_client(handler)).force_off()

    assert result.ok is True
    assert result.attempts == 2
    assert result.status_code == 202
    assert result.state == "Off"
    assert result.status == VERIFIED_OFF
    assert result.simulated is True


def test_a_202_whose_power_state_stays_on_times_out(cfg) -> None:
    """202 is acceptance, not power-off; a body that says Off is not evidence."""

    polls = {"get": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            polls["get"] += 1
            return httpx.Response(200, json=_body("On"))
        return httpx.Response(202, json=_body("Off"))

    section = dict(cfg["bmc"], poll_interval_ms=1, power_off_timeout_ms=20)
    result = RedfishClient(section, client=_client(handler)).force_off()

    assert result.ok is False
    assert result.status == VERIFICATION_TIMEOUT
    assert result.status_code == 202
    assert result.state == "On"
    assert polls["get"] > 1  # it kept polling instead of trusting the 202
    assert "still On" in result.detail


def test_a_202_is_verified_off_only_after_the_readback_flips(cfg) -> None:
    polls = {"get": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            polls["get"] += 1
            return httpx.Response(200, json=_body("On" if polls["get"] == 1 else "Off"))
        return httpx.Response(202, json=_body("On"))

    section = dict(cfg["bmc"], poll_interval_ms=1, power_off_timeout_ms=50)
    result = RedfishClient(section, client=_client(handler)).force_off()

    assert result.ok is True
    assert result.status == VERIFIED_OFF
    assert result.state == "Off"
    assert polls["get"] == 2


def test_a_204_is_also_verified_not_assumed(cfg) -> None:
    """No body, no PowerState: ``Off`` has to come from the readback."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json=_body("On"))
        return httpx.Response(204)

    section = dict(cfg["bmc"], poll_interval_ms=1, power_off_timeout_ms=10)
    result = RedfishClient(section, client=_client(handler)).force_off()

    assert result.ok is False
    assert result.status == VERIFICATION_TIMEOUT
    assert result.status_code == 204


def test_an_unreadable_power_state_is_never_treated_as_off(cfg) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(503, json={"error": {"message": "unavailable"}})
        return httpx.Response(202, json=_body("Off"))

    section = dict(cfg["bmc"], poll_interval_ms=1, power_off_timeout_ms=10)
    result = RedfishClient(section, client=_client(handler)).force_off()

    assert result.ok is False
    assert result.status == VERIFICATION_TIMEOUT
    assert result.state == "Unknown"


def test_a_409_already_off_is_success(cfg) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json=_body("Off"))
        return httpx.Response(409, json={"error": {"code": "Base.1.0.ActionNotAvailable"}})

    result = RedfishClient(cfg["bmc"], client=_client(handler)).force_off()

    assert result.ok is True
    assert result.attempts == 1
    assert result.status_code == 409
    assert result.status == VERIFIED_OFF
    assert result.detail == "already off"


def test_a_400_is_not_retried(cfg) -> None:
    calls = {"post": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["post"] += 1
        return httpx.Response(400, json={"error": {"message": "bad reset type"}})

    result = RedfishClient(cfg["bmc"], client=_client(handler)).force_off()

    assert result.ok is False
    assert result.attempts == 1
    assert calls["post"] == 1
    assert "400" in result.detail


def test_force_off_against_the_mock_over_a_real_socket(cfg) -> None:
    async def case():
        async with loopback_simulators(cfg) as endpoints:
            client = RedfishClient(cfg["bmc"], base_url=endpoints.bmc_url)
            first = await asyncio.to_thread(client.force_off)
            second = await asyncio.to_thread(client.force_off)
            state = await asyncio.to_thread(client.power_state)
            return first, second, state

    first, second, state = asyncio.run(case())

    assert first.ok is True
    assert first.simulated is True
    assert first.state == "Off"
    assert second.ok is True
    assert second.detail == "already off"
    assert state == "Off"
