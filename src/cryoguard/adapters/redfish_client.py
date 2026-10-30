"""Redfish BMC client (P6-T05).

Only two operations are needed by the safety sequence: read ``PowerState`` and
issue ``ComputerSystem.Reset`` with ``ForceOff``. Retries are safe because
ForceOff is idempotent: a 409 that says the system is already off is success, not
a failure. A 5xx is retried; a 4xx is not.

Every response from the mock carries ``Oem.CryoGuard.simulated = true``. The
client surfaces that flag so the incident report can label the action honestly.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import httpx

from cryoguard.orchestrator.incident_fsm import PowerResult


class RedfishError(ValueError):
    """An invalid BMC configuration or an unusable response."""


def _simulated(body: dict[str, Any]) -> bool:
    oem = body.get("Oem")
    if not isinstance(oem, dict):
        return False
    cryoguard = oem.get("CryoGuard")
    if not isinstance(cryoguard, dict):
        return False
    return bool(cryoguard.get("simulated", False))


class RedfishClient:
    def __init__(
        self,
        cfg: dict[str, Any],
        *,
        base_url: str | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = str(base_url if base_url is not None else cfg["base_url"]).rstrip("/")
        self.node_id = str(cfg["node_id"])
        self.simulated = bool(cfg["simulated"])
        self.timeout_s = float(cfg["timeout_ms"]) / 1000.0
        self.max_retries = int(cfg["max_retries"])
        self.retry_backoff_s = float(cfg["retry_backoff_ms"]) / 1000.0
        self.headers = {"Content-Type": "application/json"}
        token_env = str(cfg.get("token_env", ""))
        if token_env:
            import os

            token = os.environ.get(token_env)
            if token:
                self.headers["X-Auth-Token"] = token

        self.system_url = f"{self.base_url}/redfish/v1/Systems/{self.node_id}"
        self.reset_url = f"{self.system_url}/Actions/ComputerSystem.Reset"
        self._client = client
        if self.timeout_s <= 0:
            raise RedfishError(f"timeout_ms must be positive: {cfg['timeout_ms']}")
        if self.max_retries < 0:
            raise RedfishError(f"max_retries must be non-negative: {self.max_retries}")

    @contextmanager
    def _http(self) -> Iterator[httpx.Client]:
        if self._client is not None:
            yield self._client
            return
        with httpx.Client(timeout=self.timeout_s, headers=self.headers) as client:
            yield client

    def system(self) -> dict[str, Any]:
        with self._http() as client:
            response = client.get(self.system_url)
        if response.status_code != 200:
            raise RedfishError(f"GET {self.system_url} -> {response.status_code}")
        body = response.json()
        if not isinstance(body, dict):
            raise RedfishError("system resource is not a JSON object")
        return body

    def power_state(self) -> str:
        try:
            return str(self.system().get("PowerState", "Unknown"))
        except (httpx.HTTPError, RedfishError, ValueError):
            return "Unknown"

    def force_off(self) -> PowerResult:
        attempts = 0
        last_detail = ""
        for attempt in range(self.max_retries + 1):
            attempts = attempt + 1
            try:
                with self._http() as client:
                    response = client.post(self.reset_url, json={"ResetType": "ForceOff"})
            except httpx.HTTPError as exc:
                last_detail = f"{type(exc).__name__}: {exc}"
                if attempt < self.max_retries:
                    time.sleep(self.retry_backoff_s * (attempt + 1))
                    continue
                return PowerResult(False, "Unknown", attempts, 0, self.simulated, last_detail)

            body: dict[str, Any] = {}
            try:
                parsed = response.json()
                if isinstance(parsed, dict):
                    body = parsed
            except ValueError:
                body = {}

            if response.status_code == 202:
                state = str(body.get("PowerState", "Off"))
                return PowerResult(True, state, attempts, 202, _simulated(body) or self.simulated)

            if response.status_code == 409:
                state = self.power_state()
                if state == "Off":
                    return PowerResult(
                        True,
                        "Off",
                        attempts,
                        409,
                        _simulated(body) or self.simulated,
                        "already off",
                    )
                return PowerResult(
                    False,
                    state,
                    attempts,
                    409,
                    _simulated(body) or self.simulated,
                    "conflict and the system is not off",
                )

            if 500 <= response.status_code < 600 and attempt < self.max_retries:
                last_detail = f"HTTP {response.status_code}"
                time.sleep(self.retry_backoff_s * (attempt + 1))
                continue

            detail = f"HTTP {response.status_code}"
            error = body.get("error")
            if isinstance(error, dict) and error.get("message"):
                detail = f"{detail}: {error['message']}"
            return PowerResult(
                False, "Unknown", attempts, response.status_code, self.simulated, detail
            )

        return PowerResult(
            False, "Unknown", attempts, 0, self.simulated, last_detail or "retries exhausted"
        )
