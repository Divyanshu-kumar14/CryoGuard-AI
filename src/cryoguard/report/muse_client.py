"""Post-incident explanation via Meta Muse Spark Open (P7-T02/T03/T04).

This module is only reachable from the composition root, after the action log is
sealed. Two properties are deliberate:

1. **The model explains, it does not decide.** Its output is two strings, and
   only ``root_cause`` and ``uncertainty`` are copied into the report. The
   recommended operator action stays the deterministic one.
2. **Failure is boring.** A missing key, a timeout, a 5xx, or malformed JSON
   returns the deterministic report that was already built (NFR-SAF-01). No
   exception from this module can reach the safety path.

Requests are logged to a JSONL proof file with the model id, latency, and token
counts. No line is written when no request was made, so the file cannot contain
a fabricated proof of an API call.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from cryoguard.report.schema import ActionModel, EvidenceModel

ROOT = Path(__file__).resolve().parents[3]


class MuseUnavailable(RuntimeError):
    """The explanation could not be produced; the caller keeps the fallback."""


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


class MuseBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    root_cause: str = Field(min_length=1)
    uncertainty: str = Field(min_length=1)


@dataclass(frozen=True)
class MuseResult:
    body: MuseBody
    model: str
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    attempts: int


@dataclass(frozen=True)
class PromptBook:
    system: str
    labels: dict[str, str]

    def for_label(self, label: str) -> str:
        return self.labels.get(label, "Explain the incident from the supplied facts.")


def load_prompts(path: str | Path) -> PromptBook:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "system" not in data:
        raise MuseUnavailable(f"prompt file is missing a system prompt: {path}")
    return PromptBook(system=str(data["system"]), labels=dict(data.get("labels", {})))


def incident_facts(
    *,
    incident_id: str,
    label: str,
    evidence: EvidenceModel,
    actions: list[ActionModel],
    fixture_source: str,
    trip_reason: str | None,
) -> dict[str, Any]:
    """The only facts the model sees. Nothing free-text enters this dict."""

    return {
        "incidentId": incident_id,
        "label": label,
        "tripReason": trip_reason,
        "fixtureSource": fixture_source,
        "simulated": fixture_source == "synthetic",
        "suspectedOrigin": evidence.qd_origin,
        "zone": evidence.zone,
        "evidence": evidence.model_dump(by_alias=True),
        "deterministicActionsTaken": [action.model_dump(by_alias=True) for action in actions],
    }


def build_messages(
    prompts: PromptBook,
    *,
    label: str,
    facts: dict[str, Any],
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": prompts.system},
        {
            "role": "user",
            "content": (
                f"{prompts.for_label(label)}\n\nIncident JSON:\n"
                f"{json.dumps(facts, sort_keys=True, indent=2)}"
            ),
        },
    ]


class MuseClient:
    """Async OpenAI-compatible client. Constructing it performs no network I/O."""

    def __init__(
        self,
        cfg: dict[str, Any],
        *,
        api_key: str | None = None,
        client: Any | None = None,
        prompts: PromptBook | None = None,
        proof_log: str | Path | None = None,
    ) -> None:
        self.cfg = cfg
        self.model = os.environ.get("BASETEN_MODEL") or str(cfg["model"])
        self.base_url = os.environ.get("BASETEN_BASE_URL") or str(cfg["base_url"])
        self.provider = str(cfg.get("provider", "baseten"))
        self.timeout_s = float(cfg["timeout_ms"]) / 1000.0
        self.max_tokens = int(cfg.get("max_tokens", 600))
        self.key_env = str(cfg.get("api_key_env", "MUSE_API_KEY"))
        self.primary_key_env = "BASETEN_API_KEY"
        self.prompts = prompts or load_prompts(_resolve(cfg.get("prompts", "configs/prompts.yaml")))
        log_path = proof_log if proof_log is not None else cfg.get("proof_log")
        self.proof_log = _resolve(log_path) if log_path else None

        key: str | None
        if api_key is not None:
            key = api_key
        else:
            # Baseten is the provider; MUSE_API_KEY remains a compatibility alias.
            key = os.environ.get(self.primary_key_env) or os.environ.get(self.key_env)
        self._key_present = bool(key)
        self.client = client
        if self.client is None and key:
            self.client = AsyncOpenAI(
                api_key=key,
                base_url=self.base_url,
                timeout=self.timeout_s,
                max_retries=int(cfg.get("max_retries", 1)),
            )

    @property
    def available(self) -> bool:
        return self.client is not None

    def _log(
        self,
        *,
        incident_id: str,
        status: str,
        latency_ms: float,
        prompt_tokens: int,
        completion_tokens: int,
        attempts: int,
        detail: str = "",
    ) -> None:
        if self.proof_log is None:
            return
        entry = {
            "ts": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
            "incidentId": incident_id,
            "provider": self.provider,
            "model": self.model,
            "status": status,
            "latencyMs": round(latency_ms, 3),
            "promptTokens": prompt_tokens,
            "completionTokens": completion_tokens,
            "attempts": attempts,
        }
        if detail:
            entry["detail"] = detail[:500]
        self.proof_log.parent.mkdir(parents=True, exist_ok=True)
        with self.proof_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")

    async def _request(self, messages: list[dict[str, str]]) -> tuple[str, int, int]:
        assert self.client is not None
        response = await asyncio.wait_for(
            self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.0,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"},
            ),
            timeout=self.timeout_s,
        )
        content = response.choices[0].message.content or ""
        usage = getattr(response, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        return content, prompt_tokens, completion_tokens

    async def explain(
        self,
        *,
        incident_id: str,
        label: str,
        evidence: EvidenceModel,
        actions: list[ActionModel],
        fixture_source: str,
        trip_reason: str | None = None,
    ) -> MuseResult:
        if not self.available:
            raise MuseUnavailable(
                f"{self.primary_key_env} (or {self.key_env}) is not set; no request was made"
            )

        facts = incident_facts(
            incident_id=incident_id,
            label=label,
            evidence=evidence,
            actions=actions,
            fixture_source=fixture_source,
            trip_reason=trip_reason,
        )
        messages = build_messages(self.prompts, label=label, facts=facts)
        started = time.monotonic()
        last_error = "unknown"
        prompt_tokens = completion_tokens = 0

        # One initial attempt plus at most one repair, as specified in P7-T01.
        for attempt in (1, 2):
            try:
                content, prompt_tokens, completion_tokens = await self._request(messages)
            except TimeoutError as exc:
                self._log(
                    incident_id=incident_id,
                    status="timeout",
                    latency_ms=(time.monotonic() - started) * 1000.0,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    attempts=attempt,
                    detail=str(exc) or "client timeout",
                )
                raise MuseUnavailable(
                    f"Muse request timed out after {self.timeout_s:.2f} s"
                ) from exc
            except Exception as exc:
                self._log(
                    incident_id=incident_id,
                    status="error",
                    latency_ms=(time.monotonic() - started) * 1000.0,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    attempts=attempt,
                    detail=f"{type(exc).__name__}: {exc}",
                )
                raise MuseUnavailable(f"Muse request failed: {type(exc).__name__}: {exc}") from exc

            try:
                body = MuseBody.model_validate_json(content)
            except ValidationError as exc:
                last_error = str(exc)
                self._log(
                    incident_id=incident_id,
                    status="malformed",
                    latency_ms=(time.monotonic() - started) * 1000.0,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    attempts=attempt,
                    detail=last_error,
                )
                if attempt == 1:
                    messages = [
                        *messages,
                        {"role": "assistant", "content": content[:2000]},
                        {
                            "role": "user",
                            "content": (
                                "That reply failed schema validation with: "
                                f"{last_error}. Reply with corrected JSON only: "
                                '{"root_cause": "...", "uncertainty": "..."}'
                            ),
                        },
                    ]
                    continue
                raise MuseUnavailable(f"Muse returned malformed JSON twice: {last_error}") from exc

            self._log(
                incident_id=incident_id,
                status="ok",
                latency_ms=(time.monotonic() - started) * 1000.0,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                attempts=attempt,
            )
            return MuseResult(
                body=body,
                model=self.model,
                latency_ms=(time.monotonic() - started) * 1000.0,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                attempts=attempt,
            )

        raise MuseUnavailable(f"Muse produced no usable reply: {last_error}")
