"""Post-incident explanation via Meta Muse Spark Open (P7-T02/T03/T04).

This module is only reachable from the composition root, after the action log is
sealed. Two properties are deliberate:

1. **The model explains, it does not decide.** Its output is two strings, and
   only ``root_cause`` and ``uncertainty`` are copied into the report. The
   recommended operator action stays the deterministic one.
2. **Failure is boring.** A missing key, a timeout, a 5xx, or malformed JSON
   returns the deterministic report that was already built (NFR-SAF-01). No
   exception from this module can reach the safety path.

Requests are logged to a JSONL proof file with honest metadata: model id,
provider echo, ``finish_reason``, prompt/completion tokens, response char
count, cumulative latency, attempt number, schema-validation result, and a
short failure category. No line is written when no request was made, so the
file cannot contain a fabricated proof of an API call. Secrets and full
responses are never logged; ``detail`` is a truncated validation or transport
summary (<= 500 chars).

Token budget: ``max_tokens`` defaults to 600 (config ``report.max_tokens``,
overridable via ``BASETEN_MAX_TOKENS``). Muse Spark Open is a reasoning model,
so a small budget can be consumed by reasoning before any JSON content is
emitted, surfacing as ``finish_reason == "length"`` with empty content. Do NOT
blindly raise the default: a larger budget raises cost and latency on every
incident. The client detects ``length`` as ``truncated_length`` in the log and
the result instead of misreporting it as generic malformed JSON.

Attempt vs request accounting: each log line is one *attempt*; one
:py:meth:`MuseClient.explain` call is one *request* (at most two attempts: one
initial plus one compact repair). :py:func:`health_summary` groups lines by
``incidentId`` so retries are never misrepresented as independent incidents.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import statistics
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from cryoguard.report.schema import ActionModel, EvidenceModel

ROOT = Path(__file__).resolve().parents[3]

MAX_FIELD_CHARS = 2000
DETAIL_TRUNCATE = 500
SHORT_ERROR_TRUNCATE = 180

# Heuristic guard against the model issuing operator instructions or invented
# identifiers. This is a complement to the system prompt, not a substitute:
# the schema already forbids extra keys and caps field length.
BLOCKED_PHRASES: tuple[str, ...] = (
    "open the valve",
    "close the valve",
    "increase power",
    "decrease power",
    "power cycle",
    "restart the node",
    "reschedule the job",
    "part number",
)


class MuseUnavailable(RuntimeError):
    """The explanation could not be produced; the caller keeps the fallback."""


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


class MuseBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    root_cause: str = Field(min_length=1, max_length=MAX_FIELD_CHARS)
    uncertainty: str = Field(min_length=1, max_length=MAX_FIELD_CHARS)

    @field_validator("root_cause", "uncertainty", mode="after")
    @classmethod
    def _stripped_nonempty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        lowered = value.strip().lower()
        for phrase in BLOCKED_PHRASES:
            if phrase in lowered:
                raise ValueError(f"unsupported assertion: {phrase!r}")
        return value


@dataclass(frozen=True)
class MuseResult:
    body: MuseBody
    model: str
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    attempts: int
    finish_reason: str = "unknown"
    response_chars: int = 0
    provider: str = "baseten"


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


def _short_error(exc: ValidationError) -> str:
    """First pydantic error as a short, log-safe summary (no response echo)."""
    try:
        errors = exc.errors()
    except Exception:
        return str(exc)[:SHORT_ERROR_TRUNCATE]
    if not errors:
        return str(exc)[:SHORT_ERROR_TRUNCATE]
    first = errors[0]
    loc = ".".join(str(part) for part in first.get("loc", ())) or "<root>"
    msg = str(first.get("msg", "invalid"))
    return f"{loc}: {msg}"[:SHORT_ERROR_TRUNCATE]


def _failure_category(exc: ValidationError, content: str) -> str:
    """Map a pydantic failure to a stable log category."""
    if not content.strip():
        return "empty"
    try:
        errors = exc.errors()
    except Exception:
        return "invalid_json"
    for error in errors:
        if "unsupported assertion" in str(error.get("msg", "")).lower():
            return "unsupported_assertion"
    for error in errors:
        if error.get("type") == "json_invalid":
            return "invalid_json"
    for error in errors:
        if error.get("type") == "missing":
            return "missing_fields"
    for error in errors:
        if error.get("type") == "extra_forbidden":
            return "extra_fields"
    for error in errors:
        if error.get("type") == "string_too_long":
            return "too_long"
    for error in errors:
        if error.get("type") == "string_too_short":
            return "empty_strings"
    for error in errors:
        msg = str(error.get("msg", "")).lower()
        if "must not be blank" in msg or "too short" in msg or "at least 1 character" in msg:
            return "empty_strings"
        if "too long" in msg or f"at most {MAX_FIELD_CHARS}" in msg:
            return "too_long"
    return "invalid_json"


def _repair_message(category: str, short: str) -> dict[str, str]:
    """Compact repair instruction: failure + required shape, no echo."""
    return {
        "role": "user",
        "content": (
            f"Previous reply failed schema validation ({category}): {short}. "
            'Reply JSON only {"root_cause": "...", "uncertainty": "..."}. '
            f"Each field 1-{MAX_FIELD_CHARS} chars, no extra keys."
        ),
    }


def _p95(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * 0.95) - 1))
    return float(ordered[index])


def health_summary(
    entries: Iterable[Mapping[str, Any]] | str | Path | None = None,
    *,
    proof_log: str | Path | None = None,
) -> dict[str, Any]:
    """Summarise proof-log attempts without inflating retries into incidents.

    Args:
        entries: parsed proof-log dicts, or a path to a JSONL proof file.
        proof_log: alternative path to a JSONL proof file.

    Returns a dict with request/attempt separation: ``requests`` counts
    distinct ``incidentId`` values (completed requests), ``attempts`` counts
    log lines. ``valid_rate`` is ok-requests / requests, ``malformed_rate``
    is malformed-attempts / attempts, ``retry_rate`` is retried-requests /
    requests. Latency (``median_latency_ms``/``p95_latency_ms``) and token
    averages are computed over the terminal attempt of each request so a
    retry is not double-counted.
    """
    source: str | Path | None = proof_log
    raw: Iterable[Mapping[str, Any]] | str | Path | None = entries
    if isinstance(raw, (str, Path)):
        if source is not None:
            raise ValueError("health_summary got two proof_log paths")
        source = raw
        raw = None
    parsed: list[Mapping[str, Any]]
    if raw is None:
        if source is None:
            raise ValueError("health_summary needs entries or a proof_log path")
        path = _resolve(source)
        if not path.is_file():
            return {
                "requests": 0,
                "attempts": 0,
                "ok_requests": 0,
                "failed_requests": 0,
                "valid_rate": 0.0,
                "malformed_rate": 0.0,
                "retry_rate": 0.0,
                "median_latency_ms": None,
                "p95_latency_ms": None,
                "avg_prompt_tokens": 0.0,
                "avg_completion_tokens": 0.0,
                "by_status": {},
                "by_failure_category": {},
                "truncated_count": 0,
            }
        parsed = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        parsed = list(raw)

    by_incident: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for entry in parsed:
        by_incident[str(entry.get("incidentId", "?"))].append(entry)

    requests = len(by_incident)
    attempts = len(parsed)
    ok_requests = sum(
        1 for group in by_incident.values() if any(e.get("status") == "ok" for e in group)
    )
    retried = sum(1 for group in by_incident.values() if len(group) > 1)
    malformed_attempts = sum(1 for e in parsed if e.get("status") == "malformed")
    terminals = [group[-1] for group in by_incident.values()]
    latencies = [float(t["latencyMs"]) for t in terminals if "latencyMs" in t]
    avg_prompt = (
        sum(float(t.get("promptTokens", 0) or 0) for t in terminals) / len(terminals)
        if terminals
        else 0.0
    )
    avg_completion = (
        sum(float(t.get("completionTokens", 0) or 0) for t in terminals) / len(terminals)
        if terminals
        else 0.0
    )
    by_status = dict(Counter(str(e.get("status", "unknown")) for e in parsed))
    by_failure = dict(
        Counter(str(e.get("failureCategory", "")) for e in parsed if e.get("failureCategory"))
    )
    truncated = sum(
        1
        for e in parsed
        if e.get("failureCategory") == "truncated_length" or e.get("finishReason") == "length"
    )
    return {
        "requests": requests,
        "attempts": attempts,
        "ok_requests": ok_requests,
        "failed_requests": requests - ok_requests,
        "valid_rate": (ok_requests / requests) if requests else 0.0,
        "malformed_rate": (malformed_attempts / attempts) if attempts else 0.0,
        "retry_rate": (retried / requests) if requests else 0.0,
        "median_latency_ms": float(statistics.median(latencies)) if latencies else None,
        "p95_latency_ms": _p95(latencies),
        "avg_prompt_tokens": float(avg_prompt),
        "avg_completion_tokens": float(avg_completion),
        "by_status": by_status,
        "by_failure_category": by_failure,
        "truncated_count": truncated,
    }


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
        self.max_tokens = int(os.environ.get("BASETEN_MAX_TOKENS") or cfg.get("max_tokens", 600))
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
        finish_reason: str = "unknown",
        response_chars: int = 0,
        failure_category: str = "",
        schema_valid: bool | None = None,
        detail: str = "",
    ) -> None:
        if self.proof_log is None:
            return
        entry: dict[str, Any] = {
            "ts": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
            "incidentId": incident_id,
            "provider": self.provider,
            "model": self.model,
            "status": status,
            "latencyMs": round(latency_ms, 3),
            "promptTokens": prompt_tokens,
            "completionTokens": completion_tokens,
            "attempts": attempts,
            "finishReason": finish_reason,
            "responseChars": response_chars,
        }
        if failure_category:
            entry["failureCategory"] = failure_category
        if schema_valid is not None:
            entry["schemaValid"] = schema_valid
        if detail:
            entry["detail"] = detail[:DETAIL_TRUNCATE]
        self.proof_log.parent.mkdir(parents=True, exist_ok=True)
        with self.proof_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")

    async def _request(self, messages: list[dict[str, str]]) -> tuple[str, int, int, str]:
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
        choices = getattr(response, "choices", []) or []
        first = choices[0] if choices else None
        finish = str(getattr(first, "finish_reason", None) or "unknown")
        message = getattr(first, "message", None) if first is not None else None
        content = getattr(message, "content", None) if message is not None else None
        usage = getattr(response, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        return (content or ""), prompt_tokens, completion_tokens, finish

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

        # One initial attempt plus at most one compact repair, as in P7-T01.
        # The repair carries only the validation failure and the required
        # shape; the previous malformed body is never echoed back.
        for attempt in (1, 2):
            try:
                content, prompt_tokens, completion_tokens, finish_reason = await self._request(
                    messages
                )
            except TimeoutError as exc:
                self._log(
                    incident_id=incident_id,
                    status="timeout",
                    latency_ms=(time.monotonic() - started) * 1000.0,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    attempts=attempt,
                    finish_reason="unknown",
                    response_chars=0,
                    failure_category="timeout",
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
                    finish_reason="unknown",
                    response_chars=0,
                    failure_category="http_error",
                    detail=f"{type(exc).__name__}: {exc}",
                )
                raise MuseUnavailable(f"Muse request failed: {type(exc).__name__}: {exc}") from exc

            response_chars = len(content)
            truncated = finish_reason == "length"

            if not content.strip():
                category = "truncated_length" if truncated else "empty"
                short = (
                    "response truncated by max_tokens (finish_reason=length)"
                    if truncated
                    else "empty response"
                )
                last_error = short
                self._log(
                    incident_id=incident_id,
                    status="malformed",
                    latency_ms=(time.monotonic() - started) * 1000.0,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    attempts=attempt,
                    finish_reason=finish_reason,
                    response_chars=response_chars,
                    failure_category=category,
                    schema_valid=False,
                    detail=short,
                )
                if attempt == 1:
                    messages = [*messages, _repair_message(category, short)]
                    continue
                raise MuseUnavailable(f"Muse returned malformed JSON twice: {last_error}")

            try:
                body = MuseBody.model_validate_json(content)
            except ValidationError as exc:
                category = "truncated_length" if truncated else _failure_category(exc, content)
                short = _short_error(exc)
                if truncated:
                    short = f"truncated by max_tokens (finish_reason=length); {short}"
                last_error = f"{category}: {short}"
                self._log(
                    incident_id=incident_id,
                    status="malformed",
                    latency_ms=(time.monotonic() - started) * 1000.0,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    attempts=attempt,
                    finish_reason=finish_reason,
                    response_chars=response_chars,
                    failure_category=category,
                    schema_valid=False,
                    detail=last_error,
                )
                if attempt == 1:
                    messages = [*messages, _repair_message(category, short)]
                    continue
                raise MuseUnavailable(f"Muse returned malformed JSON twice: {last_error}") from exc

            if truncated:
                category = "truncated_length"
                last_error = (
                    f"{category}: response truncated by max_tokens "
                    "(finish_reason=length) though JSON parsed"
                )
                self._log(
                    incident_id=incident_id,
                    status="malformed",
                    latency_ms=(time.monotonic() - started) * 1000.0,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    attempts=attempt,
                    finish_reason=finish_reason,
                    response_chars=response_chars,
                    failure_category=category,
                    schema_valid=False,
                    detail=last_error,
                )
                if attempt == 1:
                    messages = [
                        *messages,
                        _repair_message(category, "response was cut off by max_tokens"),
                    ]
                    continue
                raise MuseUnavailable(f"Muse returned malformed JSON twice: {last_error}")

            self._log(
                incident_id=incident_id,
                status="ok",
                latency_ms=(time.monotonic() - started) * 1000.0,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                attempts=attempt,
                finish_reason=finish_reason,
                response_chars=response_chars,
                schema_valid=True,
                detail="",
            )
            return MuseResult(
                body=body,
                model=self.model,
                latency_ms=(time.monotonic() - started) * 1000.0,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                attempts=attempt,
                finish_reason=finish_reason,
                response_chars=response_chars,
                provider=self.provider,
            )

        raise MuseUnavailable(f"Muse produced no usable reply: {last_error}")
