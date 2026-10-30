"""P7: Muse Spark Open is a post-incident explainer, never a safety dependency."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from cryoguard.config import load
from cryoguard.pipeline import _explained_report
from cryoguard.report.fallback import build_fallback_report
from cryoguard.report.muse_client import (
    MuseClient,
    MuseUnavailable,
    PromptBook,
    build_messages,
    incident_facts,
    load_prompts,
)
from cryoguard.report.schema import ActionModel, EvidenceModel

ROOT = Path(__file__).resolve().parents[1]
GOOD = json.dumps({"root_cause": "Optical growth at QD-04.", "uncertainty": "Mask quality high."})


@pytest.fixture(scope="module")
def cfg() -> dict:
    return load(ROOT / "configs" / "default.yaml")


@pytest.fixture(scope="module")
def prompts() -> PromptBook:
    return load_prompts(ROOT / "configs" / "prompts.yaml")


@pytest.fixture(autouse=True)
def clear_provider_env(monkeypatch) -> None:
    """A developer's .env must not decide what these tests assert."""

    for name in (
        "BASETEN_API_KEY",
        "MUSE_API_KEY",
        "BASETEN_BASE_URL",
        "BASETEN_MODEL",
        "BASETEN_MAX_TOKENS",
    ):
        monkeypatch.delenv(name, raising=False)


def evidence() -> EvidenceModel:
    return EvidenceModel(
        label="CONFIRMED_LEAK",
        zone=0,
        qd_origin="QD-04",
        p_leak=0.85,
        phi_visual=0.63,
        phi_hydraulic=0.41,
        phi_thermal=1.0,
        wetness=1.0,
        area_mm2=739.0,
        growth_mm2_per_s=91.25,
        mask_quality=1.0,
        pressure_psi=32.66,
        pressure_rate_psi_per_s=0.3,
        flow_imbalance_lpm=0.39,
        surface_c=41.61,
        dew_point_c=11.32,
    )


def actions() -> list[ActionModel]:
    return [
        ActionModel(seq=1, t_s=7.6, actor="NVML", command="CLAMP", status="READBACK_VERIFIED"),
        ActionModel(
            seq=2, t_s=7.64, actor="PLC", command="VALVE_CLOSE", status="LIMIT_SWITCH_CLOSED"
        ),
    ]


def reply(content: str = GOOD, *, prompt_tokens: int = 210, completion_tokens: int = 48):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
    )


class FakeCompletions:
    def __init__(self, responses) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        if asyncio.iscoroutine(response):
            return await response
        return response


def fake_client(*responses) -> tuple[SimpleNamespace, FakeCompletions]:
    completions = FakeCompletions(responses)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return client, completions


def test_without_a_key_no_request_is_made(cfg, prompts, tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("MUSE_API_KEY", raising=False)
    proof = tmp_path / "proof.jsonl"
    client = MuseClient(cfg["report"], prompts=prompts, proof_log=proof)

    assert client.available is False
    with pytest.raises(MuseUnavailable, match="BASETEN_API_KEY"):
        asyncio.run(
            client.explain(
                incident_id="INC-RACK-08B-20261030-01",
                label="CONFIRMED_LEAK",
                evidence=evidence(),
                actions=actions(),
                fixture_source="synthetic",
            )
        )
    assert not proof.exists(), "no request was made, so no proof line may exist"


def test_baseten_env_overrides_the_config_and_is_the_primary_key(cfg, prompts, monkeypatch) -> None:
    monkeypatch.setenv("BASETEN_API_KEY", "baseten-key")
    monkeypatch.setenv("MUSE_API_KEY", "legacy-key")
    monkeypatch.setenv("BASETEN_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("BASETEN_MODEL", "test-model")

    client = MuseClient(cfg["report"], prompts=prompts)

    assert client.available is True
    assert client.provider == "baseten"
    assert client.model == "test-model"
    assert client.base_url == "https://example.invalid/v1"
    assert client.client.api_key == "baseten-key"


def test_muse_api_key_remains_a_compatibility_fallback(cfg, prompts, monkeypatch) -> None:
    monkeypatch.setenv("MUSE_API_KEY", "legacy-key")

    client = MuseClient(cfg["report"], prompts=prompts)

    assert client.available is True
    assert client.client.api_key == "legacy-key"
    assert client.model == cfg["report"]["model"]
    assert client.base_url == cfg["report"]["base_url"]


def test_the_env_model_is_used_in_the_request_and_the_proof(
    cfg, prompts, tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("BASETEN_MODEL", "test-model")
    fake, completions = fake_client(reply())
    proof = tmp_path / "proof.jsonl"
    client = MuseClient(cfg["report"], client=fake, prompts=prompts, proof_log=proof)

    asyncio.run(
        client.explain(
            incident_id="INC-RACK-08B-20261030-01",
            label="CONFIRMED_LEAK",
            evidence=evidence(),
            actions=actions(),
            fixture_source="synthetic",
        )
    )

    assert completions.calls[0]["model"] == "test-model"
    assert json.loads(proof.read_text(encoding="utf-8").strip())["model"] == "test-model"


def test_the_max_tokens_override_is_used(cfg, prompts, tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("BASETEN_MAX_TOKENS", "2400")
    fake, completions = fake_client(reply())
    client = MuseClient(
        cfg["report"], client=fake, prompts=prompts, proof_log=tmp_path / "proof.jsonl"
    )

    asyncio.run(
        client.explain(
            incident_id="INC-RACK-08B-20261030-01",
            label="CONFIRMED_LEAK",
            evidence=evidence(),
            actions=actions(),
            fixture_source="synthetic",
        )
    )

    assert client.max_tokens == 2400
    assert completions.calls[0]["max_tokens"] == 2400


def test_a_valid_reply_is_one_request_with_usage_and_latency(cfg, prompts, tmp_path) -> None:
    fake, completions = fake_client(reply())
    proof = tmp_path / "proof.jsonl"
    client = MuseClient(cfg["report"], client=fake, prompts=prompts, proof_log=proof)

    result = asyncio.run(
        client.explain(
            incident_id="INC-RACK-08B-20261030-01",
            label="CONFIRMED_LEAK",
            evidence=evidence(),
            actions=actions(),
            fixture_source="synthetic",
        )
    )

    assert result.body.root_cause.startswith("Optical growth")
    assert result.attempts == 1
    assert result.prompt_tokens == 210
    assert result.completion_tokens == 48
    assert result.latency_ms >= 0.0
    assert completions.calls[0]["model"] == cfg["report"]["model"]
    assert completions.calls[0]["temperature"] == 0.0

    line = json.loads(proof.read_text(encoding="utf-8").strip())
    assert line["status"] == "ok"
    assert line["model"] == cfg["report"]["model"]
    assert line["promptTokens"] == 210
    assert line["incidentId"] == "INC-RACK-08B-20261030-01"


def test_malformed_json_gets_one_repair_and_then_succeeds(cfg, prompts, tmp_path) -> None:
    fake, completions = fake_client(reply("not json at all"), reply())
    client = MuseClient(cfg["report"], client=fake, prompts=prompts, proof_log=tmp_path / "p.jsonl")

    result = asyncio.run(
        client.explain(
            incident_id="INC-RACK-08B-20261030-01",
            label="CONFIRMED_LEAK",
            evidence=evidence(),
            actions=actions(),
            fixture_source="synthetic",
        )
    )

    assert result.attempts == 2
    assert len(completions.calls) == 2
    repair = completions.calls[1]["messages"]
    assert any("failed schema validation" in message["content"] for message in repair)
    lines = (tmp_path / "p.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert [json.loads(line)["status"] for line in lines] == ["malformed", "ok"]


def test_two_malformed_replies_raise_and_the_caller_keeps_the_fallback(
    cfg, prompts, tmp_path
) -> None:
    fake, completions = fake_client(reply("{}"), reply("{}"))
    client = MuseClient(cfg["report"], client=fake, prompts=prompts, proof_log=tmp_path / "p.jsonl")

    with pytest.raises(MuseUnavailable, match="malformed JSON twice"):
        asyncio.run(
            client.explain(
                incident_id="INC-RACK-08B-20261030-01",
                label="CONFIRMED_LEAK",
                evidence=evidence(),
                actions=actions(),
                fixture_source="synthetic",
            )
        )
    assert len(completions.calls) == 2


def test_a_server_error_is_not_retried_and_is_logged(cfg, prompts, tmp_path) -> None:
    fake, completions = fake_client(RuntimeError("HTTP 500"))
    proof = tmp_path / "p.jsonl"
    client = MuseClient(cfg["report"], client=fake, prompts=prompts, proof_log=proof)

    with pytest.raises(MuseUnavailable, match="HTTP 500"):
        asyncio.run(
            client.explain(
                incident_id="INC-RACK-08B-20261030-01",
                label="CONFIRMED_LEAK",
                evidence=evidence(),
                actions=actions(),
                fixture_source="synthetic",
            )
        )

    assert len(completions.calls) == 1
    assert json.loads(proof.read_text().strip())["status"] == "error"


def test_a_hung_endpoint_times_out_and_is_logged(cfg, prompts, tmp_path) -> None:
    async def hang():
        await asyncio.sleep(5.0)
        return reply()

    fake, _ = fake_client(hang())
    section = dict(cfg["report"], timeout_ms=50)
    proof = tmp_path / "p.jsonl"
    client = MuseClient(section, client=fake, prompts=prompts, proof_log=proof)

    started = time.monotonic()
    with pytest.raises(MuseUnavailable, match="timed out"):
        asyncio.run(
            client.explain(
                incident_id="INC-RACK-08B-20261030-01",
                label="CONFIRMED_LEAK",
                evidence=evidence(),
                actions=actions(),
                fixture_source="synthetic",
            )
        )

    assert time.monotonic() - started < 2.0
    assert json.loads(proof.read_text().strip())["status"] == "timeout"


def test_the_r4_prompt_requires_no_visual_evidence(prompts) -> None:
    facts = incident_facts(
        incident_id="INC-RACK-08B-20261030-04",
        label="HIDDEN_HYDRAULIC_EVENT",
        evidence=evidence(),
        actions=[],
        fixture_source="synthetic",
        trip_reason=None,
    )
    messages = build_messages(prompts, label="HIDDEN_HYDRAULIC_EVENT", facts=facts)

    assert "No visual evidence was available." in messages[1]["content"]


def test_the_system_prompt_forbids_actuation_and_invention(prompts) -> None:
    system = prompts.system
    assert "Never propose, describe, or recommend machine actuation" in system
    assert "Do not invent measurements" in system
    assert "SIMULATED" in system


def test_the_prompt_facts_are_only_the_incident_numbers(prompts) -> None:
    facts = incident_facts(
        incident_id="INC-RACK-08B-20261030-01",
        label="CONFIRMED_LEAK",
        evidence=evidence(),
        actions=actions(),
        fixture_source="synthetic",
        trip_reason=None,
    )
    messages = build_messages(prompts, label="CONFIRMED_LEAK", facts=facts)
    payload = json.loads(messages[1]["content"].split("Incident JSON:\n", 1)[1])

    assert payload["evidence"]["pressurePsi"] == pytest.approx(32.66)
    assert payload["simulated"] is True
    assert payload["fixtureSource"] == "synthetic"
    assert [action["command"] for action in payload["deterministicActionsTaken"]] == [
        "CLAMP",
        "VALVE_CLOSE",
    ]
    assert set(payload) == {
        "incidentId",
        "label",
        "tripReason",
        "fixtureSource",
        "simulated",
        "suspectedOrigin",
        "zone",
        "evidence",
        "deterministicActionsTaken",
    }


def _fallback_report():
    from datetime import UTC, datetime

    return build_fallback_report(
        evidence=evidence(),
        actions=actions(),
        incident_time=datetime(2026, 10, 30, 0, 0, 7, 800_000, tzinfo=UTC),
        rack_id="RACK-08B",
        run_index=1,
        config_hash="deadbeefdeadbeef",
        clip_id="slow_leak_qd4",
        frame_range=(0, 10),
        fixture_source="synthetic",
    )


def test_an_llm_explanation_never_changes_actions_or_the_operator_instruction(
    cfg, monkeypatch, tmp_path
) -> None:
    fake, _ = fake_client(reply())

    def factory(section):
        return MuseClient(
            section,
            client=fake,
            prompts=load_prompts(ROOT / "configs" / "prompts.yaml"),
            proof_log=tmp_path / "proof.jsonl",
        )

    monkeypatch.setattr("cryoguard.pipeline.MuseClient", factory)
    original = _fallback_report()

    explained = asyncio.run(
        _explained_report(
            report=original,
            cfg=cfg,
            evidence=evidence(),
            actions=actions(),
            label="CONFIRMED_LEAK",
            fixture_source="synthetic",
            trip_reason=None,
        )
    )

    assert explained.report_source == "llm"
    assert explained.root_cause_explanation == "Optical growth at QD-04."
    assert explained.llm_model == cfg["report"]["model"]
    assert explained.recommended_operator_action == original.recommended_operator_action
    assert explained.deterministic_actions_taken == original.deterministic_actions_taken
    assert explained.physical_evidence == original.physical_evidence


def test_a_broken_report_service_returns_the_fallback(cfg, monkeypatch) -> None:
    class Broken:
        available = True

        async def explain(self, **kwargs):
            raise RuntimeError("bug in the report service")

    monkeypatch.setattr("cryoguard.pipeline.MuseClient", lambda section: Broken())
    original = _fallback_report()

    result = asyncio.run(
        _explained_report(
            report=original,
            cfg=cfg,
            evidence=evidence(),
            actions=actions(),
            label="CONFIRMED_LEAK",
            fixture_source="synthetic",
            trip_reason=None,
        )
    )

    assert result is original
    assert result.report_source == "deterministic_fallback"


def test_a_broken_client_constructor_returns_the_fallback(cfg, monkeypatch) -> None:
    def broken(_section):
        raise RuntimeError("bad provider configuration")

    monkeypatch.setattr("cryoguard.pipeline.MuseClient", broken)
    original = _fallback_report()

    result = asyncio.run(
        _explained_report(
            report=original,
            cfg=cfg,
            evidence=evidence(),
            actions=actions(),
            label="CONFIRMED_LEAK",
            fixture_source="synthetic",
            trip_reason=None,
        )
    )

    assert result is original
    assert result.report_source == "deterministic_fallback"


# --- Structured-output reliability: honest metadata + compact retry ---


def reply_with_finish(
    content: str = GOOD,
    *,
    finish_reason: str | None = "stop",
    prompt_tokens: int = 210,
    completion_tokens: int = 48,
):
    choice = SimpleNamespace(
        message=SimpleNamespace(content=content),
        **({"finish_reason": finish_reason} if finish_reason is not None else {}),
    )
    return SimpleNamespace(
        choices=[choice],
        usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
    )


def run_explain(cfg, prompts, fake, proof) -> object:
    client = MuseClient(cfg["report"], client=fake, prompts=prompts, proof_log=proof)
    return asyncio.run(
        client.explain(
            incident_id="INC-RACK-08B-20261030-01",
            label="CONFIRMED_LEAK",
            evidence=evidence(),
            actions=actions(),
            fixture_source="synthetic",
        )
    )


def test_ok_result_carries_honest_metadata(cfg, prompts, tmp_path) -> None:
    from cryoguard.report.muse_client import health_summary

    fake, _ = fake_client(reply_with_finish(GOOD, finish_reason="stop"))
    proof = tmp_path / "p.jsonl"
    result = run_explain(cfg, prompts, fake, proof)

    assert result.finish_reason == "stop"
    assert result.response_chars == len(GOOD)
    assert result.provider == "baseten"
    assert result.model == cfg["report"]["model"]
    line = json.loads(proof.read_text(encoding="utf-8").strip())
    assert line["status"] == "ok"
    assert line["finishReason"] == "stop"
    assert line["responseChars"] == len(GOOD)
    assert line["schemaValid"] is True
    assert line["provider"] == "baseten"
    assert line["model"] == cfg["report"]["model"]
    assert "failureCategory" not in line
    assert "detail" not in line
    summary = health_summary(proof_log=proof)
    assert summary["requests"] == 1
    assert summary["attempts"] == 1
    assert summary["valid_rate"] == 1.0


def test_repair_is_compact_with_no_assistant_echo(cfg, prompts, tmp_path) -> None:
    fake, completions = fake_client(reply_with_finish("not json"), reply_with_finish())
    proof = tmp_path / "p.jsonl"
    run_explain(cfg, prompts, fake, proof)

    assert len(completions.calls) == 2
    repair_messages = completions.calls[1]["messages"]
    # Original 2 messages plus one compact user repair (no assistant echo).
    assert len(repair_messages) == 3
    assert [m["role"] for m in repair_messages] == ["system", "user", "user"]
    assert "failed schema validation" in repair_messages[2]["content"]
    assert "Reply JSON only" in repair_messages[2]["content"]
    assert "not json" not in repair_messages[2]["content"]
    lines = [json.loads(line) for line in proof.read_text(encoding="utf-8").splitlines()]
    assert lines[0]["status"] == "malformed"
    assert lines[0]["failureCategory"] == "invalid_json"
    assert lines[0]["schemaValid"] is False
    assert lines[0]["finishReason"] == "stop"
    assert lines[0]["responseChars"] == len("not json")


def test_empty_response_is_categorised(cfg, prompts, tmp_path) -> None:
    fake, _ = fake_client(reply_with_finish(""), reply_with_finish("   "))
    proof = tmp_path / "p.jsonl"
    client = MuseClient(cfg["report"], client=fake, prompts=prompts, proof_log=proof)
    with pytest.raises(MuseUnavailable, match="malformed JSON twice"):
        asyncio.run(
            client.explain(
                incident_id="INC-RACK-08B-20261030-01",
                label="CONFIRMED_LEAK",
                evidence=evidence(),
                actions=actions(),
                fixture_source="synthetic",
            )
        )
    lines = [json.loads(line) for line in proof.read_text(encoding="utf-8").splitlines()]
    assert [line["failureCategory"] for line in lines] == ["empty", "empty"]


def test_truncated_length_is_detected_and_logged(cfg, prompts, tmp_path) -> None:
    fake, completions = fake_client(
        reply_with_finish("", finish_reason="length"),
        reply_with_finish(GOOD, finish_reason="stop"),
    )
    proof = tmp_path / "p.jsonl"
    result = run_explain(cfg, prompts, fake, proof)

    assert result.attempts == 2
    assert result.finish_reason == "stop"
    lines = [json.loads(line) for line in proof.read_text(encoding="utf-8").splitlines()]
    assert lines[0]["failureCategory"] == "truncated_length"
    assert lines[0]["finishReason"] == "length"
    assert lines[1]["status"] == "ok"
    # Compact retry even for truncation: no assistant echo.
    assert len(completions.calls[1]["messages"]) == 3


def test_truncated_valid_json_still_retries(cfg, prompts, tmp_path) -> None:
    fake, _ = fake_client(
        reply_with_finish(GOOD, finish_reason="length"),
        reply_with_finish(GOOD, finish_reason="length"),
    )
    proof = tmp_path / "p.jsonl"
    client = MuseClient(cfg["report"], client=fake, prompts=prompts, proof_log=proof)
    with pytest.raises(MuseUnavailable, match="malformed JSON twice"):
        asyncio.run(
            client.explain(
                incident_id="INC-RACK-08B-20261030-01",
                label="CONFIRMED_LEAK",
                evidence=evidence(),
                actions=actions(),
                fixture_source="synthetic",
            )
        )
    lines = [json.loads(line) for line in proof.read_text(encoding="utf-8").splitlines()]
    assert all(line["failureCategory"] == "truncated_length" for line in lines)


def test_extra_keys_missing_fields_and_empty_strings(cfg, prompts, tmp_path) -> None:
    cases = [
        (
            json.dumps({"root_cause": "a", "uncertainty": "b", "extra": "c"}),
            "extra_fields",
        ),
        (json.dumps({"root_cause": "only one"}), "missing_fields"),
        (json.dumps({"root_cause": "", "uncertainty": "b"}), "empty_strings"),
        (json.dumps({"root_cause": "   ", "uncertainty": "b"}), "empty_strings"),
    ]
    for payload, category in cases:
        fake, _ = fake_client(reply_with_finish(payload), reply_with_finish(payload))
        proof = tmp_path / f"{category}.jsonl"
        client = MuseClient(cfg["report"], client=fake, prompts=prompts, proof_log=proof)
        with pytest.raises(MuseUnavailable, match="malformed JSON twice"):
            asyncio.run(
                client.explain(
                    incident_id="INC-RACK-08B-20261030-01",
                    label="CONFIRMED_LEAK",
                    evidence=evidence(),
                    actions=actions(),
                    fixture_source="synthetic",
                )
            )
        lines = [json.loads(line) for line in proof.read_text(encoding="utf-8").splitlines()]
        assert lines[0]["failureCategory"] == category, payload


def test_too_long_and_unsupported_assertion(cfg, prompts, tmp_path) -> None:
    long_payload = json.dumps({"root_cause": "x" * 2001, "uncertainty": "ok"})
    fake, _ = fake_client(reply_with_finish(long_payload), reply_with_finish(long_payload))
    proof = tmp_path / "long.jsonl"
    client = MuseClient(cfg["report"], client=fake, prompts=prompts, proof_log=proof)
    with pytest.raises(MuseUnavailable, match="malformed JSON twice"):
        asyncio.run(
            client.explain(
                incident_id="INC-RACK-08B-20261030-01",
                label="CONFIRMED_LEAK",
                evidence=evidence(),
                actions=actions(),
                fixture_source="synthetic",
            )
        )
    assert json.loads(proof.read_text().splitlines()[0])["failureCategory"] == "too_long"

    bad = json.dumps({"root_cause": "Please open the valve now.", "uncertainty": "ok"})
    fake2, _ = fake_client(reply_with_finish(bad), reply_with_finish(bad))
    proof2 = tmp_path / "blocked.jsonl"
    client2 = MuseClient(cfg["report"], client=fake2, prompts=prompts, proof_log=proof2)
    with pytest.raises(MuseUnavailable, match="malformed JSON twice"):
        asyncio.run(
            client2.explain(
                incident_id="INC-RACK-08B-20261030-01",
                label="CONFIRMED_LEAK",
                evidence=evidence(),
                actions=actions(),
                fixture_source="synthetic",
            )
        )
    assert (
        json.loads(proof2.read_text().splitlines()[0])["failureCategory"] == "unsupported_assertion"
    )


def test_timeout_and_http_error_categories(cfg, prompts, tmp_path) -> None:
    async def hang():
        await asyncio.sleep(5.0)
        return reply_with_finish()

    fake, _ = fake_client(hang())
    section = dict(cfg["report"], timeout_ms=50)
    proof = tmp_path / "t.jsonl"
    client = MuseClient(section, client=fake, prompts=prompts, proof_log=proof)
    with pytest.raises(MuseUnavailable, match="timed out"):
        asyncio.run(
            client.explain(
                incident_id="INC-RACK-08B-20261030-01",
                label="CONFIRMED_LEAK",
                evidence=evidence(),
                actions=actions(),
                fixture_source="synthetic",
            )
        )
    line = json.loads(proof.read_text().strip())
    assert line["status"] == "timeout"
    assert line["failureCategory"] == "timeout"

    fake2, _ = fake_client(RuntimeError("HTTP 500"))
    proof2 = tmp_path / "e.jsonl"
    client2 = MuseClient(cfg["report"], client=fake2, prompts=prompts, proof_log=proof2)
    with pytest.raises(MuseUnavailable, match="HTTP 500"):
        asyncio.run(
            client2.explain(
                incident_id="INC-RACK-08B-20261030-01",
                label="CONFIRMED_LEAK",
                evidence=evidence(),
                actions=actions(),
                fixture_source="synthetic",
            )
        )
    line2 = json.loads(proof2.read_text().strip())
    assert line2["status"] == "error"
    assert line2["failureCategory"] == "http_error"


def test_exhausted_retries_keep_two_malformed_lines(cfg, prompts, tmp_path) -> None:
    fake, completions = fake_client(reply_with_finish("{}"), reply_with_finish("{}"))
    proof = tmp_path / "p.jsonl"
    client = MuseClient(cfg["report"], client=fake, prompts=prompts, proof_log=proof)
    with pytest.raises(MuseUnavailable, match="malformed JSON twice"):
        asyncio.run(
            client.explain(
                incident_id="INC-RACK-08B-20261030-01",
                label="CONFIRMED_LEAK",
                evidence=evidence(),
                actions=actions(),
                fixture_source="synthetic",
            )
        )
    assert len(completions.calls) == 2
    lines = [json.loads(line) for line in proof.read_text(encoding="utf-8").splitlines()]
    assert [line["status"] for line in lines] == ["malformed", "malformed"]
    assert lines[0]["failureCategory"] == "missing_fields"


def test_health_summary_separates_attempts_from_requests(tmp_path) -> None:
    from cryoguard.report.muse_client import health_summary

    entries = [
        {
            "incidentId": "A",
            "status": "malformed",
            "latencyMs": 100.0,
            "promptTokens": 10,
            "completionTokens": 5,
            "failureCategory": "invalid_json",
        },
        {
            "incidentId": "A",
            "status": "ok",
            "latencyMs": 200.0,
            "promptTokens": 10,
            "completionTokens": 6,
            "finishReason": "stop",
        },
        {
            "incidentId": "B",
            "status": "malformed",
            "latencyMs": 300.0,
            "promptTokens": 20,
            "completionTokens": 600,
            "failureCategory": "truncated_length",
            "finishReason": "length",
        },
        {
            "incidentId": "B",
            "status": "malformed",
            "latencyMs": 400.0,
            "promptTokens": 20,
            "completionTokens": 600,
            "failureCategory": "truncated_length",
            "finishReason": "length",
        },
    ]
    summary = health_summary(entries)
    # Two requests (A, B), four attempts: retries are not new incidents.
    assert summary["requests"] == 2
    assert summary["attempts"] == 4
    assert summary["ok_requests"] == 1
    assert summary["failed_requests"] == 1
    assert summary["valid_rate"] == 0.5
    assert summary["malformed_rate"] == pytest.approx(0.75)
    assert summary["retry_rate"] == 1.0
    assert summary["median_latency_ms"] == pytest.approx(300.0)
    assert summary["p95_latency_ms"] is not None
    # Terminal attempts only: A-ok (10/6) and B-second (20/600).
    assert summary["avg_prompt_tokens"] == pytest.approx(15.0)
    assert summary["avg_completion_tokens"] == pytest.approx(303.0)
    assert summary["truncated_count"] == 2

    path = tmp_path / "proof.jsonl"
    path.write_text("\n".join(json.dumps(e) for e in entries) + "\n", encoding="utf-8")
    from_file = health_summary(proof_log=path)
    assert from_file["requests"] == 2
    assert health_summary(path)["attempts"] == 4


def test_proof_log_never_contains_secrets_or_full_responses(cfg, prompts, tmp_path) -> None:
    secret_body = json.dumps(
        {"root_cause": "a", "uncertainty": "b", "secret": "sk-secret-should-not-log"}
    )
    fake, _ = fake_client(reply_with_finish(secret_body), reply_with_finish(secret_body))
    proof = tmp_path / "p.jsonl"
    client = MuseClient(cfg["report"], client=fake, prompts=prompts, proof_log=proof)
    with pytest.raises(MuseUnavailable):
        asyncio.run(
            client.explain(
                incident_id="INC-RACK-08B-20261030-01",
                label="CONFIRMED_LEAK",
                evidence=evidence(),
                actions=actions(),
                fixture_source="synthetic",
            )
        )
    text = proof.read_text(encoding="utf-8")
    assert "sk-secret" not in text
    assert "MUSE_API_KEY" not in text and "BASETEN_API_KEY" not in text
