# Post-incident reports

Two sources, one schema, one write.

| Source | When | What the reader gets |
| :-- | :-- | :-- |
| `deterministic_fallback` | always, built first | every sentence from measured values in the incident |
| `llm` | `MUSE_API_KEY` is set and the endpoint answers valid JSON | adds `rootCauseExplanation`, `llmUncertainty`, `llmModel` |

The deterministic report is not a toggle: it is constructed before any network
call, and it is what ships if the model is absent, slow, malformed, or broken
(NFR-SAF-01). One incident is written once; a second emission with different
content is refused by `writer.write_incident`.

## Ordering

```
actions sealed (ActionLog.seal)
  -> deterministic report built from the frozen action snapshot
  -> Muse request, bounded by report.timeout_ms
  -> one incident JSON written, then chmod 0444
```

The request happens **after** the safety sequence finished and the log is
sealed. A 15 s timeout can delay the filing of the report; it cannot delay,
alter, or reorder a valve, a cordon, or a power cut. `pipeline._explained_report`
catches every exception from the client and returns the fallback unchanged.

## What the model may change

Only two fields, both nullable and both additive:

- `rootCauseExplanation` — the operator-facing explanation
- `llmUncertainty` — what was not observed or is weakly supported

The actions, the evidence, `tripReason`, `reportSource` semantics, and the
`recommendedOperatorAction` stay deterministic. The response schema
(`MuseBody`, `extra="forbid"`) has no operator-action field, so the model cannot
even express an actuation instruction. The prompt additionally forbids it.

## Prompts

`configs/prompts.yaml`. The user message is the incident facts as sorted JSON:
id, label, trip reason, fixture provenance, suspected origin, zone, evidence,
and the sealed action list. No free text from the operator or the model enters
the request. Per-label instructions:

- `CONFIRMED_LEAK` — explain the optical evidence and hydraulic corroboration.
- `EMERGENCY_TRIP` — hardware trip, state the recorded reason and value.
- `HIDDEN_HYDRAULIC_EVENT` — begin with "No visual evidence was available."

## Failure matrix

| Failure | Result | Proof line |
| :-- | :-- | :-- |
| `MUSE_API_KEY` absent | fallback, no request | none (nothing was called) |
| request times out | fallback | `status: timeout` |
| HTTP/transport error | fallback, no retry beyond the SDK | `status: error` |
| invalid JSON | one repair retry | `status: malformed`, then `ok` or `malformed` |
| two invalid replies | fallback | two `malformed` lines |
| exception inside the report layer | fallback | depends on where it failed |

The proof log is `report.proof_log` (default
`artifacts/open_weights_proof.jsonl`), one JSON object per request with model id,
provider, latency, token counts, and status. A line is only written when a
request was actually attempted — the repository ships no proof line, because no
key is available in this environment.

## Enabling a live call

```bash
export MUSE_API_KEY=...          # never in the repository
make demo SCENARIO=leak
```

The incident then carries `reportSource: llm` and the proof log has an `ok`
line. If the key is absent the same command prints
`report    deterministic_fallback`.

## Tests

- `tests/test_muse_client.py` — no-key path, success with usage, one repair
  retry, two malformed replies, 500, timeout, prompt constraints, and the
  guarantee that an LLM explanation cannot change the actions or the operator
  instruction.
- `tests/test_report_schema.py` — camelCase, UTC `Z`, round trip.
- `tests/test_action_log.py` — sealed log and the file-level immutability guard.
