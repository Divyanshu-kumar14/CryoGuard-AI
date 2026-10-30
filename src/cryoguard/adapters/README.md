# Protocol adapters

The orchestrator never talks to a vendor SDK directly. Each port has one client
here, and the FSM's ordering guarantees are enforced above this layer.

| Port | Client | Transport | Target |
| :-- | :-- | :-- | :-- |
| NVML | `nvml.RealNvml` / `nvml.FakeNvml` | local library | GPU power limit |
| PLC | `plc_client.PlcClient` | Modbus-TCP | `sims/plc_server.py`, port 5020 by default |
| BMC | `redfish_client.RedfishClient` | HTTP (Redfish) | `sims/bmc_server.py`, port 8443 by default |
| Scheduler | `scheduler.SchedulerAdapter` | in-process | `sims/slurm_mock.py` |

The PLC and BMC clients are exercised against the mocks on real loopback
sockets. The scheduler mock is in-process because the real Slurm interface is
`scontrol`, not an HTTP service. The NVML fake is used unless `nvml.backend` is
`real`; `pynvml` is imported lazily and the real backend fails safe when it is
missing or when the readback is outside `orchestrator.clamp_tolerance_w`.

Every simulated value is labelled. BMC responses carry
`Oem.CryoGuard.simulated = true`, checkpoint manifests carry
`simulated: true`, and the action log says `(SIMULATED BMC)` / `(mock)`.

## Modes

`orchestrator.protocol_mode` (or `--protocols`):

| Mode | Behaviour |
| :-- | :-- |
| `loopback` | Starts fresh PLC and BMC mocks on ephemeral ports for the run. `make demo` uses this, so the demo is self-contained. |
| `external` | Connects to `plc.host:plc.port` and `bmc.base_url`. Start them first with `make sims`. |
| `memory` | No sockets; the in-memory ports in `orchestrator/adapters.py`. Used by unit tests and offline CI. |

`make demo --protocols external` is not a Make target flag; run
`make sims` in one terminal and
`PYTHONPATH=src .venv/bin/python -m cryoguard.app --protocols external` in another.

## PLC wire map

Addresses are 0-based on the wire and match `sims/plc_server.py`.

| Object | Address | Meaning |
| :-- | :-- | :-- |
| Coil 0001 | 0 | valve solenoid; writing `False` commands the spring-return close |
| Coil 0002 | 1 | vacuum drain |
| Coil 0003 | 2 | `VALVE_FAULT` alarm, read-only output |
| DI 1001 | 0 | limit switch closed |
| DI 1002 | 1 | line depressurized |
| HR 4001 | 0 | pressure, int16 psi × 10 |
| HR 4002 | 1 | flow, uint16 L/min × 100 |

Travel is measured from the coil write, not from process start. The bands are
`<= 1200 ms` `VERIFIED`, `<= 1500 ms` `VERIFIED_SLOW`, above that `FAULT`
(D-15). A transport failure is returned as `FAULT`; the FSM then cuts power
rather than assuming the coolant stopped.

## Ordering and fault paths

The confirmed-leak sequence is:

```
checkpoint.start (parallel, before the clamp)
  -> CLAMP (readback verified)
  -> VALVE_CLOSE (limit switch)
  -> SCHEDULER_DRAIN (reversible cordon, before power-off)
  -> CHECKPOINT_VERIFY (SHA-256)
  -> FORCE_OFF (Redfish)
  -> REQUEUE (standby node, verified step)
```

- Clamp readback fails: no valve command, cordon, `FORCE_OFF`, exit
  `HARD_TRIP`; exit `FAILED_SAFE` if the power cut is also rejected.
- Valve `FAULT`: cordon, `FORCE_OFF`, exit `HARD_TRIP`; exit `FAILED_SAFE` if
  the power cut is also rejected (T-FSM-04). Isolation is never claimed.
- Valve `VERIFIED_SLOW`: proceed and record `LIMIT_SWITCH_CLOSED_SLOW`.
- A BMC rejection after isolation: `FAILED_SAFE`, never `REQUEUED`.
- A failed scheduler cordon is recorded as `FAILED` but never delays the power
  cut. The cordon is reversible; the cut is not optional.
- A corrupt checkpoint: clamp and valve still run, the cordon runs, the
  checkpoint is recorded as `FAILED`, and the sequence exits `FAILED_SAFE`
  without a power-off or requeue (C-04).

## Checkpoints

`CheckpointAdapter` selects the newest manifest whose SHA-256 and size both
verify. If the newest is corrupt it resumes N−1 and the reason string names the
rejected manifest. If every manifest is corrupt it fails closed: the orchestrator
still clamps and closes the valve, records `CHECKPOINT_VERIFY` as `FAILED`, does
not power the node off, and exits `FAILED_SAFE`. A bad checkpoint must never
leave coolant flowing. When the demo directory is empty the adapter writes a
deterministic SIMULATED step (default 4200) so `make demo` remains
self-contained.

The scenario clock advances by the **configured nominal** valve travel, not the
measured wall time. The measured travel still selects the VERIFIED / SLOW /
FAULT band, but it does not enter the incident JSON; this keeps two runs of the
same clip byte-identical (T-DET-01).

## Tests

- `tests/test_adapters_plc.py` — register decode and the 950 / 1300 / 1600 ms
  bands over a real socket.
- `tests/test_adapters_redfish.py` — 503→202 retry, 409 already-off, 400
  no-retry, and ForceOff against the live mock.
- `tests/test_checkpoint_adapter.py` — N−1 fallback, all-corrupt fail-closed,
  and requeue on the standby node.
- `tests/test_closed_loop.py` — the full sequence, clamp-failure HARD_TRIP,
  valve-fault HARD_TRIP, and FAILED_SAFE when the power cut also fails.
- `tests/test_architecture.py` — `adapters/` cannot import `report/` or
  `openai`.

## Known limits

- `RealNvml` is not exercised in CI; this host has no NVIDIA device and
  `pynvml` is not a required dependency.
- The BMC mock has no TLS or authentication. `bmc.token_env` reads an optional
  token from the environment; no secret is stored in the repository.
- The R1 emergency path and the independent thermal watchdog (P6-T07) are not
  wired yet. The fault paths above are R1b-style hard trips inside the
  confirmed-leak sequence.
