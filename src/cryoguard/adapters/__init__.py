"""Live protocol adapters.

These are the clients the orchestrator uses outside the in-memory unit tests:
Modbus-TCP for the PLC, Redfish for the BMC, NVML for the power clamp, and the
Slurm checkpoint/requeue adapter. The PLC and BMC clients are exercised against
the simulators on real loopback sockets; the Slurm mock is in-process because the
real interface is ``scontrol``, not an HTTP service.

Nothing in this package imports ``cryoguard.report`` or ``openai``; the
architecture test enforces it.
"""
