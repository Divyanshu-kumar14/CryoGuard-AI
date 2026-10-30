"""Start the SIMULATED PLC and BMC on ephemeral loopback ports.

The closed-loop demo must exercise real Modbus-TCP and Redfish sockets without
requiring a separate ``make sims`` process. Binding port 0 and reading the bound
port back removes the free-port race that a bind-close-rebind helper has.

Both servers are mocks. Every endpoint returned here is marked ``simulated`` and
the BMC responses carry ``Oem.CryoGuard.simulated = true``.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from typing import Any

import uvicorn

from cryoguard.sims.bmc_server import BmcState, create_app
from cryoguard.sims.plc_server import PlcError, PlcModbusServer, PlcSimulator


@dataclass(frozen=True)
class SimEndpoints:
    plc_host: str
    plc_port: int
    bmc_url: str
    simulated: bool = True

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> SimEndpoints:
        return cls(
            plc_host=str(cfg["plc"]["host"]),
            plc_port=int(cfg["plc"]["port"]),
            bmc_url=str(cfg["bmc"]["base_url"]),
        )


async def _wait_started(server: uvicorn.Server, task: asyncio.Task[None]) -> None:
    for _ in range(500):
        if server.started:
            return
        if task.done():
            task.result()  # re-raise the startup failure
        await asyncio.sleep(0.01)
    raise PlcError("BMC mock did not start within 5 s")


@asynccontextmanager
async def loopback_simulators(
    cfg: dict[str, Any], *, plc_travel_ms: float | None = None
) -> AsyncIterator[SimEndpoints]:
    simulator = (
        PlcSimulator(cfg["plc"], travel_ms=plc_travel_ms) if plc_travel_ms is not None else None
    )
    plc = PlcModbusServer(cfg["plc"], port=0, simulator=simulator)
    await plc.start()

    app = create_app(BmcState(node_id=str(cfg["bmc"]["node_id"])))
    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=0,
        log_level="error",
        access_log=False,
    )
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())

    try:
        await _wait_started(server, task)
        bmc_port = int(server.servers[0].sockets[0].getsockname()[1])
        yield SimEndpoints(
            plc_host="127.0.0.1",
            plc_port=plc.bound_port,
            bmc_url=f"http://127.0.0.1:{bmc_port}",
        )
    finally:
        server.should_exit = True
        try:
            await asyncio.wait_for(task, timeout=5.0)
        except Exception:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        await plc.stop()
