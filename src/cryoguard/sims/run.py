"""Run the simulated PLC and BMC together: ``make sims``.

Both are mocks. The banner says so, and every BMC response carries
``Oem.CryoGuard.simulated = true``.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import uvicorn

from cryoguard.config import load
from cryoguard.sims.bmc_server import BmcState, create_app
from cryoguard.sims.plc_server import PlcModbusServer

ROOT = Path(__file__).resolve().parents[3]


async def serve(cfg: dict) -> None:
    plc = PlcModbusServer(cfg["plc"])
    await plc.start()
    print(
        f"[SIMULATED] PLC  modbus-tcp {cfg['plc']['host']}:{cfg['plc']['port']} "
        f"unit {cfg['plc']['unit_id']}",
        flush=True,
    )

    app = create_app(BmcState(node_id=cfg["bmc"]["node_id"]))
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host=str(cfg["bmc"]["host"]),
            port=int(cfg["bmc"]["port"]),
            log_level="warning",
        )
    )
    print(
        f"[SIMULATED] BMC  redfish http://{cfg['bmc']['host']}:{cfg['bmc']['port']}/redfish/v1",
        flush=True,
    )
    try:
        await server.serve()
    finally:
        await plc.stop()


def main() -> int:
    cfg = load(ROOT / "configs" / "default.yaml")
    print("CryoGuard simulators: no hardware is connected.", flush=True)
    asyncio.run(serve(cfg))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
