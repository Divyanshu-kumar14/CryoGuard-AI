"""Incident file naming and writing.

The id is deterministic: rack, scenario date, and a run index from config. A
second run of the same scenario overwrites the same file rather than inventing a
number from the wall clock, so the benchmark can diff two runs byte for byte.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from cryoguard.report.schema import IncidentReport

INCIDENT_ID_PATTERN = re.compile(r"^INC-[A-Z0-9-]+-\d{8}-\d{2}$")


def incident_id(rack_id: str, timestamp: datetime, run_index: int) -> str:
    if not 1 <= run_index <= 99:
        raise ValueError(f"run index must be in 1..99, got {run_index}")
    return f"INC-{rack_id}-{timestamp.strftime('%Y%m%d')}-{run_index:02d}"


def write_incident(report: IncidentReport, out_dir: str | Path) -> Path:
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{report.incident_id}.json"

    payload = report.model_dump_json(by_alias=True, indent=2) + "\n"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)
    return path
