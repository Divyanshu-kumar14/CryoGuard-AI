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


class IncidentImmutableError(RuntimeError):
    """A closed incident file may be re-emitted identically, never rewritten."""


def incident_id(rack_id: str, timestamp: datetime, run_index: int) -> str:
    if not 1 <= run_index <= 99:
        raise ValueError(f"run index must be in 1..99, got {run_index}")
    return f"INC-{rack_id}-{timestamp.strftime('%Y%m%d')}-{run_index:02d}"


def write_incident(report: IncidentReport, out_dir: str | Path) -> Path:
    """Emit once, read-only; an identical re-run is a no-op.

    The demo reruns the same scenario and must stay byte-identical, so identical
    content is accepted. Different content under the same incident id means two
    different incidents are claiming one identity; that is refused, not merged.
    """

    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{report.incident_id}.json"

    payload = (report.model_dump_json(by_alias=True, indent=2) + "\n").encode("utf-8")
    if path.exists():
        if path.read_bytes() != payload:
            raise IncidentImmutableError(
                f"{path} is already closed with different content; refusing to overwrite. "
                "Delete it deliberately if this is a new emission."
            )
        path.chmod(0o444)
        return path

    temporary = path.with_suffix(".json.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)
    path.chmod(0o444)
    return path
