"""YAML config loading and the hash that makes every run traceable."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

REQUIRED_SECTIONS = (
    "system",
    "camera",
    "perception",
    "sam",
    "telemetry",
    "fusion",
    "decision",
    "orchestrator",
    "thermal_model",
    "nvml",
    "plc",
    "bmc",
    "scheduler",
    "report",
    "demo",
)


class ConfigError(ValueError):
    """Raised when a config file is missing, malformed, or incomplete."""


def load(path: str | Path) -> dict[str, Any]:
    candidate = Path(path)
    if not candidate.is_file():
        raise ConfigError(f"config not found: {candidate}")

    data = yaml.safe_load(candidate.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ConfigError(f"{candidate} must contain a mapping at the top level")

    missing = [section for section in REQUIRED_SECTIONS if section not in data]
    if missing:
        raise ConfigError(f"{candidate} is missing sections: {', '.join(missing)}")
    return data


def get(config: dict[str, Any], dotted: str) -> Any:
    """Read a nested key, failing loudly instead of returning None."""

    node: Any = config
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            raise ConfigError(f"missing config key: {dotted}")
        node = node[part]
    return node


def config_hash(config: dict[str, Any]) -> str:
    """Stable 16-hex digest of the effective configuration."""

    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
