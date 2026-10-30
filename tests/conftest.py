"""Shared fixtures: the real config tree, never a hand-built dict.

Tests load the same YAML the demo loads, so a config edit that breaks the
contract fails here instead of on stage.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from cryoguard import config

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def cfg() -> dict:
    return config.load(ROOT / "configs" / "default.yaml")


@pytest.fixture(scope="session")
def scenario_path():
    def _load(name: str) -> dict:
        path = ROOT / "configs" / "scenarios" / f"{name}.yaml"
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert isinstance(data, dict), f"{path} must contain a mapping"
        return data

    return _load
