"""Local credential loading (`.env`), presence checks, and secret hygiene."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from cryoguard import env

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def clean_credentials():  # type: ignore[no-untyped-def]
    names = env.CREDENTIALS + env.PROVIDER_SETTINGS
    saved = {name: os.environ.get(name) for name in names}
    for name in names:
        os.environ.pop(name, None)
    yield
    for name, value in saved.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


def test_env_loads_values_from_the_file(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_text(
        "META_API_KEY=local-meta\n"
        "BASETEN_API_KEY=local-baseten\n"
        "BASETEN_BASE_URL=https://example.invalid/v1\n"
        "BASETEN_MODEL=test-model\n"
        "MUSE_API_KEY='local muse'\n"
        "CRYOGUARD_BMC_TOKEN=local-bmc\n",
        encoding="utf-8",
    )

    loaded = env.load_env(path)

    assert loaded == path
    assert os.environ["META_API_KEY"] == "local-meta"
    assert os.environ["BASETEN_API_KEY"] == "local-baseten"
    assert os.environ["BASETEN_BASE_URL"] == "https://example.invalid/v1"
    assert os.environ["BASETEN_MODEL"] == "test-model"
    assert os.environ["MUSE_API_KEY"] == "local muse"
    assert os.environ["CRYOGUARD_BMC_TOKEN"] == "local-bmc"


def test_the_process_environment_wins_over_the_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("META_API_KEY", "from-process")
    path = tmp_path / ".env"
    path.write_text("META_API_KEY=from-file\n", encoding="utf-8")

    env.load_env(path)

    assert os.environ["META_API_KEY"] == "from-process"


def test_a_missing_file_is_not_an_error(tmp_path: Path) -> None:
    assert env.load_env(tmp_path / "missing.env") is None
    assert env.credential_status() == {name: False for name in env.CREDENTIALS}


def test_status_is_presence_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BASETEN_API_KEY", "do-not-print-this")

    status = env.credential_status()

    assert status == {
        "META_API_KEY": False,
        "BASETEN_API_KEY": True,
        "MUSE_API_KEY": False,
        "CRYOGUARD_BMC_TOKEN": False,
    }
    assert all(isinstance(value, bool) for value in status.values())


def test_the_example_pins_the_baseten_endpoint_and_model() -> None:
    text = (ROOT / ".env.example").read_text(encoding="utf-8")

    assert "BASETEN_BASE_URL=https://inference.baseten.co/v1" in text
    assert "BASETEN_MODEL=meta-models/muse-spark-open-1026" in text
    assert "BASETEN_MAX_TOKENS=2400" in text
    assert "MUSE_API_KEY=" in text, "the compatibility alias must remain documented"


def test_the_example_lists_every_credential_with_an_empty_value() -> None:
    text = (ROOT / ".env.example").read_text(encoding="utf-8")

    for name in env.CREDENTIALS:
        assert f"{name}=" in text
        for line in text.splitlines():
            if line.startswith(f"{name}="):
                assert line.split("=", 1)[1].strip() == ""


def test_dot_env_is_gitignored_and_the_example_is_not() -> None:
    lines = [line.strip() for line in (ROOT / ".gitignore").read_text().splitlines()]

    assert ".env" in lines
    assert ".env.*" in lines
    assert "!.env.example" in lines


def test_the_entry_points_load_env_before_constructing_a_client() -> None:
    app = (ROOT / "src" / "cryoguard" / "app.py").read_text(encoding="utf-8")
    server = (ROOT / "src" / "cryoguard" / "server.py").read_text(encoding="utf-8")

    for source in (app, server):
        assert "from cryoguard.env import load_env" in source
        assert "load_env()" in source


def test_the_presence_command_never_prints_a_value(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / ".env"
    path.write_text("META_API_KEY=super-secret-value\n", encoding="utf-8")

    assert env.main(["--file", str(path)]) == 0
    output = capsys.readouterr().out

    assert "META_API_KEY: present" in output
    assert "super-secret-value" not in output


def test_the_presence_command_reports_provider_settings_without_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("BASETEN_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("BASETEN_MODEL", "test-model")
    monkeypatch.setenv("BASETEN_MAX_TOKENS", "2400")
    path = tmp_path / ".env"
    path.write_text("", encoding="utf-8")

    assert env.main(["--file", str(path)]) == 0
    output = capsys.readouterr().out

    assert "BASETEN_BASE_URL: configured" in output
    assert "BASETEN_MODEL: configured" in output
    assert "BASETEN_MAX_TOKENS: configured" in output
    assert "https://example.invalid/v1" not in output
    assert "test-model" not in output
