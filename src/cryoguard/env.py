"""Local credential loading for the executable entry points.

The library deliberately does **not** load ``.env``: the test suite and the
benchmark must never pick up a developer's live keys. ``app.py`` and
``server.py`` call :func:`load_env` explicitly before constructing a client.

Rules:

* the process environment wins over ``.env`` (``override=False``);
* ``.env`` is gitignored and must never be committed;
* nothing in this module prints a credential value. ``credential_status``
  returns booleans only, so ``make env-check`` can confirm presence safely.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
CREDENTIALS = ("META_API_KEY", "BASETEN_API_KEY", "MUSE_API_KEY", "CRYOGUARD_BMC_TOKEN")
PROVIDER_SETTINGS = ("BASETEN_BASE_URL", "BASETEN_MODEL")


def load_env(path: str | Path | None = None) -> Path | None:
    """Load ``.env`` into ``os.environ`` without overriding real environment.

    Returns the file that was loaded, or ``None`` when it is absent. A missing
    file is not an error: the fallback report and the cache-only SAM path are
    designed to run without any credential.
    """

    candidate = Path(path) if path is not None else ROOT / ".env"
    if not candidate.is_file():
        return None
    load_dotenv(dotenv_path=candidate, override=False, interpolate=False)
    return candidate


def credential_status() -> dict[str, bool]:
    """Presence only. Never return or log the value itself."""

    return {name: bool(os.environ.get(name)) for name in CREDENTIALS}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Report credential presence without printing any value"
    )
    parser.add_argument("--file", default=None, help="alternate .env path")
    args = parser.parse_args(argv)

    loaded = load_env(args.file)
    print(f"env file: {loaded if loaded is not None else 'not found'}")
    for name, present in credential_status().items():
        print(f"{name}: {'present' if present else 'absent'}")
    for name in PROVIDER_SETTINGS:
        state = "configured" if os.environ.get(name) else "default from configs/default.yaml"
        print(f"{name}: {state}")
    print(
        "note: the process environment overrides .env; live SAM also needs "
        "sam.mode=live, and the report model is optional."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
