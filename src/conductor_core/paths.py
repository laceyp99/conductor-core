"""Side-effect-free data path policy for Conductor Core."""

import os
from pathlib import Path

PROJECT_ID = "core"
PROJECT_DATA_ENV = "CONDUCTOR_CORE_DATA_DIR"
SUITE_HOME_ENV = "CONDUCTOR_HOME"


def _environment_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else None


def resolve_conductor_home() -> Path:
    """Return the shared Conductor suite root without creating it."""
    return _environment_path(SUITE_HOME_ENV) or Path.home() / ".conductor"


def resolve_data_dir() -> Path:
    """Return Core's complete data directory without creating it."""
    return _environment_path(PROJECT_DATA_ENV) or resolve_conductor_home() / PROJECT_ID


def resolve_default_artifact_root() -> Path:
    """Return Core's default generation-history root without creating it."""
    return resolve_data_dir() / "generations"
