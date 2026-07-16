import os
import subprocess
import sys
from pathlib import Path

from conductor_core import paths
from conductor_core.config import EngineConfig
from conductor_core.storage import FilesystemArtifactStore


def _clear_data_environment(monkeypatch):
    monkeypatch.delenv("CONDUCTOR_CORE_DATA_DIR", raising=False)
    monkeypatch.delenv("CONDUCTOR_HOME", raising=False)


def test_default_data_directory_uses_home(monkeypatch, tmp_path):
    _clear_data_environment(monkeypatch)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    assert paths.resolve_data_dir() == tmp_path / ".conductor" / "core"
    assert paths.resolve_default_artifact_root() == tmp_path / ".conductor" / "core" / "generations"
    assert EngineConfig.from_defaults().artifact_root == paths.resolve_default_artifact_root()


def test_conductor_home_overrides_suite_root(monkeypatch, tmp_path):
    _clear_data_environment(monkeypatch)
    suite_root = tmp_path / "suite"
    monkeypatch.setenv("CONDUCTOR_HOME", str(suite_root))

    assert paths.resolve_data_dir() == suite_root / "core"


def test_project_data_directory_overrides_conductor_home(monkeypatch, tmp_path):
    project_root = tmp_path / "project"
    monkeypatch.setenv("CONDUCTOR_HOME", str(tmp_path / "suite"))
    monkeypatch.setenv("CONDUCTOR_CORE_DATA_DIR", str(project_root))

    assert paths.resolve_data_dir() == project_root
    assert paths.resolve_default_artifact_root() == project_root / "generations"


def test_explicit_artifact_root_overrides_data_directory_policy(monkeypatch, tmp_path):
    monkeypatch.setenv("CONDUCTOR_CORE_DATA_DIR", str(tmp_path / "project"))
    explicit_root = tmp_path / "portable-generations"

    assert EngineConfig.from_defaults(artifact_root=explicit_root).artifact_root == explicit_root
    assert FilesystemArtifactStore(explicit_root).artifact_root == str(explicit_root)
    assert not explicit_root.exists()


def test_environment_paths_expand_tilde(monkeypatch, tmp_path):
    _clear_data_environment(monkeypatch)
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("CONDUCTOR_HOME", "~/suite")
    assert paths.resolve_data_dir() == tmp_path / "suite" / "core"

    monkeypatch.setenv("CONDUCTOR_CORE_DATA_DIR", "~/project")
    assert paths.resolve_data_dir() == tmp_path / "project"


def test_resolution_is_independent_of_cwd_and_module_location(monkeypatch, tmp_path):
    _clear_data_environment(monkeypatch)
    home = tmp_path / "home"
    cwd = tmp_path / "elsewhere"
    cwd.mkdir()
    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.setattr(
        paths, "__file__", str(tmp_path / "site-packages" / "conductor_core" / "paths.py")
    )
    monkeypatch.chdir(cwd)

    assert paths.resolve_data_dir() == home / ".conductor" / "core"


def test_import_and_resolution_do_not_create_directories(tmp_path):
    suite_root = tmp_path / "isolated-suite"
    env = os.environ.copy()
    env["CONDUCTOR_HOME"] = str(suite_root)
    env.pop("CONDUCTOR_CORE_DATA_DIR", None)
    code = "import conductor_core; print(conductor_core.resolve_default_artifact_root())"

    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=env,
        cwd=tmp_path,
    )

    assert result.stdout.strip() == str(suite_root / "core" / "generations")
    assert not suite_root.exists()


def test_reading_missing_history_does_not_create_directories(tmp_path):
    artifact_root = tmp_path / "missing" / "generations"

    assert FilesystemArtifactStore(artifact_root).load_history() == []
    assert not artifact_root.exists()
