import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def release_check():
    # CI runs this standalone script with the runner's Python 3.11 or newer.
    pytest.importorskip("tomllib", reason="release-check requires Python 3.11+")
    script = Path(__file__).parents[1] / ".github/scripts/release_check.py"
    spec = importlib.util.spec_from_file_location("release_check", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "body",
    [
        "### Added\n\n### Fixed",
        "<!-- Write release notes here. -->",
        "<!--\nWrite release notes here.\n-->",
        "### Fixed\n<!--\n- Hidden placeholder.\n-->\n<!-- Another placeholder. -->",
        "<!--\nUnfinished hidden placeholder.",
    ],
)
def test_changelog_rejects_sections_without_visible_entries(
    release_check, tmp_path, monkeypatch, body
):
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text(f"## [0.8.4] - 2026-10-10\n\n{body}\n", encoding="utf-8")
    monkeypatch.setattr(release_check, "CHANGELOG_PATH", str(changelog))

    with pytest.raises(SystemExit, match="1"):
        release_check.changelog_was_handled("0.8.4")


@pytest.mark.parametrize(
    "body",
    [
        "### Fixed\n\n- Fixed MIDI export.",
        "<!--\nHidden placeholder.\n-->\n### Fixed\n\n- Fixed MIDI export.",
        "<!-- Hidden -->- Fixed MIDI export.<!-- Hidden too -->",
    ],
)
def test_changelog_accepts_visible_entries(release_check, tmp_path, monkeypatch, body):
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text(f"## [0.8.4] - 2026-10-10\n\n{body}\n", encoding="utf-8")
    monkeypatch.setattr(release_check, "CHANGELOG_PATH", str(changelog))

    release_check.changelog_was_handled("0.8.4")
