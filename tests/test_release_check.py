import importlib.util
from itertools import pairwise
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


@pytest.mark.parametrize(
    "suffix",
    [
        "",
        " - not-a-date",
        " - 2026-2-03",
        " - 20261010",
        " - 2026-10-10T12:00:00",
        " - 2026-02-29",
        " - 2026-04-31",
        " - 0000-01-01",
    ],
)
def test_changelog_rejects_invalid_heading_dates(
    release_check, tmp_path, monkeypatch, capsys, suffix
):
    changelog = tmp_path / "CHANGELOG.md"
    text = f"## [0.8.4]{suffix}\n\n- Fixed MIDI export.\n"
    changelog.write_text(text, encoding="utf-8")
    monkeypatch.setattr(release_check, "CHANGELOG_PATH", str(changelog))

    # The gate and release publication use the same date validation.
    with pytest.raises(SystemExit, match="1"):
        release_check.changelog_was_handled("0.8.4")
    with pytest.raises(SystemExit, match="1"):
        release_check.section_body(text, "0.8.4")
    assert "[0.8.4] heading needs a valid YYYY-MM-DD date" in capsys.readouterr().err


@pytest.mark.parametrize("heading_date", ["2026-10-10", "2024-02-29"])
def test_changelog_accepts_valid_heading_dates(
    release_check, tmp_path, monkeypatch, heading_date
):
    changelog = tmp_path / "CHANGELOG.md"
    text = (
        "## [Unreleased]\n\n- Future changes.\n\n"
        f"## [0.8.4] - {heading_date}\n\n- Fixed MIDI export.\n\n"
        "## [0.8.3] - not-a-date\n\n- Older notes.\n"
    )
    changelog.write_text(text, encoding="utf-8")
    monkeypatch.setattr(release_check, "CHANGELOG_PATH", str(changelog))

    release_check.changelog_was_handled("0.8.4")
    assert release_check.section_body(text, "0.8.4").strip() == "- Fixed MIDI export."


# SemVer 2.0.0 section 11.4 example ordering, lowest to highest.
SEMVER_ORDER = [
    "1.0.0-alpha",
    "1.0.0-alpha.1",
    "1.0.0-alpha.beta",
    "1.0.0-beta",
    "1.0.0-beta.2",
    "1.0.0-beta.11",
    "1.0.0-rc.1",
    "1.0.0",
]


@pytest.mark.parametrize(("lower", "higher"), list(pairwise(SEMVER_ORDER)))
def test_greater_than_follows_semver_prerelease_precedence(
    release_check, lower, higher
):
    assert release_check.greater_than(higher, lower)
    assert not release_check.greater_than(lower, higher)
