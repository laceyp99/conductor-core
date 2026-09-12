#!/usr/bin/env python3
"""Validate pull-request release metadata.

A non-skip pull request must:

1. declare a valid Semantic Versioning version in ``pyproject.toml``,
2. bump that version above the version on the base branch, and
3. update ``CHANGELOG.md`` with an ``[Unreleased]`` entry or a matching
   ``[x.y.z]`` section for the new version.

Pull requests are skipped when they carry the ``skip-release`` label or use a
Conventional Commit title whose type is test, ci, docs, style, or chore. The
script is intentionally stdlib-only so CI can run it without environment setup.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import NoReturn

import tomllib

CHANGELOG_PATH = "CHANGELOG.md"
PYPROJECT_PATH = "pyproject.toml"

SKIP_LABEL = "skip-release"
SKIP_TYPES = {"test", "ci", "docs", "style", "chore"}

CONVENTIONAL_TITLE = re.compile(r"^(?P<type>[a-z]+)(?:\([^)]+\))?!?: ")

# Semantic Versioning 2.0.0, https://semver.org
SEMVER = re.compile(
    r"^(?P<major>0|[1-9]\d*)\.(?P<minor>0|[1-9]\d*)\.(?P<patch>0|[1-9]\d*)"
    r"(?:-(?P<prerelease>(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)"
    r"(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?"
    r"(?:\+(?P<build>[0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$"
)

# A Keep a Changelog version heading, e.g. "## [0.6.0] - 2026-09-11".
HEADING = re.compile(r"^##\s+\[(?P<version>[^\]]+)\](?:\s*-\s*(?P<date>.+))?\s*$")


def fail(message: str) -> NoReturn:
    print(f"release-check: {message}", file=sys.stderr)
    sys.exit(1)


def git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        fail(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout


def git_optional(*args: str) -> str | None:
    """Return git output, or None when the requested object does not exist."""
    result = subprocess.run(
        ["git", *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout


def declared_version(raw: str, source: str) -> str:
    try:
        data = tomllib.loads(raw)
    except tomllib.TOMLDecodeError as error:
        fail(f"could not parse {source}: {error}")
    try:
        return str(data["project"]["version"])
    except KeyError:
        fail(f"{source} does not declare project.version")


def is_skipped() -> str | None:
    labels = json.loads(os.environ.get("PR_LABELS", "[]"))
    if SKIP_LABEL in labels:
        return f"the {SKIP_LABEL!r} label is present"

    title = os.environ.get("PR_TITLE", "")
    match = CONVENTIONAL_TITLE.match(title)
    if match and match.group("type") in SKIP_TYPES:
        return f"the PR title type {match.group('type')!r} does not affect releases"
    return None


def parse_semver(version: str) -> tuple[int, int, int, list[str] | None] | None:
    match = SEMVER.match(version)
    if match is None:
        return None
    prerelease = match.group("prerelease")
    return (
        int(match.group("major")),
        int(match.group("minor")),
        int(match.group("patch")),
        None if prerelease is None else prerelease.split("."),
    )


def greater_than(left: str, right: str) -> bool:
    left_key = parse_semver(left)
    right_key = parse_semver(right)
    if left_key is None or right_key is None:
        return False

    if left_key[:3] != right_key[:3]:
        return left_key[:3] > right_key[:3]

    left_pre, right_pre = left_key[3], right_key[3]
    if left_pre is None or right_pre is None:
        return left_pre is None and right_pre is not None

    for left_part, right_part in zip(left_pre, right_pre, strict=False):
        if left_part == right_part:
            continue
        left_numeric = left_part.isdigit()
        right_numeric = right_part.isdigit()
        if left_numeric and right_numeric:
            return int(left_part) > int(right_part)
        if left_numeric != right_numeric:
            return left_numeric
        return left_part > right_part
    return len(left_pre) > len(right_pre)


def sections(text: str) -> list[tuple[str, int]]:
    """Return (version, start index) for each version heading."""
    found = []
    for index, line in enumerate(text.splitlines()):
        match = HEADING.match(line)
        if match:
            found.append((match.group("version"), index))
    return found


def section_body(text: str, version: str) -> str | None:
    lines = text.splitlines()
    found = sections(text)
    for position, (label, start) in enumerate(found):
        if label != version:
            continue
        end = found[position + 1][1] if position + 1 < len(found) else len(lines)
        return "\n".join(lines[start + 1 : end])
    return None


def has_version_section(text: str, version: str) -> bool:
    return any(label == version for label, _ in sections(text))


def entry_lines(body: str) -> list[str]:
    """Return trimmed, non-empty lines so whitespace-only edits are ignored."""
    return [line.strip() for line in body.splitlines() if line.strip()]


def changelog_was_handled(new_version: str) -> None:
    if not Path(CHANGELOG_PATH).exists():
        fail(f"{CHANGELOG_PATH} does not exist")

    head_text = Path(CHANGELOG_PATH).read_text(encoding="utf-8")

    if has_version_section(head_text, new_version):
        print(
            f"release-check: {CHANGELOG_PATH} has a [{new_version}] section "
            "matching pyproject.toml."
        )
        return

    base_ref = os.environ.get("BASE_REF", "main")
    base_text = git_optional("show", f"origin/{base_ref}:{CHANGELOG_PATH}") or ""
    base_entries = set(entry_lines(section_body(base_text, "Unreleased") or ""))
    head_entries = entry_lines(section_body(head_text, "Unreleased") or "")

    if not head_entries:
        fail(
            f"{CHANGELOG_PATH} needs either an [Unreleased] entry or a "
            f"[{new_version}] section for this change."
        )
    if all(entry in base_entries for entry in head_entries):
        fail(
            f"{CHANGELOG_PATH} [Unreleased] has no new content. Add a "
            f"user-facing entry under [Unreleased] or a [{new_version}] section."
        )

    print(f"release-check: {CHANGELOG_PATH} [Unreleased] section was updated.")


def main() -> None:
    skipped_reason = is_skipped()
    if skipped_reason is not None:
        print(f"release-check: skipped because {skipped_reason}.")
        return

    base_ref = os.environ.get("BASE_REF", "main")
    new_version = declared_version(
        Path(PYPROJECT_PATH).read_text(encoding="utf-8"), PYPROJECT_PATH
    )
    if parse_semver(new_version) is None:
        fail(
            f"pyproject.toml version {new_version!r} is not valid Semantic Versioning."
        )

    base_version = declared_version(
        git("show", f"origin/{base_ref}:{PYPROJECT_PATH}"),
        f"origin/{base_ref}:{PYPROJECT_PATH}",
    )
    if parse_semver(base_version) is None:
        fail(
            f"the {base_version!r} version on {base_ref} is not valid Semantic "
            "Versioning."
        )
    if not greater_than(new_version, base_version):
        fail(
            f"pyproject.toml version {new_version!r} must be greater than the "
            f"{base_version!r} version on {base_ref}."
        )

    changelog_was_handled(new_version)
    print(
        f"release-check: version {base_version} -> {new_version} and "
        f"{CHANGELOG_PATH} are consistent."
    )


if __name__ == "__main__":
    main()
