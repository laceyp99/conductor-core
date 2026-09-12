# Summary

<!-- One or two sentences: the problem this solves and how. Link an issue if
relevant. -->

# Changes

<!-- Bullet the user-visible outcomes. If this change warrants a changelog
entry, this is a good draft of it. -->

# Validation

<!-- Commands run, and anything skipped or not run with why. -->

- [ ] `uv run --locked --all-extras ruff format --check .`
- [ ] `uv run --locked --all-extras ruff check .`
- [ ] `uv run --locked --all-extras pytest -q`

# Release metadata

<!-- The branch carries these; release-check validates them. Skipped for
test/ci/docs/style/chore or with the skip-release label. -->

- [ ] `pyproject.toml` version bumped above `main` (`patch`, `minor`, or `major`)
- [ ] `CHANGELOG.md` updated under `[Unreleased]` with a user-facing entry

# Notes

<!-- Risks, follow-ups, reviewer call-outs. Remove if empty. -->
