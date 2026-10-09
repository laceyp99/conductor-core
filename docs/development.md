# Development

## Set up the environment

Core uses [uv](https://docs.astral.sh/uv/) 0.11.16 or newer. 

From the repository
root, install Core, development tools, and all optional provider and playback
dependencies:

```powershell
uv sync --all-extras
```

You do not need to activate the environment. Use `uv sync` if you need only the
base package and development tools.

## Validate changes

Run the narrowest relevant test module while iterating. Before handing off a
change, run the complete project checks:

```powershell
uv run --locked --all-extras ruff format --check .
uv run --locked --all-extras ruff check .
uv run --locked --all-extras ty check
uv run --locked --all-extras pytest -q
uv build --clear
uv run --locked --all-extras python scripts/check_wheel_typing.py
```

The test suite is deterministic. It does not make live provider calls or
require FluidSynth or FFmpeg.

Ty is a development-only dependency, with its exact version resolved in `uv.lock`.
The [Ty configuration](https://docs.astral.sh/ty/reference/configuration/) targets
Python 3.10 and checks `src/`; tests and examples are not part of the required
source baseline. Errors fail the command. Warnings remain visible but do not
fail it. Do not use blanket or file-wide suppressions: an unavoidable suppression
must name its specific rule and include a comment explaining why it is safe.

The consumer command checks the wheel in `dist/` that matches the current project
version; pass a wheel path to check a different build. It creates a temporary
bare-wheel installation outside the checkout and checks `tests/typing/consumer.py` with the locked Ty executable.
It verifies existing and variation APIs, named routing results, progress callbacks,
artifact storage, and the packaged `py.typed` marker, and confirms that an invalid
request is rejected by the checker. It does not call a provider or render audio.
The CI quality/build job runs both the Ty source baseline and the installed-wheel
typing check on pull requests and pushes to `main`.

When intentionally updating dependencies, run `uv lock --upgrade`, review the
lockfile diff, and rerun the checks. Never edit `uv.lock` by hand.

## Release metadata

Every pull request must carry its own release metadata. The required
`release-check` CI job fails a pull request unless it:

- declares a valid Semantic Versioning version in `pyproject.toml`,
- sets that version higher than the version on the base branch, and
- adds a non-empty `## [x.y.z] - YYYY-MM-DD` section to `CHANGELOG.md`
  matching the new version.

The version bump is your choice: use `patch` for backward-compatible fixes,
`minor` for new backward-compatible features, and `major` for breaking changes.
Keep changelog entries user-facing; omit them for test-only, CI, formatting, or
internal refactor changes.

Skip `release-check` when a pull request does not affect releases by adding the
`skip-release` label or by using a Conventional Commit title whose type is
`test`, `ci`, `docs`, `style`, or `chore`. The check only validates; it never
edits your branch.

## Publishing releases

Merging to `main` publishes the release. The `Release` workflow reads the
version from `pyproject.toml`; if no `vX.Y.Z` tag exists yet, it creates the
tag and a GitHub release whose notes are that version's changelog section.
Merges that keep the same version, such as skipped `docs` or `chore` changes,
publish nothing. Consumers can pin the new tag right after the workflow runs.

Release runs execute one at a time, with up to 100 pending runs queued so newer
pushes do not replace waiting releases. HTML comments and subheadings do not
count as user-facing changelog entries.

## Preview the documentation

Install development dependencies, then serve the site locally:

```powershell
uv sync --all-extras
uv run mkdocs serve
```

Build the static documentation and fail on warnings with:

```powershell
uv run mkdocs build --strict
```

## Examples

- [`scripts/generate_midi.py`](https://github.com/laceyp99/conductor-core/blob/main/scripts/generate_midi.py): complete online
  generation workflow.
- [`scripts/generate_variations.py`](https://github.com/laceyp99/conductor-core/blob/main/scripts/generate_variations.py): one-request online
  workflow for generating and inspecting several loop alternatives.
- [`scripts/inspect_models.py`](https://github.com/laceyp99/conductor-core/blob/main/scripts/inspect_models.py): model and
  capability inspection without a provider call.
- [`scripts/midi_loop_roundtrip.py`](https://github.com/laceyp99/conductor-core/blob/main/scripts/midi_loop_roundtrip.py): offline
  MIDI conversion.

Release history, compatibility notes, and migration guidance live in the
[changelog](https://github.com/laceyp99/conductor-core/blob/main/CHANGELOG.md).
Maintainer release guidance lives in
[`RELEASE_TEMPLATE.md`](https://github.com/laceyp99/conductor-core/blob/main/.github/RELEASE_TEMPLATE.md).
