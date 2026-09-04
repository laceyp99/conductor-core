# Development

## Set up the environment

Core uses [uv](https://docs.astral.sh/uv/) 0.11.16 or newer. From the repository
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
uv run --locked --all-extras pytest -q
uv build
```

The test suite is deterministic. It does not make live provider calls or
require FluidSynth or FFmpeg.

When intentionally updating dependencies, run `uv lock --upgrade`, review the
lockfile diff, and rerun the checks. Never edit `uv.lock` by hand.

## Preview the documentation

Install development dependencies, then serve the site locally:

```powershell
uv sync
uv run mkdocs serve
```

Build the static documentation and fail on warnings with:

```powershell
uv run mkdocs build --strict
```

## Examples

- [`scripts/generate_midi.py`](https://github.com/laceyp99/conductor-core/blob/main/scripts/generate_midi.py): complete online
  generation workflow.
- [`scripts/inspect_models.py`](https://github.com/laceyp99/conductor-core/blob/main/scripts/inspect_models.py): model and
  capability inspection without a provider call.
- [`scripts/midi_loop_roundtrip.py`](https://github.com/laceyp99/conductor-core/blob/main/scripts/midi_loop_roundtrip.py): offline
  MIDI conversion.

Release history, compatibility notes, and migration guidance live in the
[changelog](https://github.com/laceyp99/conductor-core/blob/main/CHANGELOG.md).
Maintainer release guidance lives in
[`RELEASE_TEMPLATE.md`](https://github.com/laceyp99/conductor-core/blob/main/.github/RELEASE_TEMPLATE.md).
