# Changelog

All notable changes to Conductor Core are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
while its public API is still in initial development.

## [0.2.0] - 2026-07-19

Version 0.2.0 hardens and extends the reusable Core library now that other
Conductor repositories install and depend on it directly. It focuses on safer
artifact persistence, more reliable provider routing, and stricter musical data
handling without adding application or evaluation dependencies to Core.

### Added

- Shared Conductor data-directory resolution through `CONDUCTOR_HOME` and
  Core-specific overrides through `CONDUCTOR_CORE_DATA_DIR`.
- Public helpers for resolving the Conductor home, Core data directory, and
  default artifact root.
- Configurable generation-history retention through `max_generations`, including
  unlimited retention with `None`.
- Cross-bar note durations of up to 64 sixteenth notes, with validation against
  the four-bar loop boundary.
- Consumer examples for generation, model inspection, MIDI round trips, and
  copied-history verification.
- Regression coverage for routing, MIDI conversion, storage boundaries, audio
  failures, public configuration, and example scripts.

### Changed

- The default generation-history location moved from a project-local
  `generations/` directory to `~/.conductor/core/generations/`.
- Provider identity is derived from the route actually used for the selected
  model instead of caller-supplied request metadata.
- Hosted model metadata is evaluated before Ollama discovery, avoiding an
  unnecessary local-service dependency for hosted routes.
- Model effort values are validated against packaged capability metadata.
- MIDI import and export use exact PPQ calculations and explicitly reject
  unsupported timing divisions instead of silently approximating them.
- Copied generation histories reconstruct artifact paths under their current
  store root rather than trusting paths persisted on another machine.

### Deprecated

- `GenerationRequest.provider` is ignored and retained temporarily for source
  compatibility. Consumers should select a supported model and allow Core to
  derive its provider.

### Fixed

- Sustained notes now survive MIDI conversion across bar boundaries and are
  clipped safely at the end of the four-bar loop.
- Enharmonic spellings such as C-flat and B-sharp now preserve octave
  boundaries correctly.
- Low-PPQ MIDI imports no longer divide by a rounded-down zero tick interval.
- Google responses tolerate missing usage-token counts, and Anthropic responses
  preserve unknown generation costs.
- Successful audio metadata records the resolved SoundFont, while skipped or
  failed updates preserve existing metadata appropriately.
- Failed MP3 renders no longer leave partial artifacts in generation history.

### Security

- Artifact operations validate generation IDs and confine workspaces to the
  configured store root.
- History loading rebinds persisted paths to validated local artifacts and
  rejects mismatched metadata, symbolic links, reparse points, and hard links.
- Metadata and copied audio files are replaced atomically to reduce partial
  writes and unsafe destination handling.

### Upgrade notes

- Existing project-local `generations/` directories are not moved automatically.
  Pass `artifact_root="generations"` to preserve that layout, or copy reviewed
  history into the new data directory explicitly.
- Remove `GenerationRequest.provider` from new integrations. Its value no longer
  influences routing.
- Check effort values against model metadata before submitting requests; invalid
  values now raise `ValueError`.
- Consumers that construct timing models directly should accept the expanded
  duration enums and the stricter four-bar boundary validation.
- Update Git references in dependent repositories from `v0.1.0` to `v0.2.0`
  after the release tag is available.

## [0.1.0] - 2026-07-11

Version 0.1.0 was primarily the transition point from LoopGPT into the broader
Conductor project suite. It established `conductor-core` as the installable,
UI-independent library for generation contracts, provider routing, music
models, MIDI conversion, artifact history, and optional playback helpers so the
other Conductor repositories could build on a shared engine.

### Added

- The initial `conductor-core` Python package and public generation API.
- Provider adapters and model metadata for OpenAI, Anthropic, Google, and
  Ollama.
- Validated four-bar music models and MIDI conversion utilities.
- Filesystem generation artifacts, history metadata, and optional audio
  rendering.
- Deterministic tests and package-boundary checks suitable for reuse outside the
  original LoopGPT application.

[0.2.0]: https://github.com/laceyp99/conductor-core/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/laceyp99/conductor-core/tree/v0.1.0
