# Changelog

All notable changes to Conductor Core are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
while its public API is still in initial development.

## [Unreleased]

### Changed

- `GenerationRequest` now validates every public field at construction: wrong
  Python types raise `TypeError`, accepted types with invalid values raise
  `ValueError`, and boolean fields no longer coerce truthy strings or integers.
- Request effort now defaults to `None`. Disabled thinking resolves directly to
  the selected model's lowest supported effort, while enabled thinking still
  requires a supported effort for models with discrete options.
- Request SoundFont overrides accept `os.PathLike` objects and normalize them
  only when audio is requested. Byte-valued or invalid path protocol results
  preserve generated MIDI and return an audio warning.

Integrations should parse form, command-line, and environment-variable strings
before constructing `GenerationRequest` objects.

## [0.4.0] - 2026-08-02

Version 0.4.0 makes generation inputs, provider failures, model capabilities,
and MIDI export behavior more explicit. It also adopts a locked uv workflow for
reproducible development, testing, and package builds.

### Added

- Validation for generation keys, scales, and finite temperatures from 0.0
  through 2.0.
- MIDI export warnings are now included in `GenerationResult.warnings` for
  notes that are changed or dropped.
- A normalized `rate_limits` object with `RPM`, `TPM`, and `RPD` fields for
  every packaged cloud model, plus validation of packaged model metadata.
- Weekly compatibility testing against the newest supported dependencies on
  Python 3.10 through 3.14.

### Changed

- Note octave and velocity inputs must be integers that can produce valid MIDI
  note-on events; valid low enharmonic spellings remain supported.
- `loop_to_midi()` returns its export warnings while continuing to modify the
  supplied MIDI file in place.
- Anthropic always-on adaptive thinking behavior is selected from model
  metadata instead of a provider-side model-name list.
- Unsupported non-default thinking and effort options emit warnings, while
  invalid effort values for configurable models still fail before a provider
  call.
- Development, CI, dependency locking, and package builds now use uv; the
  project lockfile replaces the former CI-only known-good constraints file.
- Application branding assets were refreshed.

### Fixed

- OpenAI, Anthropic, and Google fail before SDK client construction when their
  API key is missing or blank, consistently raising
  `ProviderAuthenticationError`.
- Invalid or out-of-range notes are dropped during MIDI export instead of being
  encoded as zero-velocity events, and excessive velocities are clamped with a
  warning.
- Message JSON helpers accept paths with or without an existing `.json` suffix
  without appending the extension twice.
- Cloud-model rate limits use consistent field names and conservative baseline
  values instead of mixing incompatible provider-specific representations.

### Upgrade notes

- Review any `GenerationRequest` construction that can receive unvalidated
  user input. Invalid keys, scales, and temperatures now raise `ValueError`
  during request construction.
- Consumers that construct `Note` or `Note_G` directly must provide integer
  octave and velocity values. Velocity must be from 1 through 127, and the
  pitch and octave together must map to MIDI note 0 through 127.
- Code that calls `loop_to_midi()` may inspect its returned warning list;
  callers that previously ignored its `None` return value can continue to
  ignore the result.
- Catch `ProviderAuthenticationError` for missing or blank hosted-provider API
  keys as well as credentials rejected by a provider.
- Update Git references in dependent repositories from `v0.3.0` to `v0.4.0`
  after the release tag is available.

## [0.3.0] - 2026-07-26

Version 0.3.0 improves provider failure handling, request configuration,
generation-history fidelity, dependency testing, and model metadata. It also
completes the removal of the deprecated caller-supplied provider field.

### Added

- A public provider-error hierarchy for authentication, rate-limit, timeout,
  connection, and request failures across OpenAI, Anthropic, Google, and Ollama.
- Configurable provider request timeouts through `EngineConfig.request_timeout`.
- Persisted `use_thinking` and `effort` generation metadata, with optional
  fields so histories written by earlier Core versions remain readable.
- Claude Opus 5 and the current Gemini 3.5/3.6 Flash model family in the
  packaged model registry.
- A known-good dependency constraint set and CI coverage for Python 3.10
  through 3.13, minimal installs, built-wheel consumers, and Windows storage
  security.
- Bundled SoundFont provenance documentation and Conductor Core logo assets.

### Changed

- Runtime dependencies now use compatible version ranges; CI separately tests
  both a pinned known-good set and the latest compatible versions.
- The bundled FM Piano SoundFont filename no longer contains spaces.
- Model metadata no longer advertises retired Anthropic and Gemini models.
- Generation workspaces pin their resolved artifact root for their full
  lifecycle, while module-level storage helpers resolve defaults at call time.
- Provider adapters consistently pass configured timeouts and translate SDK
  failures into Core's provider-independent exceptions.

### Removed

- The deprecated `GenerationRequest.provider` field. Provider identity is
  derived from the selected model and the route actually used.

### Fixed

- Provider client-initialization `TypeError` exceptions are no longer mistaken
  for legacy initializer signatures.
- Ollama connection and HTTP failures are normalized with the other provider
  adapters.
- Generation storage is no longer redirected when relevant environment
  variables change after a workspace is created.
- The built wheel is now exercised from an isolated consumer process in CI.

### Security

- Windows CI now exercises storage confinement behavior, including escaping
  directory-symlink rejection.

### Upgrade notes

- Remove the `provider` argument from every `GenerationRequest`; select a
  supported `model` and let Core derive the provider.
- Update Git references in dependent repositories from `v0.2.0` to `v0.3.0`
  after the release tag is available.
- If a consumer references the bundled SoundFont by filename, update it to
  `FM-Piano1-20190916.sf2`. Consumers using Core's default do not need a change.
- Consumers may catch the new `ProviderError` subclasses instead of individual
  provider SDK exceptions.
- Existing generation histories require no migration.

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

[0.4.0]: https://github.com/laceyp99/conductor-core/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/laceyp99/conductor-core/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/laceyp99/conductor-core/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/laceyp99/conductor-core/tree/v0.1.0
