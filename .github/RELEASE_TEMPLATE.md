# Conductor Core vX.Y.Z

Briefly describe the purpose of this release and how it affects repositories
that install Conductor Core.

## What changed

- Summarize the most important additions, behavior changes, and fixes.
- Link to the matching version of
  [`CHANGELOG.md`](https://github.com/laceyp99/conductor-core/blob/vX.Y.Z/CHANGELOG.md)
  for complete details.

## Compatibility

- **Python:** State the supported Python versions.
- **Public API:** Identify compatible, deprecated, or intentionally changed
  contracts.
- **Consumers:** Note any relevant compatibility expectations for other
  Conductor repositories.

## Installation reference

Pin dependent projects to the release tag and include only the extras they use:

```text
conductor-core[providers] @ git+https://github.com/laceyp99/conductor-core.git@vX.Y.Z
```

## Upgrade and migration

- List required reference updates, configuration changes, and data migrations.
- State explicitly when no migration is required.
- Call out deprecated behavior and its replacement.

## Validation

- List the release checks that passed for the tagged commit.

## Source archives

GitHub automatically provides source-code archives for the tag. Conductor Core
does not attach release-specific wheels, binaries, or other downloads.
