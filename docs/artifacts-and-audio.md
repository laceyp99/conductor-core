# Artifacts and audio

## Data directory

Core stores durable history under one predictable Conductor suite root:

```text
~/.conductor/
  core/
    generations/
      gen_<id>/
        loop.mid
        loop.mp3          # only when audio rendering succeeds
        messages.json     # when provider messages are available
        metadata.json
```

On Windows, the default is `%USERPROFILE%\.conductor\core`. Selection follows
this precedence:

1. `CONDUCTOR_CORE_DATA_DIR` selects Core's complete project data directory.
2. `CONDUCTOR_HOME` selects the shared suite root; Core appends `core`.
3. Core uses `Path.home() / ".conductor" / "core"`.

Both environment variables support `~` expansion.

```mermaid
flowchart TD
    explicit{Explicit artifact root?}
    core_env{CONDUCTOR_CORE_DATA_DIR set?}
    home_env{CONDUCTOR_HOME set?}
    explicit -->|Yes| chosen[Use explicit root]
    explicit -->|No| core_env
    core_env -->|Yes| core_dir[Use CONDUCTOR_CORE_DATA_DIR]
    core_env -->|No| home_env
    home_env -->|Yes| suite_dir[Use CONDUCTOR_HOME / core]
    home_env -->|No| default_dir[Use ~/.conductor/core]
```

```powershell
$env:CONDUCTOR_HOME = "D:\ConductorData"
$env:CONDUCTOR_CORE_DATA_DIR = "D:\ConductorData\custom-core"
```

An explicit `EngineConfig.artifact_root` or `FilesystemArtifactStore` root
overrides the default. Resolving or importing paths does not create directories;
Core creates the history directory only when writing a generation workspace.
Packaged prompts, metadata, and the bundled SoundFont remain read-only package
resources.

## Retention

New generation metadata uses timezone-aware UTC timestamps, and new
timestamp-based generation IDs use UTC. Batch manifests already use UTC.
History and retention compare timestamps as instants rather than local clock
text. Existing timestamps without an offset remain readable and are interpreted
as local time in the current system timezone, matching their previous retention
behavior; reading history does not rewrite them. Their original timezone and
repeated daylight saving hour cannot be recovered from those legacy records.
Consumers displaying new timestamps can convert them to their preferred timezone.

Core retains the newest 20 generations by default. MIDI, JSON, and especially
MP3 files can still consume substantial space. Configure retention on the
engine or store; use `None` only when the calling application owns disk policy.

```python
from conductor_core import EngineConfig
from conductor_core.storage import FilesystemArtifactStore

config = EngineConfig.from_defaults(max_generations=100)
unlimited_store = FilesystemArtifactStore("my-output", max_generations=None)
```

The limit counts saved generations, not provider requests or disk bytes. Each
variation in a batch counts separately, so a request for four variations needs
`max_generations >= 4` or `None` for unlimited retention. Core rejects a batch
that exceeds the actual store's finite limit before provider work begins.
An allowed variation batch takes priority during its own retention passes;
Core fills remaining capacity with the newest prior history, even if a clock
change makes older work appear newer than the current batch. Ordinary retention
passes keep the newest generations by timestamp. Later requests can prune artifacts from a
previously returned batch. Variation manifests have a separate lifetime and
are not removed by generation retention; see [Variation history](variations.md#variation-history).

## Generation results

| Attribute | Contents |
| --- | --- |
| `generation_id` | Unique filesystem generation identifier. |
| `loop` | Validated provider-independent loop object. |
| `midi_path` | Persisted MIDI path. |
| `audio_path` | Persisted MP3 path when rendering succeeds. |
| `messages` | Provider conversation messages, each a dict with a `role` (`system`, `user`, or `assistant`) and string `content`. |
| `cost` | Provider-reported estimated cost when available. |
| `metadata` | Persisted generation metadata. |
| `warnings` | Non-fatal issues such as skipped audio. |

Use `FilesystemArtifactStore` for custom roots, loading or deleting saved
generations, and updating saved audio metadata.

## Audio previews

Set `render_audio=True` to render an MP3 after MIDI generation. Install the
`playback` extra and place FluidSynth and FFmpeg on `PATH`. With no
`soundfont_path`, Core uses its packaged SoundFont; set the request path or
`EngineConfig.default_soundfont_path` to choose another.

Audio failure does not discard successful MIDI generation. Core returns a
warning and `audio_path=None`. Lower-level discovery and rendering helpers live
in `conductor_core.playback`. Rendered audio is trimmed to the MIDI endpoint;
SoundFont release tails are intentionally excluded so the preview duration
matches the loop boundary.

## Direct MIDI utilities

Consumers can convert existing MIDI to Core's four-bar loop model and write it
back without calling a provider. See
[`scripts/midi_loop_roundtrip.py`](https://github.com/laceyp99/conductor-core/blob/main/scripts/midi_loop_roundtrip.py)
for an
offline example that normalizes note starts and durations to sixteenth-note
integer positions.

Related modules include:

- `conductor_core.models` for loop, bar, note, and timing models;
- `conductor_core.music` for model metadata, prompts, scales, and durations;
- `conductor_core.routing` for lower-level routing, where `generate_midi()`
  returns a `LoopProviderResult` with `loop`, `messages`, `cost`, and
  `provider` fields;
- `conductor_core.storage` for artifacts and history; and
- `conductor_core.playback` for optional audio operations.

Prefer `LoopGenerationEngine` for complete generation workflows so persistence,
cleanup, prompt handling, and provider behavior remain consistent.
