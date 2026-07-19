# Conductor Core

`conductor-core` is the reusable prompt-to-MIDI engine behind the Conductor
applications. It can be embedded in a CLI, notebook, backend service, test
harness, or another UI without importing Gradio, Dash, Plotly, or the evaluation
package.

Core owns:

- provider routing for OpenAI, Anthropic, Google, and Ollama
- validated four-bar loop models and provider response parsing
- prompt assembly and model capability metadata
- loop-to-MIDI and MIDI-to-loop conversion
- generation workspaces, messages, metadata, and history persistence
- optional SoundFont discovery and MIDI-to-audio rendering
- structured generation results and progress events

## Installation

From the `conductor-core` project directory on Windows:

```
py -3.12 -m venv .venv
.\.venv\Scripts\activate
pip install --upgrade pip
pip install -e .
```

The base install supports deterministic music models and MIDI operations. Add
only the capabilities your consumer needs:

| Extra | Adds | Example use |
|---|---|---|
| `openai` | OpenAI SDK | An OpenAI-only service |
| `anthropic` | Anthropic SDK | A Claude-only client |
| `google` | Google Gen AI SDK | A Gemini-only notebook |
| `ollama` | Ollama SDK | Local generation |
| `providers` | All four provider SDKs | A client with model switching |
| `playback` | MIDI synthesis and MP3 helpers | Audio previews |
| `dev` | Pytest | Core development |

```
# All providers
pip install -e ".[providers]"

# One provider plus playback
pip install -e ".[google,playback]"

# Complete local development install
pip install -e ".[providers,playback,dev]"
```

Using the venv interpreter explicitly is intentional. `py -3.12 -m pip`
selects the registered global Python even when a virtual environment exists.

### Install as a project dependency

Other Conductor repositories should pin Core to a release tag instead of an
unreleased branch or moving commit. Include only the extras the consumer uses:

```text
conductor-core[providers] @ git+https://github.com/laceyp99/conductor-core.git@v0.2.0
```

To install the same reference directly:

```powershell
python -m pip install "conductor-core[providers] @ git+https://github.com/laceyp99/conductor-core.git@v0.2.0"
```

Upgrade a dependent project by changing its pinned tag and reinstalling its
dependencies. Review [`CHANGELOG.md`](CHANGELOG.md) before changing versions,
especially while Core remains below version 1.0.

## Basic generation

```python
from conductor_core import EngineConfig, GenerationRequest, LoopGenerationEngine

engine = LoopGenerationEngine(EngineConfig.from_defaults())
result = engine.generate(
    GenerationRequest(
        key="C",
        scale="Major",
        description="warm neo-soul electric piano chords",
        model="gemini-3.1-flash-lite",
        temperature=0.3,
    )
)

print(result.generation_id)
print(result.midi_path)
print(result.cost)
```

`generate()` is synchronous. It calls the selected provider, converts the
validated loop to MIDI, and persists the resulting artifacts before returning.

For a complete editable workflow—including prompt customization, progress
events, persisted result fields, and optional audio rendering—see
[`scripts/generate_midi.py`](scripts/generate_midi.py). Running that example
makes a real provider call and may incur usage charges.

## Credentials and provider selection

Credentials can be injected by the calling application:

```python
from conductor_core import EngineConfig, ProviderCredentials

config = EngineConfig.from_defaults(
    artifact_root="my-output",
    provider_credentials=ProviderCredentials(
        openai_api_key="...",
        google_api_key="...",
        anthropic_api_key="...",
        ollama_host="http://localhost:11434",
    ),
)
```

If a credential is not injected, provider modules fall back to these environment
variables:

```ini
OPENAI_API_KEY="..."
GEMINI_API_KEY="..."
ANTHROPIC_API_KEY="..."
OLLAMA_API_HOST_ADDRESS="http://localhost:11434"
```

The provider is derived from the route actually used for `model`. The
`GenerationRequest.provider` field is deprecated, ignored, and retained only
for temporary compatibility with existing callers. To inspect available
providers, models, and capabilities without contacting a provider, run
[`scripts/inspect_models.py`](scripts/inspect_models.py).

## Generation request options

| Field | Purpose |
|---|---|
| `key`, `scale`, `description` | Musical request added to the model prompt |
| `model` | Packaged model identifier used for routing and response handling |
| `provider` | Deprecated compatibility field; ignored |
| `temperature` | Sampling temperature for models that support it |
| `use_thinking` | Toggle-style reasoning control for supported models |
| `effort` | Model-specific reasoning effort such as `minimal`, `low`, or `high` |
| `prompt_override` | System prompt override for only this request |
| `render_audio` | Request an MP3 preview after MIDI generation |
| `soundfont_path` | SoundFont name or path for this request |

Model capabilities differ. Consumers can inspect
`conductor_core.music.get_model_info()` or run
[`scripts/inspect_models.py`](scripts/inspect_models.py) instead of assuming
every model accepts temperature or the same reasoning settings.

## Prompt customization

Core ships with a default loop-generation prompt. Set `prompt_override` on
`EngineConfig` for every request made by an engine or on `GenerationRequest`
for one request. The request override takes precedence over the engine override,
which takes precedence over the packaged prompt. The generation script contains
a commented prompt override ready to edit.

## Progress reporting

Pass a callback to `generate(..., progress_callback=...)` to adapt synchronous
Core work to logs, a progress bar, a queue, or an asynchronous UI wrapper.
Current stages include provider generation, MIDI processing, and audio
rendering. The callback reports progress but does not cancel an in-flight
provider request. The generation script prints each event as it arrives.

## Audio rendering

Set `render_audio=True` on a request to render an MP3 after MIDI generation.
Install the `playback` extra and provide FluidSynth and FFmpeg on the system
`PATH`. Leaving `soundfont_path` unset uses Core's default packaged SoundFont;
set it on the request or `default_soundfont_path` on `EngineConfig` to choose
another. Audio failure does not discard a successful MIDI generation: Core
returns the MIDI with a warning and `audio_path=None`.

Lower-level discovery and rendering helpers live in `conductor_core.playback`.
The generation script enables audio with the default SoundFont and reports both
the MIDI and audio result paths.

## Results and persisted artifacts

### Data directory

Core stores durable generation history under one predictable Conductor suite
root. The default layout is:

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

On Windows, `~/.conductor/core` is
`%USERPROFILE%\.conductor\core`. Path selection has this precedence:

1. `CONDUCTOR_CORE_DATA_DIR` selects Core's complete project data directory.
2. `CONDUCTOR_HOME` selects the shared suite root; Core appends `core`.
3. Otherwise Core uses `Path.home() / ".conductor" / "core"`.

Both environment variables support `~` expansion. PowerShell examples:

```powershell
# Relocate every participating Conductor project under one suite root.
$env:CONDUCTOR_HOME = "D:\ConductorData"

# Relocate only Core; this takes precedence over CONDUCTOR_HOME.
$env:CONDUCTOR_CORE_DATA_DIR = "D:\ConductorData\custom-core"
```

An explicit `EngineConfig.artifact_root` or `FilesystemArtifactStore` root still
overrides the default generation location. Request- and engine-specific prompt
or SoundFont choices keep their existing precedence, and caller-added SoundFont
search directories remain separate from Core's packaged read-only resources.
Packaged prompts, model metadata, and the bundled SoundFont are not copied or
moved into the data directory. Core currently owns no persistent configuration
or disposable disk cache.

Resolving or importing these paths does not create directories. Core creates
`generations/` only when a generation workspace is written. It does not migrate,
overwrite, or delete an existing project-local `generations/` directory. To keep
using that portable layout, pass `artifact_root="generations"`; to migrate data,
copy it manually after reviewing destination contents.

Generation history can grow through MIDI, JSON, and especially optional MP3
files. Core retains the newest 20 generations by default, but custom artifact
stores and manually retained files still consume space at their selected
location.

Configure retention on the engine or store. Use `None` only when the calling
application owns its disk-usage policy:

```python
from conductor_core import EngineConfig
from conductor_core.storage import FilesystemArtifactStore

config = EngineConfig.from_defaults(max_generations=100)
unlimited_store = FilesystemArtifactStore("my-output", max_generations=None)
```

`GenerationResult` contains:

| Attribute | Contents |
|---|---|
| `generation_id` | Unique filesystem generation identifier |
| `loop` | Validated provider-independent loop object |
| `midi_path` | Persisted MIDI path |
| `audio_path` | Persisted MP3 path, when rendering succeeds |
| `messages` | Provider conversation/response messages |
| `cost` | Provider-reported estimated cost, when available |
| `metadata` | Persisted generation metadata |
| `warnings` | Non-fatal issues such as skipped audio |

Each generation workspace contains `loop.mid`, `messages.json`,
`metadata.json`, and optionally `loop.mp3`. Use `FilesystemArtifactStore` for
custom history roots, loading saved generations, deleting generations, and
updating saved audio metadata. By default, history retains the newest 20
generations. The generation script shows the most commonly consumed result
fields after a run.

## Direct MIDI and music utilities

Consumers can convert existing MIDI into Core's four-bar loop model and write it
back without a provider call. See
[`scripts/midi_loop_roundtrip.py`](scripts/midi_loop_roundtrip.py) for an
offline example that normalizes note starts and durations to sixteenth-note
integer positions.

Additional packaged utilities and models are available from:

- `conductor_core.models` for loop, bar, note, and timing models;
- `conductor_core.music` for model metadata, prompts, scales, and durations;
- `conductor_core.routing` for lower-level provider routing;
- `conductor_core.storage` for artifact and history management;
- `conductor_core.playback` for optional audio operations.

Prefer `LoopGenerationEngine` for complete generation workflows so persistence,
cleanup, prompt handling, and provider behavior stay consistent.

## Error behavior

Provider, parsing, and MIDI conversion errors are raised to the caller. If an
error occurs after a workspace is allocated, Core removes the unfinished
workspace. Callers should catch exceptions at their application boundary and
decide how to display, retry, or log them.

## Logging

Core emits log records under the `conductor_core` logger namespace and never
configures handlers or global logging itself (a `NullHandler` is attached so
unconfigured consumers see no warnings). To surface Core logs, configure
logging in the application:

```python
import logging

logging.basicConfig(level=logging.INFO)          # everything to the console
# or route only Core records somewhere specific:
logging.getLogger("conductor_core").addHandler(my_handler)
```

## Validate Core independently

```
python -m pytest -q
```

The tests are deterministic and do not make live provider calls or require the
audio toolchain.

## Releases

Release history, compatibility notes, and migration guidance live in
[`CHANGELOG.md`](CHANGELOG.md). GitHub releases provide the automatic source
archives for each tag; this project does not attach release-specific wheels or
binary downloads.
