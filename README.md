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

## Basic generation

```python
from conductor_core import EngineConfig, GenerationRequest, LoopGenerationEngine

engine = LoopGenerationEngine(
    EngineConfig.from_defaults(artifact_root="generations")
)
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
