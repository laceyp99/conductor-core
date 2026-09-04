# Getting started

## Install as a dependency

Conductor Core is pre-release. Pin it to a release tag and choose only the
optional features your application needs. Use `providers` for every provider,
a provider name such as `google` for one provider, and `playback` for audio
helpers.

```powershell
# uv-managed project
uv add "conductor-core[providers] @ git+https://github.com/laceyp99/conductor-core.git@v0.5.1"

# pip-managed environment
python -m pip install "conductor-core[providers] @ git+https://github.com/laceyp99/conductor-core.git@v0.5.1"
```

Provider extras are `openai`, `anthropic`, `google`, and `ollama`. Extras can be
combined; for example, `[google,playback]` enables Gemini generation and audio
previews.

## Generate a loop

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
validated loop to MIDI, and persists the artifacts before returning. See
[`scripts/generate_midi.py`](https://github.com/laceyp99/conductor-core/blob/main/scripts/generate_midi.py)
for an editable
example with prompt customization, progress events, persisted result fields,
and optional audio. That script makes a real provider call and may incur usage
charges.

## Credentials

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

If a credential is not injected, provider modules use these environment
variables:

```ini
OPENAI_API_KEY="..."
GEMINI_API_KEY="..."
ANTHROPIC_API_KEY="..."
OLLAMA_API_HOST_ADDRESS="http://localhost:11434"
```

The provider is derived from the route used for `model`;
`GenerationRequest` does not accept a caller-supplied provider. Inspect models
and capabilities without contacting a provider with:

```powershell
uv run python scripts/inspect_models.py
```
