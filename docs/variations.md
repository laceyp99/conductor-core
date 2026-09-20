# Generate loop variations

Core can generate 2 through 8 ordered alternatives for one musical brief with a
single provider request. Use the dedicated `VariationGenerationRequest`; it is
separate from `GenerationRequest` so count and batch prompt behavior remain
explicit.

```python
from conductor_core import (
    LoopGenerationEngine,
    VariationGenerationRequest,
)

engine = LoopGenerationEngine()
result = engine.generate_variations(
    VariationGenerationRequest(
        key="C",
        scale="Major",
        description="warm neo-soul electric piano",
        model="gpt-4o-mini",
        count=4,
    )
)

for item in result.items:
    print(item.index, item.generation.midi_path)
```

`count` defaults to 4. `validate_variation_count()` and the request constructor
accept only integers from 2 through 8; booleans, floats, and strings are not
coerced. Core sends one shared brief and count to the selected OpenAI,
Anthropic, Google, or Ollama adapter. The provider returns one internal JSON
object with an ordered `variations` array, allowing later items to follow the
earlier output context. Provider SDK objects never enter the public result.

## All-or-nothing validation

The complete collection is parsed as typed `Loop` data before any artifact is
created. If one sibling is malformed, validation raises and Core writes no
generation workspace or variation manifest. Setup, credential, transport,
service, and storage failures also raise; Core does not retry or synthesize
fallback music.

A decoded collection with a different length returns an empty failed
`VariationBatchResult` with a `wrong_count` diagnostic. It creates no child
artifacts and no manifest. A successful exact-count result contains ordered
`VariationResult` items, each with its validated loop, finalized ordinary
`GenerationMetadata`, and any non-fatal MIDI or audio warnings.

## Accounting, messages, and prompts

`VariationBatchMetadata` records the batch ID, provider, model, requested and
received counts, canonical messages, nullable token usage, and nullable total
cost for the single provider request. Child generation metadata always has
`cost=None`; Core never divides or duplicates the batch cost.

The shared provider messages are stored once in the variation manifest. Child
generation workspaces contain `loop.mid`, optional `loop.mp3`, and
`metadata.json`, but no duplicate `messages.json`.

Prompt precedence is the request's `prompt_override`, then the shared
`EngineConfig.prompt_override`, then the packaged `variation_gen_v1` prompt.
Either override replaces the complete variation system prompt and records
`prompt_version="override"`.

## Progress

Pass `progress_callback` to `generate_variations()` to receive correlated
`ProgressEvent` values. A batch emits `started`; each item then emits
`persisting` and `complete` in provider order; the batch finally emits
`complete`. A failure emits batch `failed` before Core returns or raises. Batch
events have no variation index, while item events use zero-based indexes.

## Variation history

Successful children remain ordinary generations under the configured artifact
root. Core stores one immutable, versioned batch manifest in the sibling
`variations/` directory (by default `~/.conductor/core/variations/`). It contains
the ordered generation IDs and batch request metadata, not copied MIDI, audio,
or `Loop` data.

Use `list_variation_history()`, `get_variation_history()`,
`delete_variation_history()`, and `clear_variation_history()` from the top-level
package, or the corresponding `FilesystemArtifactStore` methods. Loading a
record resolves all surviving ordinary generations and explicitly reports
`missing_generation_ids` in manifest order. Generation deletion and retention
never rewrite a manifest, and deleting a manifest never deletes a generation.

Variation manifests are retained indefinitely and can accumulate. They include
provider messages, so applications should apply their own privacy and lifecycle
policy and explicitly delete or clear them when no longer needed.
