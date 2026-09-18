# Variation contracts (preparation for batch generation)

Core exposes variation data contracts, a canonical prompt, and lower-level
provider routing for structured variation batches. The public
`generate_variations(request, count=4)` engine operation, per-item validation,
and batch persistence are follow-up work; they are not available yet.
The planned operation reuses `GenerationRequest` for one shared musical brief.
`validate_variation_count()` defaults to 4 and accepts only integers from 2 to 8;
booleans, floats, and strings are rejected without coercion.

`VariationResult` contains a zero-based `index`, derived `status` (`valid` or
`invalid`), `loop`, optional `generation` (`GenerationMetadata` with artifact
information), and optional `diagnostic`. Invalid items require `loop=None`, a
diagnostic, and no generation information. Valid items can exist before artifacts
are persisted and must not carry a validation diagnostic. Optional audio failure
does not make a valid MIDI loop invalid. Per-item `warnings` default to an empty
sequence and preserve non-fatal MIDI or audio messages independently of item
and batch status. Warnings serialize as a JSON array, including `[]` when empty.

`VariationDiagnostic` uses a nonblank `code` and `message`, plus a nullable
`location` sequence of JSON field names and array indexes. Initial producers
use `invalid_json`, `invalid_top_level`, `wrong_count`, and `missing_output` for
provider-level structural failures. Per-item validation remains engine-owned and
can use codes such as `invalid_loop`. Codes are extensible strings; consumers
should tolerate unknown codes. A location such as
`["Bar_1", "notes", 0, "pitch"]` is relative to the item.

`VariationBatchResult` contains ordered `items`, `metadata`, derived `status`,
and a nullable batch-level `diagnostic`. It derives `complete` when all items
are valid, `partial` when some are valid, and `failed` when none are valid.
Callers may omit statuses; contradictory explicit statuses are rejected.
Items must retain every original position in contiguous order, including invalid
neighbors. A wrong top-level count requires an empty item sequence and a batch
diagnostic, so it cannot reference generation artifacts. A non-array response
uses `received_count=None`. An all-invalid array retains its indexed diagnostics.
These contracts never create files or substitute musical fallback content.

`VariationBatchMetadata` contains `batch_id`, `model`, `provider`, `prompt_version`,
`requested_count`, `received_count`, ordered JSON `messages`, nullable `usage`,
and nullable `cost`. Usage holds nullable `input_tokens`, `output_tokens`, and
`total_tokens`. Cost and usage describe the one shared request, including failed
results; unavailable values remain `None`, not zero. Do not sum or divide these
values from per-item generation records. The batch exposes read-only
`requested_count` and `received_count` properties forwarding its metadata.

The new Pydantic models reject unknown fields and support `model_dump_json()`
and `model_validate_json()`. JSON has the shape
`{"metadata": {...}, "items": [...], "status": "partial", "diagnostic": null}`;
counts occur only inside `metadata`, locations/items become arrays, and optional
fields remain explicit `null` values by default. Existing `Loop` parsing is unchanged.

`conductor_core.music.get_variation_prompt()` loads the packaged
`variation_gen_v1.txt`; `VARIATION_PROMPT_VERSION` identifies it. Every provider
is instructed and schema-constrained to return one object whose only top-level
property is `items`, an array with exactly the requested length. Adapters unwrap
that object while retaining each raw JSON item at its original index. Each item
contains exactly a nonblank musical `description` and a complete `loop`; `Loop`
validation does not happen at this boundary. An override replaces the entire
system prompt, so it is responsible for preserving the structured-output
instructions. Routing composes the requested count and unmodified shared brief
into one provider-neutral user message. Retry prompts are outside this contract.

`conductor_core.routing.generate_variations()` is the internal routing entry
point. It validates the count before model lookup or provider interaction,
selects OpenAI, Google, Anthropic, or Ollama with the same model registry and
availability behavior as single-loop generation, and returns a frozen internal
result. Successful provider calls with malformed, missing, or wrong-length model
output return structural diagnostics in-band. Authentication, timeout,
rate-limit, connection, schema-rejection, and other provider-call failures remain
exceptions. Usage and cost remain nullable when the provider does not report the
primary token counts; Ollama cost is always `0.0`.

`ProgressEvent` adds nullable `batch_id`, `variation_index`, and `status` while
preserving the existing three positional fields and extensible `stage` string.
The correlation status vocabulary is `started`, `validating`, `persisting`,
`valid`, `invalid`, `complete`, `partial`, and `failed`. The planned synchronous
event order is: batch `started`; each item in array order emits `validating`,
then either `invalid` or `persisting` followed by `valid`; finally the batch emits
`complete`, `partial`, or `failed`. Batch events have no variation index; item
events share the batch ID and original zero-based index. A top-level response
failure goes directly from `started` to `failed`. Request/setup/transport errors
remain exceptions. Actual emission belongs to the engine follow-up; current
single-generation events retain `None` for all three new fields.
