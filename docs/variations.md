# Variation contracts (preparation for batch generation)

Core now exposes variation data contracts and a canonical initial prompt.
The `generate_variations(request, count=4)` engine operation, provider array
handling, and batch persistence are follow-up work; they are not available yet.
The planned operation reuses `GenerationRequest` for one shared musical brief.
`validate_variation_count()` defaults to 4 and accepts only integers from 1 to 8;
booleans, floats, and strings are rejected without coercion.

`VariationResult` contains a zero-based `index`, derived `status` (`valid` or
`invalid`), `loop`, optional `generation` (`GenerationMetadata` with artifact
information), and optional `diagnostic`. Invalid items require `loop=None`, a
diagnostic, and no generation information. Valid items can exist before artifacts
are persisted and must not carry a validation diagnostic. Optional audio failure
does not make a valid MIDI loop invalid.

`VariationDiagnostic` uses a nonblank `code` and `message`, plus a nullable
`location` sequence of JSON field names and array indexes. Initial producers
should use `invalid_loop`, `invalid_response`, and `wrong_count` for those three
failure cases. Codes are extensible strings; consumers should tolerate unknown
codes. A location such as `["Bar_1", "notes", 0, "pitch"]` is relative to the item.

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
`variation_gen_v1.txt`; `VARIATION_PROMPT_VERSION` identifies it. The planned
engine operation uses the existing request override, then engine override, then
this packaged default. An override replaces the entire system prompt and should
be recorded with `prompt_version="override"`. The requested count and shared
brief belong in the provider request, not string interpolation of the resource.
Retry prompts are outside this contract.

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
