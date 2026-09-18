"""Shared SDK-free mechanics for structured provider variation batches."""

import json
from collections.abc import Mapping, Sequence
from copy import deepcopy

from pydantic import BaseModel, ConfigDict, JsonValue

from conductor_core.models import Loop
from conductor_core.variations import NonblankString, VariationDiagnostic


class VariationItem(BaseModel):
    """Private provider-schema item; adapters never instantiate this model."""

    model_config = ConfigDict(extra="forbid")

    description: NonblankString
    loop: Loop


class VariationBatch(BaseModel):
    """Private provider-schema wrapper used only to generate JSON Schema."""

    model_config = ConfigDict(extra="forbid")

    items: list[VariationItem]


def _close_object_schemas(value: JsonValue) -> None:
    """Require exact properties for every object in a JSON schema tree."""
    if isinstance(value, dict):
        if value.get("type") == "object":
            value["additionalProperties"] = False
        for child in value.values():
            _close_object_schemas(child)
    elif isinstance(value, list):
        for child in value:
            _close_object_schemas(child)


def _inline_local_references(value, definitions):
    """Replace Pydantic's local definitions with their complete schema trees."""
    if isinstance(value, dict):
        reference = value.get("$ref")
        if reference is not None:
            if not isinstance(reference, str) or not reference.startswith("#/$defs/"):
                raise ValueError(f"Unsupported variation schema reference: {reference}")
            definition_name = reference.removeprefix("#/$defs/")
            try:
                resolved = deepcopy(definitions[definition_name])
            except KeyError as exc:
                raise ValueError(
                    f"Missing variation schema definition: {definition_name}"
                ) from exc
            resolved.update(
                {key: child for key, child in value.items() if key != "$ref"}
            )
            return _inline_local_references(resolved, definitions)
        return {
            key: _inline_local_references(child, definitions)
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [_inline_local_references(child, definitions) for child in value]
    return value


def build_variation_schema(count: int) -> dict[str, JsonValue]:
    """Return the closed, fully inlined provider variation schema."""
    generated = VariationBatch.model_json_schema()
    definitions = generated.pop("$defs", {})
    schema = _inline_local_references(generated, definitions)
    schema["properties"]["items"]["minItems"] = count
    schema["properties"]["items"]["maxItems"] = count
    _close_object_schemas(schema)
    return schema


def normalize_variation_output(
    output: str | Mapping[str, JsonValue] | None,
    count: int,
) -> tuple[
    tuple[JsonValue, ...] | None,
    int | None,
    VariationDiagnostic | None,
]:
    """Parse and structurally validate a provider's variation wrapper."""
    if output is None or (isinstance(output, str) and not output.strip()):
        return (
            None,
            None,
            VariationDiagnostic(
                code="missing_output",
                message="The provider response did not contain usable output.",
                location=None,
            ),
        )

    if isinstance(output, str):
        try:
            parsed = json.loads(output)
        except json.JSONDecodeError:
            return (
                None,
                None,
                VariationDiagnostic(
                    code="invalid_json",
                    message="The provider output was not valid JSON.",
                    location=None,
                ),
            )
    else:
        parsed = output

    if (
        not isinstance(parsed, Mapping)
        or set(parsed) != {"items"}
        or not isinstance(parsed["items"], Sequence)
        or isinstance(parsed["items"], (str, bytes, bytearray))
    ):
        return (
            None,
            None,
            VariationDiagnostic(
                code="invalid_top_level",
                message='The provider output must be an object containing only an "items" array.',
                location=(),
            ),
        )

    items = tuple(parsed["items"])
    received_count = len(items)
    if received_count != count:
        return (
            None,
            received_count,
            VariationDiagnostic(
                code="wrong_count",
                message=(
                    f"Expected {count} variation items but received {received_count}."
                ),
                location=("items",),
            ),
        )
    return items, received_count, None
