"""Shared SDK-free mechanics for structured provider variation batches."""

import json
from collections.abc import Mapping, Sequence

from pydantic import JsonValue

from conductor_core.models import Loop
from conductor_core.variations import VariationDiagnostic


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


def build_variation_schema(count: int) -> dict[str, JsonValue]:
    """Return the exact provider-neutral ``items`` wrapper schema."""
    loop_schema = Loop.model_json_schema()
    definitions = loop_schema.pop("$defs", {})
    schema = {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": loop_schema,
                "minItems": count,
                "maxItems": count,
            }
        },
        "required": ["items"],
        "additionalProperties": False,
    }
    if definitions:
        schema["$defs"] = definitions
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
