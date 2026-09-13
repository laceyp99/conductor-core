import json
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from conductor_core import (
    ProgressEvent,
    VariationBatchMetadata,
    VariationBatchResult,
    VariationDiagnostic,
    VariationResult,
    VariationUsage,
    validate_variation_count,
)
from conductor_core.storage import GenerationMetadata


def metadata(**overrides):
    values = {
        "batch_id": "batch-97",
        "model": "test-model",
        "provider": "test-provider",
        "requested_count": 4,
        "received_count": None,
    }
    values.update(overrides)
    return VariationBatchMetadata(**values)


def diagnostic(**overrides):
    values = {"code": "invalid_loop", "message": "The loop was malformed."}
    values.update(overrides)
    return VariationDiagnostic(**values)


def valid_item(index, loop):
    return VariationResult(index=index, loop=loop)


def invalid_item(index):
    return VariationResult(
        index=index, diagnostic=diagnostic(location=(index, "Bar_1"))
    )


@pytest.mark.parametrize("count", range(2, 9))
def test_validate_variation_count_accepts_default_minimum_and_maximum(count):
    assert validate_variation_count(count) == count


def test_validate_variation_count_defaults_to_four():
    assert validate_variation_count() == 4


@pytest.mark.parametrize("count", [0, 1, -1, 9, 100])
def test_validate_variation_count_rejects_out_of_range_integers(count):
    with pytest.raises(ValueError, match="between 2 and 8"):
        validate_variation_count(count)


@pytest.mark.parametrize("count", [True, False, 4.0, 1.5, "4", None])
def test_validate_variation_count_does_not_coerce_values(count):
    with pytest.raises(TypeError):
        validate_variation_count(count)


@pytest.mark.parametrize(
    ("contract", "values"),
    [
        (VariationDiagnostic, {"code": "", "message": "message"}),
        (VariationDiagnostic, {"code": "code", "message": "   "}),
        (VariationUsage, {"input_tokens": -1}),
        (VariationUsage, {"output_tokens": True}),
        (VariationBatchMetadata, {"batch_id": 97, "model": "m", "provider": "p"}),
        (VariationBatchMetadata, {"batch_id": "b", "model": " ", "provider": "p"}),
        (
            VariationBatchMetadata,
            {"batch_id": "b", "model": "m", "provider": "p", "cost": "1.2"},
        ),
        (
            VariationBatchMetadata,
            {
                "batch_id": "b",
                "model": "m",
                "provider": "p",
                "requested_count": True,
            },
        ),
        (
            VariationBatchMetadata,
            {
                "batch_id": "b",
                "model": "m",
                "provider": "p",
                "received_count": -1,
            },
        ),
        (VariationResult, {"index": "0", "diagnostic": {"code": "c", "message": "m"}}),
        (VariationResult, {"index": True, "diagnostic": {"code": "c", "message": "m"}}),
    ],
)
def test_contracts_reject_invalid_fields_and_coercion(contract, values):
    with pytest.raises(ValidationError):
        contract(**values)


def test_contracts_forbid_unknown_fields():
    with pytest.raises(ValidationError):
        VariationDiagnostic(code="code", message="message", severity="error")


def test_item_status_is_derived_and_rejects_inconsistent_literal(sample_loop):
    assert valid_item(0, sample_loop).status == "valid"
    assert invalid_item(0).status == "invalid"

    with pytest.raises(ValidationError):
        VariationResult(index=0, loop=sample_loop, status="invalid")
    with pytest.raises(ValidationError):
        VariationResult(index=0, diagnostic=diagnostic(), status="valid")
    with pytest.raises(ValidationError):
        VariationResult(index=0, diagnostic=diagnostic(), status="unknown")


def test_invalid_item_requires_diagnostic_and_cannot_have_artifacts():
    generation = GenerationMetadata(
        id="generation-1",
        timestamp=datetime(2026, 1, 2, tzinfo=timezone.utc),
        prompt="prompt",
        key="C",
        scale="major",
        model="model",
        provider="provider",
        temperature=0.0,
        midi_path="loop.mid",
    )
    with pytest.raises(ValidationError):
        VariationResult(index=0)
    with pytest.raises(ValidationError):
        VariationResult(index=0, diagnostic=diagnostic(), generation=generation)


def test_valid_item_allows_optional_generation_information(sample_loop):
    generation = GenerationMetadata(
        id="generation-1",
        timestamp=datetime(2026, 1, 2, tzinfo=timezone.utc),
        prompt="prompt",
        key="C",
        scale="major",
        model="model",
        provider="provider",
        temperature=0.0,
        midi_path="loop.mid",
        audio_path=None,
    )
    assert valid_item(0, sample_loop).generation is None
    assert (
        VariationResult(index=0, loop=sample_loop, generation=generation).generation
        == generation
    )


def test_complete_partial_and_all_invalid_batch_outcomes(sample_loop):
    complete = VariationBatchResult(
        metadata=metadata(requested_count=2, received_count=2),
        items=(valid_item(0, sample_loop), valid_item(1, sample_loop)),
    )
    partial = VariationBatchResult(
        metadata=metadata(requested_count=2, received_count=2),
        items=(valid_item(0, sample_loop), invalid_item(1)),
    )
    failed = VariationBatchResult(
        metadata=metadata(requested_count=2, received_count=2),
        items=(invalid_item(0), invalid_item(1)),
    )
    assert (complete.status, partial.status, failed.status) == (
        "complete",
        "partial",
        "failed",
    )
    assert complete.requested_count == 2
    assert complete.received_count == 2


def test_metadata_defaults_to_four_requested_variations():
    batch_metadata = VariationBatchMetadata(
        batch_id="batch-97", model="model", provider="provider"
    )
    assert batch_metadata.requested_count == 4
    assert batch_metadata.received_count is None


@pytest.mark.parametrize("count", [1, 9, True, "2", 2.0])
def test_metadata_rejects_invalid_requested_count(count):
    with pytest.raises(ValidationError):
        metadata(requested_count=count)


@pytest.mark.parametrize("count", [2, 8])
def test_metadata_accepts_requested_count_boundaries(count):
    assert metadata(requested_count=count).requested_count == count


def test_item_warnings_default_to_empty_sequence(sample_loop):
    for item in (valid_item(0, sample_loop), invalid_item(0)):
        assert item.warnings == ()
        assert json.loads(item.model_dump_json())["warnings"] == []


def test_audio_warning_preserves_complete_batch_through_json(sample_loop):
    warning = "Audio rendering was skipped or failed. FluidSynth is unavailable."
    item = VariationResult(index=0, loop=sample_loop, warnings=[warning])
    batch = VariationBatchResult(
        metadata=metadata(requested_count=2, received_count=2),
        items=(item, valid_item(1, sample_loop)),
    )
    assert item.status == "valid"
    assert item.diagnostic is None
    assert item.warnings == (warning,)
    assert batch.status == "complete"

    encoded = batch.model_dump_json()
    assert json.loads(encoded)["items"][0]["warnings"] == [warning]
    decoded = VariationBatchResult.model_validate_json(encoded)
    assert decoded == batch
    assert decoded.status == "complete"
    assert decoded.items[0].status == "valid"
    assert decoded.items[0].warnings == (warning,)


def test_warning_does_not_make_invalid_item_valid():
    item = VariationResult(
        index=0, diagnostic=diagnostic(), warnings=("Provider output was truncated.",)
    )
    batch = VariationBatchResult(
        metadata=metadata(requested_count=2, received_count=2),
        items=(item, invalid_item(1)),
    )
    assert item.status == "invalid"
    assert batch.status == "failed"
    assert VariationBatchResult.model_validate_json(batch.model_dump_json()) == batch


@pytest.mark.parametrize("received_count", [None, 0, 1, 3])
def test_nonarray_and_wrong_count_are_top_level_failures(received_count):
    result = VariationBatchResult(
        metadata=metadata(requested_count=2, received_count=received_count),
        diagnostic=diagnostic(code="wrong_count"),
    )
    assert result.status == "failed"
    assert result.items == ()


def test_top_level_failure_requires_diagnostic_and_rejects_items(sample_loop):
    with pytest.raises(ValidationError):
        VariationBatchResult(metadata=metadata(requested_count=2, received_count=None))
    with pytest.raises(ValidationError):
        VariationBatchResult(
            metadata=metadata(requested_count=2, received_count=1),
            items=(valid_item(0, sample_loop),),
            diagnostic=diagnostic(),
        )


@pytest.mark.parametrize("indexes", [(1, 0), (0, 2), (0, 0)])
def test_batch_rejects_reordered_holey_and_duplicate_indexes(sample_loop, indexes):
    with pytest.raises(ValidationError):
        VariationBatchResult(
            metadata=metadata(requested_count=2, received_count=2),
            items=tuple(valid_item(index, sample_loop) for index in indexes),
        )


def test_batch_rejects_inconsistent_status(sample_loop):
    with pytest.raises(ValidationError):
        VariationBatchResult(
            metadata=metadata(requested_count=2, received_count=2),
            items=(valid_item(0, sample_loop), valid_item(1, sample_loop)),
            status="partial",
        )


def test_batch_rejects_lazy_items_before_status_derivation(sample_loop):
    with pytest.raises(ValidationError, match="ordered list or tuple"):
        VariationBatchResult(
            metadata=metadata(requested_count=2, received_count=2),
            items=(valid_item(index, sample_loop) for index in range(2)),
        )


def test_partial_batch_json_round_trip_keeps_invalid_position(sample_loop):
    batch = VariationBatchResult(
        metadata=metadata(requested_count=2, received_count=2, cost=0.25),
        items=[valid_item(0, sample_loop), invalid_item(1)],
    )
    serialized = json.loads(batch.model_dump_json())
    assert serialized["items"][1]["loop"] is None
    assert serialized["items"][1]["generation"] is None
    assert serialized["metadata"]["requested_count"] == 2
    assert "requested_count" not in serialized
    assert VariationBatchResult.model_validate_json(batch.model_dump_json()) == batch


def test_json_round_trip_preserves_nullable_metadata_and_unicode(sample_loop):
    batch = VariationBatchResult(
        metadata=metadata(
            requested_count=2,
            received_count=2,
            messages=[
                {"role": "user", "content": "変奏 🎵"},
                {
                    "role": "assistant",
                    "content": {"label": "été", "parts": [1, None, True]},
                },
            ],
            usage=None,
            cost=None,
        ),
        items=(valid_item(0, sample_loop), valid_item(1, sample_loop)),
    )
    encoded = batch.model_dump_json()
    decoded = VariationBatchResult.model_validate_json(encoded)

    assert decoded == batch
    assert json.loads(encoded)["metadata"]["messages"][0]["content"] == "変奏 🎵"
    assert decoded.metadata.prompt_version == "variation_gen_v1"
    assert decoded.metadata.usage is None
    assert decoded.metadata.cost is None


def test_usage_round_trip_accepts_nullable_individual_measurements():
    usage = VariationUsage(input_tokens=12, output_tokens=None, total_tokens=None)
    assert VariationUsage.model_validate_json(usage.model_dump_json()) == usage


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"batch_id": "batch-1"},
        {"variation_index": 0},
        {"variation_index": 7, "status": "complete"},
    ],
)
def test_progress_event_variation_context_is_optional(kwargs):
    event = ProgressEvent(stage="provider.custom-stage", message="Working", **kwargs)
    assert event.stage == "provider.custom-stage"
    assert event.message == "Working"
    assert event.detail is None
    assert event.batch_id == kwargs.get("batch_id")
    assert event.variation_index == kwargs.get("variation_index")
    assert event.status == kwargs.get("status")


@pytest.mark.parametrize("variation_index", [-1, 8, True, 1.0, "1"])
def test_progress_event_rejects_invalid_variation_index(variation_index):
    with pytest.raises((TypeError, ValueError)):
        ProgressEvent(
            stage="validation", message="Working", variation_index=variation_index
        )


@pytest.mark.parametrize("status", ["queued", "done", 1])
def test_progress_event_rejects_invalid_variation_status(status):
    with pytest.raises((TypeError, ValueError)):
        ProgressEvent(stage="validation", message="Working", status=status)


def test_public_api_exports_variation_contracts():
    import conductor_core

    expected = {
        "VariationStatus",
        "VariationBatchStatus",
        "VariationProgressStatus",
        "VariationDiagnostic",
        "VariationUsage",
        "VariationBatchMetadata",
        "VariationResult",
        "VariationBatchResult",
        "validate_variation_count",
    }
    assert expected <= set(conductor_core.__all__)
    assert all(hasattr(conductor_core, name) for name in expected)
