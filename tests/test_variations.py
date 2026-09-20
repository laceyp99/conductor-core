import json
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from conductor_core import (
    ProgressEvent,
    VariationBatchMetadata,
    VariationBatchResult,
    VariationDiagnostic,
    VariationGenerationRequest,
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
        "requested_count": 2,
        "received_count": 2,
    }
    values.update(overrides)
    return VariationBatchMetadata(**values)


def generation(index=0):
    return GenerationMetadata(
        id=f"generation-{index}",
        timestamp=datetime(2026, 1, 2, tzinfo=timezone.utc),
        prompt="prompt",
        key="C",
        scale="major",
        model="model",
        provider="provider",
        temperature=0.0,
        cost=None,
        midi_path=f"loop-{index}.mid",
    )


def item(index, loop):
    return VariationResult(index=index, loop=loop, generation=generation(index))


@pytest.mark.parametrize("count", range(2, 9))
def test_validate_variation_count_accepts_supported_range(count):
    assert validate_variation_count(count) == count


@pytest.mark.parametrize("count", [0, 1, -1, 9, 100])
def test_validate_variation_count_rejects_out_of_range(count):
    with pytest.raises(ValueError, match="between 2 and 8"):
        validate_variation_count(count)


@pytest.mark.parametrize("count", [True, False, 4.0, "4", None])
def test_validate_variation_count_does_not_coerce(count):
    with pytest.raises(TypeError):
        validate_variation_count(count)


def test_variation_request_is_distinct_and_strict():
    request = VariationGenerationRequest(
        key="C",
        scale="Major",
        description="warm keys",
        model="test-model",
    )
    assert request.count == 4

    with pytest.raises(TypeError):
        VariationGenerationRequest(
            key="C",
            scale="Major",
            description="warm keys",
            model="test-model",
            count=True,
        )


def test_complete_batch_round_trip_preserves_order_warnings_and_unicode(sample_loop):
    batch = VariationBatchResult(
        metadata=metadata(
            messages=[{"role": "user", "content": "variation 🎵"}],
            usage=VariationUsage(
                input_tokens=12, output_tokens=None, total_tokens=None
            ),
            cost=None,
        ),
        items=(
            VariationResult(
                index=0,
                loop=sample_loop,
                generation=generation(0),
                warnings=("Audio rendering was skipped.",),
            ),
            item(1, sample_loop),
        ),
    )

    encoded = batch.model_dump_json()
    assert json.loads(encoded)["metadata"]["messages"][0]["content"] == "variation 🎵"
    assert VariationBatchResult.model_validate_json(encoded) == batch
    assert batch.status == "complete"
    assert batch.items[0].generation.cost is None


@pytest.mark.parametrize("received_count", [0, 1, 3])
def test_wrong_count_is_an_empty_failed_result(received_count):
    result = VariationBatchResult(
        metadata=metadata(received_count=received_count),
        diagnostic=VariationDiagnostic(
            code="wrong_count", message="The provider returned the wrong count."
        ),
    )
    assert result.status == "failed"
    assert result.items == ()


def test_complete_batch_rejects_missing_reordered_or_diagnostic_items(sample_loop):
    with pytest.raises(ValidationError, match="requested item count"):
        VariationBatchResult(metadata=metadata(), items=(item(0, sample_loop),))
    with pytest.raises(ValidationError, match="provider-order"):
        VariationBatchResult(
            metadata=metadata(), items=(item(1, sample_loop), item(0, sample_loop))
        )
    with pytest.raises(ValidationError, match="cannot contain a diagnostic"):
        VariationBatchResult(
            metadata=metadata(),
            items=(item(0, sample_loop), item(1, sample_loop)),
            diagnostic=VariationDiagnostic(code="unexpected", message="bad"),
        )


def test_wrong_count_requires_specific_diagnostic_and_rejects_items(sample_loop):
    with pytest.raises(ValidationError, match="requires its diagnostic"):
        VariationBatchResult(metadata=metadata(received_count=1))
    with pytest.raises(ValidationError, match="must not contain items"):
        VariationBatchResult(
            metadata=metadata(received_count=1),
            items=(item(0, sample_loop),),
            diagnostic=VariationDiagnostic(code="wrong_count", message="bad"),
        )


@pytest.mark.parametrize("status", ["queued", "validating", "valid", "partial"])
def test_progress_event_rejects_removed_variation_statuses(status):
    with pytest.raises((TypeError, ValueError)):
        ProgressEvent(stage="variations", message="Working", status=status)


@pytest.mark.parametrize("status", ["started", "persisting", "complete", "failed"])
def test_progress_event_accepts_variation_statuses(status):
    assert (
        ProgressEvent(stage="variations", message="Working", status=status).status
        == status
    )


def test_public_api_exports_adopted_variation_contracts():
    import conductor_core

    expected = {
        "VariationBatchStatus",
        "VariationProgressStatus",
        "VariationDiagnostic",
        "VariationUsage",
        "VariationBatchMetadata",
        "VariationResult",
        "VariationBatchResult",
        "VariationGenerationRequest",
        "VariationHistoryManifest",
        "VariationHistoryRecord",
        "VariationHistoryUsage",
        "list_variation_history",
        "get_variation_history",
        "delete_variation_history",
        "clear_variation_history",
        "validate_variation_count",
    }
    assert expected <= set(conductor_core.__all__)
    assert "VariationStatus" not in conductor_core.__all__
