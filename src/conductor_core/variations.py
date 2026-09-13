"""Provider-independent variation outcomes; no generation or storage side effects."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from conductor_core.models import Loop
from conductor_core.storage import GenerationMetadata

VariationStatus = Literal["valid", "invalid"]
VariationBatchStatus = Literal["complete", "partial", "failed"]
VariationProgressStatus = Literal[
    "started",
    "validating",
    "persisting",
    "valid",
    "invalid",
    "complete",
    "partial",
    "failed",
]
VariationCount = Annotated[int, Field(strict=True, ge=2, le=8)]
NonnegativeInt = Annotated[int, Field(strict=True, ge=0)]
NonblankString = Annotated[str, Field(strict=True, min_length=1, pattern=r"\S")]


class _VariationContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class VariationDiagnostic(_VariationContract):
    """Machine-readable code and human-readable explanation; location is a JSON path."""

    code: NonblankString
    message: NonblankString
    location: tuple[str | NonnegativeInt, ...] | None = None


class VariationUsage(_VariationContract):
    """Batch-level token totals; unavailable measurements remain null."""

    input_tokens: NonnegativeInt | None = None
    output_tokens: NonnegativeInt | None = None
    total_tokens: NonnegativeInt | None = None


class VariationBatchMetadata(_VariationContract):
    """Information from the single shared provider request, never per-item estimates."""

    batch_id: NonblankString
    model: NonblankString
    provider: NonblankString
    prompt_version: NonblankString = "variation_gen_v1"
    requested_count: VariationCount = 4
    received_count: NonnegativeInt | None = None
    messages: list[dict[str, JsonValue]] = Field(default_factory=list)
    usage: VariationUsage | None = None
    cost: Annotated[float, Field(strict=True, ge=0, allow_inf_nan=False)] | None = None


class VariationResult(_VariationContract):
    """One original zero-based array position, including malformed items.

    ``generation`` holds optional persisted artifact information. A validated
    loop can exist before persistence; an invalid item cannot reference artifacts.
    """

    index: Annotated[int, Field(strict=True, ge=0, le=7)]
    status: VariationStatus = "invalid"
    loop: Loop | None = None
    generation: GenerationMetadata | None = None
    warnings: tuple[str, ...] = ()
    diagnostic: VariationDiagnostic | None = None

    @model_validator(mode="before")
    @classmethod
    def derive_status(cls, value):
        if isinstance(value, dict):
            value = dict(value)
            expected = "valid" if value.get("loop") is not None else "invalid"
            if "status" in value and value["status"] != expected:
                raise ValueError("item status must agree with loop availability")
            value["status"] = expected
        return value

    @model_validator(mode="after")
    def validate_outcome(self):
        if self.loop is None:
            if self.diagnostic is None:
                raise ValueError("an invalid item requires a validation diagnostic")
            if self.generation is not None:
                raise ValueError("an invalid item cannot have generation artifacts")
        elif self.diagnostic is not None:
            raise ValueError("a valid item cannot have a validation diagnostic")
        return self


class VariationBatchResult(_VariationContract):
    """Ordered outcomes or an empty top-level failure, with no musical fallback.

    Counts describe the provider array, not just its valid loops. A non-array
    response has ``received_count=None``. A wrong count must yield no items.
    """

    metadata: VariationBatchMetadata
    items: tuple[VariationResult, ...] = ()
    status: VariationBatchStatus = "failed"
    diagnostic: VariationDiagnostic | None = None

    @property
    def requested_count(self) -> int:
        return self.metadata.requested_count

    @property
    def received_count(self) -> int | None:
        return self.metadata.received_count

    @model_validator(mode="before")
    @classmethod
    def derive_status(cls, value):
        if isinstance(value, dict):
            value = dict(value)
            items = value.get("items", ())
            if not isinstance(items, (tuple, list)):
                raise ValueError("items must be an ordered list or tuple")
            valid = sum(
                (
                    item.get("loop")
                    if isinstance(item, dict)
                    else getattr(item, "loop", None)
                )
                is not None
                for item in items
            )
            expected = (
                "failed"
                if not valid
                else "complete"
                if valid == len(items)
                else "partial"
            )
            if "status" in value and value["status"] != expected:
                raise ValueError("batch status must agree with item outcomes")
            value["status"] = expected
        return value

    @model_validator(mode="after")
    def validate_outcome(self):
        if self.received_count != self.requested_count:
            if self.items:
                raise ValueError(
                    "a top-level count failure must have no items or artifacts"
                )
            if self.diagnostic is None:
                raise ValueError("a top-level failure requires a diagnostic")
        else:
            if len(self.items) != self.requested_count:
                raise ValueError("every received item must retain its original index")
            if tuple(item.index for item in self.items) != tuple(
                range(self.requested_count)
            ):
                raise ValueError(
                    "items must be ordered by contiguous zero-based indexes"
                )
            if self.diagnostic is not None:
                raise ValueError(
                    "item validation diagnostics belong on individual items"
                )
        return self
