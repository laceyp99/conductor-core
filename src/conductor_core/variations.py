"""Provider-independent contracts for complete variation batches."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from conductor_core.models import Loop
from conductor_core.storage import GenerationMetadata

VariationBatchStatus = Literal["complete", "failed"]
VariationProgressStatus = Literal["started", "persisting", "complete", "failed"]
VariationCount = Annotated[int, Field(strict=True, ge=2, le=8)]
NonnegativeInt = Annotated[int, Field(strict=True, ge=0)]
NonblankString = Annotated[str, Field(strict=True, min_length=1, pattern=r"\S")]


class _VariationContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class VariationDiagnostic(_VariationContract):
    """Machine-readable code and human-readable explanation."""

    code: NonblankString
    message: NonblankString
    location: tuple[str | NonnegativeInt, ...] | None = None


class VariationUsage(_VariationContract):
    """Batch-level token totals; unavailable measurements remain null."""

    input_tokens: NonnegativeInt | None = None
    output_tokens: NonnegativeInt | None = None
    total_tokens: NonnegativeInt | None = None


class VariationBatchMetadata(_VariationContract):
    """Information from the single shared provider request."""

    batch_id: NonblankString
    model: NonblankString
    provider: NonblankString
    prompt_version: NonblankString = "variation_gen_v1"
    requested_count: VariationCount = 4
    received_count: NonnegativeInt
    messages: list[dict[str, JsonValue]] = Field(default_factory=list)
    usage: VariationUsage | None = None
    cost: Annotated[float, Field(strict=True, ge=0, allow_inf_nan=False)] | None = None


class VariationResult(_VariationContract):
    """One successfully validated and finalized loop in provider order."""

    index: Annotated[int, Field(strict=True, ge=0, le=7)]
    loop: Loop
    generation: GenerationMetadata
    warnings: tuple[str, ...] = ()


class VariationBatchResult(_VariationContract):
    """A complete exact-count batch or an empty decoded count failure."""

    metadata: VariationBatchMetadata
    items: tuple[VariationResult, ...] = ()
    status: VariationBatchStatus = "failed"
    diagnostic: VariationDiagnostic | None = None

    @property
    def requested_count(self) -> int:
        return self.metadata.requested_count

    @property
    def received_count(self) -> int:
        return self.metadata.received_count

    @model_validator(mode="before")
    @classmethod
    def derive_status(cls, value):
        if isinstance(value, dict):
            value = dict(value)
            items = value.get("items", ())
            if not isinstance(items, (tuple, list)):
                raise ValueError("items must be an ordered list or tuple")
            expected = "complete" if items else "failed"
            if "status" in value and value["status"] != expected:
                raise ValueError("batch status must agree with item outcomes")
            value["status"] = expected
        return value

    @model_validator(mode="after")
    def validate_outcome(self):
        exact_count = self.received_count == self.requested_count
        if not exact_count:
            if self.items:
                raise ValueError("a wrong-count failure must not contain items")
            if self.diagnostic is None or self.diagnostic.code != "wrong_count":
                raise ValueError("a wrong-count failure requires its diagnostic")
            return self

        if len(self.items) != self.requested_count:
            raise ValueError("a complete batch must contain the requested item count")
        if tuple(item.index for item in self.items) != tuple(
            range(self.requested_count)
        ):
            raise ValueError("items must use contiguous provider-order indexes")
        if self.diagnostic is not None:
            raise ValueError("a complete batch cannot contain a diagnostic")
        return self
