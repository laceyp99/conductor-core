"""Private provider-routing result contracts."""

from dataclasses import dataclass
from typing import Literal

from pydantic import JsonValue

from conductor_core.models import Loop
from conductor_core.variations import VariationDiagnostic, VariationUsage

ProviderId = Literal["OpenAI", "Anthropic", "Google", "Ollama"]


@dataclass(frozen=True)
class ProviderLoopResult:
    """Normalized result from a single-loop provider request."""

    loop: Loop
    messages: list[dict[str, JsonValue]]
    cost: float | None
    provider: ProviderId


@dataclass(frozen=True)
class ProviderVariationResult:
    """Normalized result from a provider variation request."""

    provider: ProviderId
    model: str
    messages: list[dict[str, JsonValue]]
    usage: VariationUsage | None
    cost: float | None
    items: tuple[JsonValue, ...] | None
    received_count: int | None
    structural_diagnostic: VariationDiagnostic | None
