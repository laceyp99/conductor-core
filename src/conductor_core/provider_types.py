"""Shared typing contracts for provider-facing values Core creates.

This module intentionally imports nothing from ``conductor_core`` so routing,
provider adapters, errors, and public contracts can all depend on it without
import cycles.
"""

from typing import Literal

from typing_extensions import TypedDict

ProviderId = Literal["OpenAI", "Anthropic", "Google", "Ollama"]
"""Display name of a provider Core can route to.

This is a checker-facing type. It adds no runtime validation, so persisted or
loaded provider names remain ordinary strings.
"""


class ProviderMessage(TypedDict):
    """One provider conversation message Core records for a generation.

    At runtime each message is an ordinary ``dict`` with ``role`` and
    ``content`` keys, so it serializes to JSON unchanged.
    """

    role: Literal["system", "user", "assistant"]
    content: str
