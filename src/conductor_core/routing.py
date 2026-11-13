"""Provider routing for Conductor Core."""

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from conductor_core.config import ProviderCredentials
from conductor_core.models import Loop
from conductor_core.music import get_model_info
from conductor_core.provider_types import ProviderId, ProviderMessage
from conductor_core.providers import anthropic as claude_api
from conductor_core.providers import google as gemini_api
from conductor_core.providers import ollama as ollama_api
from conductor_core.providers import openai as openai_api
from conductor_core.providers._reasoning import _resolve_discrete_effort
from conductor_core.variations import VariationUsage

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LoopProviderResult:
    """Normalized result of a single provider loop request."""

    loop: Loop
    messages: list[ProviderMessage]
    cost: float | None
    provider: ProviderId


@dataclass(frozen=True)
class VariationProviderResult:
    """Normalized result of a single provider variation request."""

    variations: tuple[Loop, ...]
    provider: ProviderId
    messages: list[ProviderMessage]
    usage: VariationUsage | None
    cost: float | None


def _resolve_reasoning_effort(
    model_choice: str,
    model_config: Mapping[str, Any],
    use_thinking: bool,
    effort: str | None,
) -> str | None:
    """Validate reasoning options and return the effective provider effort."""
    effort_options = model_config.get("effort_options") or []
    effort = _resolve_discrete_effort(
        model_choice, effort_options, use_thinking, effort
    )
    if not effort_options and effort not in (None, "low"):
        logger.warning(
            "Effort %r was requested for %s, but this model does not support "
            "configurable effort; the setting will be ignored.",
            effort,
            model_choice,
        )

    if use_thinking and not model_config.get("extended_thinking"):
        logger.warning(
            "Thinking was requested for %s, but this model does not support "
            "extended thinking; the setting will be ignored.",
            model_choice,
        )

    return effort


_CLOUD_PROVIDERS: tuple[ProviderId, ...] = ("OpenAI", "Google", "Anthropic")
# Cloud adapters keep their own loop_gen/variations_gen so wire schemas stay explicit.
_CLOUD_ADAPTERS = {
    "OpenAI": openai_api,
    "Google": gemini_api,
    "Anthropic": claude_api,
}


@dataclass(frozen=True)
class _Route:
    """Provider choice and reasoning settings resolved once per request."""

    provider: ProviderId
    effort: str | None
    # Inspected Ollama capabilities; None for cloud models.
    model_capabilities: Mapping[str, Any] | None


def _resolve_route(
    model_choice: str,
    credentials: ProviderCredentials,
    use_thinking: bool,
    effort: str | None,
    request_timeout: float | None,
    ollama_num_ctx: int | None,
) -> _Route:
    """Pick the provider for a model and resolve its effective effort.

    Cloud models are looked up in model metadata. Anything else is treated as
    an installed Ollama model, whose capabilities are inspected live.
    """
    model_info = get_model_info()
    for provider in _CLOUD_PROVIDERS:
        model_config = model_info["models"][provider].get(model_choice)
        if model_config is None:
            continue
        if ollama_num_ctx is not None:
            logger.warning(
                "ollama_num_ctx=%r only applies to Ollama models; ignoring it for %s.",
                ollama_num_ctx,
                model_choice,
            )
        return _Route(
            provider=provider,
            effort=_resolve_reasoning_effort(
                model_choice, model_config, use_thinking, effort
            ),
            model_capabilities=None,
        )

    status = ollama_api.get_model_status(
        model_choice,
        host_address=credentials.ollama_host,
        request_timeout=request_timeout,
    )
    if status is None:
        raise ValueError("Invalid Model Selected")
    if status.error is not None:
        raise status.error
    model_capabilities = status.capabilities
    # Successful inspection is required before routing reasoning controls.
    assert model_capabilities is not None
    return _Route(
        provider="Ollama",
        effort=_resolve_reasoning_effort(
            model_choice, model_capabilities, use_thinking, effort
        ),
        model_capabilities=model_capabilities,
    )


def _cloud_api_key(
    credentials: ProviderCredentials, provider: ProviderId
) -> str | None:
    """Return the configured API key for a cloud provider."""
    return {
        "OpenAI": credentials.openai_api_key,
        "Google": credentials.google_api_key,
        "Anthropic": credentials.anthropic_api_key,
    }[provider]


def generate_midi(
    model_choice: str,
    prompt: str,
    temp: float = 0.0,
    use_thinking: bool = False,
    effort: str | None = "low",
    provider_credentials: ProviderCredentials | None = None,
    request_timeout: float | None = None,
    system_prompt: str | None = None,
    ollama_num_ctx: int | None = None,
) -> LoopProviderResult:
    """Generate loop data by routing a prompt to the selected provider.

    Returns:
        A :class:`LoopProviderResult` with the generated loop, provider
        messages, total cost, and the name of the provider that handled the
        request.
    """
    credentials = provider_credentials or ProviderCredentials()
    route = _resolve_route(
        model_choice,
        credentials,
        use_thinking,
        effort,
        request_timeout,
        ollama_num_ctx,
    )
    if route.provider == "Ollama":
        loop, messages, loop_cost = ollama_api.loop_gen(
            prompt=prompt,
            model=model_choice,
            temp=temp,
            host_address=credentials.ollama_host,
            system_prompt=system_prompt,
            use_thinking=use_thinking,
            effort=route.effort,
            model_capabilities=route.model_capabilities,
            request_timeout=request_timeout,
            num_ctx=ollama_num_ctx,
        )
    else:
        loop, messages, loop_cost = _CLOUD_ADAPTERS[route.provider].loop_gen(
            prompt=prompt,
            model=model_choice,
            temp=temp,
            use_thinking=use_thinking,
            effort=route.effort,
            api_key=_cloud_api_key(credentials, route.provider),
            system_prompt=system_prompt,
            request_timeout=request_timeout,
        )
    return LoopProviderResult(
        loop=loop, messages=messages, cost=loop_cost, provider=route.provider
    )


def generate_variations(
    model_choice: str,
    prompt: str,
    count: int,
    temp: float = 0.0,
    use_thinking: bool = False,
    effort: str | None = "low",
    provider_credentials: ProviderCredentials | None = None,
    request_timeout: float | None = None,
    system_prompt: str | None = None,
    ollama_num_ctx: int | None = None,
) -> VariationProviderResult:
    """Route one ordered variation collection request to a provider."""
    credentials = provider_credentials or ProviderCredentials()
    route = _resolve_route(
        model_choice,
        credentials,
        use_thinking,
        effort,
        request_timeout,
        ollama_num_ctx,
    )
    variation_prompt = f"Requested variation count: {count}\n\n{prompt}"
    if route.provider == "Ollama":
        collection, messages, cost, usage = ollama_api.variations_gen(
            prompt=variation_prompt,
            model=model_choice,
            temp=temp,
            host_address=credentials.ollama_host,
            system_prompt=system_prompt,
            use_thinking=use_thinking,
            effort=route.effort,
            model_capabilities=route.model_capabilities,
            request_timeout=request_timeout,
            num_ctx=ollama_num_ctx,
        )
    else:
        collection, messages, cost, usage = _CLOUD_ADAPTERS[
            route.provider
        ].variations_gen(
            prompt=variation_prompt,
            model=model_choice,
            temp=temp,
            use_thinking=use_thinking,
            effort=route.effort,
            api_key=_cloud_api_key(credentials, route.provider),
            system_prompt=system_prompt,
            request_timeout=request_timeout,
        )
    return VariationProviderResult(
        variations=tuple(collection.variations),
        provider=route.provider,
        messages=messages,
        usage=usage,
        cost=cost,
    )
