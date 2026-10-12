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
    if effort_options and not use_thinking:
        return effort_options[0]

    if effort_options and effort not in effort_options:
        supported_values = ", ".join(effort_options)
        raise ValueError(
            f"Invalid effort {effort!r} for {model_choice}. Expected one of: {supported_values}"
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


def _warn_ignored_num_ctx(
    model_choice: str,
    model_info: Mapping[str, Any],
    ollama_num_ctx: int | None,
) -> None:
    """Log when an Ollama-only context size is set for a cloud model."""
    if ollama_num_ctx is None:
        return
    if any(model_choice in models for models in model_info["models"].values()):
        logger.warning(
            "ollama_num_ctx=%r only applies to Ollama models; ignoring it for %s.",
            ollama_num_ctx,
            model_choice,
        )


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
    model_info = get_model_info()
    _warn_ignored_num_ctx(model_choice, model_info, ollama_num_ctx)
    provider: ProviderId

    if model_choice in model_info["models"]["OpenAI"]:
        effective_effort = _resolve_reasoning_effort(
            model_choice,
            model_info["models"]["OpenAI"][model_choice],
            use_thinking,
            effort,
        )
        provider = "OpenAI"
        loop, messages, loop_cost = openai_api.loop_gen(
            prompt=prompt,
            model=model_choice,
            temp=temp,
            use_thinking=use_thinking,
            effort=effective_effort,
            api_key=credentials.openai_api_key,
            system_prompt=system_prompt,
            **(
                {"request_timeout": request_timeout}
                if request_timeout is not None
                else {}
            ),
        )
    elif model_choice in model_info["models"]["Google"]:
        effective_effort = _resolve_reasoning_effort(
            model_choice,
            model_info["models"]["Google"][model_choice],
            use_thinking,
            effort,
        )
        provider = "Google"
        loop, messages, loop_cost = gemini_api.loop_gen(
            prompt=prompt,
            model=model_choice,
            temp=temp,
            use_thinking=use_thinking,
            effort=effective_effort,
            api_key=credentials.google_api_key,
            system_prompt=system_prompt,
            **(
                {"request_timeout": request_timeout}
                if request_timeout is not None
                else {}
            ),
        )
    elif model_choice in model_info["models"]["Anthropic"]:
        effective_effort = _resolve_reasoning_effort(
            model_choice,
            model_info["models"]["Anthropic"][model_choice],
            use_thinking,
            effort,
        )
        provider = "Anthropic"
        loop, messages, loop_cost = claude_api.loop_gen(
            prompt=prompt,
            model=model_choice,
            temp=temp,
            use_thinking=use_thinking,
            effort=effective_effort,
            api_key=credentials.anthropic_api_key,
            system_prompt=system_prompt,
            **(
                {"request_timeout": request_timeout}
                if request_timeout is not None
                else {}
            ),
        )
    else:
        ollama_status = ollama_api.get_model_status(
            model_choice,
            host_address=credentials.ollama_host,
            **(
                {"request_timeout": request_timeout}
                if request_timeout is not None
                else {}
            ),
        )

        if exception := ollama_status.get("exception"):
            raise exception
        if ollama_status["installed"]:
            model_capabilities = ollama_status["model_capabilities"]
            # Successful inspection is required before routing reasoning controls.
            assert model_capabilities is not None
            effective_effort = _resolve_reasoning_effort(
                model_choice, model_capabilities, use_thinking, effort
            )
            provider = "Ollama"
            loop, messages, loop_cost = ollama_api.loop_gen(
                prompt=prompt,
                model=model_choice,
                temp=temp,
                host_address=credentials.ollama_host,
                system_prompt=system_prompt,
                use_thinking=use_thinking,
                effort=effective_effort,
                model_capabilities=model_capabilities,
                **(
                    {"request_timeout": request_timeout}
                    if request_timeout is not None
                    else {}
                ),
                **({"num_ctx": ollama_num_ctx} if ollama_num_ctx is not None else {}),
            )
        elif not ollama_status["available"]:
            raise ValueError(
                "Invalid Model Selected. If you intended to use Ollama, it is currently unavailable."
            )
        else:
            raise ValueError("Invalid Model Selected")

    return LoopProviderResult(
        loop=loop, messages=messages, cost=loop_cost, provider=provider
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
    model_info = get_model_info()
    _warn_ignored_num_ctx(model_choice, model_info, ollama_num_ctx)
    variation_prompt = f"Requested variation count: {count}\n\n{prompt}"
    provider: ProviderId
    if model_choice in model_info["models"]["OpenAI"]:
        provider = "OpenAI"
        collection, messages, cost, usage = openai_api.variations_gen(
            prompt=variation_prompt,
            model=model_choice,
            temp=temp,
            use_thinking=use_thinking,
            effort=_resolve_reasoning_effort(
                model_choice,
                model_info["models"][provider][model_choice],
                use_thinking,
                effort,
            ),
            api_key=credentials.openai_api_key,
            system_prompt=system_prompt,
            **(
                {"request_timeout": request_timeout}
                if request_timeout is not None
                else {}
            ),
        )
    elif model_choice in model_info["models"]["Google"]:
        provider = "Google"
        collection, messages, cost, usage = gemini_api.variations_gen(
            prompt=variation_prompt,
            model=model_choice,
            temp=temp,
            use_thinking=use_thinking,
            effort=_resolve_reasoning_effort(
                model_choice,
                model_info["models"][provider][model_choice],
                use_thinking,
                effort,
            ),
            api_key=credentials.google_api_key,
            system_prompt=system_prompt,
            **(
                {"request_timeout": request_timeout}
                if request_timeout is not None
                else {}
            ),
        )
    elif model_choice in model_info["models"]["Anthropic"]:
        provider = "Anthropic"
        collection, messages, cost, usage = claude_api.variations_gen(
            prompt=variation_prompt,
            model=model_choice,
            temp=temp,
            use_thinking=use_thinking,
            effort=_resolve_reasoning_effort(
                model_choice,
                model_info["models"][provider][model_choice],
                use_thinking,
                effort,
            ),
            api_key=credentials.anthropic_api_key,
            system_prompt=system_prompt,
            **(
                {"request_timeout": request_timeout}
                if request_timeout is not None
                else {}
            ),
        )
    else:
        status = ollama_api.get_model_status(
            model_choice,
            host_address=credentials.ollama_host,
            **(
                {"request_timeout": request_timeout}
                if request_timeout is not None
                else {}
            ),
        )
        if exception := status.get("exception"):
            raise exception
        if not status["installed"]:
            if not status["available"]:
                raise ValueError(
                    "Invalid Model Selected. If you intended to use Ollama, it is currently unavailable."
                )
            raise ValueError("Invalid Model Selected")
        model_capabilities = status["model_capabilities"]
        assert model_capabilities is not None
        effective_effort = _resolve_reasoning_effort(
            model_choice, model_capabilities, use_thinking, effort
        )
        provider = "Ollama"
        collection, messages, cost, usage = ollama_api.variations_gen(
            prompt=variation_prompt,
            model=model_choice,
            temp=temp,
            host_address=credentials.ollama_host,
            system_prompt=system_prompt,
            use_thinking=use_thinking,
            effort=effective_effort,
            model_capabilities=model_capabilities,
            **(
                {"request_timeout": request_timeout}
                if request_timeout is not None
                else {}
            ),
            **({"num_ctx": ollama_num_ctx} if ollama_num_ctx is not None else {}),
        )

    return VariationProviderResult(
        variations=tuple(collection.variations),
        provider=provider,
        messages=messages,
        usage=usage,
        cost=cost,
    )
