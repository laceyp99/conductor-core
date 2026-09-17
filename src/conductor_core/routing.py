"""Provider routing for Conductor Core."""

import logging

from conductor_core._internal_types import ProviderLoopResult, ProviderVariationResult
from conductor_core.config import ProviderCredentials, validate_variation_count
from conductor_core.music import get_model_info, get_variation_prompt
from conductor_core.providers import anthropic as claude_api
from conductor_core.providers import google as gemini_api
from conductor_core.providers import ollama as ollama_api
from conductor_core.providers import openai as openai_api

logger = logging.getLogger(__name__)


def _resolve_reasoning_effort(model_choice, model_config, use_thinking, effort):
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


def generate_midi(
    model_choice,
    prompt,
    temp=0.0,
    use_thinking=False,
    effort="low",
    provider_credentials: ProviderCredentials | None = None,
    request_timeout: float | None = None,
    system_prompt: str | None = None,
):
    """Generate loop data by routing a prompt to the selected provider.

    Returns:
        A named provider result containing the loop, messages, cost, and provider.
    """
    credentials = provider_credentials or ProviderCredentials()
    model_info = get_model_info()

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
        ollama_status = ollama_api.get_ollama_status(
            host_address=credentials.ollama_host,
            **(
                {"request_timeout": request_timeout}
                if request_timeout is not None
                else {}
            ),
        )

        if model_choice in ollama_status["models"]:
            _resolve_reasoning_effort(model_choice, {}, use_thinking, effort)
            provider = "Ollama"
            loop, messages, loop_cost = ollama_api.loop_gen(
                prompt,
                model_choice,
                temp=temp,
                host_address=credentials.ollama_host,
                system_prompt=system_prompt,
                **(
                    {"request_timeout": request_timeout}
                    if request_timeout is not None
                    else {}
                ),
            )
        elif not ollama_status["available"]:
            raise ValueError(
                "Invalid Model Selected. If you intended to use Ollama, it is currently unavailable."
            )
        else:
            raise ValueError("Invalid Model Selected")

    return ProviderLoopResult(
        loop=loop,
        messages=messages,
        cost=loop_cost,
        provider=provider,
    )


def generate_variations(
    model_choice,
    brief,
    count=4,
    temp=0.0,
    use_thinking=False,
    effort="low",
    provider_credentials: ProviderCredentials | None = None,
    request_timeout: float | None = None,
    system_prompt: str | None = None,
) -> ProviderVariationResult:
    """Generate one structured batch of variations through the selected provider.

    An explicit ``system_prompt`` completely replaces the canonical variation
    prompt, so callers experimenting with overrides are responsible for retaining
    the current structured-output instructions.
    """
    count = validate_variation_count(count)
    credentials = provider_credentials or ProviderCredentials()
    model_info = get_model_info()
    variation_prompt = (
        get_variation_prompt() if system_prompt is None else system_prompt
    )
    user_message = f"Requested variation count: {count}\n\nMusical brief: {brief}"
    common_args = {
        "prompt": user_message,
        "count": count,
        "model": model_choice,
        "temp": temp,
        "use_thinking": use_thinking,
        "effort": effort,
        "system_prompt": variation_prompt,
    }
    if request_timeout is not None:
        common_args["request_timeout"] = request_timeout

    if model_choice in model_info["models"]["OpenAI"]:
        common_args["effort"] = _resolve_reasoning_effort(
            model_choice,
            model_info["models"]["OpenAI"][model_choice],
            use_thinking,
            effort,
        )
        return openai_api.variation_gen(
            **common_args,
            api_key=credentials.openai_api_key,
        )
    if model_choice in model_info["models"]["Google"]:
        common_args["effort"] = _resolve_reasoning_effort(
            model_choice,
            model_info["models"]["Google"][model_choice],
            use_thinking,
            effort,
        )
        return gemini_api.variation_gen(
            **common_args,
            api_key=credentials.google_api_key,
        )
    if model_choice in model_info["models"]["Anthropic"]:
        common_args["effort"] = _resolve_reasoning_effort(
            model_choice,
            model_info["models"]["Anthropic"][model_choice],
            use_thinking,
            effort,
        )
        return claude_api.variation_gen(
            **common_args,
            api_key=credentials.anthropic_api_key,
        )

    ollama_status = ollama_api.get_ollama_status(
        host_address=credentials.ollama_host,
        **({"request_timeout": request_timeout} if request_timeout is not None else {}),
    )
    if model_choice in ollama_status["models"]:
        _resolve_reasoning_effort(model_choice, {}, use_thinking, effort)
        common_args.pop("use_thinking")
        common_args.pop("effort")
        return ollama_api.variation_gen(
            **common_args,
            host_address=credentials.ollama_host,
        )
    if not ollama_status["available"]:
        raise ValueError(
            "Invalid Model Selected. If you intended to use Ollama, it is currently unavailable."
        )
    raise ValueError("Invalid Model Selected")
