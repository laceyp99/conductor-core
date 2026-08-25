"""Provider routing for Conductor Core."""

import logging

from conductor_core.config import ProviderCredentials
from conductor_core.music import get_model_info
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
    _return_provider: bool = False,
):
    """Generate MIDI loop data by routing a prompt to the selected provider."""
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
            force_refresh=True,
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

    if _return_provider:
        return loop, messages, loop_cost, provider
    return loop, messages, loop_cost
