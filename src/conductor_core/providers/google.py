"""Google Gemini provider adapter for Conductor Core."""

import logging
import os

from conductor_core import models as objects
from conductor_core import music as utils
from conductor_core.errors import (
    ProviderAuthenticationError,
    ProviderConnectionError,
    ProviderTimeoutError,
    error_for_status,
)

try:
    import httpx
    from google import genai
    from google.genai import errors as genai_errors
    from google.genai import types
except ImportError:  # pragma: no cover - exercised only in minimal installs
    genai = None
    genai_errors = None
    httpx = None
    types = None

logger = logging.getLogger(__name__)


def _raise_google_error(exc: Exception, operation: str) -> None:
    if isinstance(exc, httpx.TimeoutException):
        error = ProviderTimeoutError("Google", str(exc), operation=operation)
    elif isinstance(exc, httpx.NetworkError):
        error = ProviderConnectionError("Google", str(exc), operation=operation)
    else:
        error = error_for_status(
            "Google",
            str(exc),
            getattr(exc, "code", None),
            operation=operation,
        )
    raise error from exc


def initialize_gemini_client(api_key: str | None = None, timeout: float | None = None):
    """Initialize and return a Gemini client."""
    if genai is None:
        raise ImportError("Install conductor-core[google] to use Google models.")

    resolved_api_key = api_key or os.getenv("GEMINI_API_KEY")
    if not resolved_api_key or not resolved_api_key.strip():
        raise ProviderAuthenticationError(
            "Google",
            "GEMINI_API_KEY is not set and no usable api_key was provided",
            operation="client initialization",
        )

    client_args = {"api_key": resolved_api_key}
    if timeout is not None:
        client_args["http_options"] = types.HttpOptions(
            timeout=max(1, round(timeout * 1000))
        )
    try:
        return genai.Client(**client_args)
    except (genai_errors.APIError, httpx.TimeoutException, httpx.NetworkError) as exc:
        _raise_google_error(exc, "client initialization")


def calc_cost(model, usage):
    """Calculate the cost for a Gemini completion based on token usage."""
    model_info = utils.get_model_info()
    model_cost = model_info["models"]["Google"][model]["cost"]
    prompt_tokens = usage.prompt_token_count or 0
    output_tokens = usage.candidates_token_count or 0
    cached = usage.cached_content_token_count or 0

    if isinstance(model_cost["input"], dict):
        if prompt_tokens <= 200000:
            input_cost = model_cost["input"]["<=200k"] / 1000000
            output_cost = model_cost["output"]["<=200k"] / 1000000
            cache_cost = model_cost["cache"]["<=200k"] / 1000000
        else:
            input_cost = model_cost["input"][">200k"] / 1000000
            output_cost = model_cost["output"][">200k"] / 1000000
            cache_cost = model_cost["cache"][">200k"] / 1000000
    else:
        input_cost = model_cost["input"] / 1000000
        output_cost = model_cost["output"] / 1000000
        cache_cost = (
            model_cost["cache"]["text"] / 1000000 if "cache" in model_cost else 0
        )

    new_input_tokens, cached = utils.split_reported_cache_tokens(
        prompt_tokens,
        cached,
    )
    return (
        new_input_tokens * input_cost
        + output_tokens * output_cost
        + cached * cache_cost
    )


def process_output(response):
    final_result = ""
    thinking_content = ""
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        raise ValueError("Google response did not include any candidates.")

    content = getattr(candidates[0], "content", None)
    parts = getattr(content, "parts", None) or []
    if not parts:
        raise ValueError("Google response did not include generated content parts.")

    for part in parts:
        text = getattr(part, "text", None)
        if not text:
            continue
        if getattr(part, "thought", False):
            thinking_content += text
        else:
            final_result += text
    if not final_result:
        raise ValueError("Google response did not include final loop content.")
    return final_result, thinking_content


def loop_gen(
    prompt,
    model,
    temp=0.0,
    use_thinking=None,
    effort=None,
    api_key: str | None = None,
    system_prompt: str | None = None,
    request_timeout: float | None = None,
):
    """Generate a MIDI loop using the specified Gemini model and prompt."""
    client = initialize_gemini_client(
        api_key=api_key,
        **({"timeout": request_timeout} if request_timeout is not None else {}),
    )
    loop_prompt = system_prompt or utils.get_loop_prompt()

    model_info = utils.get_model_info()
    model_config = model_info["models"]["Google"][model]
    config = {
        "system_instruction": loop_prompt,
        "response_mime_type": "application/json",
        "response_schema": objects.Loop_G,
    }
    if model_config.get("temperature_supported", True):
        config["temperature"] = temp
    model_with_thinking = model_config["extended_thinking"]
    effort_options = model_config.get("effort_options", [])

    if effort_options:
        if effort in effort_options:
            config.update(
                {
                    "thinking_config": types.ThinkingConfig(
                        thinking_level=effort,
                        include_thoughts=True,
                    )
                }
            )
        else:
            logger.warning(
                "Effort %r is not supported by model %s; using default thinking configuration.",
                effort,
                model,
            )
    elif model_with_thinking and use_thinking:
        config.update(
            {
                "thinking_config": types.ThinkingConfig(
                    thinking_budget=model_config["max_thinking_budget"],
                    include_thoughts=True,
                )
            }
        )
    elif model_with_thinking and not use_thinking:
        config.update(
            {
                "thinking_config": types.ThinkingConfig(
                    thinking_budget=model_config["min_thinking_budget"],
                    include_thoughts=True,
                )
            }
        )

    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=config,
        )
    except (genai_errors.APIError, httpx.TimeoutException, httpx.NetworkError) as exc:
        logger.error("Google request failed: %s", exc)
        _raise_google_error(exc, "request")
    content, thinking_content = process_output(response)
    midi_loop: objects.Loop_G = response.parsed
    if midi_loop is None:
        raise ValueError("Google response did not include parsed loop content.")

    messages = [
        {"role": "system", "content": loop_prompt},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": content},
    ]
    if thinking_content:
        messages.insert(2, {"role": "assistant", "content": thinking_content})

    return midi_loop, messages, calc_cost(model, response.usage_metadata)
