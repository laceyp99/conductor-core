"""Google Gemini provider adapter for Conductor Core."""

import logging
import os

from conductor_core import models as objects
from conductor_core import music as utils
from conductor_core._internal_types import ProviderVariationResult
from conductor_core.errors import (
    ProviderAuthenticationError,
    ProviderConnectionError,
    ProviderTimeoutError,
    error_for_status,
)
from conductor_core.providers._variations import (
    build_variation_schema,
    normalize_variation_output,
)
from conductor_core.variations import VariationUsage

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
    output_tokens = (usage.candidates_token_count or 0) + (
        usage.thoughts_token_count or 0
    )
    cached = usage.cached_content_token_count or 0

    input_cost = model_cost["input"] / 1000000
    output_cost = model_cost["output"] / 1000000
    cache_cost = model_cost["cache"]["text"] / 1000000 if "cache" in model_cost else 0

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
        "response_json_schema": objects.Loop.model_json_schema(),
    }
    if model_config.get("temperature_supported", True):
        config["temperature"] = temp
    model_with_thinking = model_config["extended_thinking"]
    effort_options = model_config.get("effort_options", [])
    if effort_options and not use_thinking:
        effort = effort_options[0]

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
    midi_loop = objects.Loop.model_validate_json(content)

    messages = [
        {"role": "system", "content": loop_prompt},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": content},
    ]
    if thinking_content:
        messages.insert(2, {"role": "assistant", "content": thinking_content})

    return midi_loop, messages, calc_cost(model, response.usage_metadata)


def _variation_usage_and_cost(model, response):
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return None, None
    input_tokens = getattr(usage, "prompt_token_count", None)
    candidate_tokens = getattr(usage, "candidates_token_count", None)
    thinking_tokens = getattr(usage, "thoughts_token_count", None)
    output_tokens = (
        None if candidate_tokens is None else candidate_tokens + (thinking_tokens or 0)
    )
    total_tokens = getattr(usage, "total_token_count", None)
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens
    normalized = VariationUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
    )
    cost = (
        calc_cost(model, usage)
        if input_tokens is not None and output_tokens is not None
        else None
    )
    return normalized, cost


def _variation_output(response):
    final_result = ""
    thinking_content = ""
    candidates = getattr(response, "candidates", None) or []
    if candidates:
        content = getattr(candidates[0], "content", None)
        for part in getattr(content, "parts", None) or []:
            text = getattr(part, "text", None)
            if not text:
                continue
            if getattr(part, "thought", False):
                thinking_content += text
            else:
                final_result += text
    return final_result or None, thinking_content


def variation_gen(
    prompt,
    count,
    model,
    temp=0.0,
    use_thinking=None,
    effort=None,
    api_key: str | None = None,
    system_prompt: str | None = None,
    request_timeout: float | None = None,
):
    """Generate and structurally normalize a batch of Gemini variations."""
    client = initialize_gemini_client(
        api_key=api_key,
        **({"timeout": request_timeout} if request_timeout is not None else {}),
    )
    variation_prompt = (
        utils.get_variation_prompt() if system_prompt is None else system_prompt
    )
    model_config = utils.get_model_info()["models"]["Google"][model]
    config = {
        "system_instruction": variation_prompt,
        "response_mime_type": "application/json",
        "response_json_schema": build_variation_schema(count),
    }
    if model_config.get("temperature_supported", True):
        config["temperature"] = temp
    effort_options = model_config.get("effort_options", [])
    if effort_options:
        config["thinking_config"] = types.ThinkingConfig(
            thinking_level=effort,
            include_thoughts=True,
        )
    elif model_config.get("extended_thinking"):
        config["thinking_config"] = types.ThinkingConfig(
            thinking_budget=(
                model_config["max_thinking_budget"]
                if use_thinking
                else model_config["min_thinking_budget"]
            ),
            include_thoughts=True,
        )

    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=config,
        )
    except (genai_errors.APIError, httpx.TimeoutException, httpx.NetworkError) as exc:
        logger.error("Google variation request failed: %s", exc)
        _raise_google_error(exc, "variation request")

    raw_output, thinking_content = _variation_output(response)
    messages = [
        {"role": "system", "content": variation_prompt},
        {"role": "user", "content": prompt},
    ]
    if thinking_content:
        messages.append({"role": "assistant", "content": thinking_content})
    if raw_output:
        messages.append({"role": "assistant", "content": raw_output})
    items, received_count, diagnostic = normalize_variation_output(raw_output, count)
    usage, cost = _variation_usage_and_cost(model, response)
    return ProviderVariationResult(
        provider="Google",
        model=model,
        messages=messages,
        usage=usage,
        cost=cost,
        items=items,
        received_count=received_count,
        structural_diagnostic=diagnostic,
    )
