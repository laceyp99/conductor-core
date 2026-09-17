"""OpenAI provider adapter for Conductor Core."""

import logging
import os

from conductor_core import models as objects
from conductor_core import music as utils
from conductor_core._internal_types import ProviderVariationResult
from conductor_core.errors import (
    ProviderAuthenticationError,
    ProviderConnectionError,
    ProviderRateLimitError,
    ProviderRequestError,
    ProviderTimeoutError,
)
from conductor_core.providers._variations import (
    build_variation_schema,
    normalize_variation_output,
)
from conductor_core.variations import VariationUsage

try:
    from openai import (
        APIConnectionError,
        APIError,
        APITimeoutError,
        AuthenticationError,
        OpenAI,
        RateLimitError,
    )
except ImportError:  # pragma: no cover - exercised only in minimal installs
    APIConnectionError = APIError = APITimeoutError = AuthenticationError = (
        RateLimitError
    ) = ()
    OpenAI = None

logger = logging.getLogger(__name__)


def _raise_openai_error(exc: Exception, operation: str) -> None:
    if isinstance(exc, AuthenticationError):
        error = ProviderAuthenticationError("OpenAI", str(exc), operation=operation)
    elif isinstance(exc, RateLimitError):
        error = ProviderRateLimitError("OpenAI", str(exc), operation=operation)
    elif isinstance(exc, APITimeoutError):
        error = ProviderTimeoutError("OpenAI", str(exc), operation=operation)
    elif isinstance(exc, APIConnectionError):
        error = ProviderConnectionError("OpenAI", str(exc), operation=operation)
    else:
        error = ProviderRequestError("OpenAI", str(exc), operation=operation)
    raise error from exc


def initialize_openai_client(api_key: str | None = None, timeout: float | None = None):
    """Initialize and return an OpenAI client."""
    if OpenAI is None:
        raise ImportError("Install conductor-core[openai] to use OpenAI models.")

    resolved_api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not resolved_api_key or not resolved_api_key.strip():
        raise ProviderAuthenticationError(
            "OpenAI",
            "OPENAI_API_KEY is not set and no usable api_key was provided",
            operation="client initialization",
        )
    client_args = {"api_key": resolved_api_key}
    if timeout is not None:
        client_args["timeout"] = timeout
    try:
        return OpenAI(**client_args)
    except (
        AuthenticationError,
        RateLimitError,
        APITimeoutError,
        APIConnectionError,
        APIError,
    ) as exc:
        _raise_openai_error(exc, "client initialization")


def calc_price(model, response):
    """Calculate the cost for a given response based on token usage."""
    model_info = utils.get_model_info()
    usage = response.usage
    model_cost = model_info["models"]["OpenAI"][model]["cost"]
    input_tokens = max(0, usage.input_tokens or 0)
    output_tokens = max(0, usage.output_tokens or 0)
    input_cost = model_cost.get("input", 0) / 1000000
    output_cost = model_cost.get("output", 0) / 1000000
    cached_input_cost = model_cost.get("cached input", 0) / 1000000
    cache_write_cost = model_cost.get("cache write", model_cost.get("input", 0))
    cache_write_cost /= 1000000
    input_details = getattr(usage, "input_tokens_details", None)
    reported_cache_writes = max(0, getattr(input_details, "cache_write_tokens", 0) or 0)
    reported_cached_tokens = max(0, getattr(input_details, "cached_tokens", 0) or 0)
    cache_write_tokens = min(input_tokens, reported_cache_writes)
    cached_tokens = min(
        input_tokens - cache_write_tokens,
        reported_cached_tokens,
    )
    new_input_tokens = input_tokens - cache_write_tokens - cached_tokens

    return (
        input_cost * new_input_tokens
        + output_cost * output_tokens
        + cached_input_cost * cached_tokens
        + cache_write_cost * cache_write_tokens
    )


def extract_reasoning(response):
    reasoning = ""
    for item in getattr(response, "output", []):
        if getattr(item, "type", None) == "reasoning":
            for summary in getattr(item, "summary", []) or []:
                text = getattr(summary, "text", None)
                if text:
                    reasoning += text + "\n"
    return reasoning


def loop_gen(
    prompt,
    model,
    temp=0.0,
    use_thinking=False,
    effort=None,
    api_key: str | None = None,
    system_prompt: str | None = None,
    request_timeout: float | None = None,
):
    """Generate a MIDI loop using the specified OpenAI model and prompt."""
    client = initialize_openai_client(
        api_key=api_key,
        **({"timeout": request_timeout} if request_timeout is not None else {}),
    )
    loop_prompt = system_prompt or utils.get_loop_prompt()
    messages = [
        {"role": "system", "content": loop_prompt},
        {"role": "user", "content": prompt},
    ]

    model_info = utils.get_model_info()
    request_params = {
        "model": model,
        "instructions": loop_prompt,
        "input": prompt,
        "text_format": objects.Loop,
        "store": False,
    }

    model_config = model_info["models"]["OpenAI"][model]
    effort_options = model_config.get("effort_options") or []
    if effort_options and not use_thinking:
        effort = effort_options[0]
    if model_config.get("extended_thinking") and effort:
        request_params["reasoning"] = {"effort": effort, "summary": "auto"}
    else:
        request_params["temperature"] = temp

    try:
        response = client.responses.parse(**request_params)
    except (
        AuthenticationError,
        RateLimitError,
        APITimeoutError,
        APIConnectionError,
        APIError,
    ) as exc:
        logger.error("OpenAI request failed: %s", exc)
        _raise_openai_error(exc, "request")

    if response.output_parsed is None:
        raise ValueError("OpenAI response did not include parsed loop content.")

    reasoning = extract_reasoning(response)
    if reasoning:
        messages.append({"role": "assistant", "content": reasoning})
    messages.append({"role": "assistant", "content": str(response.output_parsed)})

    return response.output_parsed, messages, calc_price(model, response)


def _json_compatible(value):
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, list):
        return [_json_compatible(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _variation_usage_and_cost(model, response):
    usage = getattr(response, "usage", None)
    if usage is None:
        return None, None
    input_tokens = getattr(usage, "input_tokens", None)
    output_tokens = getattr(usage, "output_tokens", None)
    total_tokens = getattr(usage, "total_tokens", None)
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens
    normalized = VariationUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
    )
    cost = (
        calc_price(model, response)
        if input_tokens is not None and output_tokens is not None
        else None
    )
    return normalized, cost


def variation_gen(
    prompt,
    count,
    model,
    temp=0.0,
    use_thinking=False,
    effort=None,
    api_key: str | None = None,
    system_prompt: str | None = None,
    request_timeout: float | None = None,
):
    """Generate and structurally normalize a batch of OpenAI variations."""
    client = initialize_openai_client(
        api_key=api_key,
        **({"timeout": request_timeout} if request_timeout is not None else {}),
    )
    variation_prompt = (
        utils.get_variation_prompt() if system_prompt is None else system_prompt
    )
    messages = [
        {"role": "system", "content": variation_prompt},
        {"role": "user", "content": prompt},
    ]
    request_params = {
        "model": model,
        "instructions": variation_prompt,
        "input": prompt,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "variation_batch",
                "schema": build_variation_schema(count),
                "strict": True,
            }
        },
        "store": False,
    }
    model_config = utils.get_model_info()["models"]["OpenAI"][model]
    if model_config.get("extended_thinking") and effort:
        request_params["reasoning"] = {"effort": effort, "summary": "auto"}
    else:
        request_params["temperature"] = temp

    try:
        response = client.responses.create(**request_params)
    except (
        AuthenticationError,
        RateLimitError,
        APITimeoutError,
        APIConnectionError,
        APIError,
    ) as exc:
        logger.error("OpenAI variation request failed: %s", exc)
        _raise_openai_error(exc, "variation request")

    reasoning = extract_reasoning(response)
    if reasoning:
        messages.append({"role": "assistant", "content": reasoning})
    raw_output = getattr(response, "output_text", None)
    if raw_output:
        messages.append({"role": "assistant", "content": raw_output})
    else:
        evidence = _json_compatible(getattr(response, "output", []))
        if evidence:
            messages.append({"role": "assistant", "content": evidence})
    items, received_count, diagnostic = normalize_variation_output(raw_output, count)
    usage, cost = _variation_usage_and_cost(model, response)
    return ProviderVariationResult(
        provider="OpenAI",
        model=model,
        messages=messages,
        usage=usage,
        cost=cost,
        items=items,
        received_count=received_count,
        structural_diagnostic=diagnostic,
    )
