"""Anthropic provider adapter for Conductor Core."""

import json
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


class _UnavailableProviderError(Exception):
    """Placeholder exception class until the optional SDK is loaded."""


APIConnectionError = APIError = APITimeoutError = AuthenticationError = (
    RateLimitError
) = _UnavailableProviderError
Anthropic = None

logger = logging.getLogger(__name__)

ANTHROPIC_CACHE_CONTROL_MIN_CHARS = 4096


def _load_anthropic_sdk() -> None:
    global APIConnectionError, APIError, APITimeoutError, AuthenticationError
    global Anthropic, RateLimitError
    if Anthropic is not None:
        return
    try:
        from anthropic import (
            Anthropic as sdk_anthropic,
        )
        from anthropic import (
            APIConnectionError as sdk_api_connection_error,
        )
        from anthropic import (
            APIError as sdk_api_error,
        )
        from anthropic import (
            APITimeoutError as sdk_api_timeout_error,
        )
        from anthropic import (
            AuthenticationError as sdk_authentication_error,
        )
        from anthropic import (
            RateLimitError as sdk_rate_limit_error,
        )
    except ImportError as exc:  # pragma: no cover - exercised in minimal installs
        raise ImportError(
            "Install conductor-core[anthropic] to use Anthropic models."
        ) from exc
    Anthropic = sdk_anthropic
    APIConnectionError = sdk_api_connection_error
    APIError = sdk_api_error
    APITimeoutError = sdk_api_timeout_error
    AuthenticationError = sdk_authentication_error
    RateLimitError = sdk_rate_limit_error


def _anthropic_exception_types():
    return (
        APIConnectionError,
        APIError,
        APITimeoutError,
        AuthenticationError,
        RateLimitError,
    )


def _raise_anthropic_error(exc: Exception, operation: str) -> None:
    if isinstance(exc, AuthenticationError):
        error = ProviderAuthenticationError("Anthropic", str(exc), operation=operation)
    elif isinstance(exc, RateLimitError):
        error = ProviderRateLimitError("Anthropic", str(exc), operation=operation)
    elif isinstance(exc, APITimeoutError):
        error = ProviderTimeoutError("Anthropic", str(exc), operation=operation)
    elif isinstance(exc, APIConnectionError):
        error = ProviderConnectionError("Anthropic", str(exc), operation=operation)
    else:
        error = ProviderRequestError("Anthropic", str(exc), operation=operation)
    raise error from exc


def initialize_anthropic_client(
    api_key: str | None = None, timeout: float | None = None
):
    """Initialize and return an Anthropic client."""
    _load_anthropic_sdk()

    resolved_api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
    if not resolved_api_key or not resolved_api_key.strip():
        raise ProviderAuthenticationError(
            "Anthropic",
            "ANTHROPIC_API_KEY is not set and no usable api_key was provided",
            operation="client initialization",
        )
    client_args = {"api_key": resolved_api_key}
    if timeout is not None:
        client_args["timeout"] = timeout
    try:
        return Anthropic(**client_args)
    except _anthropic_exception_types() as exc:
        _raise_anthropic_error(exc, "client initialization")


def calc_price(model, output):
    """Calculate the cost for a completion based on token usage."""
    model_info = utils.get_model_info()
    anthropic_models = model_info["models"]["Anthropic"]
    if model not in anthropic_models:
        logger.warning("Model %s not found in model info.", model)
        return None

    model_cost = anthropic_models[model]["cost"]
    input_cost = model_cost["input"] / 1000000
    output_cost = model_cost["output"] / 1000000
    cached_5min = model_cost.get("5m cache input", 0) / 1000000
    cached_1hour = model_cost.get("1h cache input", 0) / 1000000
    cache_hits = model_cost.get("cache hits/refreshes", 0) / 1000000

    input_tokens = max(0, output.get("input_tokens", 0) or 0)
    output_tokens = max(0, output.get("output_tokens", 0) or 0)
    cache_creation = max(0, output.get("cache_creation", 0) or 0)
    cache_read = max(0, output.get("cache_read", 0) or 0)
    reported_1hour = max(0, output.get("cache_creation_1h", 0) or 0)

    # The aggregate cache_creation count is authoritative. Bill the reported 1h
    # portion at the 1h rate and everything else at the 5m rate, so cache writes
    # without a TTL breakdown deliberately land on the cheaper rate.
    cache_creation_1hour = min(cache_creation, reported_1hour)
    cache_creation_5min = cache_creation - cache_creation_1hour

    return (
        input_tokens * input_cost
        + output_tokens * output_cost
        + cache_creation_5min * cached_5min
        + cache_creation_1hour * cached_1hour
        + cache_read * cache_hits
    )


def build_system_prompt_block(loop_prompt):
    """Build Anthropic's system text block, marking only likely-cacheable prompts."""
    block = {"type": "text", "text": loop_prompt}
    if len(loop_prompt) >= ANTHROPIC_CACHE_CONTROL_MIN_CHARS:
        block["cache_control"] = {"type": "ephemeral"}
    return block


def process_streaming_response(completion):
    """Extract text, tool JSON, and token usage from a streaming response."""
    output = {
        "loop": "",
        "text": "",
        "thinking_content": "",
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_creation": 0,
        "cache_creation_5m": 0,
        "cache_creation_1h": 0,
        "cache_read": 0,
    }

    usage_fields = {
        "input_tokens": "input_tokens",
        "output_tokens": "output_tokens",
        "cache_creation": "cache_creation_input_tokens",
        "cache_read": "cache_read_input_tokens",
    }
    ttl_fields = {
        "cache_creation_5m": "ephemeral_5m_input_tokens",
        "cache_creation_1h": "ephemeral_1h_input_tokens",
    }

    def apply_usage(usage):
        # Anthropic usage snapshots are cumulative message totals: message_start
        # reports the baseline and message_delta reports the running total, so
        # each reported (non-None) field replaces the prior value instead of
        # being added to it.
        for key, attr in usage_fields.items():
            value = getattr(usage, attr, None)
            if value is not None:
                output[key] = value
        cache_creation = getattr(usage, "cache_creation", None)
        if cache_creation is not None:
            for key, attr in ttl_fields.items():
                value = getattr(cache_creation, attr, None)
                if value is not None:
                    output[key] = value

    for chunk in completion:
        if chunk.type == "message_start":
            if hasattr(chunk, "message") and hasattr(chunk.message, "usage"):
                apply_usage(chunk.message.usage)
        elif chunk.type == "content_block_delta":
            if hasattr(chunk.delta, "thinking"):
                output["thinking_content"] += chunk.delta.thinking
            elif hasattr(chunk.delta, "text"):
                output["text"] += chunk.delta.text
            elif hasattr(chunk.delta, "partial_json"):
                output["loop"] += chunk.delta.partial_json
        elif chunk.type == "message_delta":
            if hasattr(chunk, "usage"):
                apply_usage(chunk.usage)
        elif chunk.type == "message_stop":
            break
    return output


def loop_gen(
    prompt,
    model,
    temp=0.0,
    use_thinking=False,
    effort="low",
    api_key: str | None = None,
    system_prompt: str | None = None,
    request_timeout: float | None = None,
):
    """Generate a MIDI loop using the specified Anthropic model and prompt."""
    client = initialize_anthropic_client(
        api_key=api_key,
        **({"timeout": request_timeout} if request_timeout is not None else {}),
    )
    loop_prompt = system_prompt or utils.get_loop_prompt()
    tools = [
        {
            "name": "build_MIDI_loop",
            "description": "builds a music loop in MIDI format",
            "input_schema": objects.Loop.model_json_schema(),
        }
    ]

    model_info = utils.get_model_info()
    model_config = model_info["models"]["Anthropic"][model]
    always_on_adaptive_thinking = model_config.get("always_on_adaptive_thinking", False)
    effort_options = model_config.get("effort_options") or []
    if effort_options and not use_thinking:
        effort = effort_options[0]
    api_params = {
        "model": model,
        "max_tokens": model_config["max_tokens"],
        "system": [build_system_prompt_block(loop_prompt)],
        "messages": [{"role": "user", "content": prompt}],
        "tools": tools,
        "tool_choice": {"type": "tool", "name": "build_MIDI_loop"},
        "stream": True,
    }
    if not always_on_adaptive_thinking:
        api_params["temperature"] = temp

    if effort_options:
        api_params["tool_choice"] = {"type": "auto"}
        if not always_on_adaptive_thinking:
            api_params["thinking"] = {"type": "adaptive"}
        api_params["output_config"] = {"effort": effort}
        if not always_on_adaptive_thinking:
            api_params["temperature"] = 1.0
    elif use_thinking and model_config.get("extended_thinking"):
        api_params["tool_choice"] = {"type": "auto"}
        api_params["thinking"] = {
            "type": "enabled",
            "budget_tokens": model_config["max_thinking_budget"],
        }
        api_params["temperature"] = 1.0
    elif use_thinking and not model_config.get("extended_thinking"):
        logger.warning(
            "Extended thinking requested but not supported by model: %s", model
        )

    try:
        completion = client.messages.create(**api_params)
    except _anthropic_exception_types() as exc:
        logger.error("Anthropic request failed: %s", exc)
        _raise_anthropic_error(exc, "request")

    try:
        output = process_streaming_response(completion)
    except _anthropic_exception_types() as exc:
        logger.error("Anthropic stream failed: %s", exc)
        _raise_anthropic_error(exc, "stream")
    if not output["loop"]:
        raise ValueError(
            f"Model {model} did not call the build_MIDI_loop tool. "
            f"Response text: {output['text'][:200]}"
        )
    loop = objects.Loop.model_validate_json(output["loop"])

    messages = [
        {"role": "system", "content": loop_prompt},
        {"role": "user", "content": prompt},
    ]
    if output["thinking_content"]:
        messages.append({"role": "assistant", "content": output["thinking_content"]})
    messages.append({"role": "assistant", "content": output["loop"]})

    return loop, messages, calc_price(model, output)


def _process_variation_stream(completion):
    output = {
        "structured": "",
        "text": "",
        "thinking_content": "",
        "input_tokens": None,
        "output_tokens": None,
        "cache_creation": None,
        "cache_creation_5m": None,
        "cache_creation_1h": None,
        "cache_read": None,
    }
    usage_fields = {
        "input_tokens": "input_tokens",
        "output_tokens": "output_tokens",
        "cache_creation": "cache_creation_input_tokens",
        "cache_read": "cache_read_input_tokens",
    }
    ttl_fields = {
        "cache_creation_5m": "ephemeral_5m_input_tokens",
        "cache_creation_1h": "ephemeral_1h_input_tokens",
    }

    def apply_usage(usage):
        for key, attr in usage_fields.items():
            value = getattr(usage, attr, None)
            if value is not None:
                output[key] = value
        cache_creation = getattr(usage, "cache_creation", None)
        if cache_creation is not None:
            for key, attr in ttl_fields.items():
                value = getattr(cache_creation, attr, None)
                if value is not None:
                    output[key] = value

    for chunk in completion:
        if chunk.type == "message_start":
            usage = getattr(getattr(chunk, "message", None), "usage", None)
            if usage is not None:
                apply_usage(usage)
        elif chunk.type == "content_block_delta":
            if hasattr(chunk.delta, "thinking"):
                output["thinking_content"] += chunk.delta.thinking
            elif hasattr(chunk.delta, "partial_json"):
                output["structured"] += chunk.delta.partial_json
            elif hasattr(chunk.delta, "text"):
                output["text"] += chunk.delta.text
        elif chunk.type == "message_delta":
            usage = getattr(chunk, "usage", None)
            if usage is not None:
                apply_usage(usage)
        elif chunk.type == "message_stop":
            break
    return output


def _variation_usage_and_cost(model, output):
    primary_input = output["input_tokens"]
    output_tokens = output["output_tokens"]
    usage_values = (
        primary_input,
        output_tokens,
        output["cache_creation"],
        output["cache_read"],
    )
    if all(value is None for value in usage_values):
        return None, None
    input_tokens = (
        None
        if primary_input is None
        else primary_input
        + (output["cache_creation"] or 0)
        + (output["cache_read"] or 0)
    )
    total_tokens = (
        input_tokens + output_tokens
        if input_tokens is not None and output_tokens is not None
        else None
    )
    usage = VariationUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
    )
    cost = (
        calc_price(model, output)
        if primary_input is not None and output_tokens is not None
        else None
    )
    return usage, cost


def variation_gen(
    prompt,
    count,
    model,
    temp=0.0,
    use_thinking=False,
    effort="low",
    api_key: str | None = None,
    system_prompt: str | None = None,
    request_timeout: float | None = None,
):
    """Generate and structurally normalize a batch of Anthropic variations."""
    client = initialize_anthropic_client(
        api_key=api_key,
        **({"timeout": request_timeout} if request_timeout is not None else {}),
    )
    variation_prompt = (
        utils.get_variation_prompt() if system_prompt is None else system_prompt
    )
    tool_name = "build_variations"
    model_config = utils.get_model_info()["models"]["Anthropic"][model]
    always_on_adaptive = model_config.get("always_on_adaptive_thinking", False)
    effort_options = model_config.get("effort_options") or []
    api_params = {
        "model": model,
        "max_tokens": model_config["max_tokens"],
        "system": [build_system_prompt_block(variation_prompt)],
        "messages": [{"role": "user", "content": prompt}],
        "tools": [
            {
                "name": tool_name,
                "description": "builds a batch of music loop variations",
                "input_schema": build_variation_schema(count),
            }
        ],
        "tool_choice": {"type": "tool", "name": tool_name},
        "stream": True,
    }
    if not always_on_adaptive:
        api_params["temperature"] = temp
    if effort_options:
        api_params["tool_choice"] = {"type": "auto"}
        if not always_on_adaptive:
            api_params["thinking"] = {"type": "adaptive"}
        api_params["output_config"] = {"effort": effort}
        if not always_on_adaptive:
            api_params["temperature"] = 1.0
    elif use_thinking and model_config.get("extended_thinking"):
        api_params["tool_choice"] = {"type": "auto"}
        api_params["thinking"] = {
            "type": "enabled",
            "budget_tokens": model_config["max_thinking_budget"],
        }
        api_params["temperature"] = 1.0

    try:
        completion = client.messages.create(**api_params)
    except _anthropic_exception_types() as exc:
        logger.error("Anthropic variation request failed: %s", exc)
        _raise_anthropic_error(exc, "variation request")
    try:
        output = _process_variation_stream(completion)
    except _anthropic_exception_types() as exc:
        logger.error("Anthropic variation stream failed: %s", exc)
        _raise_anthropic_error(exc, "variation stream")

    raw_output = output["structured"] or None
    messages = [
        {"role": "system", "content": variation_prompt},
        {"role": "user", "content": prompt},
    ]
    if output["thinking_content"]:
        messages.append({"role": "assistant", "content": output["thinking_content"]})
    if raw_output:
        try:
            evidence = json.loads(raw_output)
        except json.JSONDecodeError:
            evidence = raw_output
        messages.append({"role": "assistant", "content": evidence})
    elif output["text"]:
        messages.append({"role": "assistant", "content": output["text"]})

    items, received_count, diagnostic = normalize_variation_output(raw_output, count)
    usage, cost = _variation_usage_and_cost(model, output)
    return ProviderVariationResult(
        provider="Anthropic",
        model=model,
        messages=messages,
        usage=usage,
        cost=cost,
        items=items,
        received_count=received_count,
        structural_diagnostic=diagnostic,
    )
