"""Anthropic provider adapter for Conductor Core."""

import logging
import os
from typing import Any

from conductor_core import models as objects
from conductor_core import music as utils
from conductor_core.errors import (
    ProviderAuthenticationError,
    ProviderConnectionError,
    ProviderRateLimitError,
    ProviderRequestError,
    ProviderTimeoutError,
)

Anthropic: Any
APIConnectionError: Any
APIError: Any
APITimeoutError: Any
AuthenticationError: Any
RateLimitError: Any

try:
    import anthropic as _anthropic
except ImportError:  # pragma: no cover - exercised only in minimal installs
    APIConnectionError = APIError = APITimeoutError = AuthenticationError = (
        RateLimitError
    ) = ()
    Anthropic = None
else:
    Anthropic = _anthropic.Anthropic
    APIConnectionError = _anthropic.APIConnectionError
    APIError = _anthropic.APIError
    APITimeoutError = _anthropic.APITimeoutError
    AuthenticationError = _anthropic.AuthenticationError
    RateLimitError = _anthropic.RateLimitError

logger = logging.getLogger(__name__)

ANTHROPIC_CACHE_CONTROL_MIN_CHARS = 4096


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
    if Anthropic is None:
        raise ImportError("Install conductor-core[anthropic] to use Anthropic models.")

    resolved_api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
    if not resolved_api_key or not resolved_api_key.strip():
        raise ProviderAuthenticationError(
            "Anthropic",
            "ANTHROPIC_API_KEY is not set and no usable api_key was provided",
            operation="client initialization",
        )
    try:
        if timeout is None:
            return Anthropic(api_key=resolved_api_key)
        return Anthropic(api_key=resolved_api_key, timeout=timeout)
    except (
        AuthenticationError,
        RateLimitError,
        APITimeoutError,
        APIConnectionError,
        APIError,
    ) as exc:
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
    cache_hits = model_cost.get("cache hits/refreshes", 0) / 1000000

    return (
        output["input_tokens"] * input_cost
        + output["output_tokens"] * output_cost
        + output["cache_creation"] * cached_5min
        + output["cache_read"] * cache_hits
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
        "cache_read": 0,
    }

    for chunk in completion:
        if chunk.type == "message_start":
            if hasattr(chunk, "message") and hasattr(chunk.message, "usage"):
                output["input_tokens"] += getattr(
                    chunk.message.usage, "input_tokens", 0
                )
                output["output_tokens"] += getattr(
                    chunk.message.usage, "output_tokens", 0
                )
                output["cache_creation"] += getattr(
                    chunk.message.usage,
                    "cache_creation_input_tokens",
                    0,
                )
                output["cache_read"] += getattr(
                    chunk.message.usage,
                    "cache_read_input_tokens",
                    0,
                )
        elif chunk.type == "content_block_delta":
            if hasattr(chunk.delta, "thinking"):
                output["thinking_content"] += chunk.delta.thinking
            elif hasattr(chunk.delta, "text"):
                output["text"] += chunk.delta.text
            elif hasattr(chunk.delta, "partial_json"):
                output["loop"] += chunk.delta.partial_json
        elif chunk.type == "message_delta":
            if hasattr(chunk, "usage"):
                output["input_tokens"] += getattr(chunk.usage, "input_tokens", 0)
                output["output_tokens"] += getattr(chunk.usage, "output_tokens", 0)
                output["cache_creation"] += getattr(
                    chunk.usage,
                    "cache_creation_input_tokens",
                    0,
                )
                output["cache_read"] += getattr(
                    chunk.usage,
                    "cache_read_input_tokens",
                    0,
                )
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

    if model_config.get("effort_options"):
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
    except (
        AuthenticationError,
        RateLimitError,
        APITimeoutError,
        APIConnectionError,
        APIError,
    ) as exc:
        logger.error("Anthropic request failed: %s", exc)
        _raise_anthropic_error(exc, "request")

    try:
        output = process_streaming_response(completion)
    except (
        AuthenticationError,
        RateLimitError,
        APITimeoutError,
        APIConnectionError,
        APIError,
    ) as exc:
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
