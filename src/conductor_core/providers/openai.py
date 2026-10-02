"""OpenAI provider adapter for Conductor Core."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, NoReturn, cast

from typing_extensions import NotRequired, TypedDict

if TYPE_CHECKING:
    from openai import OpenAI as OpenAIClient
    from openai.types.shared.reasoning_effort import ReasoningEffort
    from openai.types.shared_params.reasoning import Reasoning

from conductor_core import models as objects
from conductor_core import music as utils
from conductor_core.errors import (
    ProviderAuthenticationError,
    ProviderConnectionError,
    ProviderRateLimitError,
    ProviderRequestError,
    ProviderTimeoutError,
)
from conductor_core.provider_types import ProviderMessage
from conductor_core.providers._variations import VariationCollection
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
    # Keep exception names as classes so handlers remain valid in minimal installs.
    class APIError(Exception):
        pass

    class APIConnectionError(APIError):
        pass

    class APITimeoutError(APIError):
        pass

    class AuthenticationError(APIError):
        pass

    class RateLimitError(APIError):
        pass

    OpenAI = None

logger = logging.getLogger(__name__)


class _LoopParseKwargs(TypedDict):
    model: str
    instructions: str
    input: str
    text_format: type[objects.Loop]
    store: bool
    reasoning: NotRequired[Reasoning]
    temperature: NotRequired[float]


class _VariationParseKwargs(TypedDict):
    model: str
    instructions: str
    input: str
    text_format: type[VariationCollection]
    store: bool
    reasoning: NotRequired[Reasoning]
    temperature: NotRequired[float]


def _reasoning_params(effort: str) -> Reasoning:
    if effort not in ("none", "minimal", "low", "medium", "high", "xhigh", "max"):
        raise ValueError(f"Unsupported OpenAI reasoning effort: {effort}")
    return {"effort": cast("ReasoningEffort", effort), "summary": "auto"}


def _raise_openai_error(exc: Exception, operation: str) -> NoReturn:
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


def initialize_openai_client(
    api_key: str | None = None, timeout: float | None = None
) -> OpenAIClient:
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
    try:
        if timeout is None:
            return OpenAI(api_key=resolved_api_key)
        return OpenAI(api_key=resolved_api_key, timeout=timeout)
    except (
        AuthenticationError,
        RateLimitError,
        APITimeoutError,
        APIConnectionError,
        APIError,
    ) as exc:
        _raise_openai_error(exc, "client initialization")


def calc_price(model, response) -> float:
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


def extract_reasoning(response) -> str:
    reasoning = ""
    for item in getattr(response, "output", []):
        if getattr(item, "type", None) == "reasoning":
            for summary in getattr(item, "summary", []) or []:
                text = getattr(summary, "text", None)
                if text:
                    reasoning += text + "\n"
    return reasoning


def loop_gen(
    prompt: str,
    model: str,
    temp: float = 0.0,
    use_thinking: bool = False,
    effort: str | None = None,
    api_key: str | None = None,
    system_prompt: str | None = None,
    request_timeout: float | None = None,
) -> tuple[objects.Loop, list[ProviderMessage], float]:
    """Generate a MIDI loop using the specified OpenAI model and prompt."""
    client = initialize_openai_client(
        api_key=api_key,
        **({"timeout": request_timeout} if request_timeout is not None else {}),
    )
    loop_prompt = system_prompt or utils.get_loop_prompt()
    messages: list[ProviderMessage] = [
        {"role": "system", "content": loop_prompt},
        {"role": "user", "content": prompt},
    ]

    model_info = utils.get_model_info()
    request_params: _LoopParseKwargs = {
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
        request_params["reasoning"] = _reasoning_params(effort)
    elif model_config.get("temperature_supported", True):
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
    messages.append(
        {"role": "assistant", "content": response.output_parsed.model_dump_json()}
    )

    return response.output_parsed, messages, calc_price(model, response)


def variations_gen(
    prompt: str,
    model: str,
    temp: float = 0.0,
    use_thinking: bool = False,
    effort: str | None = None,
    api_key: str | None = None,
    system_prompt: str | None = None,
    request_timeout: float | None = None,
) -> tuple[
    VariationCollection, list[ProviderMessage], float | None, VariationUsage | None
]:
    """Generate an ordered collection of loops in one OpenAI response."""
    client = initialize_openai_client(
        api_key=api_key,
        **({"timeout": request_timeout} if request_timeout is not None else {}),
    )
    loop_prompt = system_prompt or utils.get_variation_prompt()
    messages: list[ProviderMessage] = [
        {"role": "system", "content": loop_prompt},
        {"role": "user", "content": prompt},
    ]
    request_params: _VariationParseKwargs = {
        "model": model,
        "instructions": loop_prompt,
        "input": prompt,
        "text_format": VariationCollection,
        "store": False,
    }
    model_config = utils.get_model_info()["models"]["OpenAI"][model]
    effort_options = model_config.get("effort_options") or []
    if effort_options and not use_thinking:
        effort = effort_options[0]
    if model_config.get("extended_thinking") and effort:
        request_params["reasoning"] = _reasoning_params(effort)
    elif model_config.get("temperature_supported", True):
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
        _raise_openai_error(exc, "request")
    if response.output_parsed is None:
        raise ValueError("OpenAI response did not include parsed variation content.")
    collection = VariationCollection.model_validate(response.output_parsed)
    reasoning = extract_reasoning(response)
    if reasoning:
        messages.append({"role": "assistant", "content": reasoning})
    messages.append({"role": "assistant", "content": collection.model_dump_json()})
    raw_usage = getattr(response, "usage", None)
    input_tokens = getattr(raw_usage, "input_tokens", None)
    output_tokens = getattr(raw_usage, "output_tokens", None)
    usage = (
        VariationUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=(
                input_tokens + output_tokens
                if input_tokens is not None and output_tokens is not None
                else None
            ),
        )
        if raw_usage is not None
        else None
    )
    cost = calc_price(model, response) if raw_usage is not None else None
    return collection, messages, cost, usage
