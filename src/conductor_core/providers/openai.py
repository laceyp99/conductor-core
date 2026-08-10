"""OpenAI provider adapter for Conductor Core."""

import logging
import os

from conductor_core import models as objects
from conductor_core import music as utils
from conductor_core.errors import (
    ProviderAuthenticationError,
    ProviderConnectionError,
    ProviderRateLimitError,
    ProviderRequestError,
    ProviderTimeoutError,
)

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
    input_cost = model_info["models"]["OpenAI"][model]["cost"].get("input", 0) / 1000000
    output_cost = (
        model_info["models"]["OpenAI"][model]["cost"].get("output", 0) / 1000000
    )
    cached_input_cost = (
        model_info["models"]["OpenAI"][model]["cost"].get("cached input", 0) / 1000000
    )

    if (
        hasattr(usage, "input_tokens_details")
        and usage.input_tokens_details
        and hasattr(usage.input_tokens_details, "cached_tokens")
    ):
        new_input_tokens, cached_tokens = utils.split_reported_cache_tokens(
            usage.input_tokens,
            usage.input_tokens_details.cached_tokens,
        )
    else:
        new_input_tokens = usage.input_tokens
        cached_tokens = 0

    return (
        input_cost * new_input_tokens
        + output_cost * usage.output_tokens
        + cached_input_cost * cached_tokens
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
