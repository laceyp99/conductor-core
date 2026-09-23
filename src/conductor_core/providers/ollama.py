"""Ollama provider adapter for Conductor Core."""

import logging
import os

from conductor_core import models as objects
from conductor_core import music as utils
from conductor_core.errors import (
    ProviderConnectionError,
    ProviderRequestError,
    ProviderTimeoutError,
    error_for_status,
)
from conductor_core.providers._variations import VariationCollection
from conductor_core.variations import VariationUsage

try:
    import httpx
    import ollama
except ImportError:  # pragma: no cover - exercised only in minimal installs
    httpx = None
    ollama = None

logger = logging.getLogger(__name__)


def _get_model_capabilities(client, model_name, host):
    try:
        model_info = client.show(model_name)
        capabilities = getattr(model_info, "capabilities", None) or []
        supports_thinking = "thinking" in capabilities
    except Exception as exc:
        logger.warning(
            "Could not inspect Ollama model %s at %s: %s", model_name, host, exc
        )
        supports_thinking = False
    # Ollama's show capabilities only reports thinking support, not whether the
    # model accepts effort levels, so expose on/off only.
    return {
        "extended_thinking": supports_thinking,
        "effort_options": [],
        "temperature_supported": True,
    }


def _resolve_host(host_address: str | None = None) -> str:
    return (
        host_address or os.getenv("OLLAMA_API_HOST_ADDRESS") or "http://localhost:11434"
    )


def _raise_ollama_error(exc: Exception, operation: str) -> None:
    if isinstance(exc, httpx.TimeoutException):
        error = ProviderTimeoutError("Ollama", str(exc), operation=operation)
    elif isinstance(exc, (httpx.NetworkError, ConnectionError)):
        error = ProviderConnectionError("Ollama", str(exc), operation=operation)
    elif isinstance(exc, ollama.ResponseError):
        error = error_for_status(
            "Ollama",
            str(exc),
            exc.status_code,
            operation=operation,
        )
    else:
        error = ProviderRequestError("Ollama", str(exc), operation=operation)
    raise error from exc


def initialize_ollama_client(
    host_address: str | None = None, timeout: float | None = None
):
    """Initialize and return an Ollama client."""
    if ollama is None:
        raise ImportError("Install conductor-core[ollama] to use Ollama models.")

    client_args = {"host": _resolve_host(host_address)}
    if timeout is not None:
        client_args["timeout"] = timeout
    try:
        return ollama.Client(**client_args)
    except (
        httpx.TimeoutException,
        httpx.NetworkError,
        ConnectionError,
        ollama.RequestError,
        ollama.ResponseError,
    ) as exc:
        _raise_ollama_error(exc, "client initialization")


def get_ollama_status(
    host_address: str | None = None,
    request_timeout: float | None = None,
):
    """Get the current Ollama availability and discovered models."""
    host = _resolve_host(host_address)
    status = {
        "available": False,
        "models": [],
        "model_capabilities": {},
        "host": host,
        "error": None,
    }

    if ollama is None:
        status["error"] = "Install conductor-core[ollama] to use Ollama models."
        return status

    try:
        client = initialize_ollama_client(
            host_address=host,
            **({"timeout": request_timeout} if request_timeout is not None else {}),
        )
        status["models"] = [model.model for model in client.list().models]
        status["model_capabilities"] = {
            model_name: _get_model_capabilities(client, model_name, host)
            for model_name in status["models"]
        }
        status["available"] = True
    except Exception as exc:
        status["error"] = str(exc)
        logger.warning("Ollama unavailable at %s: %s", host, exc)

    return status


def get_model_list(host_address: str | None = None):
    """Get the available Ollama model names."""
    return get_ollama_status(host_address=host_address)["models"]


def _thinking_option(model_capabilities, use_thinking, effort):
    """Return the Ollama ``think`` value, or None when unsupported/unknown."""
    if not model_capabilities or not model_capabilities.get("extended_thinking"):
        return None
    effort_options = model_capabilities.get("effort_options") or []
    return effort if effort_options else bool(use_thinking)


def loop_gen(
    prompt,
    model,
    temp=0.0,
    host_address: str | None = None,
    system_prompt: str | None = None,
    request_timeout: float | None = None,
    use_thinking: bool = False,
    effort: str | None = "low",
    model_capabilities: dict | None = None,
):
    """Generate a MIDI loop using the specified Ollama model and prompt."""
    client = initialize_ollama_client(
        host_address=host_address,
        **({"timeout": request_timeout} if request_timeout is not None else {}),
    )
    loop_prompt = system_prompt or utils.get_loop_prompt()
    messages = [
        {"role": "system", "content": loop_prompt},
        {"role": "user", "content": prompt},
    ]
    if model_capabilities is None:
        model_capabilities = _get_model_capabilities(
            client, model, _resolve_host(host_address)
        )
    think = _thinking_option(model_capabilities, use_thinking, effort)
    chat_options = {
        "model": model,
        "messages": messages,
        "format": objects.Loop.model_json_schema(),
        "options": {"temperature": temp},
    }
    if think is not None:
        chat_options["think"] = think
    try:
        completion = client.chat(**chat_options)
    except (
        httpx.TimeoutException,
        httpx.NetworkError,
        ConnectionError,
        ollama.RequestError,
        ollama.ResponseError,
    ) as exc:
        logger.error("Ollama request failed: %s", exc)
        _raise_ollama_error(exc, "request")
    message = getattr(completion, "message", None)
    content = getattr(message, "content", None)
    if not content:
        raise ValueError("Ollama response did not include generated content.")

    midi_loop = objects.Loop.model_validate_json(content)
    thinking = getattr(message, "thinking", None)
    if thinking:
        messages.append({"role": "assistant", "content": thinking})
    messages.append({"role": "assistant", "content": str(midi_loop)})
    return midi_loop, messages, 0


def variations_gen(
    prompt,
    model,
    temp=0.0,
    host_address: str | None = None,
    system_prompt: str | None = None,
    request_timeout: float | None = None,
    use_thinking: bool = False,
    effort: str | None = "low",
    model_capabilities: dict | None = None,
):
    """Generate an ordered collection of loops in one Ollama response."""
    client = initialize_ollama_client(
        host_address=host_address,
        **({"timeout": request_timeout} if request_timeout is not None else {}),
    )
    loop_prompt = system_prompt or utils.get_variation_prompt()
    messages = [
        {"role": "system", "content": loop_prompt},
        {"role": "user", "content": prompt},
    ]
    if model_capabilities is None:
        model_capabilities = _get_model_capabilities(
            client, model, _resolve_host(host_address)
        )
    think = _thinking_option(model_capabilities, use_thinking, effort)
    chat_options = {
        "model": model,
        "messages": messages,
        "format": VariationCollection.model_json_schema(),
        "options": {"temperature": temp},
    }
    if think is not None:
        chat_options["think"] = think
    try:
        completion = client.chat(**chat_options)
    except (
        httpx.TimeoutException,
        httpx.NetworkError,
        ConnectionError,
        ollama.RequestError,
        ollama.ResponseError,
    ) as exc:
        _raise_ollama_error(exc, "request")
    message = getattr(completion, "message", None)
    content = getattr(message, "content", None)
    if not content:
        raise ValueError("Ollama response did not include generated content.")
    collection = VariationCollection.model_validate_json(content)
    thinking = getattr(message, "thinking", None)
    if thinking:
        messages.append({"role": "assistant", "content": thinking})
    messages.append({"role": "assistant", "content": content})
    input_tokens = getattr(completion, "prompt_eval_count", None)
    output_tokens = getattr(completion, "eval_count", None)
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
        if input_tokens is not None or output_tokens is not None
        else None
    )
    return collection, messages, 0, usage
