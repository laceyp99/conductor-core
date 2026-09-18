"""Ollama provider adapter for Conductor Core."""

import logging
import os

from conductor_core import models as objects
from conductor_core import music as utils
from conductor_core._internal_types import ProviderVariationResult
from conductor_core.errors import (
    ProviderConnectionError,
    ProviderRequestError,
    ProviderTimeoutError,
    error_for_status,
)
from conductor_core.providers._variations import (
    build_variation_schema,
    normalize_variation_output,
)
from conductor_core.variations import VariationUsage


class _UnavailableProviderError(Exception):
    """Placeholder exception class until the optional SDK is loaded."""


class _PlaceholderOllama:
    Client = None
    RequestError = _UnavailableProviderError
    ResponseError = _UnavailableProviderError


class _PlaceholderHttpx:
    TimeoutException = _UnavailableProviderError
    NetworkError = _UnavailableProviderError


httpx = _PlaceholderHttpx()
ollama = _PlaceholderOllama()

logger = logging.getLogger(__name__)


def _load_ollama_sdk() -> None:
    global httpx, ollama
    if getattr(ollama, "Client", None) is not None:
        return
    try:
        import httpx as sdk_httpx
        import ollama as sdk_ollama
    except ImportError as exc:  # pragma: no cover - exercised in minimal installs
        raise ImportError(
            "Install conductor-core[ollama] to use Ollama models."
        ) from exc
    httpx = sdk_httpx
    ollama = sdk_ollama


def _ollama_exception_types():
    return (
        httpx.TimeoutException,
        httpx.NetworkError,
        ConnectionError,
        ollama.RequestError,
        ollama.ResponseError,
    )


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
    _load_ollama_sdk()

    client_args = {"host": _resolve_host(host_address)}
    if timeout is not None:
        client_args["timeout"] = timeout
    try:
        return ollama.Client(**client_args)
    except _ollama_exception_types() as exc:
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
        "host": host,
        "error": None,
    }

    if getattr(ollama, "Client", None) is None:
        try:
            _load_ollama_sdk()
        except ImportError as exc:
            status["error"] = str(exc)
            return status

    try:
        client = initialize_ollama_client(
            host_address=host,
            **({"timeout": request_timeout} if request_timeout is not None else {}),
        )
        status["models"] = [model.model for model in client.list().models]
        status["available"] = True
    except Exception as exc:
        status["error"] = str(exc)
        logger.warning("Ollama unavailable at %s: %s", host, exc)

    return status


def get_model_list(host_address: str | None = None):
    """Get the available Ollama model names."""
    return get_ollama_status(host_address=host_address)["models"]


def loop_gen(
    prompt,
    model,
    temp=0.0,
    host_address: str | None = None,
    system_prompt: str | None = None,
    request_timeout: float | None = None,
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
    try:
        completion = client.chat(
            model=model,
            messages=messages,
            format=objects.Loop.model_json_schema(),
            options={"temperature": temp},
        )
    except _ollama_exception_types() as exc:
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


def _variation_usage(completion):
    input_tokens = getattr(completion, "prompt_eval_count", None)
    output_tokens = getattr(completion, "eval_count", None)
    if input_tokens is None and output_tokens is None:
        return None
    total_tokens = (
        input_tokens + output_tokens
        if input_tokens is not None and output_tokens is not None
        else None
    )
    return VariationUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
    )


def variation_gen(
    prompt,
    count,
    model,
    temp=0.0,
    host_address: str | None = None,
    system_prompt: str | None = None,
    request_timeout: float | None = None,
):
    """Generate and structurally normalize a batch of Ollama variations."""
    client = initialize_ollama_client(
        host_address=host_address,
        **({"timeout": request_timeout} if request_timeout is not None else {}),
    )
    variation_prompt = (
        utils.get_variation_prompt() if system_prompt is None else system_prompt
    )
    messages = [
        {"role": "system", "content": variation_prompt},
        {"role": "user", "content": prompt},
    ]
    try:
        completion = client.chat(
            model=model,
            messages=messages,
            format=build_variation_schema(count),
            options={"temperature": temp},
        )
    except _ollama_exception_types() as exc:
        logger.error("Ollama variation request failed: %s", exc)
        _raise_ollama_error(exc, "variation request")

    message = getattr(completion, "message", None)
    raw_output = getattr(message, "content", None)
    thinking = getattr(message, "thinking", None)
    if thinking:
        messages.append({"role": "assistant", "content": thinking})
    if raw_output:
        messages.append({"role": "assistant", "content": raw_output})
    items, received_count, diagnostic = normalize_variation_output(raw_output, count)
    return ProviderVariationResult(
        provider="Ollama",
        model=model,
        messages=messages,
        usage=_variation_usage(completion),
        cost=0.0,
        items=items,
        received_count=received_count,
        structural_diagnostic=diagnostic,
    )
