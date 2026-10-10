"""Ollama provider adapter for Conductor Core."""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal, NoReturn

from typing_extensions import TypedDict

if TYPE_CHECKING:
    from ollama import Client as OllamaClient

from conductor_core import models as objects
from conductor_core import music as utils
from conductor_core.errors import (
    ProviderConnectionError,
    ProviderContextLengthError,
    ProviderError,
    ProviderRequestError,
    ProviderTimeoutError,
    error_for_status,
)
from conductor_core.provider_types import ProviderMessage
from conductor_core.providers._variations import VariationCollection
from conductor_core.variations import VariationUsage

try:
    import httpx
    import ollama
except ImportError:  # pragma: no cover - exercised only in minimal installs
    httpx = None
    ollama = None

logger = logging.getLogger(__name__)


class OllamaStatus(TypedDict):
    available: bool
    models: list[str]
    model_capabilities: dict[str, dict[str, object] | None]
    model_errors: dict[str, ProviderError]
    host: str
    error: str | None
    exception: ProviderError | ImportError | None


class OllamaModelStatus(TypedDict):
    available: bool
    installed: bool
    model_capabilities: dict[str, object] | None
    host: str
    error: str | None
    exception: ProviderError | ImportError | None


def _get_thinking_metadata(client, model_name, model_info):
    """Read Ollama's per-model think values when the SDK preserves them.

    Returns the supported effort levels and whether ``think=False`` is
    accepted, or ``None`` when Ollama did not report its think values.
    """
    thinking = getattr(model_info, "thinking", None)

    # ollama-python currently parses /api/show into ShowResponse, which drops
    # the API's `thinking` field. Use its raw request path so SDK auth, host,
    # and timeout settings are retained. Keep this optional for older SDKs and
    # lightweight client implementations.
    if thinking is None:
        request_raw = getattr(client, "_request_raw", None)
        if callable(request_raw):
            try:
                response = request_raw("POST", "/api/show", json={"model": model_name})
                thinking = response.json().get("thinking")
            except Exception as exc:
                _raise_ollama_error(
                    exc, f"thinking metadata inspection for {model_name!r}"
                )

    if not isinstance(thinking, dict):
        return [], None
    values = thinking.get("values") or []
    # The API exposes explicit accepted values. Only advertise the supported
    # effort vocabulary Core can route; never infer it from a model name.
    supported = {value for value in values if isinstance(value, str)}
    effort_options = [
        level for level in ("low", "medium", "high") if level in supported
    ]
    return effort_options, any(value is False for value in values)


def _get_model_capabilities(client, model_name, host):
    effort_options = []
    accepts_think_false = None
    try:
        model_info = client.show(model_name)
        capabilities = getattr(model_info, "capabilities", None) or []
        effort_options, accepts_think_false = _get_thinking_metadata(
            client, model_name, model_info
        )
        supports_thinking = "thinking" in capabilities or bool(effort_options)
    except Exception as exc:
        _raise_ollama_error(exc, f"capability inspection for {model_name!r} at {host}")
    if not supports_thinking:
        thinking_off = None
    elif accepts_think_false is False:
        # Ollama reports only effort levels (for example gpt-oss).
        thinking_off = "lowest_effort"
    else:
        # Reported values include false, or none were reported and the model
        # takes a boolean think value.
        thinking_off = "disabled"
    return {
        "extended_thinking": supports_thinking,
        "effort_options": effort_options,
        "temperature_supported": True,
        "thinking_fixed_temperature": None,
        "thinking_off": thinking_off,
    }


def _resolve_host(host_address: str | None = None) -> str:
    return (
        host_address or os.getenv("OLLAMA_API_HOST_ADDRESS") or "http://localhost:11434"
    )


def _ollama_error(exc: Exception, operation: str) -> ProviderError | ImportError:
    """Normalize SDK failures while retaining already normalized errors and causes."""
    if isinstance(exc, (ProviderError, ImportError)):
        return exc
    assert httpx is not None
    assert ollama is not None
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
    elif isinstance(exc, httpx.HTTPStatusError):
        error = error_for_status(
            "Ollama", str(exc), exc.response.status_code, operation=operation
        )
    else:
        error = ProviderRequestError("Ollama", str(exc), operation=operation)
    error.__cause__ = exc
    return error


def _raise_ollama_error(exc: Exception, operation: str) -> NoReturn:
    error = _ollama_error(exc, operation)
    if error is exc:
        raise error
    raise error from exc


def initialize_ollama_client(
    host_address: str | None = None, timeout: float | None = None
) -> OllamaClient:
    """Initialize and return an Ollama client."""
    if ollama is None:
        raise ImportError("Install conductor-core[ollama] to use Ollama models.")
    assert httpx is not None

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
) -> OllamaStatus:
    """Report listing availability and per-model inspection results.

    Failed inspections have unknown (None) capabilities and a typed entry in
    model_errors. Listing failures populate error and exception instead.
    """
    host = _resolve_host(host_address)
    status: OllamaStatus = {
        "available": False,
        "models": [],
        "model_capabilities": {},
        "model_errors": {},
        "host": host,
        "error": None,
        "exception": None,
    }

    if ollama is None:
        status["exception"] = ImportError(
            "Install conductor-core[ollama] to use Ollama models."
        )
        status["error"] = str(status["exception"])
        return status

    try:
        client = initialize_ollama_client(
            host_address=host,
            **({"timeout": request_timeout} if request_timeout is not None else {}),
        )
        status["models"] = _list_models(client)
        status["available"] = True
    except Exception as exc:
        status["exception"] = _ollama_error(exc, "model listing")
        status["error"] = str(status["exception"])
        logger.warning("Ollama unavailable at %s: %s", host, exc)
        return status

    for model_name in status["models"]:
        try:
            status["model_capabilities"][model_name] = _get_model_capabilities(
                client, model_name, host
            )
        # Each network inspection must fail independently to retain partial results.
        except ProviderError as exc:  # noqa: PERF203
            status["model_capabilities"][model_name] = None
            status["model_errors"][model_name] = exc

    return status


def get_model_status(
    model_name: str,
    host_address: str | None = None,
    request_timeout: float | None = None,
) -> OllamaModelStatus:
    """Check whether one model is installed and inspect only that model.

    Unlike :func:`get_ollama_status`, this lists installed models and then
    requests details for ``model_name`` alone, so generation does not pay for
    inspecting every installed model.

    available and installed remain true after a failed inspection; capabilities
    are then None, with a display error and the original typed exception.
    """
    host = _resolve_host(host_address)
    status: OllamaModelStatus = {
        "available": False,
        "installed": False,
        "model_capabilities": None,
        "host": host,
        "error": None,
        "exception": None,
    }
    if ollama is None:
        status["exception"] = ImportError(
            "Install conductor-core[ollama] to use Ollama models."
        )
        status["error"] = str(status["exception"])
        return status

    try:
        client = initialize_ollama_client(
            host_address=host,
            **({"timeout": request_timeout} if request_timeout is not None else {}),
        )
        models = _list_models(client)
        status["available"] = True
        if model_name in models:
            status["installed"] = True
            status["model_capabilities"] = _get_model_capabilities(
                client, model_name, host
            )
    except Exception as exc:
        status["exception"] = _ollama_error(exc, "model listing")
        status["error"] = str(status["exception"])
        logger.warning("Ollama status failed at %s: %s", host, exc)

    return status


def _list_models(client) -> list[str]:
    """List named models, preserving the SDK failure and operation."""
    try:
        return [model.model for model in client.list().models if model.model]
    except Exception as exc:
        _raise_ollama_error(exc, "model listing")


def get_model_list(host_address: str | None = None) -> list[str | None]:
    """Get the available Ollama model names without inspecting each model."""
    if ollama is None:
        return []
    host = _resolve_host(host_address)
    try:
        client = initialize_ollama_client(host_address=host)
        return [model.model for model in client.list().models]
    except Exception as exc:
        logger.warning("Ollama unavailable at %s: %s", host, exc)
        return []


def _thinking_option(
    model_capabilities: Mapping[str, Any] | None,
    use_thinking: bool,
    effort: str | None,
) -> bool | Literal["low", "medium", "high"] | None:
    """Return the Ollama ``think`` value, or None when unsupported/unknown."""
    if not model_capabilities or not model_capabilities.get("extended_thinking"):
        return None
    effort_options = model_capabilities.get("effort_options") or []
    validated_effort: Literal["low", "medium", "high"] | None = None
    if effort_options and effort is not None:
        if effort not in effort_options:
            raise ValueError(f"Unsupported Ollama reasoning effort for model: {effort}")
        if effort not in ("low", "medium", "high"):
            raise ValueError(f"Unsupported Ollama reasoning effort: {effort}")
        validated_effort = effort
    if use_thinking:
        return validated_effort if effort_options else True
    thinking_off = model_capabilities.get(
        "thinking_off", "lowest_effort" if effort_options else "disabled"
    )
    if thinking_off == "disabled":
        return False
    # Reasoning cannot be turned off: send the lowest level, or leave the
    # model's default when it has no levels.
    return validated_effort if effort_options else None


def _chat_options(temp, num_ctx):
    """Build Ollama model options, leaving context size to Ollama by default."""
    options = {"temperature": temp}
    if num_ctx is not None:
        options["num_ctx"] = num_ctx
    return options


def _raise_if_out_of_context(completion, model, num_ctx):
    """Raise a clear error when Ollama stopped because the context was full."""
    if getattr(completion, "done_reason", None) != "length":
        return
    raise ProviderContextLengthError(
        "Ollama",
        model,
        prompt_tokens=getattr(completion, "prompt_eval_count", None),
        output_tokens=getattr(completion, "eval_count", None),
        context_length=num_ctx,
        operation="response",
    )


def loop_gen(
    prompt: str,
    model: str,
    temp: float = 0.0,
    host_address: str | None = None,
    system_prompt: str | None = None,
    request_timeout: float | None = None,
    use_thinking: bool = False,
    effort: str | None = "low",
    model_capabilities: Mapping[str, Any] | None = None,
    num_ctx: int | None = None,
) -> tuple[objects.Loop, list[ProviderMessage], float]:
    """Generate a MIDI loop using the specified Ollama model and prompt."""
    client = initialize_ollama_client(
        host_address=host_address,
        **({"timeout": request_timeout} if request_timeout is not None else {}),
    )
    assert httpx is not None
    assert ollama is not None
    loop_prompt = system_prompt or utils.get_loop_prompt()
    messages: list[ProviderMessage] = [
        {"role": "system", "content": loop_prompt},
        {"role": "user", "content": prompt},
    ]
    if model_capabilities is None:
        model_capabilities = _get_model_capabilities(
            client, model, _resolve_host(host_address)
        )
    think = _thinking_option(model_capabilities, use_thinking, effort)
    try:
        completion = client.chat(
            model=model,
            messages=messages,
            format=objects.Loop.model_json_schema(),
            options=_chat_options(temp, num_ctx),
            **({"think": think} if think is not None else {}),
        )
    except (
        httpx.TimeoutException,
        httpx.NetworkError,
        ConnectionError,
        ollama.RequestError,
        ollama.ResponseError,
    ) as exc:
        logger.error("Ollama request failed: %s", exc)
        _raise_ollama_error(exc, "request")
    _raise_if_out_of_context(completion, model, num_ctx)
    message = getattr(completion, "message", None)
    content = getattr(message, "content", None)
    if not content:
        raise ValueError("Ollama response did not include generated content.")

    midi_loop = objects.Loop.model_validate_json(content)
    thinking = getattr(message, "thinking", None)
    if thinking:
        messages.append({"role": "assistant", "content": thinking})
    messages.append({"role": "assistant", "content": midi_loop.model_dump_json()})
    return midi_loop, messages, 0


def variations_gen(
    prompt: str,
    model: str,
    temp: float = 0.0,
    host_address: str | None = None,
    system_prompt: str | None = None,
    request_timeout: float | None = None,
    use_thinking: bool = False,
    effort: str | None = "low",
    model_capabilities: Mapping[str, Any] | None = None,
    num_ctx: int | None = None,
) -> tuple[VariationCollection, list[ProviderMessage], float, VariationUsage | None]:
    """Generate an ordered collection of loops in one Ollama response."""
    client = initialize_ollama_client(
        host_address=host_address,
        **({"timeout": request_timeout} if request_timeout is not None else {}),
    )
    assert httpx is not None
    assert ollama is not None
    loop_prompt = system_prompt or utils.get_variation_prompt()
    messages: list[ProviderMessage] = [
        {"role": "system", "content": loop_prompt},
        {"role": "user", "content": prompt},
    ]
    if model_capabilities is None:
        model_capabilities = _get_model_capabilities(
            client, model, _resolve_host(host_address)
        )
    think = _thinking_option(model_capabilities, use_thinking, effort)
    try:
        completion = client.chat(
            model=model,
            messages=messages,
            format=VariationCollection.model_json_schema(),
            options=_chat_options(temp, num_ctx),
            **({"think": think} if think is not None else {}),
        )
    except (
        httpx.TimeoutException,
        httpx.NetworkError,
        ConnectionError,
        ollama.RequestError,
        ollama.ResponseError,
    ) as exc:
        _raise_ollama_error(exc, "request")
    _raise_if_out_of_context(completion, model, num_ctx)
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
