"""Offline discovery failures must survive status inspection and routing."""

from types import SimpleNamespace

import httpx
import pytest

from conductor_core import routing
from conductor_core.errors import (
    ProviderAuthenticationError,
    ProviderConnectionError,
    ProviderRequestError,
    ProviderTimeoutError,
)
from conductor_core.providers import ollama


@pytest.mark.parametrize("capabilities", [None, {}])
def test_inspection_rejects_ambiguous_capability_and_error_state(capabilities):
    error = (
        ProviderRequestError("Ollama", "failed") if capabilities is not None else None
    )
    with pytest.raises(ValueError, match="either capabilities or an error"):
        ollama.OllamaModelInspection("local", capabilities, error)


def test_selected_model_status_forwards_host_and_timeout(monkeypatch):
    captured = {}
    client = SimpleNamespace(
        list=lambda: SimpleNamespace(models=[SimpleNamespace(model="local")]),
        show=lambda name: SimpleNamespace(capabilities=[]),
    )
    monkeypatch.setattr(
        ollama,
        "initialize_ollama_client",
        lambda **kwargs: captured.update(kwargs) or client,
    )
    inspection = ollama.get_model_status(
        "local", host_address="http://ollama.test", request_timeout=2.5
    )
    assert inspection.name == "local"
    assert inspection.error is None
    assert captured == {"host_address": "http://ollama.test", "timeout": 2.5}


@pytest.mark.parametrize("generator", ["generate_midi", "generate_variations"])
@pytest.mark.parametrize("operation", ["list", "show", "raw"])
@pytest.mark.parametrize(
    ("failure", "error_type"),
    [
        (httpx.ReadTimeout("slow"), ProviderTimeoutError),
        (ConnectionError("refused"), ProviderConnectionError),
        (
            httpx.HTTPStatusError(
                "auth",
                request=httpx.Request("POST", "http://test"),
                response=httpx.Response(401),
            ),
            ProviderAuthenticationError,
        ),
        (RuntimeError("bad request"), ProviderRequestError),
    ],
)
def test_status_and_both_routes_preserve_failures(
    monkeypatch, generator, operation, failure, error_type
):
    def fail(*args, **kwargs):
        raise failure

    client = SimpleNamespace(
        list=fail
        if operation == "list"
        else lambda: SimpleNamespace(models=[SimpleNamespace(model="local")]),
        show=fail
        if operation == "show"
        else lambda name: SimpleNamespace(capabilities=["thinking"]),
        _request_raw=fail
        if operation == "raw"
        else lambda *a, **k: SimpleNamespace(json=dict),
        chat=lambda **kwargs: pytest.fail(
            "generation must not run after discovery failure"
        ),
    )
    monkeypatch.setattr(ollama, "initialize_ollama_client", lambda **kwargs: client)
    monkeypatch.setattr(
        routing,
        "get_model_info",
        lambda: {"models": {"OpenAI": {}, "Google": {}, "Anthropic": {}}},
    )

    if operation == "list":
        with pytest.raises(error_type) as raised:
            ollama.get_model_status("local", request_timeout=2)
        error = raised.value
    else:
        status = ollama.get_model_status("local", request_timeout=2)
        assert status.name == "local"
        assert status.capabilities is None
        error = status.error
        assert isinstance(error, error_type)
    assert error.__cause__ is failure
    expected_operation = {
        "list": "model listing",
        "show": "capability inspection",
        "raw": "thinking metadata inspection",
    }[operation]
    assert expected_operation in error.operation

    args = (
        ("local", "prompt") if generator == "generate_midi" else ("local", "prompt", 2)
    )
    with pytest.raises(error_type) as raised:
        getattr(routing, generator)(*args, request_timeout=2, use_thinking=True)
    assert raised.value.__cause__ is failure
    assert expected_operation in raised.value.operation


def test_broad_listing_failure_is_typed(monkeypatch):
    failure = httpx.ReadTimeout("slow listing")

    def fail():
        raise failure

    monkeypatch.setattr(
        ollama, "initialize_ollama_client", lambda **kwargs: SimpleNamespace(list=fail)
    )
    with pytest.raises(ProviderTimeoutError) as raised:
        ollama.get_ollama_status()
    assert raised.value.__cause__ is failure


@pytest.mark.parametrize("generator", ["loop_gen", "variations_gen"])
def test_direct_adapters_stop_after_inspection_timeout(monkeypatch, generator):
    failure = httpx.ReadTimeout("slow inspection")

    def show(name):
        raise failure

    client = SimpleNamespace(
        show=show,
        chat=lambda **kwargs: pytest.fail("generation must not run"),
    )
    monkeypatch.setattr(ollama, "initialize_ollama_client", lambda **kwargs: client)
    with pytest.raises(ProviderTimeoutError) as raised:
        getattr(ollama, generator)("prompt", "local", use_thinking=True)
    assert raised.value.__cause__ is failure


@pytest.mark.parametrize("generator", ["generate_midi", "generate_variations"])
def test_missing_model_is_invalid_selection_after_successful_listing(
    monkeypatch, generator
):
    client = SimpleNamespace(
        list=lambda: SimpleNamespace(
            models=[SimpleNamespace(model=None), SimpleNamespace(model="installed")]
        )
    )
    monkeypatch.setattr(ollama, "initialize_ollama_client", lambda **kwargs: client)
    status = ollama.get_model_status("missing")
    assert status is None
    args = (
        ("missing", "prompt")
        if generator == "generate_midi"
        else ("missing", "prompt", 2)
    )
    with pytest.raises(ValueError, match="Invalid Model Selected"):
        getattr(routing, generator)(*args)


@pytest.mark.parametrize("generator", ["generate_midi", "generate_variations"])
def test_missing_sdk_is_actionable_and_import_safe(monkeypatch, generator):
    monkeypatch.setattr(ollama, "ollama", None)
    monkeypatch.setattr(ollama, "httpx", None)
    with pytest.raises(ImportError, match=r"Install conductor-core\[ollama\]"):
        ollama.get_ollama_status()
    args = (
        ("local", "prompt") if generator == "generate_midi" else ("local", "prompt", 2)
    )
    with pytest.raises(ImportError, match=r"Install conductor-core\[ollama\]"):
        getattr(routing, generator)(*args)


@pytest.mark.parametrize("raw_available", [False, True])
def test_absent_optional_metadata_retains_boolean_thinking(monkeypatch, raw_available):
    client = SimpleNamespace(
        list=lambda: SimpleNamespace(models=[SimpleNamespace(model="local")]),
        show=lambda name: SimpleNamespace(capabilities=["thinking"]),
    )
    if raw_available:
        client._request_raw = lambda *a, **k: SimpleNamespace(json=dict)
    monkeypatch.setattr(ollama, "initialize_ollama_client", lambda **kwargs: client)
    status = ollama.get_model_status("local")
    assert status.error is None
    assert status.capabilities["extended_thinking"] is True
    assert status.capabilities["thinking_off"] == "disabled"
