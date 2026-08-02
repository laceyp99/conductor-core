from types import SimpleNamespace

import pytest

from conductor_core import (
    EngineConfig,
    ProviderAuthenticationError,
    ProviderConnectionError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    routing,
)
from conductor_core.providers import anthropic as anthropic_api
from conductor_core.providers import google as google_api
from conductor_core.providers import ollama as ollama_api
from conductor_core.providers import openai as openai_api

HOSTED_PROVIDERS = [
    pytest.param(
        openai_api,
        "initialize_openai_client",
        "OpenAI",
        "OPENAI_API_KEY",
        ("OpenAI",),
        id="openai",
    ),
    pytest.param(
        anthropic_api,
        "initialize_anthropic_client",
        "Anthropic",
        "ANTHROPIC_API_KEY",
        ("Anthropic",),
        id="anthropic",
    ),
    pytest.param(
        google_api,
        "initialize_gemini_client",
        "Google",
        "GEMINI_API_KEY",
        ("genai", "Client"),
        id="google",
    ),
]


def _patch_client_constructor(monkeypatch, module, client_path, replacement):
    target = module
    for attribute in client_path[:-1]:
        target = getattr(target, attribute)
    monkeypatch.setattr(target, client_path[-1], replacement)


def test_engine_config_rejects_invalid_request_timeouts():
    for timeout in (0, -1, float("inf"), "5", True):
        with pytest.raises(
            ValueError, match="request_timeout must be None or a positive finite number"
        ):
            EngineConfig.from_defaults(request_timeout=timeout)


@pytest.mark.parametrize(
    ("module", "initializer_name", "provider", "env_var", "client_path"),
    HOSTED_PROVIDERS,
)
@pytest.mark.parametrize("api_key", [None, "   "], ids=["missing", "blank"])
def test_hosted_provider_initializers_fail_fast_without_credentials(
    monkeypatch,
    module,
    initializer_name,
    provider,
    env_var,
    client_path,
    api_key,
):
    monkeypatch.delenv(env_var, raising=False)
    client_called = False

    def construct_client(**kwargs):
        nonlocal client_called
        client_called = True
        return object()

    _patch_client_constructor(monkeypatch, module, client_path, construct_client)

    with pytest.raises(ProviderAuthenticationError) as raised:
        getattr(module, initializer_name)(api_key=api_key)

    assert raised.value.provider == provider
    assert raised.value.operation == "client initialization"
    assert str(raised.value) == (
        f"{provider} client initialization failed: {env_var} is not set "
        "and no usable api_key was provided"
    )
    assert client_called is False


@pytest.mark.parametrize(
    ("module", "initializer_name", "provider", "env_var", "client_path"),
    HOSTED_PROVIDERS,
)
def test_hosted_provider_initializers_reject_blank_environment_credentials(
    monkeypatch,
    module,
    initializer_name,
    provider,
    env_var,
    client_path,
):
    monkeypatch.setenv(env_var, "   ")
    client_called = False

    def construct_client(**kwargs):
        nonlocal client_called
        client_called = True
        return object()

    _patch_client_constructor(monkeypatch, module, client_path, construct_client)

    with pytest.raises(ProviderAuthenticationError) as raised:
        getattr(module, initializer_name)()

    assert raised.value.provider == provider
    assert raised.value.operation == "client initialization"
    assert str(raised.value) == (
        f"{provider} client initialization failed: {env_var} is not set "
        "and no usable api_key was provided"
    )
    assert client_called is False


@pytest.mark.parametrize(
    ("module", "initializer_name", "provider", "env_var", "client_path"),
    HOSTED_PROVIDERS,
)
def test_hosted_provider_initializers_prefer_explicit_credentials(
    monkeypatch,
    module,
    initializer_name,
    provider,
    env_var,
    client_path,
):
    monkeypatch.setenv(env_var, "environment-key")
    captured = {}
    _patch_client_constructor(
        monkeypatch,
        module,
        client_path,
        lambda **kwargs: captured.update(kwargs) or object(),
    )

    getattr(module, initializer_name)(api_key="explicit-key")

    assert captured["api_key"] == "explicit-key"


def test_openai_client_initialization_normalizes_authentication_error(monkeypatch):
    class OpenAIAuthenticationError(Exception):
        pass

    original = OpenAIAuthenticationError("bad key")

    def fail_client(**kwargs):
        raise original

    monkeypatch.setattr(openai_api, "AuthenticationError", OpenAIAuthenticationError)
    monkeypatch.setattr(openai_api, "OpenAI", fail_client)

    with pytest.raises(ProviderAuthenticationError) as raised:
        openai_api.initialize_openai_client(api_key="test-key")

    assert raised.value.provider == "OpenAI"
    assert raised.value.operation == "client initialization"
    assert raised.value.__cause__ is original


def test_openai_request_normalizes_rate_limit_error(monkeypatch):
    class OpenAIRateLimitError(Exception):
        pass

    original = OpenAIRateLimitError("slow down")
    client = SimpleNamespace(
        responses=SimpleNamespace(parse=lambda **kwargs: (_ for _ in ()).throw(original))
    )
    monkeypatch.setattr(openai_api, "RateLimitError", OpenAIRateLimitError)
    monkeypatch.setattr(openai_api, "initialize_openai_client", lambda **kwargs: client)

    with pytest.raises(ProviderRateLimitError) as raised:
        openai_api.loop_gen("write a loop", "gpt-4o-mini")

    assert raised.value.operation == "request"
    assert raised.value.__cause__ is original


def test_anthropic_stream_normalizes_timeout_error(monkeypatch):
    class AnthropicTimeoutError(Exception):
        pass

    original = AnthropicTimeoutError("stream timed out")

    class FailingStream:
        def __iter__(self):
            return self

        def __next__(self):
            raise original

    client = SimpleNamespace(messages=SimpleNamespace(create=lambda **kwargs: FailingStream()))
    monkeypatch.setattr(anthropic_api, "APITimeoutError", AnthropicTimeoutError)
    monkeypatch.setattr(anthropic_api, "initialize_anthropic_client", lambda **kwargs: client)

    with pytest.raises(ProviderTimeoutError) as raised:
        anthropic_api.loop_gen("write a loop", "claude-sonnet-4-5")

    assert raised.value.provider == "Anthropic"
    assert raised.value.operation == "stream"
    assert raised.value.__cause__ is original


def test_google_request_normalizes_rate_limit_error(monkeypatch):
    class GoogleAPIError(Exception):
        def __init__(self, code):
            self.code = code
            super().__init__(f"status {code}")

    original = GoogleAPIError(429)
    client = SimpleNamespace(
        models=SimpleNamespace(generate_content=lambda **kwargs: (_ for _ in ()).throw(original))
    )
    monkeypatch.setattr(google_api, "genai_errors", SimpleNamespace(APIError=GoogleAPIError))
    monkeypatch.setattr(google_api, "initialize_gemini_client", lambda **kwargs: client)

    with pytest.raises(ProviderRateLimitError) as raised:
        google_api.loop_gen("write a loop", "gemini-3.1-flash-lite")

    assert raised.value.provider == "Google"
    assert raised.value.operation == "request"
    assert raised.value.__cause__ is original


def test_ollama_request_normalizes_authentication_error(monkeypatch):
    class OllamaRequestError(Exception):
        pass

    class OllamaResponseError(Exception):
        def __init__(self, status_code):
            self.status_code = status_code
            super().__init__("unauthorized")

    original = OllamaResponseError(401)
    client = SimpleNamespace(chat=lambda **kwargs: (_ for _ in ()).throw(original))
    monkeypatch.setattr(
        ollama_api,
        "ollama",
        SimpleNamespace(RequestError=OllamaRequestError, ResponseError=OllamaResponseError),
    )
    monkeypatch.setattr(ollama_api, "initialize_ollama_client", lambda **kwargs: client)

    with pytest.raises(ProviderAuthenticationError) as raised:
        ollama_api.loop_gen("write a loop", "llama3")

    assert raised.value.provider == "Ollama"
    assert raised.value.operation == "request"
    assert raised.value.__cause__ is original


def test_ollama_request_normalizes_sdk_connection_error(monkeypatch):
    original = ConnectionError("connection refused")
    client = SimpleNamespace(chat=lambda **kwargs: (_ for _ in ()).throw(original))
    monkeypatch.setattr(ollama_api, "initialize_ollama_client", lambda **kwargs: client)

    with pytest.raises(ProviderConnectionError) as raised:
        ollama_api.loop_gen("write a loop", "llama3")

    assert raised.value.provider == "Ollama"
    assert raised.value.operation == "request"
    assert raised.value.__cause__ is original


@pytest.mark.parametrize(
    ("initializer", "client_attr"),
    [
        (openai_api.initialize_openai_client, "OpenAI"),
        (anthropic_api.initialize_anthropic_client, "Anthropic"),
    ],
)
def test_openai_and_anthropic_initializers_pass_timeout(monkeypatch, initializer, client_attr):
    module = openai_api if client_attr == "OpenAI" else anthropic_api
    captured = {}
    monkeypatch.setattr(module, client_attr, lambda **kwargs: captured.update(kwargs) or object())

    initializer(api_key="test-key", timeout=2.5)

    assert captured == {"api_key": "test-key", "timeout": 2.5}


def test_google_initializer_converts_timeout_to_milliseconds(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        google_api.genai,
        "Client",
        lambda **kwargs: captured.update(kwargs) or object(),
    )

    google_api.initialize_gemini_client(api_key="test-key", timeout=2.5)

    assert captured["api_key"] == "test-key"
    assert captured["http_options"].timeout == 2500


def test_ollama_initializer_passes_timeout(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        ollama_api.ollama,
        "Client",
        lambda **kwargs: captured.update(kwargs) or object(),
    )

    ollama_api.initialize_ollama_client(host_address="http://ollama.test", timeout=2.5)

    assert captured == {"host": "http://ollama.test", "timeout": 2.5}


def test_ollama_status_forwards_request_timeout(monkeypatch):
    captured = {}
    client = SimpleNamespace(list=lambda: SimpleNamespace(models=[]))
    monkeypatch.setattr(
        ollama_api,
        "initialize_ollama_client",
        lambda **kwargs: captured.update(kwargs) or client,
    )

    status = ollama_api.get_ollama_status(
        force_refresh=True,
        host_address="http://ollama.test",
        request_timeout=2.5,
    )

    assert status["available"] is True
    assert captured == {"host_address": "http://ollama.test", "timeout": 2.5}


@pytest.mark.parametrize(
    ("module", "initializer_name", "client_attr"),
    [
        (openai_api, "initialize_openai_client", "OpenAI"),
        (anthropic_api, "initialize_anthropic_client", "Anthropic"),
        (google_api, "initialize_gemini_client", "genai"),
        (ollama_api, "initialize_ollama_client", "ollama"),
    ],
)
def test_initializers_omit_timeout_when_unset(monkeypatch, module, initializer_name, client_attr):
    captured = {}
    if client_attr == "genai":
        monkeypatch.setattr(
            module.genai,
            "Client",
            lambda **kwargs: captured.update(kwargs) or object(),
        )
    elif client_attr == "ollama":
        monkeypatch.setattr(
            module.ollama,
            "Client",
            lambda **kwargs: captured.update(kwargs) or object(),
        )
    else:
        monkeypatch.setattr(
            module, client_attr, lambda **kwargs: captured.update(kwargs) or object()
        )

    initializer = getattr(module, initializer_name)
    if client_attr == "ollama":
        initializer()
    else:
        initializer(api_key="test-key")

    assert "timeout" not in captured
    assert "http_options" not in captured


@pytest.mark.parametrize(
    ("provider", "model", "adapter_name"),
    [
        ("OpenAI", "gpt-4o-mini", "openai_api"),
        ("Google", "gemini-3.1-flash-lite", "gemini_api"),
        ("Anthropic", "claude-sonnet-4-5", "claude_api"),
        ("Ollama", "llama3", "ollama_api"),
    ],
)
def test_routing_forwards_request_timeout(monkeypatch, provider, model, adapter_name):
    model_info = {"models": {"OpenAI": {}, "Google": {}, "Anthropic": {}}}
    if provider != "Ollama":
        model_info["models"][provider][model] = {}
    monkeypatch.setattr(routing, "get_model_info", lambda: model_info)
    status_captured = {}
    if provider == "Ollama":
        monkeypatch.setattr(
            routing.ollama_api,
            "get_ollama_status",
            lambda **kwargs: (
                status_captured.update(kwargs) or {"available": True, "models": [model]}
            ),
        )

    adapter = getattr(routing, adapter_name)
    captured = {}

    def fake_loop_gen(*args, **kwargs):
        captured.update(kwargs)
        return "loop", [], 0

    monkeypatch.setattr(adapter, "loop_gen", fake_loop_gen)

    routing.generate_midi(model, "write a loop", request_timeout=2.5)

    assert captured["request_timeout"] == 2.5
    if provider == "Ollama":
        assert status_captured["request_timeout"] == 2.5
