"""Offline request-level checks for Ollama discovery and client cleanup."""

import json

import httpx
import ollama as sdk
import pydantic
import pytest

from conductor_core.errors import (
    ProviderAuthenticationError,
    ProviderConnectionError,
    ProviderRequestError,
    ProviderTimeoutError,
)
from conductor_core.providers import ollama


class _SpyTransport(httpx.MockTransport):
    """Record whether the owning SDK client closed its HTTP transport."""

    closed = False

    def close(self):
        self.closed = True


@pytest.fixture
def mock_transport(monkeypatch):
    """Route real SDK clients through a handler; return their transports."""
    client_class = sdk.Client

    def install(handler):
        transports = []

        def create_client(**kwargs):
            transport = _SpyTransport(handler)
            transports.append(transport)
            return client_class(**kwargs, transport=transport)

        monkeypatch.setattr(sdk, "Client", create_client)
        return transports

    return install


@pytest.mark.parametrize("timeout", [None, 2.0])
def test_model_list_applies_timeout_to_list_request(mock_transport, timeout):
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(
            200,
            json={"models": [{"model": "a"}, {"model": None}, {}, {"model": ""}]},
        )

    mock_transport(respond)
    assert ollama.get_model_list("http://ollama.test", timeout) == ["a"]
    assert len(requests) == 1
    request = requests[0]
    assert request.method == "GET"
    assert str(request.url) == "http://ollama.test/api/tags"
    assert request.extensions["timeout"] == dict.fromkeys(
        ("connect", "read", "write", "pool"), timeout
    )


def test_model_list_returns_empty_catalog_on_success(mock_transport):
    mock_transport(lambda request: httpx.Response(200, json={"models": []}))
    assert ollama.get_model_list() == []


@pytest.mark.parametrize(
    ("original", "expected"),
    [
        (httpx.ReadTimeout("no response"), ProviderTimeoutError),
        (httpx.ConnectError("connection refused"), ProviderConnectionError),
        (ConnectionError("SDK connection failure"), ProviderConnectionError),
        (sdk.RequestError("invalid request"), ProviderRequestError),
    ],
)
def test_model_list_raises_typed_transport_errors(mock_transport, original, expected):
    def fail(request):
        raise original

    mock_transport(fail)
    with pytest.raises(expected) as raised:
        ollama.get_model_list(request_timeout=2.0)
    assert raised.value.provider == "Ollama"
    assert raised.value.operation == "model listing"
    # Ollama may translate HTTPX connection failures before Core sees them.
    assert raised.value.__cause__ is not None


@pytest.mark.parametrize(
    ("status", "expected"),
    [(401, ProviderAuthenticationError), (503, ProviderRequestError)],
)
def test_model_list_raises_typed_response_errors(mock_transport, status, expected):
    mock_transport(lambda request: httpx.Response(status, json={"error": "failed"}))
    with pytest.raises(expected) as raised:
        ollama.get_model_list()
    assert raised.value.operation == "model listing"
    assert isinstance(raised.value.__cause__, sdk.ResponseError)


def test_model_list_preserves_client_initialization_error(monkeypatch):
    original = ProviderConnectionError(
        "Ollama", "bad host", operation="client initialization"
    )

    def fail(**kwargs):
        raise original

    monkeypatch.setattr(ollama, "initialize_ollama_client", fail)
    with pytest.raises(ProviderConnectionError) as raised:
        ollama.get_model_list()
    assert raised.value is original


def test_model_list_reports_missing_sdk(monkeypatch):
    monkeypatch.setattr(ollama, "ollama", None)
    with pytest.raises(ImportError, match=r"conductor-core\[ollama\]"):
        ollama.get_model_list()


LOOP_CONTENT = json.dumps(
    {
        f"Bar_{number}": {
            "num": number,
            "notes": [
                {
                    "pitch": "C",
                    "octave": 4,
                    "velocity": 100,
                    "time": {"start_beat": 1, "duration": 1},
                }
            ],
        }
        for number in range(1, 5)
    }
)


def _server(failing_path=None, chat_content=LOOP_CONTENT):
    """Answer like a local Ollama, returning 503 for ``failing_path``."""

    def respond(request):
        path = request.url.path
        if path == failing_path:
            return httpx.Response(503, json={"error": "failed"})
        if path == "/api/tags":
            return httpx.Response(200, json={"models": [{"model": "local"}]})
        if path == "/api/show":
            show = {"capabilities": ["completion"], "model_info": {}}
            return httpx.Response(200, json=show)
        message = {"role": "assistant", "content": chat_content}
        return httpx.Response(200, json={"message": message, "done": True})

    return respond


@pytest.mark.parametrize(
    ("server", "operation", "expected"),
    [
        pytest.param(_server(), ollama.get_model_list, None, id="list"),
        pytest.param(_server(), ollama.get_ollama_status, None, id="status"),
        pytest.param(
            _server(),
            lambda: ollama.get_model_status("missing"),
            None,
            id="missing-model",
        ),
        pytest.param(
            _server("/api/tags"),
            ollama.get_model_list,
            ProviderRequestError,
            id="list-error",
        ),
        pytest.param(
            _server(), lambda: ollama.loop_gen("prompt", "local"), None, id="loop"
        ),
        pytest.param(
            _server("/api/chat"),
            lambda: ollama.loop_gen("prompt", "local"),
            ProviderRequestError,
            id="request-error",
        ),
        pytest.param(
            _server(chat_content="{}"),
            lambda: ollama.variations_gen("prompt", "local"),
            pydantic.ValidationError,
            id="parse-error",
        ),
    ],
)
def test_operations_close_their_clients(mock_transport, server, operation, expected):
    transports = mock_transport(server)
    if expected is None:
        operation()
    else:
        with pytest.raises(expected):
            operation()
    assert len(transports) == 1
    assert transports[0].closed
