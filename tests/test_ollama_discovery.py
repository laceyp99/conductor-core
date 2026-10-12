"""Offline request-level checks for Ollama model discovery."""

import httpx
import ollama as sdk
import pytest

from conductor_core.errors import (
    ProviderAuthenticationError,
    ProviderConnectionError,
    ProviderRequestError,
    ProviderTimeoutError,
)
from conductor_core.providers import ollama


@pytest.fixture
def mock_transport(monkeypatch):
    client_class = sdk.Client
    clients = []

    def install(handler):
        def create_client(**kwargs):
            client = client_class(**kwargs, transport=httpx.MockTransport(handler))
            clients.append(client)
            return client

        monkeypatch.setattr(sdk, "Client", create_client)

    yield install
    for client in clients:
        client._client.close()


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
