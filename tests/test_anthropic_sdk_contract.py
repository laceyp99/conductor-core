"""Check Core's Anthropic requests against the installed SDK without network calls."""

import inspect
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytest.importorskip("anthropic")

from conductor_core import music
from conductor_core.providers import anthropic


@pytest.mark.parametrize("generation", ["loop_gen", "variations_gen"])
@pytest.mark.parametrize(
    "outcome", ["success", "invalid_payload", "wrong_tool", "stream_error"]
)
def test_anthropic_generation_closes_sdk_response_and_client(
    monkeypatch, sample_loop, generation, outcome
):
    import httpx
    from anthropic import Stream
    from anthropic.types import RawMessageStreamEvent

    tool_name = (
        "build_MIDI_loop" if generation == "loop_gen" else "build_MIDI_variations"
    )
    payload = sample_loop.model_dump()
    if generation == "variations_gen":
        payload = {"variations": [payload]}
    if outcome == "invalid_payload":
        payload = {}
    events = [
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "tool_use",
                "id": "tool-1",
                "name": "unexpected_tool" if outcome == "wrong_tool" else tool_name,
                "input": {},
            },
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": json.dumps(payload)},
        },
        {"type": "content_block_stop", "index": 0},
        {"type": "message_stop"},
    ]
    original = RuntimeError("synthetic stream failure")

    class ResponseBody(httpx.SyncByteStream):
        def __iter__(self):
            for event in events:
                if outcome == "stream_error":
                    raise original
                yield f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode()

    # A synthetic streaming body exercises the installed SDK without a request.
    response = httpx.Response(200, stream=ResponseBody())
    client = anthropic.Anthropic(api_key="unused")
    stream = Stream(cast_to=RawMessageStreamEvent, response=response, client=client)
    monkeypatch.setattr(client.messages, "create", lambda **_: stream)
    monkeypatch.setattr(anthropic, "initialize_anthropic_client", lambda **_: client)

    try:
        if outcome == "success":
            result = getattr(anthropic, generation)("write music", "claude-sonnet-4-5")
            loop = result[0] if generation == "loop_gen" else result[0].variations[0]
            assert loop == sample_loop
        else:
            error_type = RuntimeError if outcome == "stream_error" else ValueError
            with pytest.raises(error_type) as raised:
                getattr(anthropic, generation)("write music", "claude-sonnet-4-5")
            if outcome == "stream_error":
                assert raised.value is original
        assert response.is_closed
        assert client.is_closed()
    finally:
        stream.close()
        client.close()


@pytest.mark.parametrize("generation", ["loop_gen", "variations_gen"])
def test_anthropic_generation_closes_client_when_request_fails(monkeypatch, generation):
    from conductor_core.errors import ProviderTimeoutError

    client = anthropic.Anthropic(api_key="unused")
    original = anthropic.APITimeoutError(request=Mock())

    def fail_request(**kwargs):
        raise original

    monkeypatch.setattr(client.messages, "create", fail_request)
    monkeypatch.setattr(anthropic, "initialize_anthropic_client", lambda **_: client)

    try:
        with pytest.raises(ProviderTimeoutError) as raised:
            getattr(anthropic, generation)("write music", "claude-sonnet-4-5")
        assert raised.value.operation == "request"
        assert raised.value.__cause__ is original
        assert client.is_closed()
    finally:
        client.close()


def test_anthropic_rejects_unknown_reasoning_effort():
    with pytest.raises(ValueError, match="Unsupported Anthropic reasoning effort"):
        anthropic._validated_effort("extreme")


@pytest.mark.parametrize("use_thinking", [False, True])
@pytest.mark.parametrize("effort", ["low", "medium", "high", "xhigh", "max"])
def test_haiku_5_5_thinking_controls(use_thinking, effort):
    config = music.get_model_info()["models"]["Anthropic"]["claude-haiku-5-5"]
    request = {"tool_choice": {"type": "tool", "name": "build_MIDI_loop"}}

    anthropic._apply_thinking_params(request, config, 0.4, use_thinking, effort)

    assert "extra_body" not in request
    if use_thinking:
        assert request["thinking"] == {"type": "adaptive"}
        assert request["output_config"] == {"effort": effort}
        assert request["tool_choice"] == {
            "type": "auto",
            "disable_parallel_tool_use": True,
        }
    else:
        assert request["thinking"] == {"type": "disabled"}
        # Omit effort so disabled thinking uses the API's valid medium default.
        assert "output_config" not in request
        assert request["tool_choice"]["type"] == "tool"


class _RequestCaptured(Exception):
    pass


@pytest.mark.parametrize("generation", ["loop", "variations"])
@pytest.mark.parametrize("use_thinking", [False, True])
@pytest.mark.parametrize("model", music.get_model_info()["models"]["Anthropic"])
def test_anthropic_request_matches_installed_sdk_signature(
    monkeypatch, model, use_thinking, generation
):
    with anthropic.Anthropic(api_key="unused") as sdk_client:
        sdk_signature = inspect.signature(sdk_client.messages.create)
    captured = {}

    def capture_create(**kwargs):
        sdk_signature.bind(**kwargs)
        captured.update(kwargs)
        raise _RequestCaptured

    client = SimpleNamespace(
        close=Mock(), messages=SimpleNamespace(create=capture_create)
    )
    monkeypatch.setattr(anthropic, "initialize_anthropic_client", lambda **_: client)
    generate = anthropic.loop_gen if generation == "loop" else anthropic.variations_gen
    model_config = music.get_model_info()["models"]["Anthropic"][model]
    effort_options = model_config.get("effort_options") or []

    with pytest.raises(_RequestCaptured):
        generate(
            "write a loop",
            model,
            temp=0.4,
            use_thinking=use_thinking,
            effort=effort_options[-1] if effort_options else None,
        )

    assert captured["tool_choice"]["disable_parallel_tool_use"] is True
    client.close.assert_called_once_with()
    assert "temperature" not in captured
    if model_config.get("temperature_supported", True):
        expected = 1.0 if use_thinking else 0.4
        assert captured["extra_body"] == {"temperature": expected}
    else:
        assert "extra_body" not in captured
