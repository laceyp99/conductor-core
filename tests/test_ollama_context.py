"""Offline Ollama context-window handling tests."""

import json
import logging
from types import SimpleNamespace

import pytest

import conductor_core
from conductor_core import routing
from conductor_core.config import GenerationRequest, VariationGenerationRequest
from conductor_core.errors import ProviderContextLengthError, ProviderRequestError
from conductor_core.providers import ollama

CAPABILITIES = {"extended_thinking": True, "effort_options": []}


def _loop_payload():
    bar = {
        "num": 1,
        "notes": [
            {
                "pitch": "C",
                "octave": 4,
                "velocity": 100,
                "time": {"start_beat": 1, "duration": 1},
            }
        ],
    }
    return {f"Bar_{number}": {**bar, "num": number} for number in range(1, 5)}


def _content(generation):
    if generation == "loop":
        return json.dumps(_loop_payload())
    return json.dumps({"variations": [_loop_payload()]})


def _generate(monkeypatch, generation, completion):
    calls = []
    client = SimpleNamespace(chat=lambda **kwargs: calls.append(kwargs) or completion)
    monkeypatch.setattr(ollama, "initialize_ollama_client", lambda **kwargs: client)
    function = ollama.loop_gen if generation == "loop" else ollama.variations_gen
    result = function(
        "prompt",
        "qwen3.5:4b",
        temp=0.4,
        use_thinking=True,
        model_capabilities=CAPABILITIES,
    )
    return result, calls


@pytest.mark.parametrize("generation", ["loop", "variations"])
@pytest.mark.parametrize("content", ["", '{"Bar_1": {"num": 1, "notes": ['])
def test_ollama_length_stop_raises_context_error(monkeypatch, generation, content):
    completion = SimpleNamespace(
        done_reason="length",
        prompt_eval_count=506,
        eval_count=3567,
        message=SimpleNamespace(content=content, thinking="private reasoning"),
    )

    with pytest.raises(ProviderContextLengthError) as exc_info:
        _generate(monkeypatch, generation, completion)

    error = exc_info.value
    assert isinstance(error, ProviderRequestError)
    assert error.provider == "Ollama"
    assert error.operation == "response"
    assert error.model == "qwen3.5:4b"
    assert error.prompt_tokens == 506
    assert error.output_tokens == 3567
    assert error.context_length is None
    message = str(error)
    assert "ran out of context" in message
    assert "Lower or disable thinking" in message
    assert "prompt (506 tokens)" in message
    assert "response (3,567 tokens)" in message
    assert "private reasoning" not in message
    if content:
        assert content not in message


def test_context_error_message_without_token_counts():
    error = ProviderContextLengthError("Ollama", "local-model", context_length=4096)

    assert "the prompt and response filled the 4,096-token context window" in str(error)


@pytest.mark.parametrize("generation", ["loop", "variations"])
def test_ollama_stop_response_is_unchanged(monkeypatch, generation):
    completion = SimpleNamespace(
        done_reason="stop",
        prompt_eval_count=10,
        eval_count=20,
        message=SimpleNamespace(content=_content(generation), thinking=None),
    )

    result, _ = _generate(monkeypatch, generation, completion)

    assert result[2] == 0


def test_context_error_is_public():
    assert conductor_core.ProviderContextLengthError is ProviderContextLengthError
    assert "ProviderContextLengthError" in conductor_core.__all__


@pytest.mark.parametrize("generation", ["loop", "variations"])
def test_ollama_default_request_leaves_num_ctx_to_ollama(monkeypatch, generation):
    completion = SimpleNamespace(
        done_reason="stop",
        message=SimpleNamespace(content=_content(generation), thinking=None),
    )

    _, calls = _generate(monkeypatch, generation, completion)

    assert calls[0]["options"] == {"temperature": 0.4}


@pytest.mark.parametrize("generation", ["generate_midi", "generate_variations"])
def test_routing_forwards_ollama_num_ctx(monkeypatch, generation):
    calls = []
    completion = SimpleNamespace(
        done_reason="stop",
        message=SimpleNamespace(
            content=_content("loop" if generation == "generate_midi" else "var"),
            thinking=None,
        ),
    )
    client = SimpleNamespace(
        list=lambda: SimpleNamespace(models=[SimpleNamespace(model="local-model")]),
        show=lambda name: SimpleNamespace(capabilities=["completion"]),
        chat=lambda **kwargs: calls.append(kwargs) or completion,
    )
    monkeypatch.setattr(ollama, "initialize_ollama_client", lambda **kwargs: client)

    if generation == "generate_midi":
        routing.generate_midi("local-model", "prompt", ollama_num_ctx=16384)
    else:
        routing.generate_variations("local-model", "prompt", 1, ollama_num_ctx=16384)

    assert calls[0]["options"]["num_ctx"] == 16384


def test_context_error_reports_requested_num_ctx(monkeypatch):
    calls = []
    completion = SimpleNamespace(
        done_reason="length",
        message=SimpleNamespace(content="", thinking=None),
    )
    client = SimpleNamespace(chat=lambda **kwargs: calls.append(kwargs) or completion)
    monkeypatch.setattr(ollama, "initialize_ollama_client", lambda **kwargs: client)

    with pytest.raises(ProviderContextLengthError) as exc_info:
        ollama.loop_gen(
            "prompt", "local-model", model_capabilities=CAPABILITIES, num_ctx=8192
        )

    assert exc_info.value.context_length == 8192
    assert "8,192-token context window" in str(exc_info.value)


def test_routing_ignores_ollama_num_ctx_for_cloud_models(monkeypatch, caplog):
    calls = []
    monkeypatch.setattr(
        routing.openai_api,
        "loop_gen",
        lambda **kwargs: calls.append(kwargs) or (None, [], 0),
    )

    with caplog.at_level(logging.WARNING, logger="conductor_core.routing"):
        routing.generate_midi("gpt-4.1", "prompt", ollama_num_ctx=16384)

    assert "num_ctx" not in calls[0]
    assert "only applies to Ollama models" in caplog.text


@pytest.mark.parametrize(
    "request_type", [GenerationRequest, VariationGenerationRequest]
)
@pytest.mark.parametrize(
    ("value", "error"),
    [(0, ValueError), (-1, ValueError), (True, TypeError), (4096.0, TypeError)],
)
def test_requests_reject_invalid_ollama_num_ctx(request_type, value, error):
    with pytest.raises(error, match="ollama_num_ctx"):
        request_type(
            key="C",
            scale="major",
            description="loop",
            model="local-model",
            ollama_num_ctx=value,
        )


@pytest.mark.parametrize(
    "request_type", [GenerationRequest, VariationGenerationRequest]
)
def test_requests_accept_ollama_num_ctx(request_type):
    request = request_type(
        key="C", scale="major", description="loop", model="m", ollama_num_ctx=16384
    )

    assert request.ollama_num_ctx == 16384
