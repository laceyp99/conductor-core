"""Offline Ollama context-window handling tests."""

import json
from types import SimpleNamespace

import pytest

import conductor_core
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
