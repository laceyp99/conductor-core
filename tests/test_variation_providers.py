import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from conductor_core import routing
from conductor_core.models import Loop
from conductor_core.providers import anthropic, google, ollama, openai
from conductor_core.providers._variations import VariationCollection
from conductor_core.variations import VariationUsage


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
    return {
        "Bar_1": bar,
        "Bar_2": {**bar, "num": 2},
        "Bar_3": {**bar, "num": 3},
        "Bar_4": {**bar, "num": 4},
    }


def test_collection_rejects_the_complete_payload_when_one_item_is_malformed():
    payload = {"variations": [_loop_payload(), {"not": "a loop"}]}

    with pytest.raises(ValidationError):
        VariationCollection.model_validate(payload)


@pytest.mark.parametrize(
    ("provider", "model", "adapter_name"),
    [
        ("OpenAI", "openai-model", "openai_api"),
        ("Google", "google-model", "gemini_api"),
        ("Anthropic", "anthropic-model", "claude_api"),
        ("Ollama", "ollama-model", "ollama_api"),
    ],
)
def test_routing_normalizes_all_variation_providers(
    monkeypatch, provider, model, adapter_name
):
    model_info = {"models": {"OpenAI": {}, "Google": {}, "Anthropic": {}}}
    if provider != "Ollama":
        model_info["models"][provider][model] = {}
    monkeypatch.setattr(routing, "get_model_info", lambda: model_info)
    if provider == "Ollama":
        monkeypatch.setattr(
            routing.ollama_api,
            "get_ollama_status",
            lambda **kwargs: {"available": True, "models": [model]},
        )
    collection = VariationCollection(
        variations=[Loop.model_validate(_loop_payload())] * 2
    )
    usage = VariationUsage(input_tokens=10, output_tokens=20, total_tokens=30)
    calls = []

    def fake_variations_gen(**kwargs):
        calls.append(kwargs)
        return collection, [{"role": "assistant", "content": "response"}], 0.25, usage

    monkeypatch.setattr(
        getattr(routing, adapter_name), "variations_gen", fake_variations_gen
    )

    result = routing.generate_variations(model, "brief", 2, request_timeout=3.0)

    assert len(calls) == 1
    assert calls[0]["request_timeout"] == 3.0
    assert calls[0]["prompt"] == "Requested variation count: 2\n\nbrief"
    assert result.variations == tuple(collection.variations)
    assert result.provider == provider
    assert result.usage == usage
    assert result.cost == 0.25
    with pytest.raises(AttributeError):
        result.provider = "changed"


def test_routing_preserves_nullable_usage(monkeypatch):
    monkeypatch.setattr(
        routing,
        "get_model_info",
        lambda: {
            "models": {"OpenAI": {"openai-model": {}}, "Google": {}, "Anthropic": {}}
        },
    )
    collection = SimpleNamespace(variations=[Loop.model_validate(_loop_payload())])
    monkeypatch.setattr(
        routing.openai_api,
        "variations_gen",
        lambda **kwargs: (collection, [], None, None),
    )

    result = routing.generate_variations("openai-model", "brief", 2)

    assert result.usage is None
    assert result.cost is None


def test_openai_requests_one_typed_collection(monkeypatch):
    calls = []
    response = SimpleNamespace(
        output_parsed={"variations": [_loop_payload()]}, output=[], usage=None
    )
    client = SimpleNamespace(
        responses=SimpleNamespace(
            parse=lambda **kwargs: calls.append(kwargs) or response
        )
    )
    monkeypatch.setattr(openai, "initialize_openai_client", lambda **kwargs: client)
    monkeypatch.setattr(
        openai.utils,
        "get_model_info",
        lambda: {"models": {"OpenAI": {"model": {}}}},
    )

    collection, _, cost, usage = openai.variations_gen("brief", "model")

    assert len(calls) == 1
    assert calls[0]["text_format"] is VariationCollection
    assert len(collection.variations) == 1
    assert cost is None
    assert usage is None


def test_anthropic_requests_one_collection_tool(monkeypatch):
    calls = []
    payload = json.dumps({"variations": [_loop_payload()]})
    completion = [
        SimpleNamespace(
            type="content_block_delta",
            delta=SimpleNamespace(partial_json=payload),
        ),
        SimpleNamespace(type="message_stop"),
    ]
    client = SimpleNamespace(
        messages=SimpleNamespace(
            create=lambda **kwargs: calls.append(kwargs) or completion
        )
    )
    monkeypatch.setattr(
        anthropic, "initialize_anthropic_client", lambda **kwargs: client
    )
    monkeypatch.setattr(
        anthropic.utils,
        "get_model_info",
        lambda: {
            "models": {
                "Anthropic": {
                    "model": {
                        "max_tokens": 1000,
                        "cost": {"input": 0, "output": 0},
                    }
                }
            }
        },
    )

    collection, _, _, _ = anthropic.variations_gen("brief", "model")

    assert len(calls) == 1
    assert (
        calls[0]["tools"][0]["input_schema"] == VariationCollection.model_json_schema()
    )
    assert len(collection.variations) == 1


def test_google_requests_one_collection_schema(monkeypatch):
    calls = []
    content = json.dumps({"variations": [_loop_payload()]})
    response = SimpleNamespace(
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(
                    parts=[SimpleNamespace(text=content, thought=False)]
                )
            )
        ],
        usage_metadata=None,
    )
    client = SimpleNamespace(
        models=SimpleNamespace(
            generate_content=lambda **kwargs: calls.append(kwargs) or response
        )
    )
    monkeypatch.setattr(google, "initialize_gemini_client", lambda **kwargs: client)
    monkeypatch.setattr(
        google.utils,
        "get_model_info",
        lambda: {"models": {"Google": {"model": {"extended_thinking": False}}}},
    )

    collection, _, cost, usage = google.variations_gen("brief", "model")

    assert len(calls) == 1
    assert (
        calls[0]["config"]["response_json_schema"]
        == VariationCollection.model_json_schema()
    )
    assert len(collection.variations) == 1
    assert cost is None
    assert usage is None


def test_ollama_requests_one_collection_format(monkeypatch):
    calls = []
    content = json.dumps({"variations": [_loop_payload()]})
    completion = SimpleNamespace(
        message=SimpleNamespace(content=content),
        prompt_eval_count=4,
        eval_count=6,
    )
    client = SimpleNamespace(chat=lambda **kwargs: calls.append(kwargs) or completion)
    monkeypatch.setattr(ollama, "initialize_ollama_client", lambda **kwargs: client)

    collection, _, cost, usage = ollama.variations_gen("brief", "model")

    assert len(calls) == 1
    assert calls[0]["format"] == VariationCollection.model_json_schema()
    assert len(collection.variations) == 1
    assert cost == 0
    assert usage.total_tokens == 10


@pytest.mark.parametrize(("use_thinking", "effort"), [(False, "low"), (True, "max")])
def test_opus_5_5_variations_use_always_on_thinking(monkeypatch, use_thinking, effort):
    calls = []
    payload = json.dumps({"variations": [_loop_payload()]})
    completion = [
        SimpleNamespace(
            type="content_block_delta",
            delta=SimpleNamespace(partial_json=payload),
        ),
        SimpleNamespace(type="message_stop"),
    ]
    client = SimpleNamespace(
        messages=SimpleNamespace(
            create=lambda **kwargs: calls.append(kwargs) or completion
        )
    )
    monkeypatch.setattr(
        anthropic, "initialize_anthropic_client", lambda **kwargs: client
    )

    collection, _, _, _ = anthropic.variations_gen(
        "brief", "claude-opus-5-5", temp=0.5, use_thinking=use_thinking, effort="max"
    )

    assert len(collection.variations) == 1
    assert calls[0]["max_tokens"] == 128000
    assert calls[0]["output_config"] == {"effort": effort}
    assert calls[0]["tool_choice"] == {"type": "auto"}
    assert "temperature" not in calls[0]
    assert "thinking" not in calls[0]
