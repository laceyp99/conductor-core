import json
from types import SimpleNamespace
from unittest.mock import Mock

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
            "get_model_status",
            lambda model_name, **kwargs: ollama.OllamaModelInspection(
                model_name, {}, None
            ),
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


@pytest.mark.parametrize("use_thinking", [False, True, None])
def test_anthropic_requests_one_collection_tool(monkeypatch, use_thinking):
    calls = []
    payload = json.dumps({"variations": [_loop_payload()]})
    completion = [
        SimpleNamespace(
            type="content_block_start",
            index=0,
            content_block=SimpleNamespace(
                type="tool_use", name="build_MIDI_variations"
            ),
        ),
        SimpleNamespace(
            type="content_block_delta",
            index=0,
            delta=SimpleNamespace(partial_json=payload),
        ),
        SimpleNamespace(type="content_block_stop", index=0),
        SimpleNamespace(type="message_stop"),
    ]
    client = SimpleNamespace(
        close=Mock(),
        messages=SimpleNamespace(
            create=lambda **kwargs: (
                calls.append(kwargs) or (event for event in completion)
            )
        ),
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
                        "extended_thinking": True,
                        "max_thinking_budget": 59000,
                        "cost": {"input": 0, "output": 0},
                    }
                }
            }
        },
    )

    collection, _, _, _ = anthropic.variations_gen(
        "brief", "model", use_thinking=use_thinking
    )

    assert len(calls) == 1
    assert (
        calls[0]["tools"][0]["input_schema"] == VariationCollection.model_json_schema()
    )
    assert len(collection.variations) == 1
    if use_thinking:
        assert calls[0]["thinking"] == {"type": "enabled", "budget_tokens": 59000}
    else:
        assert "thinking" not in calls[0]
    assert "output_config" not in calls[0]


def test_anthropic_variation_usage_includes_cache_tokens(monkeypatch):
    payload = json.dumps({"variations": [_loop_payload(), _loop_payload()]})
    start_usage = SimpleNamespace(
        input_tokens=10,
        output_tokens=1,
        cache_read_input_tokens=90,
        cache_creation_input_tokens=5,
        cache_creation=SimpleNamespace(
            ephemeral_5m_input_tokens=0, ephemeral_1h_input_tokens=5
        ),
    )
    completion = [
        SimpleNamespace(
            type="message_start", message=SimpleNamespace(usage=start_usage)
        ),
        SimpleNamespace(
            type="content_block_start",
            index=0,
            content_block=SimpleNamespace(
                type="tool_use", name="build_MIDI_variations"
            ),
        ),
        SimpleNamespace(
            type="content_block_delta",
            index=0,
            delta=SimpleNamespace(partial_json=payload),
        ),
        SimpleNamespace(type="content_block_stop", index=0),
        SimpleNamespace(type="message_delta", usage=SimpleNamespace(output_tokens=20)),
        SimpleNamespace(type="message_stop"),
    ]
    client = SimpleNamespace(
        close=Mock(),
        messages=SimpleNamespace(
            create=lambda **kwargs: (event for event in completion)
        ),
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
                        "cost": {
                            "input": 1,
                            "output": 2,
                            "1h cache input": 4,
                            "cache hits/refreshes": 8,
                        },
                    }
                }
            }
        },
    )

    _, _, cost, usage = anthropic.variations_gen("brief", "model")

    assert usage == VariationUsage(input_tokens=105, output_tokens=20, total_tokens=125)
    # Billing keeps each category at its own rate; the 1h split is not re-added.
    assert cost == pytest.approx((10 * 1 + 20 * 2 + 5 * 4 + 90 * 8) / 1_000_000)


def _fake_gemini_variations(monkeypatch, model_config):
    """Capture Gemini variation requests made against fake model metadata."""
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
        lambda: {"models": {"Google": {"model": model_config}}},
    )
    return calls


def test_google_requests_one_collection_schema(monkeypatch):
    calls = _fake_gemini_variations(monkeypatch, {"extended_thinking": False})

    collection, _, cost, usage = google.variations_gen("brief", "model")

    assert len(calls) == 1
    assert (
        calls[0]["config"]["response_json_schema"]
        == VariationCollection.model_json_schema()
    )
    assert len(collection.variations) == 1
    assert "thinking_config" not in calls[0]["config"]
    assert cost is None
    assert usage is None


@pytest.mark.parametrize(
    ("model_config", "use_thinking", "expected_level", "expected_budget"),
    [
        # Discrete models have no budget keys; see issue #148.
        (
            {"extended_thinking": True, "effort_options": ["low", "high"]},
            True,
            "HIGH",
            None,
        ),
        (
            {
                "extended_thinking": True,
                "min_thinking_budget": 128,
                "max_thinking_budget": 32768,
            },
            True,
            None,
            32768,
        ),
        (
            {
                "extended_thinking": True,
                "min_thinking_budget": 128,
                "max_thinking_budget": 32768,
            },
            False,
            None,
            128,
        ),
    ],
    ids=["discrete", "budget-on", "budget-off"],
)
def test_google_variations_thinking_config(
    monkeypatch, model_config, use_thinking, expected_level, expected_budget
):
    calls = _fake_gemini_variations(monkeypatch, model_config)

    google.variations_gen("brief", "model", use_thinking=use_thinking, effort="high")

    thinking = calls[0]["config"]["thinking_config"]
    assert getattr(thinking.thinking_level, "value", None) == expected_level
    assert thinking.thinking_budget == expected_budget


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

    collection, _, cost, usage = ollama.variations_gen(
        "brief", "model", model_capabilities={"extended_thinking": False}
    )

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
            type="content_block_start",
            index=0,
            content_block=SimpleNamespace(
                type="tool_use", name="build_MIDI_variations"
            ),
        ),
        SimpleNamespace(
            type="content_block_delta",
            index=0,
            delta=SimpleNamespace(partial_json=payload),
        ),
        SimpleNamespace(type="content_block_stop", index=0),
        SimpleNamespace(type="message_stop"),
    ]
    client = SimpleNamespace(
        close=Mock(),
        messages=SimpleNamespace(
            create=lambda **kwargs: (
                calls.append(kwargs) or (event for event in completion)
            )
        ),
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
    assert calls[0]["tool_choice"] == {
        "type": "auto",
        "disable_parallel_tool_use": True,
    }
    assert "temperature" not in calls[0]
    assert "thinking" not in calls[0]
