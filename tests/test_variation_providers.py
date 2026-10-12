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


@pytest.mark.parametrize("effort", [None, "unsupported"])
def test_google_variations_reject_invalid_effort_before_client_initialization(
    monkeypatch, effort
):
    initialize_client = Mock()
    monkeypatch.setattr(google, "initialize_gemini_client", initialize_client)

    with pytest.raises(ValueError, match="Invalid effort") as error:
        google.variations_gen(
            "brief", "gemini-3.8-flash", use_thinking=True, effort=effort
        )

    assert str(error.value) == (
        f"Invalid effort {effort!r} for gemini-3.8-flash. "
        "Expected one of: low, medium, high"
    )
    initialize_client.assert_not_called()


@pytest.mark.parametrize(
    ("model_config", "use_thinking", "effort", "expected_level", "expected_budget"),
    [
        ({"extended_thinking": False}, None, None, None, None),
        *[
            (
                {
                    "extended_thinking": True,
                    "effort_options": ["low", "medium", "high"],
                },
                use_thinking,
                effort,
                expected_level,
                None,
            )
            for use_thinking, effort, expected_level in [
                (True, "high", "HIGH"),
                (True, "medium", "MEDIUM"),
                (False, "high", "LOW"),
                (False, None, "LOW"),
                (False, "unsupported", "LOW"),
                (None, "high", "LOW"),
                (None, None, "LOW"),
                (None, "unsupported", "LOW"),
            ]
        ],
        *[
            (
                {
                    "extended_thinking": True,
                    "min_thinking_budget": minimum,
                    "max_thinking_budget": maximum,
                },
                use_thinking,
                None,
                None,
                maximum if use_thinking else minimum,
            )
            for minimum, maximum in [(128, 32768), (0, -1)]
            for use_thinking in [True, False, None]
        ],
    ],
)
def test_google_requests_one_collection_schema(
    monkeypatch, model_config, use_thinking, effort, expected_level, expected_budget
):
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

    collection, _, cost, usage = google.variations_gen(
        "brief",
        "model",
        use_thinking=use_thinking,
        **({"effort": effort} if effort is not None else {}),
    )

    assert len(calls) == 1
    assert (
        calls[0]["config"]["response_json_schema"]
        == VariationCollection.model_json_schema()
    )
    assert len(collection.variations) == 1
    assert collection.variations[0] == Loop.model_validate(_loop_payload())
    thinking = calls[0]["config"].get("thinking_config")
    if expected_level is not None:
        assert thinking.thinking_level.value == expected_level
        assert thinking.thinking_budget is None
        assert thinking.include_thoughts is True
    elif expected_budget is not None:
        assert thinking.thinking_budget == expected_budget
        assert thinking.thinking_level is None
        assert thinking.include_thoughts is True
    else:
        assert thinking is None
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
