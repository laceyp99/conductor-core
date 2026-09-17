import json
from types import SimpleNamespace

import pytest
from jsonschema import Draft202012Validator

from conductor_core.providers import anthropic as anthropic_api
from conductor_core.providers import google as google_api
from conductor_core.providers import ollama as ollama_api
from conductor_core.providers import openai as openai_api
from conductor_core.providers._variations import (
    build_variation_schema,
    normalize_variation_output,
)


def _loop_payload():
    return {f"Bar_{number}": {"num": number, "notes": []} for number in range(1, 5)}


def _representative_loop_payload():
    return {
        f"Bar_{number}": {
            "num": number,
            "notes": [
                {
                    "pitch": "C",
                    "octave": 4,
                    "velocity": 96,
                    "time": {"start_beat": 1, "duration": 4},
                }
            ],
        }
        for number in range(1, 5)
    }


def _anthropic_stream(raw_output, usage=None, text=""):
    chunks = []
    if usage is not None:
        chunks.append(
            SimpleNamespace(
                type="message_start",
                message=SimpleNamespace(usage=usage),
            )
        )
    if raw_output is not None:
        chunks.append(
            SimpleNamespace(
                type="content_block_delta",
                delta=SimpleNamespace(partial_json=raw_output),
            )
        )
    if text:
        chunks.append(
            SimpleNamespace(
                type="content_block_delta",
                delta=SimpleNamespace(text=text),
            )
        )
    chunks.append(SimpleNamespace(type="message_stop"))
    return chunks


def _invoke_provider(monkeypatch, provider, raw_output, usage=None, text=""):
    captured = {}
    if provider == "OpenAI":
        response = SimpleNamespace(output_text=raw_output, output=[], usage=usage)

        def create(**kwargs):
            captured.update(kwargs)
            return response

        client = SimpleNamespace(responses=SimpleNamespace(create=create))
        monkeypatch.setattr(
            openai_api, "initialize_openai_client", lambda **kwargs: client
        )
        result = openai_api.variation_gen(
            "user text", 4, "gpt-4o-mini", system_prompt="system text"
        )
        schema = captured["text"]["format"]["schema"]
    elif provider == "Google":
        parts = []
        if text:
            parts.append(SimpleNamespace(text=text, thought=True))
        if raw_output is not None:
            parts.append(SimpleNamespace(text=raw_output, thought=False))
        response = SimpleNamespace(
            candidates=(
                [SimpleNamespace(content=SimpleNamespace(parts=parts))] if parts else []
            ),
            usage_metadata=usage,
        )

        def generate_content(**kwargs):
            captured.update(kwargs)
            return response

        client = SimpleNamespace(
            models=SimpleNamespace(generate_content=generate_content)
        )
        monkeypatch.setattr(
            google_api, "initialize_gemini_client", lambda **kwargs: client
        )
        result = google_api.variation_gen(
            "user text", 4, "gemini-3.1-flash-lite", system_prompt="system text"
        )
        schema = captured["config"]["response_json_schema"]
    elif provider == "Anthropic":

        def create(**kwargs):
            captured.update(kwargs)
            return _anthropic_stream(raw_output, usage=usage, text=text)

        client = SimpleNamespace(messages=SimpleNamespace(create=create))
        monkeypatch.setattr(
            anthropic_api, "initialize_anthropic_client", lambda **kwargs: client
        )
        result = anthropic_api.variation_gen(
            "user text", 4, "claude-sonnet-4-5", system_prompt="system text"
        )
        schema = captured["tools"][0]["input_schema"]
    else:
        message_args = {} if raw_output is None else {"content": raw_output}
        if text:
            message_args["thinking"] = text
        completion = SimpleNamespace(message=SimpleNamespace(**message_args))
        if usage is not None:
            completion.prompt_eval_count = usage.input_tokens
            completion.eval_count = usage.output_tokens

        def chat(**kwargs):
            captured.update(kwargs)
            return completion

        client = SimpleNamespace(chat=chat)
        monkeypatch.setattr(
            ollama_api, "initialize_ollama_client", lambda **kwargs: client
        )
        result = ollama_api.variation_gen(
            "user text", 4, "llama3", system_prompt="system text"
        )
        schema = captured["format"]
    return result, captured, schema


@pytest.mark.parametrize("provider", ["OpenAI", "Google", "Anthropic", "Ollama"])
def test_all_providers_use_same_schema_and_preserve_raw_item_positions(
    monkeypatch, provider
):
    raw_items = [_loop_payload(), {"malformed": True}, "scalar", None]
    raw_output = json.dumps({"items": raw_items})

    result, _, schema = _invoke_provider(monkeypatch, provider, raw_output)

    assert schema == build_variation_schema(4)
    assert schema["additionalProperties"] is False
    assert schema["properties"]["items"]["minItems"] == 4
    assert schema["properties"]["items"]["maxItems"] == 4
    assert result.items == tuple(raw_items)
    assert result.received_count == 4
    assert result.structural_diagnostic is None
    assert result.messages[:2] == [
        {"role": "system", "content": "system text"},
        {"role": "user", "content": "user text"},
    ]


def test_shared_schema_resolves_references_and_validates_representative_batch():
    schema = build_variation_schema(2)
    batch = {"items": [_representative_loop_payload()] * 2}

    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(batch)


def test_openai_strict_schema_closes_nested_loop_objects(monkeypatch):
    raw_output = json.dumps({"items": [_loop_payload()] * 4})

    _, _, schema = _invoke_provider(monkeypatch, "OpenAI", raw_output)

    loop_schema = schema["properties"]["items"]["items"]
    assert loop_schema["additionalProperties"] is False
    for definition in ("Bar", "Note", "TimeInformation"):
        assert schema["$defs"][definition]["additionalProperties"] is False


@pytest.mark.parametrize(
    ("raw_output", "code", "location", "received_count"),
    [
        (None, "missing_output", None, None),
        ("not json", "invalid_json", None, None),
        ("[]", "invalid_top_level", (), None),
        ('{"other": []}', "invalid_top_level", (), None),
        ('{"items": [], "extra": true}', "invalid_top_level", (), None),
        ('{"items": [{}, {}]}', "wrong_count", ("items",), 2),
        ('{"items": [{}, {}, {}, {}, {}]}', "wrong_count", ("items",), 5),
    ],
)
def test_shared_normalizer_returns_stable_structural_diagnostics(
    raw_output, code, location, received_count
):
    items, actual_count, diagnostic = normalize_variation_output(raw_output, 4)

    assert items is None
    assert actual_count == received_count
    assert diagnostic.code == code
    assert diagnostic.location == location
    if code == "wrong_count":
        assert "4" in diagnostic.message
        assert str(received_count) in diagnostic.message


@pytest.mark.parametrize("provider", ["OpenAI", "Google", "Anthropic", "Ollama"])
def test_provider_missing_output_is_in_band_and_preserves_refusal_evidence(
    monkeypatch, provider
):
    result, _, _ = _invoke_provider(
        monkeypatch, provider, None, text="I cannot complete that request."
    )

    assert result.items is None
    assert result.received_count is None
    assert result.structural_diagnostic.code == "missing_output"
    assert result.structural_diagnostic.location is None
    if provider != "OpenAI":
        assert result.messages[-1]["content"] == "I cannot complete that request."


@pytest.mark.parametrize("provider", ["OpenAI", "Google", "Anthropic", "Ollama"])
def test_provider_wrong_count_retains_raw_output_and_machine_readable_count(
    monkeypatch, provider
):
    raw_output = json.dumps({"items": [_loop_payload(), None]})

    result, _, _ = _invoke_provider(monkeypatch, provider, raw_output)

    assert result.items is None
    assert result.received_count == 2
    assert result.structural_diagnostic.code == "wrong_count"
    assert result.structural_diagnostic.location == ("items",)
    assert "4" in result.structural_diagnostic.message
    assert "2" in result.structural_diagnostic.message
    assert result.messages[-1]["content"] != result.structural_diagnostic.message


def test_openai_usage_preserves_missing_partial_zero_and_complete_measurements():
    no_usage = SimpleNamespace(usage=None)
    assert openai_api._variation_usage_and_cost("gpt-4o-mini", no_usage) == (
        None,
        None,
    )

    partial = SimpleNamespace(usage=SimpleNamespace(input_tokens=10))
    usage, cost = openai_api._variation_usage_and_cost("gpt-4o-mini", partial)
    assert usage.model_dump() == {
        "input_tokens": 10,
        "output_tokens": None,
        "total_tokens": None,
    }
    assert cost is None

    for input_tokens, output_tokens in [(0, 0), (10, 5)]:
        response = SimpleNamespace(
            usage=SimpleNamespace(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                input_tokens_details=SimpleNamespace(cached_tokens=0),
            )
        )
        usage, cost = openai_api._variation_usage_and_cost("gpt-4o-mini", response)
        assert usage.total_tokens == input_tokens + output_tokens
        assert cost is not None


def test_google_usage_preserves_missing_partial_zero_and_complete_measurements():
    assert google_api._variation_usage_and_cost(
        "gemini-3.1-flash-lite", SimpleNamespace(usage_metadata=None)
    ) == (None, None)

    partial = SimpleNamespace(usage_metadata=SimpleNamespace(prompt_token_count=10))
    usage, cost = google_api._variation_usage_and_cost("gemini-3.1-flash-lite", partial)
    assert usage.input_tokens == 10
    assert usage.output_tokens is None
    assert cost is None

    for input_tokens, output_tokens in [(0, 0), (10, 5)]:
        response = SimpleNamespace(
            usage_metadata=SimpleNamespace(
                prompt_token_count=input_tokens,
                candidates_token_count=output_tokens,
                thoughts_token_count=0,
                cached_content_token_count=0,
            )
        )
        usage, cost = google_api._variation_usage_and_cost(
            "gemini-3.1-flash-lite", response
        )
        assert usage.total_tokens == input_tokens + output_tokens
        assert cost is not None


def test_anthropic_usage_sums_cached_inputs_and_preserves_partial_values():
    empty = {
        "input_tokens": None,
        "output_tokens": None,
        "cache_creation": None,
        "cache_read": None,
        "cache_creation_1h": None,
    }
    assert anthropic_api._variation_usage_and_cost("claude-sonnet-4-5", empty) == (
        None,
        None,
    )

    partial = {**empty, "input_tokens": 10, "cache_read": 3}
    usage, cost = anthropic_api._variation_usage_and_cost("claude-sonnet-4-5", partial)
    assert usage.input_tokens == 13
    assert usage.output_tokens is None
    assert cost is None

    complete = {
        **empty,
        "input_tokens": 10,
        "output_tokens": 5,
        "cache_creation": 2,
        "cache_read": 3,
        "cache_creation_1h": 0,
    }
    usage, cost = anthropic_api._variation_usage_and_cost("claude-sonnet-4-5", complete)
    assert usage.model_dump() == {
        "input_tokens": 15,
        "output_tokens": 5,
        "total_tokens": 20,
    }
    assert cost is not None


def test_ollama_usage_is_nullable_or_partial_while_cost_remains_zero(monkeypatch):
    result, _, _ = _invoke_provider(
        monkeypatch, "Ollama", json.dumps({"items": [None] * 4})
    )
    assert result.usage is None
    assert result.cost == 0.0

    usage = SimpleNamespace(input_tokens=12, output_tokens=None)
    result, _, _ = _invoke_provider(
        monkeypatch,
        "Ollama",
        json.dumps({"items": [None] * 4}),
        usage=usage,
    )
    assert result.usage.input_tokens == 12
    assert result.usage.output_tokens is None
    assert result.usage.total_tokens is None
    assert result.cost == 0.0


@pytest.mark.parametrize("provider", ["OpenAI", "Google", "Anthropic", "Ollama"])
def test_variation_provider_does_not_swallow_unexpected_or_schema_errors(
    monkeypatch, provider
):
    calls = 0

    def fail(**kwargs):
        nonlocal calls
        calls += 1
        raise RuntimeError("schema rejected")

    if provider == "OpenAI":
        client = SimpleNamespace(responses=SimpleNamespace(create=fail))
        monkeypatch.setattr(
            openai_api, "initialize_openai_client", lambda **kwargs: client
        )

        def invoke():
            return openai_api.variation_gen(
                "brief", 4, "gpt-4o-mini", system_prompt="system"
            )
    elif provider == "Google":
        client = SimpleNamespace(models=SimpleNamespace(generate_content=fail))
        monkeypatch.setattr(
            google_api, "initialize_gemini_client", lambda **kwargs: client
        )

        def invoke():
            return google_api.variation_gen(
                "brief", 4, "gemini-3.1-flash-lite", system_prompt="system"
            )
    elif provider == "Anthropic":
        client = SimpleNamespace(messages=SimpleNamespace(create=fail))
        monkeypatch.setattr(
            anthropic_api, "initialize_anthropic_client", lambda **kwargs: client
        )

        def invoke():
            return anthropic_api.variation_gen(
                "brief", 4, "claude-sonnet-4-5", system_prompt="system"
            )
    else:
        client = SimpleNamespace(chat=fail)
        monkeypatch.setattr(
            ollama_api, "initialize_ollama_client", lambda **kwargs: client
        )

        def invoke():
            return ollama_api.variation_gen(
                "brief", 4, "llama3", system_prompt="system"
            )

    with pytest.raises(RuntimeError, match="schema rejected"):
        invoke()
    assert calls == 1
