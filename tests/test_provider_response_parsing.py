import json
import logging
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from conductor_core import models as objects
from conductor_core import music, routing
from conductor_core.providers import anthropic as claude_api
from conductor_core.providers import google as gemini_api
from conductor_core.providers import ollama as ollama_api
from conductor_core.providers import openai as openai_api


class _RequestCaptured(Exception):
    """Stop an adapter after it has built its provider request."""


@pytest.mark.parametrize("generate_name", ["loop_gen", "variations_gen"])
@pytest.mark.parametrize(
    ("provider", "initialize", "provider_name"),
    [
        (openai_api, "initialize_openai_client", "OpenAI"),
        (gemini_api, "initialize_gemini_client", "Google"),
        (claude_api, "initialize_anthropic_client", "Anthropic"),
    ],
)
# 'high' is valid in the SDK vocabulary, but not for this selected model.
@pytest.mark.parametrize("effort_kwargs", [{}, {"effort": "high"}])
def test_direct_cloud_adapter_rejects_effort_not_supported_by_model(
    monkeypatch, provider, initialize, provider_name, generate_name, effort_kwargs
):
    initialize_client = Mock()
    monkeypatch.setattr(provider, initialize, initialize_client)
    monkeypatch.setattr(
        music,
        "get_model_info",
        lambda: {
            "models": {
                provider_name: {
                    "test-model": {
                        "extended_thinking": True,
                        "effort_options": ["low", "medium"],
                        "max_tokens": 1024,
                    }
                }
            }
        },
    )
    with pytest.raises(ValueError, match="Invalid effort") as error:
        getattr(provider, generate_name)(
            prompt="melody", model="test-model", use_thinking=True, **effort_kwargs
        )
    assert str(error.value) == (
        f"Invalid effort {effort_kwargs.get('effort')!r} for test-model. "
        "Expected one of: low, medium"
    )
    initialize_client.assert_not_called()


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


def _fail_save_messages(*args, **kwargs):
    raise AssertionError("provider adapters should not write message logs")


class _AnthropicStream:
    def __init__(self, events):
        self.events = events
        self.closed = False

    def __iter__(self):
        return iter(self.events)

    def close(self):
        self.closed = True


def _anthropic_completion(payload):
    usage = SimpleNamespace(
        input_tokens=100,
        output_tokens=50,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
    )
    return _AnthropicStream(
        [
            SimpleNamespace(type="message_start", message=SimpleNamespace(usage=usage)),
            SimpleNamespace(
                type="content_block_start",
                index=0,
                content_block=SimpleNamespace(type="tool_use", name="build_MIDI_loop"),
            ),
            SimpleNamespace(
                type="content_block_delta",
                index=0,
                delta=SimpleNamespace(partial_json=payload),
            ),
            SimpleNamespace(type="content_block_stop", index=0),
            SimpleNamespace(type="message_stop"),
        ]
    )


def _anthropic_tool_block(index, name, payload):
    midpoint = len(payload) // 2
    return [
        SimpleNamespace(
            type="content_block_start",
            index=index,
            content_block=SimpleNamespace(type="tool_use", name=name, input={}),
        ),
        SimpleNamespace(
            type="content_block_delta",
            index=index,
            delta=SimpleNamespace(partial_json=payload[:midpoint]),
        ),
        SimpleNamespace(
            type="content_block_delta",
            index=index,
            delta=SimpleNamespace(partial_json=payload[midpoint:]),
        ),
        SimpleNamespace(type="content_block_stop", index=index),
    ]


@pytest.mark.parametrize("generation", ["loop_gen", "variations_gen"])
def test_claude_generation_rejects_multiple_tool_calls(monkeypatch, generation):
    tool_name = (
        "build_MIDI_loop" if generation == "loop_gen" else "build_MIDI_variations"
    )
    payload = _loop_payload()
    if generation == "variations_gen":
        payload = {"variations": [payload]}
    payload_json = json.dumps(payload)
    completion = _AnthropicStream(
        [
            *_anthropic_tool_block(1, tool_name, payload_json),
            *_anthropic_tool_block(3, tool_name, payload_json),
        ]
    )
    client = SimpleNamespace(
        close=Mock(), messages=SimpleNamespace(create=lambda **kwargs: completion)
    )
    monkeypatch.setattr(
        claude_api, "initialize_anthropic_client", lambda **kwargs: client
    )

    with pytest.raises(ValueError, match="multiple tool calls") as raised:
        getattr(claude_api, generation)("write music", "claude-sonnet-4-5")

    assert f"{tool_name} (block index 1)" in str(raised.value)
    assert f"{tool_name} (block index 3)" in str(raised.value)
    assert completion.closed


@pytest.mark.parametrize("generation", ["loop_gen", "variations_gen"])
def test_claude_generation_closes_stream_after_message_stop(monkeypatch, generation):
    tool_name = (
        "build_MIDI_loop" if generation == "loop_gen" else "build_MIDI_variations"
    )
    payload = _loop_payload()
    if generation == "variations_gen":
        payload = {"variations": [payload]}
    completion = _AnthropicStream(
        [
            *_anthropic_tool_block(0, tool_name, json.dumps(payload)),
            SimpleNamespace(type="message_stop"),
        ]
    )
    client = SimpleNamespace(
        close=Mock(), messages=SimpleNamespace(create=lambda **kwargs: completion)
    )
    monkeypatch.setattr(
        claude_api, "initialize_anthropic_client", lambda **kwargs: client
    )

    getattr(claude_api, generation)("write music", "claude-sonnet-4-5")

    assert completion.closed


def test_claude_streaming_preserves_fragmented_tool_json_and_other_blocks():
    tool_name = "build_MIDI_loop"
    payload_json = json.dumps(_loop_payload())
    completion = [
        SimpleNamespace(
            type="content_block_start",
            index=0,
            content_block=SimpleNamespace(type="thinking"),
        ),
        SimpleNamespace(
            type="content_block_delta",
            index=0,
            delta=SimpleNamespace(thinking="reason"),
        ),
        SimpleNamespace(type="content_block_stop", index=0),
        SimpleNamespace(
            type="content_block_start",
            index=1,
            content_block=SimpleNamespace(type="text"),
        ),
        SimpleNamespace(
            type="content_block_delta", index=1, delta=SimpleNamespace(text="music")
        ),
        SimpleNamespace(type="content_block_stop", index=1),
        *_anthropic_tool_block(2, tool_name, payload_json),
        SimpleNamespace(type="message_stop"),
    ]

    output = claude_api.process_streaming_response(completion, tool_name)

    assert output["loop"] == payload_json
    assert output["thinking_content"] == "reason"
    assert output["text"] == "music"


def test_claude_streaming_rejects_unexpected_tool_name():
    completion = _anthropic_tool_block(4, "other_tool", json.dumps(_loop_payload()))

    with pytest.raises(
        ValueError,
        match=r"unexpected tool other_tool \(block index 4\); expected build_MIDI_loop",
    ):
        claude_api.process_streaming_response(completion, "build_MIDI_loop")


@pytest.mark.parametrize("state", ["not_started", "wrong_index", "stopped"])
def test_claude_streaming_rejects_json_outside_active_tool_block(state):
    block = _anthropic_tool_block(2, "build_MIDI_loop", "{}")
    completion = [] if state == "not_started" else [block[0]]
    if state == "stopped":
        completion.append(block[-1])
    completion.append(
        SimpleNamespace(
            type="content_block_delta",
            index=3 if state == "wrong_index" else 2,
            delta=SimpleNamespace(partial_json="{}"),
        )
    )

    with pytest.raises(ValueError, match="does not belong to an active tool call"):
        claude_api.process_streaming_response(completion, "build_MIDI_loop")


def test_openai_extract_reasoning_ignores_missing_summary():
    response = SimpleNamespace(output=[SimpleNamespace(type="reasoning")])

    assert openai_api.extract_reasoning(response) == ""


def test_openai_calc_price_uses_reported_cached_tokens():
    response = SimpleNamespace(
        usage=SimpleNamespace(
            input_tokens=1000,
            output_tokens=200,
            input_tokens_details=SimpleNamespace(cached_tokens=400),
        )
    )

    cost = openai_api.calc_price("gpt-4o-mini", response)

    expected = (
        (600 * 0.15 / 1_000_000) + (400 * 0.075 / 1_000_000) + (200 * 0.60 / 1_000_000)
    )
    assert cost == pytest.approx(expected)


def test_openai_calc_price_clamps_malformed_cached_tokens():
    response = SimpleNamespace(
        usage=SimpleNamespace(
            input_tokens=100,
            output_tokens=0,
            input_tokens_details=SimpleNamespace(cached_tokens=150),
        )
    )

    cost = openai_api.calc_price("gpt-4o-mini", response)

    assert cost == pytest.approx(100 * 0.075 / 1_000_000)


def test_openai_calc_price_partitions_cache_writes_before_reads():
    response = SimpleNamespace(
        usage=SimpleNamespace(
            input_tokens=1000,
            output_tokens=100,
            input_tokens_details=SimpleNamespace(
                cached_tokens=300,
                cache_write_tokens=200,
            ),
        )
    )

    cost = openai_api.calc_price("gpt-5.6-sol", response)

    expected = (
        (500 * 4.00 / 1_000_000)
        + (300 * 0.40 / 1_000_000)
        + (200 * 5.00 / 1_000_000)
        + (100 * 20.00 / 1_000_000)
    )
    assert cost == pytest.approx(expected)


def test_openai_calc_price_uses_input_rate_when_cache_write_rate_is_missing():
    response = SimpleNamespace(
        usage=SimpleNamespace(
            input_tokens=100,
            output_tokens=0,
            input_tokens_details=SimpleNamespace(
                cached_tokens=20,
                cache_write_tokens=30,
            ),
        )
    )

    cost = openai_api.calc_price("gpt-4o-mini", response)

    expected = (80 * 0.15 / 1_000_000) + (20 * 0.075 / 1_000_000)
    assert cost == pytest.approx(expected)


@pytest.mark.parametrize(
    "usage",
    [
        SimpleNamespace(input_tokens=100, output_tokens=20),
        SimpleNamespace(
            input_tokens=100,
            output_tokens=20,
            input_tokens_details=None,
        ),
        SimpleNamespace(
            input_tokens=100,
            output_tokens=20,
            input_tokens_details=SimpleNamespace(cached_tokens=25),
        ),
        SimpleNamespace(
            input_tokens=100,
            output_tokens=20,
            input_tokens_details=SimpleNamespace(
                cached_tokens=None,
                cache_write_tokens=None,
            ),
        ),
    ],
)
def test_openai_calc_price_tolerates_missing_or_null_input_details(usage):
    response = SimpleNamespace(usage=usage)

    cost = openai_api.calc_price("gpt-4o-mini", response)

    cached_tokens = (
        getattr(getattr(usage, "input_tokens_details", None), "cached_tokens", 0) or 0
    )
    expected = (
        ((100 - cached_tokens) * 0.15 / 1_000_000)
        + (cached_tokens * 0.075 / 1_000_000)
        + (20 * 0.60 / 1_000_000)
    )
    assert cost == pytest.approx(expected)


@pytest.mark.parametrize(
    ("input_tokens", "cached_tokens", "cache_write_tokens", "expected"),
    [
        (100, 80, 150, 100 * 5.00 / 1_000_000),
        (100, 150, -20, 100 * 0.40 / 1_000_000),
        (-100, 50, 50, 0),
        (100, -50, -50, 100 * 4.00 / 1_000_000),
    ],
)
def test_openai_calc_price_clamps_negative_and_overreported_cache_buckets(
    input_tokens, cached_tokens, cache_write_tokens, expected
):
    response = SimpleNamespace(
        usage=SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=0,
            input_tokens_details=SimpleNamespace(
                cached_tokens=cached_tokens,
                cache_write_tokens=cache_write_tokens,
            ),
        )
    )

    cost = openai_api.calc_price("gpt-5.6-sol", response)

    assert cost == pytest.approx(expected)


@pytest.mark.parametrize("output_tokens", [None, -50])
def test_openai_calc_price_clamps_null_or_negative_output_tokens(output_tokens):
    response = SimpleNamespace(
        usage=SimpleNamespace(
            input_tokens=100,
            output_tokens=output_tokens,
            input_tokens_details=None,
        )
    )

    cost = openai_api.calc_price("gpt-4o-mini", response)

    assert cost == pytest.approx(100 * 0.15 / 1_000_000)


@pytest.mark.parametrize(
    ("model", "input_rate", "output_rate", "write_rate", "read_rate"),
    [
        ("claude-sonnet-4-5", 3.00, 15.00, 3.75, 0.30),
        ("claude-sonnet-5-5", 2.00, 10.00, 2.50, 0.10),
    ],
)
def test_claude_calc_price_uses_reported_cache_creation_and_reads(
    model, input_rate, output_rate, write_rate, read_rate
):
    output = {
        "input_tokens": 1000,
        "output_tokens": 200,
        "cache_creation": 300,
        "cache_read": 400,
    }

    cost = claude_api.calc_price(model, output)

    expected = (
        (1000 * input_rate / 1_000_000)
        + (200 * output_rate / 1_000_000)
        + (300 * write_rate / 1_000_000)
        + (400 * read_rate / 1_000_000)
    )
    assert cost == pytest.approx(expected)


def test_claude_streaming_usage_preserves_cache_creation_ttl_breakdown():
    message_usage = SimpleNamespace(
        input_tokens=100,
        output_tokens=0,
        cache_creation_input_tokens=70,
        cache_read_input_tokens=30,
        cache_creation=SimpleNamespace(
            ephemeral_5m_input_tokens=40,
            ephemeral_1h_input_tokens=30,
        ),
    )
    delta_usage = SimpleNamespace(
        output_tokens=20,
        cache_creation_input_tokens=None,
        cache_read_input_tokens=None,
        cache_creation=None,
    )
    completion = [
        SimpleNamespace(
            type="message_start",
            message=SimpleNamespace(usage=message_usage),
        ),
        SimpleNamespace(type="message_delta", usage=delta_usage),
        SimpleNamespace(type="message_stop"),
    ]

    output = claude_api.process_streaming_response(completion)

    assert output["input_tokens"] == 100
    assert output["output_tokens"] == 20
    assert output["cache_creation"] == 70
    assert output["cache_creation_5m"] == 40
    assert output["cache_creation_1h"] == 30
    assert output["cache_read"] == 30


def test_claude_streaming_usage_replaces_cumulative_message_delta_totals():
    anthropic_types = pytest.importorskip("anthropic.types")
    start_usage = anthropic_types.Usage(
        input_tokens=25,
        output_tokens=1,
        cache_creation_input_tokens=1000,
        cache_read_input_tokens=2000,
        cache_creation=anthropic_types.CacheCreation(
            ephemeral_5m_input_tokens=600,
            ephemeral_1h_input_tokens=400,
        ),
    )
    delta_usage = anthropic_types.MessageDeltaUsage(
        input_tokens=25,
        output_tokens=150,
        cache_creation_input_tokens=1000,
        cache_read_input_tokens=2000,
    )
    completion = [
        SimpleNamespace(
            type="message_start",
            message=SimpleNamespace(usage=start_usage),
        ),
        SimpleNamespace(type="message_delta", usage=delta_usage),
        SimpleNamespace(type="message_stop"),
    ]

    output = claude_api.process_streaming_response(completion)

    assert output["input_tokens"] == 25
    assert output["output_tokens"] == 150
    assert output["cache_creation"] == 1000
    assert output["cache_creation_5m"] == 600
    assert output["cache_creation_1h"] == 400
    assert output["cache_read"] == 2000


def test_claude_streaming_usage_keeps_baseline_when_delta_omits_fields():
    start_usage = SimpleNamespace(
        input_tokens=100,
        output_tokens=1,
        cache_creation_input_tokens=70,
        cache_read_input_tokens=30,
        cache_creation=None,
    )
    delta_usage = SimpleNamespace(output_tokens=20)
    completion = [
        SimpleNamespace(
            type="message_start",
            message=SimpleNamespace(usage=start_usage),
        ),
        SimpleNamespace(type="message_delta", usage=delta_usage),
        SimpleNamespace(type="message_stop"),
    ]

    output = claude_api.process_streaming_response(completion)

    assert output["input_tokens"] == 100
    assert output["output_tokens"] == 20
    assert output["cache_creation"] == 70
    assert output["cache_read"] == 30


def test_claude_calc_price_uses_ttl_specific_cache_creation_rates():
    output = {
        "input_tokens": 1000,
        "output_tokens": 200,
        "cache_creation": 500,
        "cache_creation_5m": 300,
        "cache_creation_1h": 200,
        "cache_read": 400,
    }

    cost = claude_api.calc_price("claude-sonnet-4-5", output)

    expected = (
        (1000 * 3.00 / 1_000_000)
        + (200 * 15.00 / 1_000_000)
        + (300 * 3.75 / 1_000_000)
        + (200 * 6.00 / 1_000_000)
        + (400 * 0.30 / 1_000_000)
    )
    assert cost == pytest.approx(expected)


@pytest.mark.parametrize(
    ("ttl_usage", "expected_5min", "expected_1hour"),
    [
        ({}, 500, 0),
        ({"cache_creation_5m": 100, "cache_creation_1h": 200}, 300, 200),
        ({"cache_creation_5m": 400, "cache_creation_1h": 300}, 200, 300),
        ({"cache_creation_5m": -100, "cache_creation_1h": -200}, 500, 0),
        ({"cache_creation_5m": None, "cache_creation_1h": None}, 500, 0),
    ],
)
def test_claude_calc_price_reconciles_ttl_breakdown_to_aggregate(
    ttl_usage, expected_5min, expected_1hour
):
    output = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_creation": 500,
        "cache_read": 0,
        **ttl_usage,
    }

    cost = claude_api.calc_price("claude-sonnet-4-5", output)

    expected = (expected_5min * 3.75 / 1_000_000) + (expected_1hour * 6.00 / 1_000_000)
    assert cost == pytest.approx(expected)


def test_claude_calc_price_clamps_negative_usage():
    output = {
        "input_tokens": -100,
        "output_tokens": -100,
        "cache_creation": -100,
        "cache_creation_5m": 50,
        "cache_creation_1h": 50,
        "cache_read": -100,
    }

    assert claude_api.calc_price("claude-sonnet-4-5", output) == 0


def test_claude_calc_price_returns_none_for_unknown_model():
    output = {
        "input_tokens": 1,
        "output_tokens": 1,
        "cache_creation": 0,
        "cache_read": 0,
    }

    assert claude_api.calc_price("not-a-model", output) is None


def test_claude_calc_price_tolerates_missing_cache_pricing(monkeypatch):
    monkeypatch.setattr(
        claude_api.utils,
        "get_model_info",
        lambda: {
            "models": {
                "Anthropic": {"test-model": {"cost": {"input": 3.00, "output": 15.00}}}
            }
        },
    )
    output = {
        "input_tokens": 1000,
        "output_tokens": 200,
        "cache_creation": 300,
        "cache_read": 400,
    }

    cost = claude_api.calc_price("test-model", output)

    expected = (1000 * 3.00 / 1_000_000) + (200 * 15.00 / 1_000_000)
    assert cost == pytest.approx(expected)


def test_gemini_process_output_rejects_empty_candidates():
    response = SimpleNamespace(candidates=[])

    with pytest.raises(
        ValueError, match="Google response did not include any candidates"
    ):
        gemini_api.process_output(response)


def test_gemini_process_output_rejects_missing_parts():
    response = SimpleNamespace(
        candidates=[SimpleNamespace(content=SimpleNamespace(parts=[]))]
    )

    with pytest.raises(
        ValueError, match="Google response did not include generated content parts"
    ):
        gemini_api.process_output(response)


def test_claude_loop_gen_omits_cache_control_for_short_system_prompt(monkeypatch):
    captured = {}
    payload = json.dumps(_loop_payload())

    def fake_create(**kwargs):
        captured.update(kwargs)
        return _anthropic_completion(payload)

    fake_client = SimpleNamespace(
        close=Mock(), messages=SimpleNamespace(create=fake_create)
    )
    monkeypatch.setattr(
        claude_api, "initialize_anthropic_client", lambda api_key: fake_client
    )
    monkeypatch.setattr(
        claude_api.utils, "get_loop_prompt", lambda: "short system prompt"
    )
    monkeypatch.setattr(claude_api.utils, "save_messages_to_json", _fail_save_messages)

    midi_loop, messages, cost = claude_api.loop_gen(
        "write a loop",
        "claude-sonnet-4-5",
    )

    assert isinstance(midi_loop, objects.Loop)
    assert "cache_control" not in captured["system"][0]
    assert messages[0] == {"role": "system", "content": "short system prompt"}
    assert cost > 0


def test_claude_loop_gen_adds_cache_control_for_large_system_prompt(monkeypatch):
    captured = {}
    payload = json.dumps(_loop_payload())

    def fake_create(**kwargs):
        captured.update(kwargs)
        return _anthropic_completion(payload)

    fake_client = SimpleNamespace(
        close=Mock(), messages=SimpleNamespace(create=fake_create)
    )
    long_prompt = "x" * claude_api.ANTHROPIC_CACHE_CONTROL_MIN_CHARS
    monkeypatch.setattr(
        claude_api, "initialize_anthropic_client", lambda api_key: fake_client
    )
    monkeypatch.setattr(claude_api.utils, "get_loop_prompt", lambda: long_prompt)
    monkeypatch.setattr(claude_api.utils, "save_messages_to_json", _fail_save_messages)

    claude_api.loop_gen("write a loop", "claude-sonnet-4-5")

    assert captured["system"][0]["cache_control"] == {"type": "ephemeral"}


def test_claude_opus_5_uses_adaptive_thinking_and_effort(monkeypatch):
    captured = {}
    payload = json.dumps(_loop_payload())

    def fake_create(**kwargs):
        captured.update(kwargs)
        return _anthropic_completion(payload)

    fake_client = SimpleNamespace(
        close=Mock(), messages=SimpleNamespace(create=fake_create)
    )
    monkeypatch.setattr(
        claude_api, "initialize_anthropic_client", lambda api_key: fake_client
    )
    monkeypatch.setattr(claude_api.utils, "get_loop_prompt", lambda: "system prompt")
    monkeypatch.setattr(claude_api.utils, "save_messages_to_json", _fail_save_messages)

    claude_api.loop_gen(
        "write a loop", "claude-opus-5", use_thinking=True, effort="max"
    )

    assert captured["thinking"] == {"type": "adaptive"}
    assert captured["output_config"] == {"effort": "max"}
    assert "temperature" not in captured
    assert captured["tool_choice"] == {
        "type": "auto",
        "disable_parallel_tool_use": True,
    }


@pytest.mark.parametrize(
    "model", ["claude-opus-5-5", "claude-fable-5", "claude-fable-5-1"]
)
def test_claude_uses_metadata_driven_always_on_thinking(monkeypatch, model):
    captured = {}
    payload = json.dumps(_loop_payload())

    def fake_create(**kwargs):
        captured.update(kwargs)
        return _anthropic_completion(payload)

    fake_client = SimpleNamespace(
        close=Mock(), messages=SimpleNamespace(create=fake_create)
    )
    monkeypatch.setattr(
        claude_api, "initialize_anthropic_client", lambda api_key: fake_client
    )
    monkeypatch.setattr(claude_api.utils, "get_loop_prompt", lambda: "system prompt")
    monkeypatch.setattr(claude_api.utils, "save_messages_to_json", _fail_save_messages)

    claude_api.loop_gen("write a loop", model, use_thinking=True, effort="max")

    assert "thinking" not in captured
    assert "temperature" not in captured
    assert captured["output_config"] == {"effort": "max"}
    assert captured["tool_choice"] == {
        "type": "auto",
        "disable_parallel_tool_use": True,
    }


@pytest.mark.parametrize(
    "model", ["claude-opus-5-5", "claude-fable-5", "claude-fable-5-1"]
)
def test_claude_always_on_thinking_off_uses_lowest_effort(monkeypatch, model):
    captured = {}
    payload = json.dumps(_loop_payload())

    def fake_create(**kwargs):
        captured.update(kwargs)
        return _anthropic_completion(payload)

    fake_client = SimpleNamespace(
        close=Mock(), messages=SimpleNamespace(create=fake_create)
    )
    monkeypatch.setattr(
        claude_api, "initialize_anthropic_client", lambda api_key: fake_client
    )
    monkeypatch.setattr(claude_api.utils, "get_loop_prompt", lambda: "system prompt")

    claude_api.loop_gen("write a loop", model, use_thinking=False, effort="max")

    assert captured["output_config"] == {"effort": "low"}
    assert "thinking" not in captured
    assert captured["tool_choice"] == {
        "type": "auto",
        "disable_parallel_tool_use": True,
    }


def test_claude_sonnet_5_5_thinking_off_uses_between_tools(monkeypatch):
    captured = {}
    payload = json.dumps(_loop_payload())

    def fake_create(**kwargs):
        captured.update(kwargs)
        return _anthropic_completion(payload)

    fake_client = SimpleNamespace(
        close=Mock(), messages=SimpleNamespace(create=fake_create)
    )
    monkeypatch.setattr(
        claude_api, "initialize_anthropic_client", lambda api_key: fake_client
    )
    monkeypatch.setattr(claude_api.utils, "get_loop_prompt", lambda: "system prompt")

    claude_api.loop_gen(
        "write a loop", "claude-sonnet-5-5", use_thinking=False, effort="max"
    )

    assert captured["thinking"] == {"type": "between_tools"}
    assert captured["output_config"] == {"effort": "low"}
    assert captured["tool_choice"] == {
        "type": "auto",
        "disable_parallel_tool_use": True,
    }
    assert "extra_body" not in captured


@pytest.mark.parametrize("generation", ["loop", "variations"])
@pytest.mark.parametrize(
    ("model", "expected_temperature"),
    [
        ("claude-opus-5", None),
        ("claude-sonnet-5", None),
        ("claude-opus-4-8", None),
        ("claude-opus-4-7", None),
        ("claude-opus-4-6", 0.4),
        ("claude-sonnet-4-6", 0.4),
    ],
)
def test_claude_thinking_off_disables_adaptive_thinking(
    monkeypatch, model, expected_temperature, generation
):
    calls = []

    def fake_create(**kwargs):
        calls.append(kwargs)
        raise _RequestCaptured

    fake_client = SimpleNamespace(
        close=Mock(), messages=SimpleNamespace(create=fake_create)
    )
    monkeypatch.setattr(
        claude_api, "initialize_anthropic_client", lambda api_key: fake_client
    )
    function = (
        claude_api.loop_gen if generation == "loop" else claude_api.variations_gen
    )

    with pytest.raises(_RequestCaptured):
        function("prompt", model, temp=0.4, use_thinking=False, effort="max")

    request = calls[0]
    tool_name = request["tools"][0]["name"]
    assert request["thinking"] == {"type": "disabled"}
    assert "output_config" not in request
    assert request["tool_choice"] == {
        "type": "tool",
        "name": tool_name,
        "disable_parallel_tool_use": True,
    }
    assert "temperature" not in request
    assert request.get("extra_body", {}).get("temperature") == expected_temperature


_ANTHROPIC_THINKING_OPTIONAL_MODELS = [
    model
    for model, model_config in music.get_model_info()["models"]["Anthropic"].items()
    if not model_config.get("always_on_adaptive_thinking")
]


@pytest.mark.parametrize("generation", ["loop", "variations"])
@pytest.mark.parametrize("model", _ANTHROPIC_THINKING_OPTIONAL_MODELS)
def test_claude_thinking_off_always_forces_the_output_tool(
    monkeypatch, model, generation
):
    calls = []

    def fake_create(**kwargs):
        calls.append(kwargs)
        raise _RequestCaptured

    fake_client = SimpleNamespace(
        close=Mock(), messages=SimpleNamespace(create=fake_create)
    )
    monkeypatch.setattr(
        claude_api, "initialize_anthropic_client", lambda api_key: fake_client
    )
    function = (
        claude_api.loop_gen if generation == "loop" else claude_api.variations_gen
    )

    with pytest.raises(_RequestCaptured):
        function("prompt", model, use_thinking=False)

    request = calls[0]
    assert request["tool_choice"] == {
        "type": "tool",
        "name": request["tools"][0]["name"],
        "disable_parallel_tool_use": True,
    }
    assert request.get("thinking", {"type": "disabled"}) == {"type": "disabled"}


@pytest.mark.parametrize("model", ["gpt-5.6-sol", "gpt-6-sol", "gpt-6-luna"])
@pytest.mark.parametrize("use_thinking", [False, True])
def test_openai_thinking_selects_supported_effort(monkeypatch, model, use_thinking):
    captured = {}
    response = SimpleNamespace(
        output=[],
        output_parsed=objects.Loop.model_validate(_loop_payload()),
        usage=SimpleNamespace(
            input_tokens=0,
            output_tokens=0,
            input_tokens_details=SimpleNamespace(cached_tokens=0),
        ),
    )

    def fake_parse(**kwargs):
        captured.update(kwargs)
        return response

    fake_client = SimpleNamespace(responses=SimpleNamespace(parse=fake_parse))
    monkeypatch.setattr(
        openai_api, "initialize_openai_client", lambda api_key: fake_client
    )
    monkeypatch.setattr(openai_api.utils, "get_loop_prompt", lambda: "system prompt")

    openai_api.loop_gen(
        "write a loop",
        model,
        use_thinking=use_thinking,
        effort="max",
    )

    assert captured["model"] == model
    assert captured["reasoning"] == {
        "effort": "max" if use_thinking else "none",
        "summary": "auto",
    }
    assert "temperature" not in captured


def test_gemini_disabled_thinking_uses_lowest_effort(monkeypatch):
    captured = {}
    response = SimpleNamespace(
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(
                    parts=[
                        SimpleNamespace(text=json.dumps(_loop_payload()), thought=False)
                    ]
                )
            )
        ],
        usage_metadata=SimpleNamespace(
            cached_content_token_count=0,
            prompt_token_count=0,
            candidates_token_count=0,
            thoughts_token_count=0,
        ),
    )

    def fake_generate_content(**kwargs):
        captured.update(kwargs)
        return response

    fake_client = SimpleNamespace(
        models=SimpleNamespace(generate_content=fake_generate_content)
    )
    monkeypatch.setattr(
        gemini_api, "initialize_gemini_client", lambda api_key: fake_client
    )
    monkeypatch.setattr(gemini_api.utils, "get_loop_prompt", lambda: "system prompt")

    midi_loop, _, _ = gemini_api.loop_gen(
        "write a loop",
        "gemini-3.8-flash",
        use_thinking=False,
        effort="high",
    )

    assert captured["config"]["thinking_config"].thinking_level.value == "LOW"
    assert "response_schema" not in captured["config"]
    assert captured["config"]["response_json_schema"] == (
        objects.Loop.model_json_schema()
    )
    assert isinstance(midi_loop, objects.Loop)


@pytest.mark.parametrize(
    ("use_thinking", "expected_budget"),
    [(False, 128), (True, 32768)],
)
def test_gemini_budget_thinking_uses_metadata_bounds(
    monkeypatch, use_thinking, expected_budget
):
    captured = {}
    response = SimpleNamespace(
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(
                    parts=[
                        SimpleNamespace(text=json.dumps(_loop_payload()), thought=False)
                    ]
                )
            )
        ],
        usage_metadata=SimpleNamespace(
            cached_content_token_count=0,
            prompt_token_count=0,
            candidates_token_count=0,
            thoughts_token_count=0,
        ),
    )

    def fake_generate_content(**kwargs):
        captured.update(kwargs)
        return response

    fake_client = SimpleNamespace(
        models=SimpleNamespace(generate_content=fake_generate_content)
    )
    monkeypatch.setattr(
        gemini_api, "initialize_gemini_client", lambda api_key: fake_client
    )
    monkeypatch.setattr(gemini_api.utils, "get_loop_prompt", lambda: "system prompt")

    gemini_api.loop_gen("write a loop", "gemini-2.5-pro", use_thinking=use_thinking)

    assert captured["config"]["thinking_config"].thinking_budget == expected_budget


@pytest.mark.parametrize("use_thinking", [False, True, None])
def test_claude_budget_thinking_respects_toggle(monkeypatch, use_thinking):
    captured = {}
    payload = json.dumps(_loop_payload())

    def fake_create(**kwargs):
        captured.update(kwargs)
        return _anthropic_completion(payload)

    fake_client = SimpleNamespace(
        close=Mock(), messages=SimpleNamespace(create=fake_create)
    )
    monkeypatch.setattr(
        claude_api, "initialize_anthropic_client", lambda api_key: fake_client
    )
    monkeypatch.setattr(claude_api.utils, "get_loop_prompt", lambda: "system prompt")

    claude_api.loop_gen("write a loop", "claude-sonnet-4-5", use_thinking=use_thinking)

    if use_thinking:
        assert captured["thinking"] == {
            "type": "enabled",
            "budget_tokens": 59000,
        }
    else:
        assert "thinking" not in captured


@pytest.mark.parametrize("generate_name", ["loop_gen", "variations_gen"])
@pytest.mark.parametrize(
    ("adapter", "initializer", "model"),
    [
        (openai_api, "initialize_openai_client", "gpt-6-astra"),
        (gemini_api, "initialize_gemini_client", "gemini-3.8-flash"),
        (claude_api, "initialize_anthropic_client", "claude-sonnet-5-5"),
    ],
)
@pytest.mark.parametrize("use_thinking", [False, None])
def test_direct_disabled_thinking_ignores_invalid_effort(
    monkeypatch, generate_name, adapter, initializer, model, use_thinking
):
    captured = {}

    def capture_request(**kwargs):
        captured.update(kwargs)
        raise _RequestCaptured

    client = SimpleNamespace(
        close=Mock(),
        responses=SimpleNamespace(parse=capture_request),
        models=SimpleNamespace(generate_content=capture_request),
        messages=SimpleNamespace(create=capture_request),
    )
    monkeypatch.setattr(adapter, initializer, lambda **kwargs: client)

    with pytest.raises(_RequestCaptured):
        getattr(adapter, generate_name)(
            "prompt", model, use_thinking=use_thinking, effort="unsupported"
        )

    if adapter is openai_api:
        assert captured["reasoning"]["effort"] == "low"
    elif adapter is gemini_api:
        assert captured["config"]["thinking_config"].thinking_level.value == "LOW"
    else:
        assert captured["output_config"] == {"effort": "low"}


def test_ollama_loop_gen_accepts_missing_thinking(monkeypatch):
    payload = json.dumps(_loop_payload())
    completion = SimpleNamespace(message=SimpleNamespace(content=payload))
    fake_client = SimpleNamespace(chat=lambda **kwargs: completion)

    monkeypatch.setattr(
        ollama_api, "initialize_ollama_client", lambda host_address=None: fake_client
    )
    monkeypatch.setattr(ollama_api.utils, "get_loop_prompt", lambda: "system prompt")
    monkeypatch.setattr(ollama_api.utils, "save_messages_to_json", _fail_save_messages)

    midi_loop, messages, cost = ollama_api.loop_gen(
        "write a loop", "llama3", model_capabilities={"extended_thinking": False}
    )

    assert isinstance(midi_loop, objects.Loop)
    assert objects.Loop.model_validate_json(messages[-1]["content"]) == midi_loop
    assert cost == 0


def test_openai_loop_gen_does_not_write_message_log(monkeypatch):
    response = SimpleNamespace(
        output=[],
        output_parsed=objects.Loop.model_validate(_loop_payload()),
        usage=SimpleNamespace(
            input_tokens=0,
            output_tokens=0,
            input_tokens_details=SimpleNamespace(cached_tokens=0),
        ),
    )
    fake_client = SimpleNamespace(
        responses=SimpleNamespace(parse=lambda **kwargs: response)
    )

    monkeypatch.setattr(
        openai_api, "initialize_openai_client", lambda api_key: fake_client
    )
    monkeypatch.setattr(openai_api.utils, "get_loop_prompt", lambda: "system prompt")
    monkeypatch.setattr(openai_api.utils, "save_messages_to_json", _fail_save_messages)

    midi_loop, messages, cost = openai_api.loop_gen("write a loop", "gpt-4o-mini")

    assert isinstance(midi_loop, objects.Loop)
    assert objects.Loop.model_validate_json(messages[-1]["content"]) == midi_loop
    assert cost == 0


def test_gemini_loop_gen_omits_unsupported_temperature(monkeypatch):
    captured = {}
    response = SimpleNamespace(
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(
                    parts=[
                        SimpleNamespace(text=json.dumps(_loop_payload()), thought=False)
                    ]
                )
            )
        ],
        usage_metadata=SimpleNamespace(
            cached_content_token_count=0,
            prompt_token_count=0,
            candidates_token_count=0,
            thoughts_token_count=0,
        ),
    )

    def fake_generate_content(**kwargs):
        captured.update(kwargs)
        return response

    fake_client = SimpleNamespace(
        models=SimpleNamespace(generate_content=fake_generate_content)
    )
    monkeypatch.setattr(
        gemini_api, "initialize_gemini_client", lambda api_key: fake_client
    )
    monkeypatch.setattr(gemini_api.utils, "get_loop_prompt", lambda: "system prompt")

    gemini_api.loop_gen(
        "write a loop",
        "gemini-3.8-flash",
        temp=0.7,
        use_thinking=True,
        effort="medium",
    )

    assert "temperature" not in captured["config"]
    assert captured["config"]["thinking_config"].thinking_level.value == "MEDIUM"


@pytest.mark.parametrize(
    ("provider", "initializer", "model"),
    [
        (openai_api, "initialize_openai_client", "gpt-4o-mini"),
        (gemini_api, "initialize_gemini_client", "gemini-3.8-flash"),
        (claude_api, "initialize_anthropic_client", "claude-sonnet-4-5"),
    ],
)
def test_provider_loop_gen_propagates_client_initialization_type_errors(
    monkeypatch, provider, initializer, model
):
    def raise_type_error(*, api_key):
        assert api_key == "injected-key"
        raise TypeError("client initialization failed")

    monkeypatch.setattr(provider, initializer, raise_type_error)

    with pytest.raises(TypeError, match="client initialization failed"):
        provider.loop_gen("write a loop", model, api_key="injected-key")


def test_ollama_loop_gen_rejects_missing_content(monkeypatch):
    completion = SimpleNamespace(message=SimpleNamespace())
    fake_client = SimpleNamespace(chat=lambda **kwargs: completion)

    monkeypatch.setattr(
        ollama_api, "initialize_ollama_client", lambda host_address=None: fake_client
    )
    monkeypatch.setattr(ollama_api.utils, "get_loop_prompt", lambda: "system prompt")

    with pytest.raises(
        ValueError, match="Ollama response did not include generated content"
    ):
        ollama_api.loop_gen(
            "write a loop", "llama3", model_capabilities={"extended_thinking": False}
        )


def test_claude_unsupported_thinking_warns_once(monkeypatch, caplog):
    model_info = {
        "models": {
            "OpenAI": {},
            "Google": {},
            "Anthropic": {
                "claude-test": {
                    "extended_thinking": False,
                    "max_tokens": 1024,
                    "rate_limits": {"RPM": 1, "TPM": None, "RPD": None},
                }
            },
        }
    }
    monkeypatch.setattr(routing, "get_model_info", lambda: model_info)
    monkeypatch.setattr(claude_api.utils, "get_model_info", lambda: model_info)
    calls = []

    def fake_create(**kwargs):
        calls.append(kwargs)
        raise _RequestCaptured

    fake_client = SimpleNamespace(
        close=Mock(), messages=SimpleNamespace(create=fake_create)
    )
    monkeypatch.setattr(
        claude_api, "initialize_anthropic_client", lambda **kwargs: fake_client
    )

    with caplog.at_level(logging.WARNING), pytest.raises(_RequestCaptured):
        routing.generate_midi("claude-test", "prompt", use_thinking=True)

    warnings = [r for r in caplog.records if "extended thinking" in r.getMessage()]
    assert len(warnings) == 1
    assert warnings[0].name == "conductor_core.routing"
    assert "thinking" not in calls[0]


def test_openai_loop_gen_returns_null_cost_without_usage(monkeypatch):
    loop = objects.Loop.model_validate(_loop_payload())
    response = SimpleNamespace(output=[], output_parsed=loop, usage=None)
    fake_client = SimpleNamespace(
        responses=SimpleNamespace(parse=lambda **kwargs: response)
    )
    monkeypatch.setattr(
        openai_api, "initialize_openai_client", lambda api_key: fake_client
    )

    midi_loop, messages, cost = openai_api.loop_gen("write a loop", "gpt-4o-mini")

    assert midi_loop == loop
    assert objects.Loop.model_validate_json(messages[-1]["content"]) == loop
    assert cost is None


def test_gemini_loop_gen_returns_null_cost_without_usage(monkeypatch):
    payload = json.dumps(_loop_payload())
    response = SimpleNamespace(
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(
                    parts=[SimpleNamespace(text=payload, thought=False)]
                )
            )
        ],
        usage_metadata=None,
    )
    fake_client = SimpleNamespace(
        models=SimpleNamespace(generate_content=lambda **kwargs: response)
    )
    monkeypatch.setattr(
        gemini_api, "initialize_gemini_client", lambda api_key: fake_client
    )

    midi_loop, messages, cost = gemini_api.loop_gen(
        "write a loop", "gemini-3.8-flash", use_thinking=False, effort="high"
    )

    assert isinstance(midi_loop, objects.Loop)
    assert messages[-1] == {"role": "assistant", "content": payload}
    assert cost is None


def test_claude_loop_gen_returns_null_cost_without_usage(monkeypatch):
    payload = json.dumps(_loop_payload())
    completion = _anthropic_completion(payload)
    completion.events = [
        event for event in completion.events if event.type != "message_start"
    ]
    fake_client = SimpleNamespace(
        close=Mock(), messages=SimpleNamespace(create=lambda **kwargs: completion)
    )
    monkeypatch.setattr(
        claude_api, "initialize_anthropic_client", lambda api_key: fake_client
    )

    midi_loop, messages, cost = claude_api.loop_gen("write a loop", "claude-sonnet-4-5")

    assert isinstance(midi_loop, objects.Loop)
    assert messages[-1] == {"role": "assistant", "content": payload}
    assert cost is None
