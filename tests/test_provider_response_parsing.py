import json
import logging
from types import SimpleNamespace

import pytest

from conductor_core import models as objects
from conductor_core.providers import anthropic as claude_api
from conductor_core.providers import google as gemini_api
from conductor_core.providers import ollama as ollama_api
from conductor_core.providers import openai as openai_api


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


def _anthropic_completion(payload):
    usage = SimpleNamespace(
        input_tokens=100,
        output_tokens=50,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
    )
    return [
        SimpleNamespace(type="message_start", message=SimpleNamespace(usage=usage)),
        SimpleNamespace(
            type="content_block_delta", delta=SimpleNamespace(partial_json=payload)
        ),
        SimpleNamespace(type="message_stop"),
    ]


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


def test_claude_calc_price_uses_reported_cache_creation_and_reads():
    output = {
        "input_tokens": 1000,
        "output_tokens": 200,
        "cache_creation": 300,
        "cache_read": 400,
    }

    cost = claude_api.calc_price("claude-sonnet-4-5", output)

    expected = (
        (1000 * 3.00 / 1_000_000)
        + (200 * 15.00 / 1_000_000)
        + (300 * 3.75 / 1_000_000)
        + (400 * 0.30 / 1_000_000)
    )
    assert cost == pytest.approx(expected)


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

    fake_client = SimpleNamespace(messages=SimpleNamespace(create=fake_create))
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

    fake_client = SimpleNamespace(messages=SimpleNamespace(create=fake_create))
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

    fake_client = SimpleNamespace(messages=SimpleNamespace(create=fake_create))
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
    assert captured["temperature"] == 1.0
    assert captured["tool_choice"] == {"type": "auto"}


@pytest.mark.parametrize("model", ["claude-fable-5", "claude-fable-5-1"])
def test_claude_fable_uses_metadata_driven_always_on_thinking(monkeypatch, model):
    captured = {}
    payload = json.dumps(_loop_payload())

    def fake_create(**kwargs):
        captured.update(kwargs)
        return _anthropic_completion(payload)

    fake_client = SimpleNamespace(messages=SimpleNamespace(create=fake_create))
    monkeypatch.setattr(
        claude_api, "initialize_anthropic_client", lambda api_key: fake_client
    )
    monkeypatch.setattr(claude_api.utils, "get_loop_prompt", lambda: "system prompt")
    monkeypatch.setattr(claude_api.utils, "save_messages_to_json", _fail_save_messages)

    claude_api.loop_gen("write a loop", model, use_thinking=True, effort="max")

    assert "thinking" not in captured
    assert "temperature" not in captured
    assert captured["output_config"] == {"effort": "max"}
    assert captured["tool_choice"] == {"type": "auto"}


@pytest.mark.parametrize(
    ("model", "expects_thinking"),
    [
        ("claude-opus-5", True),
        ("claude-fable-5", False),
        ("claude-fable-5-1", False),
    ],
)
def test_claude_disabled_thinking_uses_lowest_effort(
    monkeypatch, model, expects_thinking
):
    captured = {}
    payload = json.dumps(_loop_payload())

    def fake_create(**kwargs):
        captured.update(kwargs)
        return _anthropic_completion(payload)

    fake_client = SimpleNamespace(messages=SimpleNamespace(create=fake_create))
    monkeypatch.setattr(
        claude_api, "initialize_anthropic_client", lambda api_key: fake_client
    )
    monkeypatch.setattr(claude_api.utils, "get_loop_prompt", lambda: "system prompt")

    claude_api.loop_gen("write a loop", model, use_thinking=False, effort="max")

    assert captured["output_config"] == {"effort": "low"}
    assert ("thinking" in captured) is expects_thinking


def test_openai_disabled_thinking_uses_lowest_effort(monkeypatch):
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
        "gpt-5.6-sol",
        use_thinking=False,
        effort="max",
    )

    assert captured["reasoning"] == {"effort": "none", "summary": "auto"}


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
        "gemini-3.7-flash",
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


@pytest.mark.parametrize("use_thinking", [False, True])
def test_claude_budget_thinking_respects_toggle(monkeypatch, use_thinking):
    captured = {}
    payload = json.dumps(_loop_payload())

    def fake_create(**kwargs):
        captured.update(kwargs)
        return _anthropic_completion(payload)

    fake_client = SimpleNamespace(messages=SimpleNamespace(create=fake_create))
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


def test_ollama_loop_gen_accepts_missing_thinking(monkeypatch):
    payload = json.dumps(_loop_payload())
    completion = SimpleNamespace(message=SimpleNamespace(content=payload))
    fake_client = SimpleNamespace(chat=lambda **kwargs: completion)

    monkeypatch.setattr(
        ollama_api, "initialize_ollama_client", lambda host_address=None: fake_client
    )
    monkeypatch.setattr(ollama_api.utils, "get_loop_prompt", lambda: "system prompt")
    monkeypatch.setattr(ollama_api.utils, "save_messages_to_json", _fail_save_messages)

    midi_loop, messages, cost = ollama_api.loop_gen("write a loop", "llama3")

    assert isinstance(midi_loop, objects.Loop)
    assert messages[-1]["content"] == str(midi_loop)
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
    assert messages[-1]["content"] == str(midi_loop)
    assert cost == 0


def test_gemini_loop_gen_logs_unsupported_effort(monkeypatch, caplog, capsys):
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
        ),
    )
    fake_client = SimpleNamespace(
        models=SimpleNamespace(generate_content=lambda **kwargs: response)
    )

    monkeypatch.setattr(
        gemini_api, "initialize_gemini_client", lambda api_key: fake_client
    )

    monkeypatch.setattr(gemini_api.utils, "get_loop_prompt", lambda: "system prompt")
    monkeypatch.setattr(gemini_api.utils, "save_messages_to_json", _fail_save_messages)

    with caplog.at_level(logging.WARNING, logger=gemini_api.__name__):
        midi_loop, messages, cost = gemini_api.loop_gen(
            "write a loop",
            "gemini-3.1-flash-lite",
            use_thinking=True,
            effort="bogus",
        )

    assert isinstance(midi_loop, objects.Loop)
    assert messages[-1]["content"] == json.dumps(_loop_payload())
    assert cost == 0
    assert capsys.readouterr().out == ""
    assert (
        "Effort 'bogus' is not supported by model gemini-3.1-flash-lite" in caplog.text
    )


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
        "gemini-3.7-flash",
        temp=0.7,
        use_thinking=True,
        effort="medium",
    )

    assert "temperature" not in captured["config"]
    assert captured["config"]["thinking_config"].thinking_level.value == "MEDIUM"


@pytest.mark.parametrize(
    ("provider", "initializer"),
    [
        (openai_api, "initialize_openai_client"),
        (gemini_api, "initialize_gemini_client"),
        (claude_api, "initialize_anthropic_client"),
    ],
)
def test_provider_loop_gen_propagates_client_initialization_type_errors(
    monkeypatch, provider, initializer
):
    def raise_type_error(*, api_key):
        assert api_key == "injected-key"
        raise TypeError("client initialization failed")

    monkeypatch.setattr(provider, initializer, raise_type_error)

    with pytest.raises(TypeError, match="client initialization failed"):
        provider.loop_gen("write a loop", "test-model", api_key="injected-key")


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
        ollama_api.loop_gen("write a loop", "llama3")
