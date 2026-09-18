from unittest.mock import Mock

import pytest

from conductor_core import ProviderCredentials
from conductor_core import routing as runs
from conductor_core._internal_types import ProviderLoopResult


@pytest.mark.parametrize(
    ("provider", "adapter_name"),
    [
        ("OpenAI", "openai_api"),
        ("Google", "gemini_api"),
        ("Anthropic", "claude_api"),
    ],
)
def test_generate_midi_rejects_unsupported_effort_before_provider_call(
    monkeypatch, provider, adapter_name
):
    model_choice = f"{provider.lower()}-model"
    model_info = {"models": {"OpenAI": {}, "Google": {}, "Anthropic": {}}}
    model_info["models"][provider][model_choice] = {"effort_options": ["low", "medium"]}
    monkeypatch.setattr(runs, "get_model_info", lambda: model_info)

    loop_gen = Mock()
    monkeypatch.setattr(getattr(runs, adapter_name), "loop_gen", loop_gen)

    with pytest.raises(
        ValueError,
        match=(
            rf"Invalid effort 'high' for {model_choice}\. "
            r"Expected one of: low, medium"
        ),
    ):
        runs.generate_midi(
            model_choice,
            "write a loop",
            use_thinking=True,
            effort="high",
        )

    loop_gen.assert_not_called()


@pytest.mark.parametrize(
    ("effort_options", "expected"),
    [
        (["none", "low", "medium"], "none"),
        (["minimal", "low", "medium"], "minimal"),
        (["low", "medium", "high"], "low"),
    ],
)
def test_resolve_reasoning_effort_uses_lowest_option_when_thinking_is_disabled(
    effort_options, expected
):
    assert (
        runs._resolve_reasoning_effort(
            "reasoning-model",
            {"extended_thinking": True, "effort_options": effort_options},
            use_thinking=False,
            effort="medium",
        )
        == expected
    )


def test_resolve_reasoning_effort_uses_lowest_option_before_validating_none():
    assert (
        runs._resolve_reasoning_effort(
            "reasoning-model",
            {
                "extended_thinking": True,
                "effort_options": ["minimal", "low", "medium"],
            },
            use_thinking=False,
            effort=None,
        )
        == "minimal"
    )


def test_resolve_reasoning_effort_preserves_supported_effort_when_enabled():
    assert (
        runs._resolve_reasoning_effort(
            "reasoning-model",
            {
                "extended_thinking": True,
                "effort_options": ["low", "medium", "high"],
            },
            use_thinking=True,
            effort="high",
        )
        == "high"
    )


def test_resolve_reasoning_effort_rejects_none_when_thinking_is_enabled():
    with pytest.raises(
        ValueError,
        match=(
            r"Invalid effort None for reasoning-model\. "
            r"Expected one of: low, medium, high"
        ),
    ):
        runs._resolve_reasoning_effort(
            "reasoning-model",
            {
                "extended_thinking": True,
                "effort_options": ["low", "medium", "high"],
            },
            use_thinking=True,
            effort=None,
        )


def test_resolve_reasoning_effort_warns_when_unsupported(caplog):
    with caplog.at_level("WARNING", logger="conductor_core.routing"):
        result = runs._resolve_reasoning_effort(
            "non-reasoning-model",
            {"extended_thinking": False},
            use_thinking=True,
            effort="high",
        )

    assert result == "high"
    assert caplog.messages == [
        (
            "Effort 'high' was requested for non-reasoning-model, but this model "
            "does not support configurable effort; the setting will be ignored."
        ),
        (
            "Thinking was requested for non-reasoning-model, but this model does "
            "not support extended thinking; the setting will be ignored."
        ),
    ]


def test_resolve_reasoning_effort_does_not_warn_for_defaults(caplog):
    with caplog.at_level("WARNING", logger="conductor_core.routing"):
        result = runs._resolve_reasoning_effort(
            "non-reasoning-model",
            {"extended_thinking": False},
            use_thinking=False,
            effort="low",
        )

    assert result == "low"
    assert caplog.messages == []


def test_generate_midi_routes_to_ollama_and_forwards_temperature(monkeypatch):
    captured = {}

    monkeypatch.setattr(
        runs,
        "get_model_info",
        lambda: {"models": {"OpenAI": {}, "Google": {}, "Anthropic": {}}},
    )
    monkeypatch.setattr(
        runs.ollama_api,
        "get_ollama_status",
        lambda host_address=None: {
            "available": True,
            "models": ["llama3"],
        },
    )

    def fake_loop_gen(prompt, model, temp=0.0, host_address=None, system_prompt=None):
        captured.update(
            {
                "prompt": prompt,
                "model": model,
                "temp": temp,
                "host_address": host_address,
                "system_prompt": system_prompt,
            }
        )
        return "loop", ["message"], 0

    monkeypatch.setattr(runs.ollama_api, "loop_gen", fake_loop_gen)

    result = runs.generate_midi(
        "llama3",
        "write a loop",
        temp=0.7,
        provider_credentials=ProviderCredentials(ollama_host="http://ollama.test"),
        system_prompt="system",
    )

    assert result == ProviderLoopResult("loop", ["message"], 0, "Ollama")
    assert captured == {
        "prompt": "write a loop",
        "model": "llama3",
        "temp": 0.7,
        "host_address": "http://ollama.test",
        "system_prompt": "system",
    }


def test_generate_midi_routes_to_openai_and_forwards_effort(monkeypatch):
    captured = {}

    monkeypatch.setattr(
        runs,
        "get_model_info",
        lambda: {
            "models": {
                "OpenAI": {"gpt-4o-mini": {}},
                "Google": {},
                "Anthropic": {},
            }
        },
    )
    ollama_status = Mock()
    monkeypatch.setattr(runs.ollama_api, "get_ollama_status", ollama_status)

    def fake_loop_gen(
        prompt,
        model,
        temp=0.0,
        use_thinking=False,
        effort=None,
        api_key=None,
        system_prompt=None,
    ):
        captured.update(
            {
                "prompt": prompt,
                "model": model,
                "temp": temp,
                "use_thinking": use_thinking,
                "effort": effort,
                "api_key": api_key,
                "system_prompt": system_prompt,
            }
        )
        return "loop", ["message"], 1.25

    monkeypatch.setattr(runs.openai_api, "loop_gen", fake_loop_gen)

    result = runs.generate_midi(
        "gpt-4o-mini",
        "write a loop",
        temp=0.2,
        use_thinking=True,
        effort="high",
        provider_credentials=ProviderCredentials(openai_api_key="openai-key"),
        system_prompt="system",
    )

    assert result == ProviderLoopResult("loop", ["message"], 1.25, "OpenAI")
    assert captured == {
        "prompt": "write a loop",
        "model": "gpt-4o-mini",
        "temp": 0.2,
        "use_thinking": True,
        "effort": "high",
        "api_key": "openai-key",
        "system_prompt": "system",
    }
    ollama_status.assert_not_called()


def test_generate_midi_routes_to_gemini_and_forwards_reasoning_options(monkeypatch):
    captured = {}

    monkeypatch.setattr(
        runs,
        "get_model_info",
        lambda: {
            "models": {
                "OpenAI": {},
                "Google": {"gemini-2.5-pro": {}},
                "Anthropic": {},
            }
        },
    )
    ollama_status = Mock()
    monkeypatch.setattr(runs.ollama_api, "get_ollama_status", ollama_status)

    def fake_loop_gen(
        prompt,
        model,
        temp=0.0,
        use_thinking=None,
        effort=None,
        api_key=None,
        system_prompt=None,
    ):
        captured.update(
            {
                "prompt": prompt,
                "model": model,
                "temp": temp,
                "use_thinking": use_thinking,
                "effort": effort,
                "api_key": api_key,
                "system_prompt": system_prompt,
            }
        )
        return "loop", ["message"], 2.5

    monkeypatch.setattr(runs.gemini_api, "loop_gen", fake_loop_gen)

    result = runs.generate_midi(
        "gemini-2.5-pro",
        "write a loop",
        temp=0.4,
        use_thinking=True,
        effort="medium",
        provider_credentials=ProviderCredentials(google_api_key="google-key"),
        system_prompt="system",
    )

    assert result == ProviderLoopResult("loop", ["message"], 2.5, "Google")
    assert captured == {
        "prompt": "write a loop",
        "model": "gemini-2.5-pro",
        "temp": 0.4,
        "use_thinking": True,
        "effort": "medium",
        "api_key": "google-key",
        "system_prompt": "system",
    }
    ollama_status.assert_not_called()


def test_generate_midi_routes_to_claude_and_forwards_reasoning_options(monkeypatch):
    captured = {}

    monkeypatch.setattr(
        runs,
        "get_model_info",
        lambda: {
            "models": {
                "OpenAI": {},
                "Google": {},
                "Anthropic": {"claude-sonnet-4-5": {}},
            }
        },
    )
    ollama_status = Mock()
    monkeypatch.setattr(runs.ollama_api, "get_ollama_status", ollama_status)

    def fake_loop_gen(
        prompt,
        model,
        temp=0.0,
        use_thinking=False,
        effort="low",
        api_key=None,
        system_prompt=None,
    ):
        captured.update(
            {
                "prompt": prompt,
                "model": model,
                "temp": temp,
                "use_thinking": use_thinking,
                "effort": effort,
                "api_key": api_key,
                "system_prompt": system_prompt,
            }
        )
        return "loop", ["message"], 3.75

    monkeypatch.setattr(runs.claude_api, "loop_gen", fake_loop_gen)

    result = runs.generate_midi(
        "claude-sonnet-4-5",
        "write a loop",
        temp=0.1,
        use_thinking=True,
        effort="high",
        provider_credentials=ProviderCredentials(anthropic_api_key="anthropic-key"),
        system_prompt="system",
    )

    assert result == ProviderLoopResult("loop", ["message"], 3.75, "Anthropic")
    assert captured == {
        "prompt": "write a loop",
        "model": "claude-sonnet-4-5",
        "temp": 0.1,
        "use_thinking": True,
        "effort": "high",
        "api_key": "anthropic-key",
        "system_prompt": "system",
    }
    ollama_status.assert_not_called()


def test_generate_midi_rejects_unknown_models_when_ollama_is_unavailable(monkeypatch):
    monkeypatch.setattr(
        runs,
        "get_model_info",
        lambda: {"models": {"OpenAI": {}, "Google": {}, "Anthropic": {}}},
    )
    monkeypatch.setattr(
        runs.ollama_api,
        "get_ollama_status",
        lambda host_address=None: {
            "available": False,
            "models": [],
        },
    )

    with pytest.raises(
        ValueError,
        match=r"Invalid Model Selected\. If you intended to use Ollama, it is currently unavailable\.",
    ):
        runs.generate_midi("unknown-model", "write a loop")


def test_generate_midi_rejects_unknown_models_when_ollama_is_available(monkeypatch):
    monkeypatch.setattr(
        runs,
        "get_model_info",
        lambda: {"models": {"OpenAI": {}, "Google": {}, "Anthropic": {}}},
    )
    monkeypatch.setattr(
        runs.ollama_api,
        "get_ollama_status",
        lambda host_address=None: {"available": True, "models": []},
    )

    with pytest.raises(ValueError, match="Invalid Model Selected"):
        runs.generate_midi("unknown-model", "write a loop")


@pytest.mark.parametrize(
    ("provider", "adapter_name", "credential_name", "credential_value"),
    [
        ("OpenAI", "openai_api", "openai_api_key", "openai-key"),
        ("Google", "gemini_api", "google_api_key", "google-key"),
        ("Anthropic", "claude_api", "anthropic_api_key", "anthropic-key"),
    ],
)
def test_generate_variations_routes_identical_composed_message_to_cloud_providers(
    monkeypatch,
    provider,
    adapter_name,
    credential_name,
    credential_value,
):
    model = f"{provider.lower()}-model"
    model_info = {"models": {"OpenAI": {}, "Google": {}, "Anthropic": {}}}
    model_info["models"][provider][model] = {}
    monkeypatch.setattr(runs, "get_model_info", lambda: model_info)
    monkeypatch.setattr(runs, "get_variation_prompt", lambda: "canonical prompt")
    captured = {}
    expected = object()

    def fake_variation_gen(**kwargs):
        captured.update(kwargs)
        return expected

    monkeypatch.setattr(
        getattr(runs, adapter_name), "variation_gen", fake_variation_gen, raising=False
    )

    credentials = ProviderCredentials(**{credential_name: credential_value})
    result = runs.generate_variations(
        model,
        "C minor brief -- unchanged",
        count=3,
        temp=0.4,
        use_thinking=True,
        effort="high",
        provider_credentials=credentials,
        request_timeout=2.5,
    )

    assert result is expected
    assert captured == {
        "prompt": (
            "Requested variation count: 3\n\nMusical brief: C minor brief -- unchanged"
        ),
        "count": 3,
        "model": model,
        "temp": 0.4,
        "use_thinking": True,
        "effort": "high",
        "system_prompt": "canonical prompt",
        "request_timeout": 2.5,
        "api_key": credential_value,
    }


def test_generate_variations_routes_to_ollama_with_prompt_override(monkeypatch):
    monkeypatch.setattr(
        runs,
        "get_model_info",
        lambda: {"models": {"OpenAI": {}, "Google": {}, "Anthropic": {}}},
    )
    monkeypatch.setattr(
        runs.ollama_api,
        "get_ollama_status",
        lambda **kwargs: {"available": True, "models": ["llama3"]},
    )
    captured = {}
    expected = object()

    def fake_variation_gen(**kwargs):
        captured.update(kwargs)
        return expected

    monkeypatch.setattr(
        runs.ollama_api, "variation_gen", fake_variation_gen, raising=False
    )

    result = runs.generate_variations(
        "llama3",
        "same brief",
        count=2,
        provider_credentials=ProviderCredentials(ollama_host="http://ollama.test"),
        request_timeout=4.0,
        system_prompt="complete override",
    )

    assert result is expected
    assert captured == {
        "prompt": "Requested variation count: 2\n\nMusical brief: same brief",
        "count": 2,
        "model": "llama3",
        "temp": 0.0,
        "system_prompt": "complete override",
        "request_timeout": 4.0,
        "host_address": "http://ollama.test",
    }


@pytest.mark.parametrize("count", [1, 9, True, 2.0, "2", None])
def test_generate_variations_validates_count_before_any_lookup(monkeypatch, count):
    model_lookup = Mock(side_effect=AssertionError("model lookup must not run"))
    ollama_lookup = Mock(side_effect=AssertionError("Ollama lookup must not run"))
    monkeypatch.setattr(runs, "get_model_info", model_lookup)
    monkeypatch.setattr(runs.ollama_api, "get_ollama_status", ollama_lookup)

    with pytest.raises((TypeError, ValueError)):
        runs.generate_variations("any-model", "brief", count=count)

    model_lookup.assert_not_called()
    ollama_lookup.assert_not_called()
