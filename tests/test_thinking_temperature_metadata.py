"""Offline checks that thinking temperature metadata matches adapter requests."""

import json
from types import SimpleNamespace

import pytest

from conductor_core import music
from conductor_core.providers import anthropic, google, ollama, openai

REQUESTED_TEMPERATURE = 0.4


class _RequestCaptured(Exception):
    """Stop an adapter after it has built its provider request."""


def _capture(calls):
    def create(**kwargs):
        calls.append(kwargs)
        raise _RequestCaptured

    return create


def _sent_temperature(provider, model, generation, monkeypatch):
    calls = []
    if provider == "Anthropic":
        client = SimpleNamespace(messages=SimpleNamespace(create=_capture(calls)))
        monkeypatch.setattr(
            anthropic, "initialize_anthropic_client", lambda **kwargs: client
        )
        module = anthropic
    elif provider == "Google":
        client = SimpleNamespace(
            models=SimpleNamespace(generate_content=_capture(calls))
        )
        monkeypatch.setattr(google, "initialize_gemini_client", lambda **kwargs: client)
        module = google
    else:
        client = SimpleNamespace(responses=SimpleNamespace(parse=_capture(calls)))
        monkeypatch.setattr(openai, "initialize_openai_client", lambda **kwargs: client)
        module = openai

    model_config = music.get_model_info()["models"][provider][model]
    effort_options = model_config.get("effort_options") or []
    function = module.loop_gen if generation == "loop" else module.variations_gen
    with pytest.raises(_RequestCaptured):
        function(
            "prompt",
            model,
            temp=REQUESTED_TEMPERATURE,
            use_thinking=True,
            effort=effort_options[-1] if effort_options else None,
        )

    request = calls[0]
    if provider == "Google":
        return request["config"].get("temperature")
    return request.get("temperature")


_THINKING_MODELS = [
    (provider, model)
    for provider, models in music.get_model_info()["models"].items()
    for model, model_config in models.items()
    if model_config.get("extended_thinking")
]


@pytest.mark.parametrize("generation", ["loop", "variations"])
@pytest.mark.parametrize(("provider", "model"), _THINKING_MODELS)
def test_thinking_fixed_temperature_matches_adapter_request(
    monkeypatch, provider, model, generation
):
    model_config = music.get_model_info()["models"][provider][model]
    fixed_temperature = model_config.get("thinking_fixed_temperature")

    sent = _sent_temperature(provider, model, generation, monkeypatch)

    if not model_config.get("temperature_supported", True):
        assert sent is None
    elif fixed_temperature is not None:
        assert sent == fixed_temperature
    else:
        assert sent == REQUESTED_TEMPERATURE


def test_models_that_reject_temperature_are_marked_unsupported():
    models = music.get_model_info()["models"]
    anthropic_models = (
        "claude-opus-5-5",
        "claude-fable-5-1",
        "claude-fable-5",
        "claude-opus-5",
        "claude-sonnet-5",
        "claude-opus-4-8",
        "claude-opus-4-7",
    )

    for model in anthropic_models:
        assert models["Anthropic"][model]["temperature_supported"] is False, model
    for model, model_config in models["OpenAI"].items():
        if model_config["extended_thinking"]:
            assert model_config["temperature_supported"] is False, model


def test_anthropic_budget_thinking_models_report_fixed_temperature():
    models = music.get_model_info()["models"]["Anthropic"]

    for model in ("claude-opus-4-5", "claude-sonnet-4-5", "claude-haiku-4-5"):
        assert models[model]["thinking_fixed_temperature"] == 1.0


def test_only_anthropic_models_report_fixed_temperature():
    for provider, models in music.get_model_info()["models"].items():
        if provider == "Anthropic":
            continue
        for model, model_config in models.items():
            assert model_config.get("thinking_fixed_temperature") is None, (
                f"{provider}/{model}"
            )


@pytest.mark.parametrize(
    "fields",
    [
        {"thinking_fixed_temperature": True},
        {"thinking_fixed_temperature": "1.0"},
        {"thinking_fixed_temperature": -0.5},
        {"thinking_fixed_temperature": 1.0, "temperature_supported": False},
        {"temperature_supported": "no"},
    ],
)
def test_model_metadata_rejects_invalid_temperature_fields(fields):
    model_info = {
        "models": {
            "Anthropic": {
                "model": {
                    **fields,
                    "rate_limits": {"RPM": 1, "TPM": None, "RPD": None},
                }
            }
        }
    }

    with pytest.raises(ValueError, match="temperature"):
        music._validate_model_info(model_info)


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


@pytest.mark.parametrize("generation", ["loop", "variations"])
def test_ollama_thinking_fixed_temperature_matches_adapter_request(
    monkeypatch, generation
):
    calls = []
    content = (
        json.dumps(_loop_payload())
        if generation == "loop"
        else json.dumps({"variations": [_loop_payload()]})
    )
    completion = SimpleNamespace(message=SimpleNamespace(content=content))
    client = SimpleNamespace(
        list=lambda: SimpleNamespace(models=[SimpleNamespace(model="local-model")]),
        show=lambda name: SimpleNamespace(capabilities=["completion", "thinking"]),
        chat=lambda **kwargs: calls.append(kwargs) or completion,
    )
    monkeypatch.setattr(ollama, "initialize_ollama_client", lambda **kwargs: client)
    capabilities = ollama.get_ollama_status()["model_capabilities"]["local-model"]

    function = ollama.loop_gen if generation == "loop" else ollama.variations_gen
    function(
        "prompt",
        "local-model",
        temp=REQUESTED_TEMPERATURE,
        use_thinking=True,
        model_capabilities=capabilities,
    )

    assert capabilities["thinking_fixed_temperature"] is None
    assert calls[0]["think"] is True
    assert calls[0]["options"]["temperature"] == REQUESTED_TEMPERATURE
