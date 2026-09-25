"""Offline checks that thinking_off metadata matches thinking-off requests."""

from types import SimpleNamespace

import pytest

from conductor_core import music
from conductor_core.providers import anthropic, google, openai


class _RequestCaptured(Exception):
    """Stop an adapter after it has built its provider request."""


def _capture(calls):
    def create(**kwargs):
        calls.append(kwargs)
        raise _RequestCaptured

    return create


def _thinking_off_request(provider, model, generation, monkeypatch):
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

    function = module.loop_gen if generation == "loop" else module.variations_gen
    with pytest.raises(_RequestCaptured):
        function("prompt", model, use_thinking=False)
    return calls[0]


def _reasoning_disabled(provider, model_config, request):
    """Return whether a request uses the provider's official no-reasoning mode."""
    if provider == "OpenAI":
        return request["reasoning"]["effort"] == "none"
    if provider == "Anthropic":
        return (
            request.get("thinking", {"type": "disabled"}) == {"type": "disabled"}
            and "output_config" not in request
        )
    thinking_config = request["config"]["thinking_config"]
    return thinking_config.thinking_budget == 0


def _lowest_effort(provider, model_config, request):
    """Return whether a request uses the model's lowest reasoning setting."""
    effort_options = model_config.get("effort_options") or []
    if provider == "OpenAI":
        return request["reasoning"]["effort"] == effort_options[0]
    if provider == "Anthropic":
        return request["output_config"] == {"effort": effort_options[0]}
    thinking_config = request["config"]["thinking_config"]
    if effort_options:
        return thinking_config.thinking_level.value == effort_options[0].upper()
    return thinking_config.thinking_budget == model_config["min_thinking_budget"]


_THINKING_MODELS = [
    (provider, model)
    for provider, models in music.get_model_info()["models"].items()
    for model, model_config in models.items()
    if model_config.get("extended_thinking")
]


@pytest.mark.parametrize("generation", ["loop", "variations"])
@pytest.mark.parametrize(("provider", "model"), _THINKING_MODELS)
def test_thinking_off_metadata_matches_adapter_request(
    monkeypatch, provider, model, generation
):
    model_config = music.get_model_info()["models"][provider][model]

    request = _thinking_off_request(provider, model, generation, monkeypatch)

    disabled = _reasoning_disabled(provider, model_config, request)
    if model_config["thinking_off"] == "disabled":
        assert disabled
    else:
        assert not disabled
        assert _lowest_effort(provider, model_config, request)


def test_non_thinking_models_do_not_report_thinking_off():
    for provider, models in music.get_model_info()["models"].items():
        for model, model_config in models.items():
            if not model_config.get("extended_thinking"):
                assert "thinking_off" not in model_config, f"{provider}/{model}"


@pytest.mark.parametrize(
    "fields",
    [
        {"extended_thinking": True},
        {"extended_thinking": True, "thinking_off": "none"},
        {"extended_thinking": False, "thinking_off": "disabled"},
    ],
)
def test_model_metadata_rejects_invalid_thinking_off(fields):
    model_info = {
        "models": {
            "OpenAI": {
                "model": {
                    **fields,
                    "rate_limits": {"RPM": 1, "TPM": None, "RPD": None},
                }
            }
        }
    }

    with pytest.raises(ValueError, match="thinking_off"):
        music._validate_model_info(model_info)
