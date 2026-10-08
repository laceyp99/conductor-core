"""Check Core's Anthropic requests against the installed SDK without network calls."""

import inspect
from types import SimpleNamespace

import pytest

pytest.importorskip("anthropic")

from conductor_core import music
from conductor_core.providers import anthropic


def test_anthropic_rejects_unknown_reasoning_effort():
    with pytest.raises(ValueError, match="Unsupported Anthropic reasoning effort"):
        anthropic._validated_effort("extreme")


def test_anthropic_rejects_effort_not_advertised_by_model():
    with pytest.raises(
        ValueError, match="Unsupported Anthropic model reasoning effort"
    ):
        anthropic._validated_effort("high", ["low", "medium"])


@pytest.mark.parametrize("use_thinking", [False, True])
@pytest.mark.parametrize("effort", ["low", "medium", "high", "xhigh", "max"])
def test_haiku_5_5_thinking_controls(use_thinking, effort):
    config = music.get_model_info()["models"]["Anthropic"]["claude-haiku-5-5"]
    request = {"tool_choice": {"type": "tool", "name": "build_MIDI_loop"}}

    anthropic._apply_thinking_params(request, config, 0.4, use_thinking, effort)

    assert "extra_body" not in request
    if use_thinking:
        assert request["thinking"] == {"type": "adaptive"}
        assert request["output_config"] == {"effort": effort}
        assert request["tool_choice"] == {"type": "auto"}
    else:
        assert request["thinking"] == {"type": "disabled"}
        # Omit effort so disabled thinking uses the API's valid medium default.
        assert "output_config" not in request
        assert request["tool_choice"]["type"] == "tool"


class _RequestCaptured(Exception):
    pass


@pytest.mark.parametrize("generation", ["loop", "variations"])
@pytest.mark.parametrize("use_thinking", [False, True])
@pytest.mark.parametrize("model", music.get_model_info()["models"]["Anthropic"])
def test_anthropic_request_matches_installed_sdk_signature(
    monkeypatch, model, use_thinking, generation
):
    sdk_create = anthropic.Anthropic(api_key="unused").messages.create
    sdk_signature = inspect.signature(sdk_create)
    captured = {}

    def capture_create(**kwargs):
        sdk_signature.bind(**kwargs)
        captured.update(kwargs)
        raise _RequestCaptured

    client = SimpleNamespace(messages=SimpleNamespace(create=capture_create))
    monkeypatch.setattr(anthropic, "initialize_anthropic_client", lambda **_: client)
    generate = anthropic.loop_gen if generation == "loop" else anthropic.variations_gen
    model_config = music.get_model_info()["models"]["Anthropic"][model]
    effort_options = model_config.get("effort_options") or []

    with pytest.raises(_RequestCaptured):
        generate(
            "write a loop",
            model,
            temp=0.4,
            use_thinking=use_thinking,
            effort=effort_options[-1] if effort_options else None,
        )

    assert "temperature" not in captured
    if model_config.get("temperature_supported", True):
        expected = 1.0 if use_thinking else 0.4
        assert captured["extra_body"] == {"temperature": expected}
    else:
        assert "extra_body" not in captured
