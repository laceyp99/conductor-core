"""Check Core's Anthropic requests against the installed SDK without network calls."""

import inspect
from types import SimpleNamespace

import pytest

pytest.importorskip("anthropic")

from conductor_core import music
from conductor_core.providers import anthropic


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
