"""Offline Ollama reasoning capability and routing tests."""

import inspect
import json
from types import SimpleNamespace

import pytest

from conductor_core import routing
from conductor_core.providers import ollama
from conductor_core.providers._variations import VariationCollection


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


@pytest.mark.parametrize(
    ("capabilities", "expected"),
    [
        (["completion", "thinking"], True),
        (["completion"], False),
    ],
)
def test_ollama_status_reports_thinking_capability(monkeypatch, capabilities, expected):
    client = SimpleNamespace(
        list=lambda: SimpleNamespace(models=[SimpleNamespace(model="local-model")]),
        show=lambda name: SimpleNamespace(capabilities=capabilities),
    )
    monkeypatch.setattr(ollama, "initialize_ollama_client", lambda **kwargs: client)

    status = ollama.get_ollama_status()

    assert status["models"] == ["local-model"]
    assert status["model_capabilities"] == {
        "local-model": {
            "extended_thinking": expected,
            "effort_options": [],
            "temperature_supported": True,
        }
    }


def test_ollama_status_reports_show_failure_as_temperature_only(monkeypatch):
    def show(name):
        if name == "broken-model":
            raise RuntimeError("show unavailable")
        return SimpleNamespace(capabilities=["thinking"])

    client = SimpleNamespace(
        list=lambda: SimpleNamespace(
            models=[
                SimpleNamespace(model="thinking-model"),
                SimpleNamespace(model="broken-model"),
            ]
        ),
        show=show,
    )
    monkeypatch.setattr(ollama, "initialize_ollama_client", lambda **kwargs: client)

    status = ollama.get_ollama_status()

    assert status["model_capabilities"]["thinking-model"]["extended_thinking"]
    assert status["model_capabilities"]["broken-model"] == {
        "extended_thinking": False,
        "effort_options": [],
        "temperature_supported": True,
    }


def test_ollama_status_discovers_effort_levels_from_raw_show(monkeypatch):
    class RawResponse:
        def json(self):
            return {
                "thinking": {
                    "values": ["low", "medium", "high"],
                    "default": "medium",
                }
            }

    raw_calls = []
    client = SimpleNamespace(
        list=lambda: SimpleNamespace(models=[SimpleNamespace(model="gpt-oss")]),
        show=lambda name: SimpleNamespace(capabilities=["completion", "thinking"]),
        _request_raw=lambda *args, **kwargs: (
            raw_calls.append((args, kwargs)) or RawResponse()
        ),
    )
    monkeypatch.setattr(ollama, "initialize_ollama_client", lambda **kwargs: client)

    status = ollama.get_ollama_status()

    assert status["model_capabilities"]["gpt-oss"] == {
        "extended_thinking": True,
        "effort_options": ["low", "medium", "high"],
        "temperature_supported": True,
    }
    assert raw_calls == [(("POST", "/api/show"), {"json": {"model": "gpt-oss"}})]


@pytest.mark.parametrize(
    ("generation", "model_capabilities", "use_thinking", "effort", "expected"),
    [
        ("loop", {"extended_thinking": True, "effort_options": []}, True, "high", True),
        (
            "variations",
            {"extended_thinking": True, "effort_options": []},
            False,
            "low",
            False,
        ),
        (
            "loop",
            {"extended_thinking": True, "effort_options": ["low", "medium", "high"]},
            True,
            "high",
            "high",
        ),
        (
            "variations",
            {"extended_thinking": False, "effort_options": []},
            True,
            "low",
            None,
        ),
    ],
)
def test_ollama_generation_passes_supported_think_values(
    monkeypatch, generation, model_capabilities, use_thinking, effort, expected
):
    calls = []
    loop_json = json.dumps(_loop_payload())
    completion = SimpleNamespace(
        message=SimpleNamespace(
            content=loop_json
            if generation == "loop"
            else json.dumps({"variations": [_loop_payload()]}),
            thinking="private reasoning",
        )
    )
    client = SimpleNamespace(chat=lambda **kwargs: calls.append(kwargs) or completion)
    monkeypatch.setattr(ollama, "initialize_ollama_client", lambda **kwargs: client)

    function = ollama.loop_gen if generation == "loop" else ollama.variations_gen
    _, messages, *_ = function(
        "prompt",
        "model",
        temp=0.4,
        use_thinking=use_thinking,
        effort=effort,
        model_capabilities=model_capabilities,
    )

    assert calls[0]["options"]["temperature"] == 0.4
    if expected is None:
        assert "think" not in calls[0]
    else:
        assert (
            calls[0]["think"] is expected
            if isinstance(expected, bool)
            else calls[0]["think"] == expected
        )
    assert {"role": "assistant", "content": "private reasoning"} in messages


@pytest.mark.parametrize("generator", ["generate_midi", "generate_variations"])
def test_ollama_routing_rejects_invalid_effort(monkeypatch, generator):
    model_info = {"models": {"OpenAI": {}, "Google": {}, "Anthropic": {}}}
    monkeypatch.setattr(routing, "get_model_info", lambda: model_info)
    monkeypatch.setattr(
        routing.ollama_api,
        "get_ollama_status",
        lambda **kwargs: {
            "available": True,
            "models": ["thinking-model"],
            "model_capabilities": {
                "thinking-model": {
                    "extended_thinking": True,
                    "effort_options": ["low", "medium", "high"],
                    "temperature_supported": True,
                }
            },
        },
    )

    def request():
        if generator == "generate_midi":
            return routing.generate_midi(
                "thinking-model", "prompt", use_thinking=True, effort="invalid"
            )
        return routing.generate_variations(
            "thinking-model", "prompt", 2, use_thinking=True, effort="invalid"
        )

    with pytest.raises(ValueError, match="Invalid effort 'invalid'"):
        request()


def test_ollama_status_contract():
    assert list(inspect.signature(ollama.get_ollama_status).parameters) == [
        "host_address",
        "request_timeout",
    ]


def test_ollama_status_return_keys(monkeypatch):
    client = SimpleNamespace(
        list=lambda: SimpleNamespace(models=[]),
        show=lambda name: SimpleNamespace(capabilities=[]),
    )
    monkeypatch.setattr(ollama, "initialize_ollama_client", lambda **kwargs: client)

    assert set(ollama.get_ollama_status()) == {
        "available",
        "models",
        "model_capabilities",
        "host",
        "error",
    }
    assert ollama.get_ollama_status()["models"] == []


def test_ollama_effort_levels_are_forwarded_by_both_adapters(monkeypatch):
    loop_calls = []
    variations_calls = []
    loop_content = json.dumps(_loop_payload())
    variations_content = json.dumps({"variations": [_loop_payload()]})
    loop_client = SimpleNamespace(
        chat=lambda **kwargs: (
            loop_calls.append(kwargs)
            or SimpleNamespace(message=SimpleNamespace(content=loop_content))
        )
    )
    variation_client = SimpleNamespace(
        chat=lambda **kwargs: (
            variations_calls.append(kwargs)
            or SimpleNamespace(message=SimpleNamespace(content=variations_content))
        )
    )
    clients = iter([loop_client, variation_client])
    monkeypatch.setattr(
        ollama, "initialize_ollama_client", lambda **kwargs: next(clients)
    )
    capability = {
        "extended_thinking": True,
        "effort_options": ["low", "medium", "high"],
    }

    ollama.loop_gen(
        "prompt",
        "model",
        use_thinking=True,
        effort="medium",
        model_capabilities=capability,
    )
    ollama.variations_gen(
        "prompt",
        "model",
        use_thinking=True,
        effort="high",
        model_capabilities=capability,
    )

    assert loop_calls[0]["think"] == "medium"
    assert variations_calls[0]["think"] == "high"
    assert len(VariationCollection.model_json_schema()) > 0
