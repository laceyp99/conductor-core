"""Offline Ollama reasoning capability and routing tests."""

import inspect
import json
from types import SimpleNamespace

import pytest

from conductor_core import routing
from conductor_core.providers import ollama
from conductor_core.providers._variations import VariationCollection


@pytest.mark.parametrize("effort", ["extreme", "high"])
def test_ollama_rejects_unknown_reasoning_effort(effort):
    capabilities = {"extended_thinking": True, "effort_options": ["low", "medium"]}

    with pytest.raises(ValueError, match="Unsupported Ollama reasoning effort"):
        ollama._thinking_option(capabilities, True, effort)


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
@pytest.mark.parametrize("effort", ["high", None, "unsupported"])
@pytest.mark.parametrize(
    ("thinking_off", "effort_options", "expected"),
    [
        ("disabled", ["low", "medium", "high"], False),
        ("lowest_effort", ["low", "medium", "high"], "low"),
        ("lowest_effort", ["medium", "high"], "medium"),
    ],
)
def test_direct_ollama_thinking_off_ignores_requested_effort(
    monkeypatch, generation, effort, thinking_off, effort_options, expected
):
    calls = []
    payload = _loop_payload()
    if generation == "variations":
        payload = {"variations": [payload]}
    completion = SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))
    client = SimpleNamespace(chat=lambda **kwargs: calls.append(kwargs) or completion)
    monkeypatch.setattr(ollama, "initialize_ollama_client", lambda **kwargs: client)

    function = ollama.loop_gen if generation == "loop" else ollama.variations_gen
    function(
        "prompt",
        "model",
        use_thinking=False,
        effort=effort,
        model_capabilities={
            "extended_thinking": True,
            "effort_options": effort_options,
            "thinking_off": thinking_off,
        },
    )

    assert len(calls) == 1
    if expected is False:
        assert calls[0]["think"] is False
    else:
        assert calls[0]["think"] == expected


@pytest.mark.parametrize("generation", ["loop", "variations"])
@pytest.mark.parametrize("effort", ["high", "unsupported"])
def test_direct_ollama_thinking_enabled_validates_effort(
    monkeypatch, generation, effort
):
    calls = []
    payload = _loop_payload()
    if generation == "variations":
        payload = {"variations": [payload]}
    completion = SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))
    client = SimpleNamespace(chat=lambda **kwargs: calls.append(kwargs) or completion)
    monkeypatch.setattr(ollama, "initialize_ollama_client", lambda **kwargs: client)

    function = ollama.loop_gen if generation == "loop" else ollama.variations_gen
    kwargs = {
        "use_thinking": True,
        "effort": effort,
        "model_capabilities": {
            "extended_thinking": True,
            "effort_options": ["low", "medium", "high"],
            "thinking_off": "lowest_effort",
        },
    }
    if effort == "unsupported":
        with pytest.raises(ValueError, match="Unsupported Ollama reasoning effort"):
            function("prompt", "model", **kwargs)
        assert calls == []
    else:
        function("prompt", "model", **kwargs)
        assert calls[0]["think"] == effort


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

    assert [model.name for model in status.models] == ["local-model"]
    assert status.models[0].capabilities == {
        "extended_thinking": expected,
        "effort_options": [],
        "temperature_supported": True,
        "thinking_fixed_temperature": None,
        "thinking_off": "disabled" if expected else None,
    }


def test_ollama_status_preserves_partial_inspection_results(monkeypatch):
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

    successful, failed = status.models
    assert successful.name == "thinking-model"
    assert successful.capabilities["extended_thinking"]
    assert successful.error is None
    assert failed.name == "broken-model"
    assert failed.capabilities is None
    assert isinstance(failed.error, ollama.ProviderRequestError)
    assert str(failed.error.__cause__) == "show unavailable"


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

    assert status.models[0].capabilities == {
        "extended_thinking": True,
        "effort_options": ["low", "medium", "high"],
        "temperature_supported": True,
        "thinking_fixed_temperature": None,
        "thinking_off": "lowest_effort",
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
        "get_model_status",
        lambda model_name, **kwargs: ollama.OllamaModelInspection(
            name=model_name,
            capabilities={
                "extended_thinking": True,
                "effort_options": ["low", "medium", "high"],
                "temperature_supported": True,
            },
            error=None,
        ),
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

    status = ollama.get_ollama_status()
    assert isinstance(status, ollama.OllamaStatus)
    assert status.models == ()
    assert isinstance(status.host, str)


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


class _RawThinking:
    def __init__(self, values):
        self.values = values

    def json(self):
        return {"thinking": {"values": self.values}}


@pytest.mark.parametrize(
    ("values", "capabilities", "expected"),
    [
        # Values reported by Ollama 0.34.4 for local models.
        ([False, True], ["completion", "thinking"], ("disabled", [])),
        (
            ["low", "medium", "high"],
            ["completion", "thinking"],
            ("lowest_effort", None),
        ),
        ([False], ["completion"], (None, [])),
        (None, ["completion", "thinking"], ("disabled", [])),
        (
            [False, True, "low", "high"],
            ["completion", "thinking"],
            ("disabled", ["low", "high"]),
        ),
    ],
)
def test_ollama_status_reports_thinking_off(
    monkeypatch, values, capabilities, expected
):
    thinking_off, effort_options = expected
    client = SimpleNamespace(
        list=lambda: SimpleNamespace(models=[SimpleNamespace(model="local-model")]),
        show=lambda name: SimpleNamespace(capabilities=capabilities),
        _request_raw=lambda *args, **kwargs: (
            _RawThinking(values) if values is not None else SimpleNamespace(json=dict)
        ),
    )
    monkeypatch.setattr(ollama, "initialize_ollama_client", lambda **kwargs: client)

    model_capabilities = ollama.get_ollama_status().models[0].capabilities

    assert model_capabilities["thinking_off"] == thinking_off
    if effort_options is not None:
        assert model_capabilities["effort_options"] == effort_options


@pytest.mark.parametrize(
    ("model_capabilities", "use_thinking", "effort", "expected"),
    [
        ({"extended_thinking": True, "thinking_off": "disabled"}, False, None, False),
        ({"extended_thinking": True, "thinking_off": "disabled"}, True, None, True),
        (
            {
                "extended_thinking": True,
                "effort_options": ["low", "medium", "high"],
                "thinking_off": "disabled",
            },
            False,
            "low",
            False,
        ),
        (
            {
                "extended_thinking": True,
                "effort_options": ["low", "medium", "high"],
                "thinking_off": "lowest_effort",
            },
            False,
            "low",
            "low",
        ),
        # Capabilities built before thinking_off existed keep the old behavior.
        ({"extended_thinking": True, "effort_options": []}, False, None, False),
        (
            {"extended_thinking": True, "effort_options": ["low", "high"]},
            False,
            "low",
            "low",
        ),
    ],
)
def test_ollama_thinking_off_selects_think_value(
    model_capabilities, use_thinking, effort, expected
):
    assert ollama._thinking_option(model_capabilities, use_thinking, effort) == expected


def _counting_client(model_names, shown):
    return SimpleNamespace(
        list=lambda: SimpleNamespace(
            models=[SimpleNamespace(model=name) for name in model_names]
        ),
        show=lambda name: (
            shown.append(name)
            or SimpleNamespace(capabilities=["completion", "thinking"])
        ),
    )


def test_model_status_inspects_only_the_requested_model(monkeypatch):
    shown = []
    client = _counting_client(["a", "b", "c"], shown)
    monkeypatch.setattr(ollama, "initialize_ollama_client", lambda **kwargs: client)

    status = ollama.get_model_status("b")

    assert shown == ["b"]
    assert status.name == "b"
    assert status.error is None
    assert status.capabilities["thinking_off"] == "disabled"


def test_model_status_skips_inspection_for_missing_model(monkeypatch):
    shown = []
    client = _counting_client(["a"], shown)
    monkeypatch.setattr(ollama, "initialize_ollama_client", lambda **kwargs: client)

    status = ollama.get_model_status("missing")

    assert shown == []
    assert status is None


def test_model_status_reports_unavailable_server(monkeypatch):
    error = ollama.ProviderConnectionError("Ollama", "refused")

    def fail(**kwargs):
        raise error

    monkeypatch.setattr(ollama, "initialize_ollama_client", fail)

    with pytest.raises(ollama.ProviderConnectionError, match="refused") as raised:
        ollama.get_model_status("a")
    assert raised.value is error


def test_model_list_does_not_inspect_models(monkeypatch):
    shown = []
    client = _counting_client(["a", "b"], shown)
    monkeypatch.setattr(ollama, "initialize_ollama_client", lambda **kwargs: client)

    assert ollama.get_model_list() == ["a", "b"]
    assert shown == []
