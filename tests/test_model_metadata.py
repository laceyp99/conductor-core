import pytest

from conductor_core import music


def test_model_info_results_are_isolated_from_caller_mutations():
    first = music.get_model_info()
    model_name = "gemini-3.7-flash"
    first_model = first["models"]["Google"][model_name]

    first["caller_only"] = True
    first_model["cost"]["input"] = -1

    second = music.get_model_info()
    second_model = second["models"]["Google"][model_name]

    assert "caller_only" not in second
    assert second_model["cost"]["input"] == 0.75
    assert first is not second
    assert first["models"] is not second["models"]
    assert first_model is not second_model
    assert first_model["cost"] is not second_model["cost"]


def test_effort_options_are_ordered_from_lowest_to_highest():
    effort_rank = {
        "none": 0,
        "minimal": 1,
        "low": 2,
        "medium": 3,
        "high": 4,
        "xhigh": 5,
        "max": 6,
    }

    for provider, models in music.get_model_info()["models"].items():
        for model, model_config in models.items():
            effort_options = model_config.get("effort_options") or []
            ranks = [effort_rank[effort] for effort in effort_options]
            assert ranks == sorted(ranks), f"{provider}/{model}"


def test_selectable_cloud_models_have_normalized_rate_limits():
    model_info = music.get_model_info()

    for provider, models in model_info["models"].items():
        assert models, f"{provider} must expose at least one selectable model"
        for model, model_config in models.items():
            rate_limits = model_config["rate_limits"]
            assert set(rate_limits) == {"RPM", "TPM", "RPD"}, f"{provider}/{model}"

            rpm = rate_limits["RPM"]
            assert rpm is None or (
                isinstance(rpm, int) and not isinstance(rpm, bool) and rpm > 0
            ), f"{provider}/{model} RPM"

            for field in ("TPM", "RPD"):
                value = rate_limits[field]
                assert value is None or (
                    isinstance(value, int) and not isinstance(value, bool) and value > 0
                ), f"{provider}/{model} {field}"


def test_gemini_3_7_flash_capabilities_match_google_documentation():
    model_config = music.get_model_info()["models"]["Google"]["gemini-3.7-flash"]

    assert model_config["extended_thinking"] is True
    assert model_config["effort_options"] == ["low", "medium", "high"]
    assert model_config["temperature_supported"] is False
    assert model_config["max_tokens"] == 65536
    assert model_config["cost"] == {
        "input": 0.75,
        "cache": {"text": 0.075, "storage hour": 0.50},
        "output": 3.75,
    }


def test_gemini_3_8_flash_capabilities_match_google_documentation():
    model_config = music.get_model_info()["models"]["Google"]["gemini-3.8-flash"]

    assert model_config["extended_thinking"] is True
    assert model_config["effort_options"] == ["low", "medium", "high"]
    assert model_config["temperature_supported"] is False
    assert model_config["max_tokens"] == 65536
    assert model_config["cost"] == {
        "input": 0.75,
        "cache": {"text": 0.075, "storage hour": 0.50},
        "output": 3.75,
    }
    assert model_config["rate_limits"] == {
        "RPM": None,
        "TPM": None,
        "RPD": None,
    }


def test_gpt_6_astra_capabilities_match_openai_documentation():
    model_config = music.get_model_info()["models"]["OpenAI"]["gpt-6-astra"]

    assert model_config["extended_thinking"] is True
    assert model_config["effort_options"] == [
        "low",
        "medium",
        "high",
        "xhigh",
        "max",
    ]
    assert model_config["max_tokens"] == 128000
    assert model_config["cost"] == {
        "input": 10.00,
        "cached input": 1.00,
        "cache write": 12.50,
        "output": 50.00,
    }


@pytest.mark.parametrize(
    ("model", "expected_cost"),
    [
        (
            "gpt-5.6-sol",
            {
                "input": 4.00,
                "cached input": 0.40,
                "cache write": 5.00,
                "output": 20.00,
            },
        ),
        (
            "gpt-5.6-terra",
            {
                "input": 2.00,
                "cached input": 0.20,
                "cache write": 2.50,
                "output": 12.00,
            },
        ),
        (
            "gpt-5.6-luna",
            {
                "input": 0.20,
                "cached input": 0.02,
                "cache write": 0.25,
                "output": 1.20,
            },
        ),
    ],
)
def test_gpt_5_6_cache_write_pricing_matches_openai_documentation(model, expected_cost):
    model_config = music.get_model_info()["models"]["OpenAI"][model]

    assert model_config["cost"] == expected_cost


def test_claude_fable_5_1_capabilities_match_anthropic_documentation():
    model_config = music.get_model_info()["models"]["Anthropic"]["claude-fable-5-1"]

    assert model_config["extended_thinking"] is True
    assert model_config["always_on_adaptive_thinking"] is True
    assert model_config["effort_options"] == [
        "low",
        "medium",
        "high",
        "xhigh",
        "max",
    ]
    assert model_config["max_tokens"] == 128000
    assert model_config["cost"] == {
        "input": 10.00,
        "5m cache input": 12.50,
        "1h cache input": 20.00,
        "cache hits/refreshes": 0.25,
        "output": 50.00,
    }


def test_fable_models_have_always_on_adaptive_thinking():
    anthropic_models = music.get_model_info()["models"]["Anthropic"]

    always_on_models = {
        model
        for model, model_config in anthropic_models.items()
        if model_config.get("always_on_adaptive_thinking")
    }

    assert always_on_models == {"claude-fable-5", "claude-fable-5-1"}


def test_model_metadata_rejects_non_boolean_always_on_adaptive_thinking():
    model_info = {
        "models": {
            "Anthropic": {
                "test-model": {
                    "always_on_adaptive_thinking": "yes",
                    "rate_limits": {"RPM": 1, "TPM": None, "RPD": None},
                }
            }
        }
    }

    with pytest.raises(
        ValueError, match="always_on_adaptive_thinking must be a boolean"
    ):
        music._validate_model_info(model_info)


@pytest.mark.parametrize(
    ("rate_limits", "message"),
    [
        ({"TPM": None, "RPD": None}, "must contain exactly"),
        ({"RPM": 0, "TPM": None, "RPD": None}, "RPM must be"),
        ({"RPM": True, "TPM": None, "RPD": None}, "RPM must be"),
        ({"RPM": 1, "TPM": 0, "RPD": None}, "TPM must be"),
        ({"RPM": 1, "TPM": None, "RPD": False}, "RPD must be"),
    ],
)
def test_model_metadata_validation_rejects_invalid_rate_limits(rate_limits, message):
    model_info = {
        "models": {
            "Cloud": {
                "selectable-model": {
                    "rate_limits": rate_limits,
                }
            }
        }
    }

    with pytest.raises(ValueError, match=message):
        music._validate_model_info(model_info)
