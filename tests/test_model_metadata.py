import pytest

from conductor_core import music


def test_selectable_cloud_models_have_normalized_rate_limits():
    model_info = music.get_model_info()

    for provider, models in model_info["models"].items():
        assert models, f"{provider} must expose at least one selectable model"
        for model, model_config in models.items():
            rate_limits = model_config["rate_limits"]
            assert set(rate_limits) == {"RPM", "TPM", "RPD"}, f"{provider}/{model}"

            rpm = rate_limits["RPM"]
            assert isinstance(rpm, int) and not isinstance(rpm, bool) and rpm > 0, (
                f"{provider}/{model}"
            )

            for field in ("TPM", "RPD"):
                value = rate_limits[field]
                assert value is None or (
                    isinstance(value, int) and not isinstance(value, bool) and value > 0
                ), f"{provider}/{model} {field}"


@pytest.mark.parametrize(
    ("rate_limits", "message"),
    [
        ({"TPM": None, "RPD": None}, "must contain exactly"),
        ({"RPM": None, "TPM": None, "RPD": None}, "RPM must be"),
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
