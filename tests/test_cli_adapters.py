import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from conductor_core import (
    GenerationRequest,
    ProviderAuthenticationError,
    ProviderRequestError,
)
from conductor_core._cli import (
    EXIT_CONFIGURATION,
    EXIT_GENERATION,
    EXIT_UNEXPECTED,
    EXIT_USAGE,
    classify_error,
    error_envelope,
    format_generation,
    format_models,
    models_envelope,
    normalize_loop,
    normalize_model_records,
    should_show_progress,
    success_envelope,
)


def test_normalize_loop_makes_standard_and_google_timing_identical(
    sample_loop, sample_loop_g
):
    assert normalize_loop(sample_loop) == normalize_loop(sample_loop_g)
    assert normalize_loop(sample_loop)["Bar_1"]["notes"][0]["time"] == {
        "start_beat": 1,
        "duration": 16,
    }


def test_normalize_model_records_is_stable_and_filters_case_insensitively():
    model_info = {
        "models": {
            "OpenAI": {
                "model-b": {
                    "extended_thinking": True,
                    "effort_options": ["low", "high"],
                    "max_tokens": 10,
                    "cost": {"input": 1.0},
                    "rate_limits": {"RPM": 1, "TPM": None, "RPD": None},
                },
                "model-a": {
                    "temperature_supported": False,
                    "max_tokens": 20,
                    "rate_limits": {"RPM": 1, "TPM": 2, "RPD": None},
                },
            },
            "Anthropic": {},
        }
    }

    records = normalize_model_records(model_info, "openai")

    assert [record["model"] for record in records] == ["model-a", "model-b"]
    assert records[0]["temperature_supported"] is False
    assert records[0]["effort_options"] == []
    assert records[1]["extended_thinking"] is True
    assert json.loads(json.dumps(models_envelope(records)))["models"] == records


def test_normalize_model_records_rejects_unknown_provider():
    with pytest.raises(ValueError, match="Unknown provider 'Other'"):
        normalize_model_records({"models": {"OpenAI": {}}}, "Other")


@pytest.mark.parametrize(
    ("json_output", "quiet", "verbose", "isatty", "expected"),
    [
        (False, False, False, True, True),
        (False, False, False, False, False),
        (False, False, True, False, True),
        (False, True, True, True, False),
        (True, False, True, True, False),
    ],
)
def test_progress_policy(json_output, quiet, verbose, isatty, expected):
    assert (
        should_show_progress(
            json_output=json_output,
            quiet=quiet,
            verbose=verbose,
            stderr_isatty=isatty,
        )
        is expected
    )


@pytest.mark.parametrize(
    ("exc", "exit_code", "code"),
    [
        (
            ProviderAuthenticationError("OpenAI", "missing key"),
            EXIT_CONFIGURATION,
            "configuration_error",
        ),
        (ImportError("missing SDK"), EXIT_CONFIGURATION, "dependency_error"),
        (
            ProviderRequestError("OpenAI", "bad response"),
            EXIT_GENERATION,
            "generation_error",
        ),
        (ValueError("Invalid Model Selected"), EXIT_USAGE, "validation_error"),
        (ValueError("internal defect"), EXIT_UNEXPECTED, "unexpected_error"),
        (RuntimeError("broken"), EXIT_UNEXPECTED, "unexpected_error"),
    ],
)
def test_error_classification_is_narrow(exc, exit_code, code):
    error = classify_error(exc)

    assert error.exit_code == exit_code
    assert error.code == code
    assert error_envelope(error)["error"]["message"] == str(exc)


def test_success_envelope_uses_json_types_and_omits_sensitive_inputs(
    tmp_path, sample_loop
):
    request = GenerationRequest(
        key="C",
        scale="Major",
        description="warm loop",
        model="gpt-4o-mini",
        prompt_override="private system prompt",
        soundfont_path="private/path.sf2",
    )
    metadata = SimpleNamespace(
        provider="OpenAI",
        timestamp=datetime(2026, 1, 2, tzinfo=timezone.utc),
        soundfont=None,
        audio_path=None,
        messages_path="relative/messages.json",
    )
    result = SimpleNamespace(
        generation_id="abc",
        loop=sample_loop,
        midi_path=str(tmp_path / "loop.mid"),
        audio_path=None,
        metadata=metadata,
        cost=None,
        warnings=["audio unavailable"],
    )

    envelope = success_envelope(request, result)
    serialized = json.dumps(envelope)

    assert envelope["schema_version"] == 1
    assert envelope["artifacts"]["audio"] is None
    assert Path(envelope["artifacts"]["messages"]).is_absolute()
    assert "private system prompt" not in serialized
    assert "private/path.sf2" not in serialized
    assert envelope["metadata"]["timestamp"] == "2026-01-02T00:00:00+00:00"


def test_human_formatters_are_plain_and_include_warnings(tmp_path):
    result = SimpleNamespace(
        generation_id="abc",
        midi_path=str(tmp_path / "loop.mid"),
        audio_path=None,
        metadata=SimpleNamespace(messages_path=str(tmp_path / "messages.json")),
        cost=None,
        warnings=["audio unavailable"],
    )
    records = [
        {
            "provider": "OpenAI",
            "model": "gpt-test",
            "extended_thinking": True,
            "effort_options": ["low", "high"],
        }
    ]

    generation_text = format_generation(result)
    models_text = format_models(records)

    assert "Cost: unavailable" in generation_text
    assert "Warning: audio unavailable" in generation_text
    assert models_text == "OpenAI:\n  gpt-test (thinking; effort=low,high)"
