import json
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import ClassVar

import pytest
from click.testing import CliRunner

from conductor_core import ProviderAuthenticationError, ProviderRequestError
from conductor_core import cli as cli_module
from conductor_core.cli import cli


def _result(tmp_path, sample_loop, *, warnings=None, audio_path=None):
    generation_dir = tmp_path / "gen_abc"
    return SimpleNamespace(
        generation_id="abc",
        loop=sample_loop,
        midi_path=str(generation_dir / "loop.mid"),
        audio_path=audio_path,
        messages=[],
        cost=0.25,
        warnings=warnings or [],
        metadata=SimpleNamespace(
            provider="OpenAI",
            timestamp=datetime(2026, 1, 2, tzinfo=timezone.utc),
            soundfont=None,
            audio_path=audio_path,
            messages_path=str(generation_dir / "messages.json"),
        ),
    )


class FakeEngine:
    result = None
    error = None
    calls: ClassVar[list] = []

    def __init__(self, config):
        self.config = config

    def generate(self, request, progress_callback=None):
        self.__class__.calls.append((self.config, request, progress_callback))
        if progress_callback is not None:
            progress_callback(SimpleNamespace(message="Generating MIDI..."))
        if self.__class__.error is not None:
            raise self.__class__.error
        return self.__class__.result


@pytest.fixture(autouse=True)
def reset_fake_engine(monkeypatch):
    FakeEngine.result = None
    FakeEngine.error = None
    FakeEngine.calls = []
    monkeypatch.setattr(cli_module, "LoopGenerationEngine", FakeEngine)


def _required_args():
    return [
        "generate",
        "--key",
        "C",
        "--scale",
        "Major",
        "--description",
        "warm loop",
        "--model",
        "gpt-4o-mini",
    ]


def test_root_help_and_version_are_available():
    runner = CliRunner()

    help_result = runner.invoke(cli, ["--help"])
    version_result = runner.invoke(cli, ["--version"])

    assert help_result.exit_code == 0
    assert "generate" in help_result.output
    assert "models" in help_result.output
    assert "conductor, version 0.4.0" in version_result.output


def test_generate_help_has_all_public_controls_and_no_credentials():
    result = CliRunner().invoke(cli, ["generate", "--help"])

    assert result.exit_code == 0
    for option in (
        "--key",
        "--scale",
        "--description",
        "--model",
        "--temperature",
        "--use-thinking",
        "--effort",
        "--prompt",
        "--prompt-file",
        "--render-audio",
        "--soundfont-path",
        "--artifact-root",
        "--request-timeout",
        "--max-generations",
        "--json",
        "--quiet",
        "--verbose",
    ):
        assert option in result.output
    assert "api-key" not in result.output.casefold()


def test_generate_maps_every_request_and_engine_option(
    tmp_path, sample_loop, monkeypatch
):
    FakeEngine.result = _result(tmp_path, sample_loop)
    monkeypatch.setattr(cli_module, "_stderr_isatty", lambda: False)
    result = CliRunner().invoke(
        cli,
        [
            *_required_args(),
            "--temperature",
            "0.5",
            "--use-thinking",
            "--effort",
            "high",
            "--prompt",
            "custom prompt",
            "--render-audio",
            "--soundfont-path",
            str(tmp_path / "soundfont.sf2"),
            "--artifact-root",
            str(tmp_path / "artifacts"),
            "--request-timeout",
            "2.5",
            "--max-generations",
            "7",
            "--verbose",
        ],
    )

    assert result.exit_code == 0, result.output
    config, request, progress_callback = FakeEngine.calls[0]
    assert config.artifact_root == tmp_path / "artifacts"
    assert config.request_timeout == 2.5
    assert config.max_generations == 7
    assert request.key == "C"
    assert request.scale == "Major"
    assert request.description == "warm loop"
    assert request.model == "gpt-4o-mini"
    assert request.temperature == 0.5
    assert request.use_thinking is True
    assert request.effort == "high"
    assert request.prompt_override == "custom prompt"
    assert request.render_audio is True
    assert request.soundfont_path == tmp_path / "soundfont.sf2"
    assert progress_callback is not None
    assert "Generating MIDI..." in result.stderr
    assert "Generation: abc" in result.stdout


def test_generate_loads_utf8_prompt_file(tmp_path, sample_loop):
    FakeEngine.result = _result(tmp_path, sample_loop)
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("play softly \u266b", encoding="utf-8")

    result = CliRunner().invoke(
        cli, [*_required_args(), "--prompt-file", str(prompt_file), "--quiet"]
    )

    assert result.exit_code == 0
    assert FakeEngine.calls[0][1].prompt_override == "play softly \u266b"
    assert result.stdout == ""


def test_prompt_sources_are_mutually_exclusive_in_json_mode(tmp_path):
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("prompt", encoding="utf-8")

    result = CliRunner().invoke(
        cli,
        [
            *_required_args(),
            "--prompt",
            "inline",
            "--prompt-file",
            str(prompt_file),
            "--json",
        ],
    )

    assert result.exit_code == 2
    assert result.stdout == ""
    payload = json.loads(result.stderr)
    assert payload["error"]["code"] == "validation_error"


def test_json_parse_error_is_structured_and_stays_on_stderr():
    result = CliRunner().invoke(cli, ["generate", "--json"])

    assert result.exit_code == 2
    assert result.stdout == ""
    payload = json.loads(result.stderr)
    assert payload["schema_version"] == 1
    assert payload["error"]["code"] == "usage_error"


def test_json_generation_is_single_clean_envelope_and_ignores_quiet(
    tmp_path, sample_loop, monkeypatch
):
    FakeEngine.result = _result(
        tmp_path, sample_loop, warnings=["Audio rendering was skipped or failed."]
    )
    monkeypatch.setattr(cli_module, "_stderr_isatty", lambda: True)

    result = CliRunner().invoke(cli, [*_required_args(), "--json", "--quiet"])

    assert result.exit_code == 0
    assert result.stderr == ""
    payload = json.loads(result.stdout)
    assert payload["status"] == "success"
    assert payload["warnings"] == ["Audio rendering was skipped or failed."]
    assert payload["artifacts"]["audio"] is None
    assert len(FakeEngine.calls) == 1
    assert FakeEngine.calls[0][2] is None


@pytest.mark.parametrize(
    ("error", "expected_exit", "expected_code"),
    [
        (
            ProviderAuthenticationError("OpenAI", "missing key"),
            3,
            "configuration_error",
        ),
        (ImportError("Install an optional provider"), 3, "dependency_error"),
        (ProviderRequestError("OpenAI", "rejected"), 4, "generation_error"),
        (ValueError("Invalid Model Selected"), 2, "validation_error"),
        (ValueError("internal defect"), 1, "unexpected_error"),
    ],
)
def test_generate_maps_failures_to_json_errors(error, expected_exit, expected_code):
    FakeEngine.error = error

    result = CliRunner().invoke(cli, [*_required_args(), "--json"])

    assert result.exit_code == expected_exit
    assert result.stdout == ""
    assert json.loads(result.stderr)["error"]["code"] == expected_code


def test_debug_prints_traceback_only_for_unexpected_failures():
    FakeEngine.error = RuntimeError("internal defect")

    normal = CliRunner().invoke(cli, _required_args())
    debug = CliRunner().invoke(cli, ["--debug", *_required_args()])

    assert normal.exit_code == 1
    assert "Traceback" not in normal.stderr
    assert "Error: internal defect" in normal.stderr
    assert debug.exit_code == 1
    assert "Traceback" in debug.stderr
    assert "RuntimeError: internal defect" in debug.stderr


def test_models_is_offline_filterable_and_json_serializable(monkeypatch):
    monkeypatch.setattr(
        cli_module,
        "get_model_info",
        lambda: {
            "models": {
                "OpenAI": {
                    "model-a": {
                        "max_tokens": 10,
                        "rate_limits": {"RPM": 1, "TPM": None, "RPD": None},
                    }
                },
                "Google": {},
            }
        },
    )

    result = CliRunner().invoke(cli, ["models", "--provider", "OPENAI", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["models"][0]["provider"] == "OpenAI"
    assert payload["models"][0]["model"] == "model-a"
    assert FakeEngine.calls == []


def test_models_unknown_provider_is_usage_error():
    result = CliRunner().invoke(cli, ["models", "--provider", "Unknown"])

    assert result.exit_code == 2
    assert "Unknown provider" in result.stderr
