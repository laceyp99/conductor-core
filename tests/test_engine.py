import json
from pathlib import Path

import pytest

from conductor_core import (
    EngineConfig,
    GenerationRequest,
    LoopGenerationEngine,
    ProviderCredentials,
)
from conductor_core import engine as engine_module


def test_engine_generates_persisted_artifacts_with_mocked_provider(
    monkeypatch,
    tmp_path,
    sample_loop,
):
    captured = {}
    progress_events = []

    def fake_generate_midi(**kwargs):
        captured.update(kwargs)
        return sample_loop, [{"role": "user", "content": "prompt"}], 0.25, "OpenAI"

    def fake_midi_to_mp3(midi_path, output_path=None, soundfont_name=None):
        Path(output_path).write_bytes(b"audio")
        captured["audio"] = {
            "midi_path": midi_path,
            "output_path": output_path,
            "soundfont_name": soundfont_name,
        }
        return engine_module.playback.MidiToMp3Result(path=output_path, error=None)

    soundfont_path = tmp_path / "custom.sf2"
    monkeypatch.setattr(engine_module.routing, "generate_midi", fake_generate_midi)
    monkeypatch.setattr(engine_module.playback, "midi_to_mp3", fake_midi_to_mp3)
    monkeypatch.setattr(
        engine_module.playback,
        "resolve_soundfont",
        lambda soundfont_name: str(soundfont_path),
    )
    monkeypatch.setattr(engine_module.music, "get_loop_prompt", lambda: "default prompt")

    engine = LoopGenerationEngine(
        EngineConfig.from_defaults(
            artifact_root=tmp_path / "generations",
            provider_credentials=ProviderCredentials(openai_api_key="openai-key"),
            prompt_override="config prompt",
            request_timeout=2.5,
        )
    )
    result = engine.generate(
        GenerationRequest(
            key="C",
            scale="Major",
            description="warm rhodes loop",
            model="gpt-4o-mini",
            temperature=0.3,
            use_thinking=False,
            effort="medium",
            render_audio=True,
            soundfont_path="custom.sf2",
        ),
        progress_callback=progress_events.append,
    )

    generation_dir = tmp_path / "generations" / f"gen_{result.generation_id}"
    metadata_json = json.loads((generation_dir / "metadata.json").read_text(encoding="utf-8"))
    loaded_metadata = engine.store.get_generation(result.generation_id)

    assert result.midi_path == str(generation_dir / "loop.mid")
    assert result.audio_path == str(generation_dir / "loop.mp3")
    assert result.metadata.soundfont == "custom.sf2"
    assert result.metadata.use_thinking is False
    assert result.metadata.effort == "medium"
    assert metadata_json["use_thinking"] is False
    assert metadata_json["effort"] == "medium"
    assert loaded_metadata is not None
    assert loaded_metadata.use_thinking is False
    assert loaded_metadata.effort == "medium"
    assert Path(result.midi_path).exists()
    assert Path(result.audio_path).read_bytes() == b"audio"
    assert json.loads((generation_dir / "messages.json").read_text(encoding="utf-8")) == [
        {"role": "user", "content": "prompt"}
    ]
    assert captured["model_choice"] == "gpt-4o-mini"
    assert captured["prompt"] == "C Major warm rhodes loop."
    assert captured["provider_credentials"].openai_api_key == "openai-key"
    assert captured["request_timeout"] == 2.5
    assert captured["system_prompt"] == "config prompt"
    assert captured["use_thinking"] is False
    assert captured["effort"] == "medium"
    assert captured["_return_provider"] is True
    assert captured["audio"]["soundfont_name"] == str(soundfont_path)
    assert [event.stage for event in progress_events] == [
        "provider_call",
        "midi",
        "audio",
    ]


def test_engine_records_resolved_default_soundfont(
    monkeypatch,
    tmp_path,
    sample_loop,
):
    captured = {}
    default_soundfont = tmp_path / "FM-Piano1-20190916.sf2"

    def fake_midi_to_mp3(midi_path, output_path=None, soundfont_name=None):
        Path(output_path).write_bytes(b"audio")
        captured["soundfont_name"] = soundfont_name
        return engine_module.playback.MidiToMp3Result(path=output_path, error=None)

    def fake_resolve_soundfont(soundfont_name):
        captured["requested_soundfont"] = soundfont_name
        return str(default_soundfont) if soundfont_name is None else None

    monkeypatch.setattr(
        engine_module.routing,
        "generate_midi",
        lambda **kwargs: (sample_loop, [], 0.25, "OpenAI"),
    )
    monkeypatch.setattr(engine_module.playback, "midi_to_mp3", fake_midi_to_mp3)
    monkeypatch.setattr(
        engine_module.playback,
        "resolve_soundfont",
        fake_resolve_soundfont,
    )

    engine = LoopGenerationEngine(
        EngineConfig.from_defaults(artifact_root=tmp_path / "generations")
    )
    result = engine.generate(
        GenerationRequest(
            key="C",
            scale="Major",
            description="warm rhodes loop",
            model="gpt-4o-mini",
            render_audio=True,
        )
    )

    assert captured["requested_soundfont"] is None
    assert captured["soundfont_name"] == str(default_soundfont)
    assert result.metadata.soundfont == default_soundfont.name


def test_engine_discards_partial_audio_when_renderer_reports_failure(
    monkeypatch,
    tmp_path,
    sample_loop,
):
    def failed_render(midi_path, output_path=None, soundfont_name=None):
        Path(output_path).write_bytes(b"partial")
        return engine_module.playback.MidiToMp3Result(
            path=None,
            error="RuntimeError: FluidSynth exited without producing audio",
        )

    monkeypatch.setattr(
        engine_module.routing,
        "generate_midi",
        lambda **kwargs: (sample_loop, [], 0.25, "OpenAI"),
    )
    monkeypatch.setattr(engine_module.playback, "midi_to_mp3", failed_render)
    monkeypatch.setattr(
        engine_module.playback,
        "resolve_soundfont",
        lambda soundfont_name: str(tmp_path / "custom.sf2"),
    )

    engine = LoopGenerationEngine(
        EngineConfig.from_defaults(artifact_root=tmp_path / "generations")
    )
    result = engine.generate(
        GenerationRequest(
            key="C",
            scale="Major",
            description="warm rhodes loop",
            model="gpt-4o-mini",
            render_audio=True,
        )
    )

    generation_dir = tmp_path / "generations" / f"gen_{result.generation_id}"
    assert result.warnings == [
        "Audio rendering was skipped or failed. "
        "RuntimeError: FluidSynth exited without producing audio"
    ]
    assert result.audio_path is None
    assert result.metadata.audio_path is None
    assert result.metadata.soundfont is None
    assert not (generation_dir / "loop.mp3").exists()


def test_engine_preserves_midi_when_soundfont_resolution_raises(
    monkeypatch,
    tmp_path,
    sample_loop,
):
    def fail_resolution(soundfont_name):
        raise OSError("resource extraction failed")

    monkeypatch.setattr(
        engine_module.routing,
        "generate_midi",
        lambda **kwargs: (sample_loop, [], 0.25, "OpenAI"),
    )
    monkeypatch.setattr(
        engine_module.playback,
        "resolve_soundfont",
        fail_resolution,
    )
    monkeypatch.setattr(
        engine_module.playback,
        "midi_to_mp3",
        lambda *args, **kwargs: pytest.fail("Rendering should not start without a SoundFont"),
    )

    engine = LoopGenerationEngine(
        EngineConfig.from_defaults(artifact_root=tmp_path / "generations")
    )
    result = engine.generate(
        GenerationRequest(
            key="C",
            scale="Major",
            description="warm rhodes loop",
            model="gpt-4o-mini",
            render_audio=True,
        )
    )

    assert result.warnings == [
        "Audio rendering was skipped or failed. OSError: resource extraction failed"
    ]
    assert Path(result.midi_path).exists()
    assert result.audio_path is None
    assert result.metadata.soundfont is None


def test_engine_surfaces_warning_for_defensively_dropped_pitch(
    monkeypatch,
    tmp_path,
    sample_loop,
):
    invalid_note = sample_loop.Bar_1.notes[0].model_copy(update={"octave": 12})
    invalid_bar = sample_loop.Bar_1.model_copy(update={"notes": [invalid_note]})
    unvalidated_loop = sample_loop.model_copy(update={"Bar_1": invalid_bar})
    monkeypatch.setattr(
        engine_module.routing,
        "generate_midi",
        lambda **kwargs: (unvalidated_loop, [], 0.25, "OpenAI"),
    )

    engine = LoopGenerationEngine(
        EngineConfig.from_defaults(artifact_root=tmp_path / "generations")
    )
    result = engine.generate(
        GenerationRequest(
            key="C",
            scale="Major",
            description="loop with a hallucinated pitch",
            model="gpt-4o-mini",
        )
    )

    assert result.warnings == ["Dropped out-of-range MIDI note C12 (156); valid range is 0-127."]


@pytest.mark.parametrize("max_generations", [None, 1])
def test_engine_config_passes_storage_limit_to_default_store(tmp_path, max_generations):
    engine = LoopGenerationEngine(
        EngineConfig.from_defaults(
            artifact_root=tmp_path / "generations",
            max_generations=max_generations,
        )
    )

    assert engine.store.max_generations == max_generations


@pytest.mark.parametrize("max_generations", [0, -1])
def test_engine_config_rejects_non_positive_storage_limit(max_generations):
    with pytest.raises(ValueError, match="max_generations must be None or a positive integer"):
        EngineConfig.from_defaults(max_generations=max_generations)


def test_engine_cleans_unfinalized_workspace_when_processing_fails(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        engine_module.routing,
        "generate_midi",
        lambda **kwargs: (None, [], 0, "OpenAI"),
    )

    engine = LoopGenerationEngine(
        EngineConfig.from_defaults(artifact_root=tmp_path / "generations")
    )

    with pytest.raises(ValueError, match="loop object is None"):
        engine.generate(
            GenerationRequest(
                key="C",
                scale="Major",
                description="broken loop",
                model="gpt-4o-mini",
            )
        )

    generations_root = tmp_path / "generations"
    assert not generations_root.exists() or list(generations_root.iterdir()) == []


def test_generation_request_rejects_removed_provider():
    with pytest.raises(TypeError, match="unexpected keyword argument 'provider'"):
        GenerationRequest(
            key="C",
            scale="Major",
            description="provider conflict",
            model="gpt-4o-mini",
            provider="Anthropic",
        )


@pytest.mark.parametrize("key", ["H", "Cbbb", "c"])
def test_generation_request_rejects_unknown_keys(key):
    with pytest.raises(
        ValueError,
        match=rf"Invalid key {key!r}\. Expected one of: B#, C, Dbb",
    ):
        GenerationRequest(
            key=key,
            scale="Major",
            description="invalid key",
            model="gpt-4o-mini",
        )


@pytest.mark.parametrize("scale", ["major", "MINOR", "Harmonic Minor", "melodic minor"])
def test_generation_request_accepts_known_scales_case_insensitively(scale):
    request = GenerationRequest(
        key="C",
        scale=scale,
        description="valid scale",
        model="gpt-4o-mini",
    )

    assert request.scale == scale


@pytest.mark.parametrize("scale", ["dorian", "", None])
def test_generation_request_rejects_unknown_scales(scale):
    expected_message = (
        f"Invalid scale {scale!r}. Expected one of: "
        "major, minor, harmonic minor, melodic minor (case-insensitive)"
    )
    with pytest.raises(ValueError, match="Invalid scale") as raised:
        GenerationRequest(
            key="C",
            scale=scale,
            description="invalid scale",
            model="gpt-4o-mini",
        )
    assert str(raised.value) == expected_message


@pytest.mark.parametrize("temperature", [-0.1, 2.1, float("inf"), "0.5", True])
def test_generation_request_rejects_invalid_temperatures(temperature):
    expected_message = (
        f"Invalid temperature {temperature!r}. "
        "Expected a finite number between 0.0 and 2.0 (inclusive)"
    )
    with pytest.raises(ValueError, match="Invalid temperature") as raised:
        GenerationRequest(
            key="C",
            scale="Major",
            description="invalid temperature",
            model="gpt-4o-mini",
            temperature=temperature,
        )
    assert str(raised.value) == expected_message


@pytest.mark.parametrize("temperature", [0.0, 0.5, 2.0])
def test_generation_request_accepts_temperature_in_range(temperature):
    request = GenerationRequest(
        key="C",
        scale="Major",
        description="valid temperature",
        model="gpt-4o-mini",
        temperature=temperature,
    )

    assert request.temperature == temperature
