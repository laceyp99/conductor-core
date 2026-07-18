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
        return output_path

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
        )
    )
    result = engine.generate(
        GenerationRequest(
            key="C",
            scale="Major",
            description="warm rhodes loop",
            model="gpt-4o-mini",
            temperature=0.3,
            render_audio=True,
            soundfont_path="custom.sf2",
        ),
        progress_callback=progress_events.append,
    )

    generation_dir = tmp_path / "generations" / f"gen_{result.generation_id}"

    assert result.midi_path == str(generation_dir / "loop.mid")
    assert result.audio_path == str(generation_dir / "loop.mp3")
    assert result.metadata.soundfont == "custom.sf2"
    assert Path(result.midi_path).exists()
    assert Path(result.audio_path).read_bytes() == b"audio"
    assert json.loads((generation_dir / "messages.json").read_text(encoding="utf-8")) == [
        {"role": "user", "content": "prompt"}
    ]
    assert captured["model_choice"] == "gpt-4o-mini"
    assert captured["prompt"] == "C Major warm rhodes loop."
    assert captured["provider_credentials"].openai_api_key == "openai-key"
    assert captured["system_prompt"] == "config prompt"
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
    default_soundfont = tmp_path / "FM-Piano1 20190916.sf2"

    def fake_midi_to_mp3(midi_path, output_path=None, soundfont_name=None):
        Path(output_path).write_bytes(b"audio")
        captured["soundfont_name"] = soundfont_name
        return output_path

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


def test_engine_ignores_deprecated_provider_and_records_actual_route(
    monkeypatch,
    tmp_path,
    sample_loop,
):
    monkeypatch.setattr(
        engine_module.routing,
        "generate_midi",
        lambda **kwargs: (sample_loop, [], 0, "OpenAI"),
    )

    engine = LoopGenerationEngine(
        EngineConfig.from_defaults(artifact_root=tmp_path / "generations")
    )

    with pytest.warns(DeprecationWarning, match="provider is deprecated and ignored"):
        request = GenerationRequest(
            key="C",
            scale="Major",
            description="provider conflict",
            model="gpt-4o-mini",
            provider="Anthropic",
        )

    result = engine.generate(request)

    assert result.metadata.provider == "OpenAI"
