from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from conductor_core import (
    EngineConfig,
    LoopGenerationEngine,
    VariationGenerationRequest,
    VariationUsage,
)
from conductor_core import engine as engine_module


def request(**overrides):
    values = {
        "key": "C",
        "scale": "Major",
        "description": "warm keys",
        "model": "test-model",
        "count": 2,
    }
    values.update(overrides)
    return VariationGenerationRequest(**values)


def provider_result(variations, **overrides):
    values = {
        "variations": tuple(variations),
        "provider": "OpenAI",
        "messages": [{"role": "user", "content": "shared"}],
        "usage": VariationUsage(input_tokens=10, output_tokens=20, total_tokens=30),
        "cost": 0.5,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_generate_variations_persists_ordered_children_and_one_manifest(
    monkeypatch, tmp_path, sample_loop
):
    captured = {}
    events = []

    def fake_generate_variations(**kwargs):
        captured.update(kwargs)
        return provider_result((sample_loop, sample_loop))

    monkeypatch.setattr(
        engine_module.routing, "generate_variations", fake_generate_variations
    )
    engine = LoopGenerationEngine(
        EngineConfig.from_defaults(
            artifact_root=tmp_path / "generations",
            prompt_override="shared override",
            request_timeout=3.0,
        )
    )

    result = engine.generate_variations(request(), progress_callback=events.append)

    assert result.status == "complete"
    assert [item.index for item in result.items] == [0, 1]
    assert all(item.generation.cost is None for item in result.items)
    assert captured["count"] == 2
    assert captured["system_prompt"] == "shared override"
    assert captured["request_timeout"] == 3.0
    assert result.metadata.prompt_version == "override"
    assert result.metadata.cost == 0.5
    assert result.metadata.usage.total_tokens == 30
    assert [event.status for event in events] == [
        "started",
        "persisting",
        "complete",
        "persisting",
        "complete",
        "complete",
    ]
    assert all(event.batch_id == result.metadata.batch_id for event in events)
    assert [event.variation_index for event in events] == [None, 0, 0, 1, 1, None]

    generation_dirs = sorted((tmp_path / "generations").glob("gen_*"))
    assert len(generation_dirs) == 2
    assert all(
        not (directory / "messages.json").exists() for directory in generation_dirs
    )
    assert len(list((tmp_path / "variations").glob("*.json"))) == 1

    record = engine.store.get_variation_history(result.metadata.batch_id)
    assert record is not None
    assert list(record.manifest.messages) == result.metadata.messages
    assert record.manifest.generation_ids == tuple(
        item.generation.id for item in result.items
    )


def test_request_prompt_override_precedes_config(monkeypatch, tmp_path, sample_loop):
    captured = {}
    monkeypatch.setattr(
        engine_module.routing,
        "generate_variations",
        lambda **kwargs: (
            captured.update(kwargs) or provider_result((sample_loop, sample_loop))
        ),
    )
    engine = LoopGenerationEngine(
        EngineConfig.from_defaults(
            artifact_root=tmp_path / "generations", prompt_override="config prompt"
        )
    )
    result = engine.generate_variations(request(prompt_override="request prompt"))
    assert captured["system_prompt"] == "request prompt"
    assert result.metadata.prompt_version == "override"


def test_packaged_prompt_and_version_are_used_without_override(
    monkeypatch, tmp_path, sample_loop
):
    captured = {}
    monkeypatch.setattr(engine_module.music, "get_variation_prompt", lambda: "packaged")
    monkeypatch.setattr(
        engine_module.routing,
        "generate_variations",
        lambda **kwargs: (
            captured.update(kwargs) or provider_result((sample_loop, sample_loop))
        ),
    )
    engine = LoopGenerationEngine(
        EngineConfig.from_defaults(artifact_root=tmp_path / "generations")
    )
    result = engine.generate_variations(request())
    assert captured["system_prompt"] == "packaged"
    assert result.metadata.prompt_version == "variation_gen_v1"


def test_wrong_count_returns_failure_without_artifacts_or_manifest(
    monkeypatch, tmp_path, sample_loop
):
    events = []
    monkeypatch.setattr(
        engine_module.routing,
        "generate_variations",
        lambda **kwargs: provider_result((sample_loop,)),
    )
    engine = LoopGenerationEngine(
        EngineConfig.from_defaults(artifact_root=tmp_path / "generations")
    )
    result = engine.generate_variations(request(), progress_callback=events.append)
    assert result.status == "failed"
    assert result.diagnostic.code == "wrong_count"
    assert result.items == ()
    assert [event.status for event in events] == ["started", "failed"]
    assert not (tmp_path / "generations").exists()
    assert not (tmp_path / "variations").exists()


def test_collection_validation_error_creates_no_artifacts(monkeypatch, tmp_path):
    events = []

    def fail_validation(**kwargs):
        raise ValidationError.from_exception_data("VariationCollection", [])

    monkeypatch.setattr(engine_module.routing, "generate_variations", fail_validation)
    engine = LoopGenerationEngine(
        EngineConfig.from_defaults(artifact_root=tmp_path / "generations")
    )
    with pytest.raises(ValidationError):
        engine.generate_variations(request(), progress_callback=events.append)
    assert [event.status for event in events] == ["started", "failed"]
    assert not (tmp_path / "generations").exists()
    assert not (tmp_path / "variations").exists()


def test_late_persistence_failure_preserves_finalized_predecessor(
    monkeypatch, tmp_path, sample_loop
):
    monkeypatch.setattr(
        engine_module.routing,
        "generate_variations",
        lambda **kwargs: provider_result((sample_loop, sample_loop)),
    )
    engine = LoopGenerationEngine(
        EngineConfig.from_defaults(artifact_root=tmp_path / "generations")
    )
    finalize = engine.store.finalize_generation
    calls = 0

    def fail_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("disk full")
        return finalize(*args, **kwargs)

    monkeypatch.setattr(engine.store, "finalize_generation", fail_second)
    with pytest.raises(OSError, match="disk full"):
        engine.generate_variations(request())

    generation_dirs = list((tmp_path / "generations").glob("gen_*"))
    assert len(generation_dirs) == 1
    assert (generation_dirs[0] / "metadata.json").exists()
    assert not (tmp_path / "variations").exists()
