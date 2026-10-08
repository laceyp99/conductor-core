from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from conductor_core import (
    EngineConfig,
    LoopGenerationEngine,
    VariationGenerationRequest,
    VariationUsage,
    storage,
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


@pytest.mark.parametrize(("count", "limit"), [(2, 1), (4, 2)])
@pytest.mark.parametrize("inject_store", [False, True])
def test_oversized_batch_is_rejected_before_provider_or_storage(
    monkeypatch, tmp_path, count, limit, inject_store
):
    events = []
    root = tmp_path / "generations"
    engine = LoopGenerationEngine(
        EngineConfig(artifact_root=root, max_generations=20 if inject_store else limit),
        store=storage.FilesystemArtifactStore(root, max_generations=limit)
        if inject_store
        else None,
    )
    provider = Mock()
    allocate = Mock()
    save_manifest = Mock()
    monkeypatch.setattr(engine_module.routing, "generate_variations", provider)
    monkeypatch.setattr(engine.store, "create_generation_workspace", allocate)
    monkeypatch.setattr(engine.store, "save_variation_history", save_manifest)

    with pytest.raises(ValueError, match=f"Requested {count} variations") as error:
        engine.generate_variations(
            request(count=count), progress_callback=events.append
        )

    assert str(error.value) == (
        f"Requested {count} variations, but this store retains at most {limit} "
        f"generations. Set max_generations >= {count} or None."
    )
    provider.assert_not_called()
    allocate.assert_not_called()
    save_manifest.assert_not_called()
    assert not root.exists()
    assert not (tmp_path / "variations").exists()
    assert [event.status for event in events] == ["started", "failed"]
    assert events[0].batch_id is not None
    assert events[0].batch_id == events[1].batch_id
    assert all(event.variation_index is None for event in events)


@pytest.mark.parametrize("clock_order", ["forward", "backward", "same"])
@pytest.mark.parametrize("limit", [2, 3, None])
def test_allowed_batch_survives_retention_of_older_generations(
    monkeypatch, tmp_path, sample_loop, limit, clock_order
):
    # With fixed IDs, each batch uses two finalization times and one manifest time.
    # Keep older IDs higher for ties so timestamp and ID sorting favor old history.
    first = datetime(2026, 1, 2)
    delta = timedelta(days=1 if clock_order == "forward" else -1)
    second = first if clock_order == "same" else first + delta
    timestamps = iter(
        [first + timedelta(seconds=index) for index in range(3)]
        + [second + timedelta(seconds=index) for index in range(3)]
    )
    generation_ids = iter(("old-a", "old-b", "new-a", "new-b"))
    monkeypatch.setattr(storage, "_generate_id", lambda: next(generation_ids))

    class SteppingDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            timestamp = next(timestamps)
            return timestamp.astimezone(tz) if tz is not None else timestamp

    monkeypatch.setattr(storage, "datetime", SteppingDatetime)
    provider = Mock(return_value=provider_result((sample_loop, sample_loop)))
    monkeypatch.setattr(engine_module.routing, "generate_variations", provider)
    store = storage.FilesystemArtifactStore(
        tmp_path / "generations", max_generations=limit
    )
    engine = LoopGenerationEngine(
        EngineConfig(artifact_root=tmp_path / "unused-config-root", max_generations=1),
        store=store,
    )
    older = engine.generate_variations(request())

    result = engine.generate_variations(request())

    assert provider.call_count == 2
    assert result.status == "complete"
    assert len(result.items) == 2
    assert all(Path(item.generation.midi_path).is_file() for item in result.items)
    lookup = store.get_variation_history(result.metadata.batch_id)
    assert lookup.record is not None
    assert lookup.diagnostics == ()
    assert lookup.record.manifest.generation_ids == tuple(
        item.generation.id for item in result.items
    )
    assert lookup.record.missing_generation_ids == ()
    assert lookup.record.invalid_generation_ids == ()

    expected_older = 2 if limit is None else limit - 2
    assert [Path(item.generation.midi_path).exists() for item in older.items] == (
        [False] * (2 - expected_older) + [True] * expected_older
    )
    assert len(store.load_history()) == 2 + expected_older
    assert not (tmp_path / "unused-config-root").exists()

    # Protection ends with this batch. An ordinary later finalization can prune
    # the batch according to timestamps without changing its saved manifest.
    monkeypatch.setattr(storage, "_generate_id", lambda: "later")
    timestamps = iter([datetime(2026, 1, 4)])
    workspace = store.create_generation_workspace()
    Path(workspace.midi_path).write_bytes(b"midi")
    store.finalize_generation(
        workspace, "later", "C", "Major", "test-model", "OpenAI", 0.5
    )
    assert len(store.load_history()) == (5 if limit is None else limit)
    assert store.get_variation_history(result.metadata.batch_id).record is not None
    if limit == 2:
        assert any(
            not Path(item.generation.midi_path).exists() for item in result.items
        )


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

    record = engine.store.get_variation_history(result.metadata.batch_id).record
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
