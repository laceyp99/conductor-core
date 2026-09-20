import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from conductor_core import storage


def _finalize(store: storage.FilesystemArtifactStore, gen_id: str, monkeypatch):
    monkeypatch.setattr(storage, "_generate_id", lambda: gen_id)
    workspace = store.create_generation_workspace()
    Path(workspace.midi_path).write_bytes(b"midi")
    return store.finalize_generation(
        workspace=workspace,
        prompt=f"prompt {gen_id}",
        key="C",
        scale="major",
        model="model",
        provider="provider",
        temperature=0.2,
    )


def _metadata(batch_id="batch-one"):
    return {
        "batch_id": batch_id,
        "model": "model",
        "provider": "provider",
        "prompt_version": "variation_gen_v1",
        "requested_count": 2,
        "received_count": 2,
        "messages": [{"role": "user", "content": "Unicode: café 🎵"}],
        "usage": {"input_tokens": 12, "output_tokens": None},
        "cost": None,
    }


def test_save_manifest_is_versioned_atomic_and_contains_only_batch_index(
    tmp_path, monkeypatch
):
    generation_root = tmp_path / "core" / "generations"
    store = storage.FilesystemArtifactStore(generation_root, max_generations=None)
    first = _finalize(store, "one", monkeypatch)
    second = _finalize(store, "two", monkeypatch)

    record = store.save_variation_history(_metadata(), [first.id, second.id])

    manifest_path = generation_root.parent / "variations" / "batch_batch-one.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["generation_ids"] == ["one", "two"]
    assert payload["messages"] == _metadata()["messages"]
    assert payload["usage"] == {
        "input_tokens": 12,
        "output_tokens": None,
        "total_tokens": None,
    }
    assert payload["cost"] is None
    assert "loop" not in payload
    assert "midi_path" not in payload
    assert not list(manifest_path.parent.glob(".variation-*.tmp"))
    assert [item.id for item in record.generations] == ["one", "two"]
    assert record.missing_generation_ids == ()


def test_loading_preserves_order_and_explicitly_reports_deleted_generations(
    tmp_path, monkeypatch
):
    store = storage.FilesystemArtifactStore(
        tmp_path / "generations", max_generations=None
    )
    first = _finalize(store, "one", monkeypatch)
    second = _finalize(store, "two", monkeypatch)
    store.save_variation_history(_metadata(), [first.id, second.id])

    assert store.delete_generation(first.id) is True
    record = store.get_variation_history("batch-one")

    assert record is not None
    assert record.manifest.generation_ids == ("one", "two")
    assert [item.id for item in record.generations] == ["two"]
    assert record.missing_generation_ids == ("one",)
    assert store.list_variation_history() == [record]


def test_generation_retention_does_not_mutate_manifest(tmp_path, monkeypatch):
    store = storage.FilesystemArtifactStore(tmp_path / "generations", max_generations=2)
    first = _finalize(store, "one", monkeypatch)
    second = _finalize(store, "two", monkeypatch)
    store.save_variation_history(_metadata(), [first.id, second.id])
    manifest_path = tmp_path / "variations" / "batch_batch-one.json"
    original = manifest_path.read_bytes()

    _finalize(store, "three", monkeypatch)

    assert manifest_path.read_bytes() == original
    record = store.get_variation_history("batch-one")
    assert record is not None
    assert record.manifest.generation_ids == ("one", "two")
    assert record.missing_generation_ids == ("one",)


def test_manifest_lifecycle_never_deletes_generation_artifacts(tmp_path, monkeypatch):
    store = storage.FilesystemArtifactStore(
        tmp_path / "generations", max_generations=None
    )
    generations = [_finalize(store, value, monkeypatch) for value in ("one", "two")]
    store.save_variation_history(_metadata("older"), [item.id for item in generations])
    store.save_variation_history(_metadata("newer"), [item.id for item in generations])
    older_path = tmp_path / "variations" / "batch_older.json"
    newer_path = tmp_path / "variations" / "batch_newer.json"
    older_payload = json.loads(older_path.read_text(encoding="utf-8"))
    newer_payload = json.loads(newer_path.read_text(encoding="utf-8"))
    older_payload["created_at"] = (datetime.now() - timedelta(days=1)).isoformat()
    newer_payload["created_at"] = datetime.now().isoformat()
    older_path.write_text(json.dumps(older_payload), encoding="utf-8")
    newer_path.write_text(json.dumps(newer_payload), encoding="utf-8")

    assert [r.manifest.batch_id for r in store.list_variation_history()] == [
        "newer",
        "older",
    ]
    assert store.delete_variation_history("newer") is True
    assert store.delete_variation_history("newer") is False
    assert store.clear_variation_history() == 1
    assert store.list_variation_history() == []
    assert [item.id for item in store.load_history()] == ["two", "one"]


def test_save_preserves_missing_ids_and_rejects_duplicate_wrong_count_and_unsafe_ids(
    tmp_path, monkeypatch
):
    store = storage.FilesystemArtifactStore(
        tmp_path / "generations", max_generations=None
    )
    first = _finalize(store, "one", monkeypatch)
    second = _finalize(store, "two", monkeypatch)

    record = store.save_variation_history(_metadata(), [first.id, "missing"])
    assert record.manifest.generation_ids == (first.id, "missing")
    assert record.missing_generation_ids == ("missing",)
    assert store.delete_variation_history("batch-one") is True
    with pytest.raises(ValueError, match="unique"):
        store.save_variation_history(_metadata(), [first.id, first.id])
    with pytest.raises(ValueError, match="count must match"):
        store.save_variation_history(
            {**_metadata(), "requested_count": 3}, [first.id, second.id]
        )
    with pytest.raises(ValueError, match="path component"):
        store.get_variation_history("../escape")

    assert list((tmp_path / "variations").glob("*.json")) == []


def test_strict_manifest_loader_skips_unknown_versions_and_extra_fields(
    tmp_path, caplog
):
    variations = tmp_path / "variations"
    variations.mkdir()
    base = {
        **_metadata(),
        "schema_version": 2,
        "created_at": datetime.now().isoformat(),
        "generation_ids": ["one", "two"],
        "unexpected": True,
    }
    (variations / "batch_bad.json").write_text(json.dumps(base), encoding="utf-8")
    store = storage.FilesystemArtifactStore(tmp_path / "generations")

    with caplog.at_level("ERROR", logger="conductor_core.storage"):
        assert store.list_variation_history() == []

    assert any(
        "Failed to load variation batch bad" in item.message for item in caplog.records
    )


def test_module_level_variation_helpers_use_default_sibling_directory(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("CONDUCTOR_CORE_DATA_DIR", str(tmp_path))
    generated = [
        _finalize(storage.FilesystemArtifactStore(), value, monkeypatch)
        for value in ("one", "two")
    ]

    storage.save_variation_history(_metadata(), [item.id for item in generated])

    assert storage.get_variation_history("batch-one") is not None
    assert len(storage.list_variation_history()) == 1
    assert storage.delete_variation_history("batch-one") is True
    assert storage.clear_variation_history() == 0
