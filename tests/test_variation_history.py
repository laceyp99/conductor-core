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
    assert record.invalid_generation_ids == ()


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
    lookup = store.get_variation_history("batch-one")
    record = lookup.record

    assert record is not None
    assert record.manifest.generation_ids == ("one", "two")
    assert [item.id for item in record.generations] == ["two"]
    assert record.missing_generation_ids == ("one",)
    assert [(item.code, item.generation_id) for item in lookup.diagnostics] == [
        ("generation_missing", "one")
    ]
    listing = store.list_variation_history()
    assert listing.records == (record,)
    assert listing.diagnostics == lookup.diagnostics


def test_generation_retention_does_not_mutate_manifest(tmp_path, monkeypatch):
    store = storage.FilesystemArtifactStore(tmp_path / "generations", max_generations=2)
    first = _finalize(store, "one", monkeypatch)
    second = _finalize(store, "two", monkeypatch)
    store.save_variation_history(_metadata(), [first.id, second.id])
    manifest_path = tmp_path / "variations" / "batch_batch-one.json"
    original = manifest_path.read_bytes()

    _finalize(store, "three", monkeypatch)

    assert manifest_path.read_bytes() == original
    record = store.get_variation_history("batch-one").record
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

    assert [r.manifest.batch_id for r in store.list_variation_history().records] == [
        "newer",
        "older",
    ]
    assert store.delete_variation_history("newer") is True
    assert store.delete_variation_history("newer") is False
    assert store.clear_variation_history() == 1
    assert store.list_variation_history().records == ()
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
        listing = store.list_variation_history()

    assert listing.records == ()
    assert [(item.code, item.batch_id) for item in listing.diagnostics] == [
        ("manifest_invalid", "bad")
    ]

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

    assert storage.get_variation_history("batch-one").record is not None
    assert len(storage.list_variation_history().records) == 1
    assert storage.delete_variation_history("batch-one") is True
    assert storage.clear_variation_history() == 0


def test_lookup_distinguishes_absent_and_malformed_manifests_without_writes(tmp_path):
    store = storage.FilesystemArtifactStore(tmp_path / "generations")
    absent = store.get_variation_history("absent")
    empty = store.list_variation_history()

    assert absent.record is None
    assert [item.code for item in absent.diagnostics] == ["manifest_missing"]
    assert empty.records == empty.diagnostics == ()
    assert not (tmp_path / "variations").exists()
    assert not (tmp_path / "generations").exists()

    variations = tmp_path / "variations"
    variations.mkdir()
    malformed = variations / "batch_bad.json"
    malformed.write_text("{broken", encoding="utf-8")
    before = malformed.read_bytes()

    lookup = store.get_variation_history("bad")
    assert lookup.record is None
    assert [(item.code, item.batch_id) for item in lookup.diagnostics] == [
        ("manifest_invalid", "bad")
    ]
    assert malformed.read_bytes() == before
    assert [item.code for item in store.list_variation_history().diagnostics] == [
        "manifest_invalid"
    ]
    assert malformed.read_bytes() == before


def test_listing_keeps_valid_neighbors_and_reports_invalid_references(
    tmp_path, monkeypatch
):
    store = storage.FilesystemArtifactStore(
        tmp_path / "generations", max_generations=None
    )
    first = _finalize(store, "invalid", monkeypatch)
    second = _finalize(store, "missing", monkeypatch)
    store.save_variation_history(_metadata("good"), [first.id, second.id])
    metadata_path = tmp_path / "generations" / "gen_invalid" / "metadata.json"
    metadata_path.write_text("{}", encoding="utf-8")
    assert store.delete_generation("missing")
    bad_path = tmp_path / "variations" / "batch_bad.json"
    bad_path.write_text('{"schema_version": 2}', encoding="utf-8")
    before = {
        path.relative_to(tmp_path): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }

    lookup = store.get_variation_history("good")
    listing = store.list_variation_history()

    assert lookup.record is not None
    assert lookup.record.generations == ()
    assert lookup.record.missing_generation_ids == ("missing",)
    assert lookup.record.invalid_generation_ids == ("invalid",)
    assert {(item.code, item.generation_id) for item in lookup.diagnostics} == {
        ("generation_missing", "missing"),
        ("generation_invalid", "invalid"),
    }
    assert listing.records == (lookup.record,)
    assert {(item.code, item.batch_id) for item in listing.diagnostics} == {
        ("manifest_invalid", "bad"),
        ("generation_missing", "good"),
        ("generation_invalid", "good"),
    }
    assert before == {
        path.relative_to(tmp_path): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }


def test_listing_limit_tie_breaker_and_strict_validation(tmp_path):
    store = storage.FilesystemArtifactStore(tmp_path / "generations")
    variations = tmp_path / "variations"
    variations.mkdir()
    manifest = {
        **_metadata(),
        "schema_version": 1,
        "created_at": "2026-09-27T12:00:00",
        "generation_ids": ["missing-a", "missing-b"],
    }
    for index in range(101):
        batch_id = f"batch-{index:03d}"
        (variations / f"batch_{batch_id}.json").write_text(
            json.dumps({**manifest, "batch_id": batch_id}), encoding="utf-8"
        )

    default = store.list_variation_history()
    assert default.limit == 20
    assert len(default.records) == 20
    assert len(default.diagnostics) == 40
    assert [item.manifest.batch_id for item in default.records] == [
        f"batch-{index:03d}" for index in range(20)
    ]
    assert len(store.list_variation_history(100).records) == 100
    assert len(store.list_variation_history(1).records) == 1
    for invalid in (True, False, 0, -1, 101, 1.0, "2", None):
        with pytest.raises(ValueError, match="integer from 1 to 100"):
            store.list_variation_history(invalid)


def test_clear_variation_history_reaches_past_default_listing_limit(tmp_path):
    store = storage.FilesystemArtifactStore(tmp_path / "generations")
    variations = tmp_path / "variations"
    variations.mkdir()
    manifest = {
        **_metadata(),
        "schema_version": 1,
        "created_at": "2026-09-27T12:00:00",
        "generation_ids": ["missing-a", "missing-b"],
    }
    for index in range(21):
        batch_id = f"batch-{index:03d}"
        (variations / f"batch_{batch_id}.json").write_text(
            json.dumps({**manifest, "batch_id": batch_id}), encoding="utf-8"
        )

    assert store.clear_variation_history() == 21
    assert store.list_variation_history().records == ()


def test_listing_orders_extreme_and_timezone_aware_dates(tmp_path):
    store = storage.FilesystemArtifactStore(tmp_path / "generations")
    variations = tmp_path / "variations"
    variations.mkdir()
    manifest = {
        **_metadata(),
        "schema_version": 1,
        "generation_ids": ["missing-a", "missing-b"],
    }
    dates = {
        "oldest": "0001-01-01T00:00:00",
        "middle": "2026-09-27T12:00:00+02:00",
        "newest": "9999-12-31T23:59:59",
    }
    for batch_id, created_at in dates.items():
        (variations / f"batch_{batch_id}.json").write_text(
            json.dumps({**manifest, "batch_id": batch_id, "created_at": created_at}),
            encoding="utf-8",
        )

    assert [
        record.manifest.batch_id for record in store.list_variation_history().records
    ] == [
        "newest",
        "middle",
        "oldest",
    ]
