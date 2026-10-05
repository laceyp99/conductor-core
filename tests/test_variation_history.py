import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from conductor_core import list_variation_history, storage


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
    assert datetime.fromisoformat(
        payload["created_at"].replace("Z", "+00:00")
    ).utcoffset() == timedelta(0)
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
    older_payload["created_at"] = (
        datetime.now(timezone.utc) - timedelta(days=1)
    ).isoformat()
    newer_payload["created_at"] = datetime.now(timezone.utc).isoformat()
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


def test_listing_returns_all_batches_and_diagnostics_in_deterministic_order(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("CONDUCTOR_CORE_DATA_DIR", str(tmp_path))
    store = storage.FilesystemArtifactStore(tmp_path / "generations")
    variations = tmp_path / "variations"
    variations.mkdir()
    manifest = {
        **_metadata(),
        "schema_version": 1,
        "created_at": "2026-09-27T12:00:00+00:00",
        "generation_ids": ["missing-a", "missing-b"],
    }
    for index in range(101):
        batch_id = f"batch-{index:03d}"
        (variations / f"batch_{batch_id}.json").write_text(
            json.dumps({**manifest, "batch_id": batch_id}), encoding="utf-8"
        )

    listing = store.list_variation_history()
    assert len(listing.records) == 101
    assert len(listing.diagnostics) == 202
    assert list_variation_history() == listing
    assert [item.manifest.batch_id for item in listing.records] == [
        f"batch-{index:03d}" for index in range(101)
    ]
    assert [
        (item.code, item.batch_id, item.generation_id) for item in listing.diagnostics
    ] == [
        ("generation_missing", f"batch-{index:03d}", generation_id)
        for index in range(101)
        for generation_id in ("missing-a", "missing-b")
    ]
    with pytest.raises(TypeError):
        store.list_variation_history(limit=1)
    with pytest.raises(TypeError):
        list_variation_history(limit=1)


def test_clear_variation_history_reaches_all_batches(tmp_path):
    store = storage.FilesystemArtifactStore(tmp_path / "generations")
    variations = tmp_path / "variations"
    variations.mkdir()
    manifest = {
        **_metadata(),
        "schema_version": 1,
        "created_at": "2026-09-27T12:00:00+00:00",
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
        "oldest": "0001-01-01T00:00:00+00:00",
        "middle": "2026-09-27T12:00:00+02:00",
        "newest": "9999-12-31T23:59:59+00:00",
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


def test_listing_uses_file_time_for_legacy_naive_manifest(tmp_path):
    store = storage.FilesystemArtifactStore(tmp_path / "generations")
    variations = tmp_path / "variations"
    variations.mkdir()
    manifest = {
        **_metadata(),
        "schema_version": 1,
        "generation_ids": ["missing-a", "missing-b"],
    }
    legacy = variations / "batch_legacy.json"
    legacy.write_text(
        json.dumps(
            {
                **manifest,
                "batch_id": "legacy",
                "created_at": "2026-09-27T12:00:00",
            }
        ),
        encoding="utf-8",
    )
    legacy_written_at = datetime(2026, 9, 27, 10, tzinfo=timezone.utc).timestamp()
    os.utime(legacy, (legacy_written_at, legacy_written_at))
    (variations / "batch_aware.json").write_text(
        json.dumps(
            {
                **manifest,
                "batch_id": "aware",
                "created_at": "2026-09-27T11:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )

    assert [
        record.manifest.batch_id for record in store.list_variation_history().records
    ] == ["aware", "legacy"]
    assert legacy.stat().st_mtime_ns // 1_000_000_000 == int(legacy_written_at)


def test_listing_returns_all_malformed_and_reference_diagnostics(tmp_path):
    store = storage.FilesystemArtifactStore(tmp_path / "generations")
    variations = tmp_path / "variations"
    variations.mkdir()
    valid = {
        **_metadata("good"),
        "schema_version": 1,
        "created_at": "2026-09-27T12:00:00+00:00",
        "generation_ids": ["missing-a", "missing-b"],
    }
    (variations / "batch_good.json").write_text(json.dumps(valid), encoding="utf-8")
    for index in range(102):
        (variations / f"batch_bad-{index:03d}.json").write_text(
            "{broken", encoding="utf-8"
        )

    listing = store.list_variation_history()

    assert [record.manifest.batch_id for record in listing.records] == ["good"]
    assert len(listing.diagnostics) == 104
    assert [
        (item.code, item.batch_id, item.generation_id)
        for item in listing.diagnostics[:2]
    ] == [
        ("generation_missing", "good", "missing-a"),
        ("generation_missing", "good", "missing-b"),
    ]
    assert [item.batch_id for item in listing.diagnostics[2:]] == [
        f"bad-{index:03d}" for index in range(102)
    ]


def test_listing_skips_unreadable_neighbor_and_is_read_only(tmp_path, monkeypatch):
    store = storage.FilesystemArtifactStore(tmp_path / "generations")
    variations = tmp_path / "variations"
    variations.mkdir()
    manifest = {
        **_metadata(),
        "schema_version": 1,
        "created_at": "2026-09-27T12:00:00+00:00",
        "generation_ids": ["missing"],
    }
    (variations / "batch_good.json").write_text(
        json.dumps({**manifest, "batch_id": "good"}), encoding="utf-8"
    )
    (variations / "batch_blocked.json").write_text(
        json.dumps({**manifest, "batch_id": "blocked"}), encoding="utf-8"
    )
    before = {
        path.relative_to(tmp_path): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    original = storage._load_variation_manifest

    def fail_one(artifact_root, batch_id):
        if batch_id == "blocked":
            raise PermissionError("test unreadable manifest")
        return original(artifact_root, batch_id)

    monkeypatch.setattr(storage, "_load_variation_manifest", fail_one)
    listing = store.list_variation_history()

    assert [record.manifest.batch_id for record in listing.records] == ["good"]
    assert [(item.code, item.batch_id) for item in listing.diagnostics] == [
        ("generation_missing", "good"),
        ("manifest_invalid", "blocked"),
    ]
    after = {
        path.relative_to(tmp_path): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    assert after == before


def test_listing_refreshes_and_matches_top_level_helper(tmp_path, monkeypatch):
    monkeypatch.setenv("CONDUCTOR_CORE_DATA_DIR", str(tmp_path))
    variations = tmp_path / "variations"
    variations.mkdir()
    manifest = {
        **_metadata("first"),
        "schema_version": 1,
        "created_at": "2026-09-27T12:00:00+00:00",
        "generation_ids": ["missing"],
    }
    (variations / "batch_first.json").write_text(json.dumps(manifest), encoding="utf-8")
    store = storage.FilesystemArtifactStore(tmp_path / "generations")
    first = store.list_variation_history()
    assert [record.manifest.batch_id for record in first.records] == ["first"]

    (variations / "batch_second.json").write_text(
        json.dumps(
            {
                **manifest,
                "batch_id": "second",
                "created_at": "2026-09-27T13:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    refreshed = store.list_variation_history()
    top_level = list_variation_history()
    assert [record.manifest.batch_id for record in refreshed.records] == [
        "second",
        "first",
    ]
    assert top_level == refreshed
