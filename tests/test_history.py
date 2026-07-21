import json
import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from conductor_core import storage as history


@pytest.fixture
def isolated_history_dir(monkeypatch, tmp_path):
    generations_dir = tmp_path / "generations"
    monkeypatch.setattr(history, "GENERATIONS_DIR", str(generations_dir))
    return generations_dir


def _write_binary_file(path: Path, content: bytes = b"test-data"):
    path.write_bytes(content)
    return path


def _write_generation_metadata(
    base_dir: Path,
    *,
    gen_id: str,
    timestamp: datetime,
    prompt: str = "prompt",
    key: str = "C",
    scale: str = "major",
    model: str = "model",
    provider: str = "OpenAI",
    temperature: float = 0.0,
    cost: float | None = None,
    soundfont: str | None = None,
):
    gen_dir = base_dir / f"gen_{gen_id}"
    gen_dir.mkdir(parents=True, exist_ok=True)

    metadata = history.GenerationMetadata(
        id=gen_id,
        timestamp=timestamp,
        prompt=prompt,
        key=key,
        scale=scale,
        model=model,
        provider=provider,
        temperature=temperature,
        cost=cost,
        midi_path=str(gen_dir / "loop.mid"),
        audio_path=str(gen_dir / "loop.mp3"),
        soundfont=soundfont,
    )
    (gen_dir / "loop.mid").write_bytes(b"midi")
    (gen_dir / "loop.mp3").write_bytes(b"audio")
    (gen_dir / "metadata.json").write_text(metadata.model_dump_json(indent=2), encoding="utf-8")
    return gen_dir


def test_create_generation_workspace_allocates_canonical_paths(isolated_history_dir, monkeypatch):
    monkeypatch.setattr(history, "_generate_id", lambda: "fixed_id")

    workspace = history.create_generation_workspace()

    gen_dir = isolated_history_dir / "gen_fixed_id"
    assert workspace.id == "fixed_id"
    assert workspace.directory == str(gen_dir)
    assert workspace.midi_path == str(gen_dir / "loop.mid")
    assert workspace.audio_path == str(gen_dir / "loop.mp3")
    assert workspace.messages_path == str(gen_dir / "messages.json")
    assert workspace.metadata_path == str(gen_dir / "metadata.json")
    assert gen_dir.exists()


def test_generation_metadata_distinguishes_unrecorded_and_explicit_reasoning_settings():
    common_fields = {
        "id": "fixed_id",
        "timestamp": datetime.now(),
        "prompt": "warm rhodes loop",
        "key": "D",
        "scale": "minor",
        "model": "gpt-4o-mini",
        "provider": "OpenAI",
        "temperature": 0.3,
        "midi_path": "loop.mid",
    }

    unrecorded = history.GenerationMetadata(**common_fields)
    explicit = history.GenerationMetadata(
        **common_fields,
        use_thinking=False,
        effort="low",
    )

    assert unrecorded.use_thinking is None
    assert unrecorded.effort is None
    assert explicit.use_thinking is False
    assert explicit.effort == "low"


def test_finalize_generation_persists_metadata_for_direct_written_artifacts(
    isolated_history_dir, monkeypatch
):
    monkeypatch.setattr(history, "_generate_id", lambda: "fixed_id")

    workspace = history.create_generation_workspace()
    _write_binary_file(Path(workspace.midi_path), b"midi")
    _write_binary_file(Path(workspace.audio_path), b"audio")
    Path(workspace.messages_path).write_text("[]", encoding="utf-8")

    metadata = history.finalize_generation(
        workspace=workspace,
        prompt="warm rhodes loop",
        key="D",
        scale="minor",
        model="gpt-4o-mini",
        provider="OpenAI",
        temperature=0.3,
        use_thinking=False,
        effort="medium",
        cost=1.5,
        soundfont="FM-Piano1 20190916.sf2",
    )

    gen_dir = isolated_history_dir / "gen_fixed_id"
    loaded_metadata = history.get_generation("fixed_id")

    assert gen_dir.exists()
    assert (gen_dir / "loop.mid").read_bytes() == b"midi"
    assert (gen_dir / "loop.mp3").read_bytes() == b"audio"
    assert (gen_dir / "messages.json").read_text(encoding="utf-8") == "[]"
    assert metadata.id == "fixed_id"
    assert metadata.prompt == "warm rhodes loop"
    assert metadata.key == "D"
    assert metadata.scale == "minor"
    assert metadata.model == "gpt-4o-mini"
    assert metadata.provider == "OpenAI"
    assert metadata.temperature == 0.3
    assert metadata.use_thinking is False
    assert metadata.effort == "medium"
    assert metadata.cost == 1.5
    assert metadata.midi_path == str(gen_dir / "loop.mid")
    assert metadata.audio_path == str(gen_dir / "loop.mp3")
    assert metadata.messages_path == str(gen_dir / "messages.json")
    assert metadata.soundfont == "FM-Piano1 20190916.sf2"
    assert loaded_metadata == metadata


def test_finalize_generation_discards_audio_after_reported_render_failure(
    isolated_history_dir, monkeypatch
):
    monkeypatch.setattr(history, "_generate_id", lambda: "fixed_id")
    workspace = history.create_generation_workspace()
    _write_binary_file(Path(workspace.midi_path), b"midi")
    _write_binary_file(Path(workspace.audio_path), b"partial")

    metadata = history.finalize_generation(
        workspace=workspace,
        prompt="warm rhodes loop",
        key="D",
        scale="minor",
        model="gpt-4o-mini",
        provider="OpenAI",
        temperature=0.3,
        soundfont="custom.sf2",
        audio_render_succeeded=False,
    )

    assert metadata.audio_path is None
    assert metadata.soundfont is None
    assert not Path(workspace.audio_path).exists()


def test_finalize_generation_requires_direct_written_midi(isolated_history_dir, monkeypatch):
    monkeypatch.setattr(history, "_generate_id", lambda: "fixed_id")
    workspace = history.create_generation_workspace()

    with pytest.raises(FileNotFoundError):
        history.finalize_generation(
            workspace=workspace,
            prompt="warm rhodes loop",
            key="D",
            scale="minor",
            model="gpt-4o-mini",
            provider="OpenAI",
            temperature=0.3,
        )

    assert not (isolated_history_dir / "gen_fixed_id" / "metadata.json").exists()


def test_finalize_generation_replaces_hard_linked_metadata_without_overwriting_target(
    tmp_path, monkeypatch
):
    store = history.FilesystemArtifactStore(tmp_path / "generations")
    monkeypatch.setattr(history, "_generate_id", lambda: "fixed_id")
    workspace = store.create_generation_workspace()
    _write_binary_file(Path(workspace.midi_path), b"midi")
    victim = tmp_path / "victim.txt"
    victim.write_text("keep", encoding="utf-8")
    os.link(victim, workspace.metadata_path)

    metadata = store.finalize_generation(
        workspace=workspace,
        prompt="prompt",
        key="C",
        scale="major",
        model="model",
        provider="OpenAI",
        temperature=0.0,
    )

    assert victim.read_text(encoding="utf-8") == "keep"
    assert not os.path.samefile(victim, workspace.metadata_path)
    assert store.get_generation(metadata.id) == metadata


def test_finalize_generation_rejects_hard_linked_midi(tmp_path, monkeypatch):
    store = history.FilesystemArtifactStore(tmp_path / "generations")
    monkeypatch.setattr(history, "_generate_id", lambda: "fixed_id")
    workspace = store.create_generation_workspace()
    source = tmp_path / "outside.mid"
    source.write_bytes(b"midi")
    os.link(source, workspace.midi_path)

    with pytest.raises(ValueError, match="must not be hard linked"):
        store.finalize_generation(
            workspace=workspace,
            prompt="prompt",
            key="C",
            scale="major",
            model="model",
            provider="OpenAI",
            temperature=0.0,
        )

    assert source.read_bytes() == b"midi"
    assert not Path(workspace.metadata_path).exists()


def test_cleanup_generation_workspace_removes_only_unfinalized_directories(
    isolated_history_dir, monkeypatch
):
    monkeypatch.setattr(history, "_generate_id", lambda: "fixed_id")
    workspace = history.create_generation_workspace()

    assert history.cleanup_generation_workspace(workspace) is True
    assert not (isolated_history_dir / "gen_fixed_id").exists()

    workspace = history.create_generation_workspace()
    _write_binary_file(Path(workspace.midi_path), b"midi")
    history.finalize_generation(
        workspace=workspace,
        prompt="warm rhodes loop",
        key="D",
        scale="minor",
        model="gpt-4o-mini",
        provider="OpenAI",
        temperature=0.3,
    )

    assert history.cleanup_generation_workspace(workspace) is False
    assert (isolated_history_dir / "gen_fixed_id").exists()


def test_store_rejects_workspace_from_another_artifact_root(tmp_path, monkeypatch):
    first_store = history.FilesystemArtifactStore(tmp_path / "first")
    second_store = history.FilesystemArtifactStore(tmp_path / "second")
    monkeypatch.setattr(history, "_generate_id", lambda: "fixed_id")
    workspace = first_store.create_generation_workspace()
    _write_binary_file(Path(workspace.midi_path), b"midi")

    with pytest.raises(ValueError, match="does not match generation"):
        second_store.finalize_generation(
            workspace=workspace,
            prompt="prompt",
            key="C",
            scale="major",
            model="model",
            provider="OpenAI",
            temperature=0.0,
        )

    with pytest.raises(ValueError, match="does not match generation"):
        second_store.cleanup_generation_workspace(workspace)

    assert Path(workspace.directory).exists()


def test_store_rejects_workspace_with_mixed_canonical_paths(tmp_path, monkeypatch):
    store = history.FilesystemArtifactStore(tmp_path / "generations")
    monkeypatch.setattr(history, "_generate_id", lambda: "fixed_id")
    workspace = store.create_generation_workspace()
    victim = tmp_path / "victim"
    victim.mkdir()
    mixed_workspace = workspace.model_copy(update={"metadata_path": str(victim / "metadata.json")})

    with pytest.raises(ValueError, match="metadata_path does not match generation"):
        store.cleanup_generation_workspace(mixed_workspace)

    assert Path(workspace.directory).exists()
    assert victim.exists()


def test_update_generation_audio_copies_audio_and_updates_soundfont(
    isolated_history_dir, monkeypatch, tmp_path
):
    monkeypatch.setattr(history, "_generate_id", lambda: "fixed_id")

    original_audio = _write_binary_file(tmp_path / "source.mp3", b"old-audio")
    updated_audio = _write_binary_file(tmp_path / "updated.mp3", b"new-audio")

    workspace = history.create_generation_workspace()
    _write_binary_file(Path(workspace.midi_path), b"midi")
    _write_binary_file(Path(workspace.audio_path), original_audio.read_bytes())
    metadata = history.finalize_generation(
        workspace=workspace,
        prompt="warm rhodes loop",
        key="D",
        scale="minor",
        model="gpt-4o-mini",
        provider="OpenAI",
        temperature=0.3,
        use_thinking=True,
        effort="high",
        cost=1.5,
        soundfont="old.sf2",
    )

    updated = history.update_generation_audio(
        metadata.id,
        str(updated_audio),
        soundfont="new.sf2",
    )
    metadata = history.get_generation(metadata.id)
    gen_dir = isolated_history_dir / "gen_fixed_id"

    assert updated is not None
    assert updated.audio_path == str(gen_dir / "loop.mp3")
    assert updated.soundfont == "new.sf2"
    assert updated.use_thinking is True
    assert updated.effort == "high"
    assert (gen_dir / "loop.mp3").read_bytes() == b"new-audio"
    assert metadata is not None
    assert metadata.soundfont == "new.sf2"
    assert metadata.use_thinking is True
    assert metadata.effort == "high"


@pytest.mark.parametrize("audio_path", [None, "missing.mp3"])
def test_update_generation_audio_preserves_soundfont_when_audio_update_is_skipped(
    isolated_history_dir, monkeypatch, tmp_path, audio_path
):
    monkeypatch.setattr(history, "_generate_id", lambda: "fixed_id")

    workspace = history.create_generation_workspace()
    _write_binary_file(Path(workspace.midi_path), b"midi")
    _write_binary_file(Path(workspace.audio_path), b"old-audio")
    metadata = history.finalize_generation(
        workspace=workspace,
        prompt="warm rhodes loop",
        key="D",
        scale="minor",
        model="gpt-4o-mini",
        provider="OpenAI",
        temperature=0.3,
        soundfont="old.sf2",
    )

    attempted_audio_path = str(tmp_path / audio_path) if audio_path else None
    updated = history.update_generation_audio(
        metadata.id,
        attempted_audio_path,
        soundfont="new.sf2",
    )
    persisted = history.get_generation(metadata.id)

    assert updated is not None
    assert updated.audio_path == str(isolated_history_dir / "gen_fixed_id" / "loop.mp3")
    assert updated.soundfont == "old.sf2"
    assert (isolated_history_dir / "gen_fixed_id" / "loop.mp3").read_bytes() == b"old-audio"
    assert persisted == updated


def test_load_history_allows_older_entries_without_soundfont(isolated_history_dir):
    now = datetime.now()
    _write_generation_metadata(
        isolated_history_dir,
        gen_id="older",
        timestamp=now - timedelta(days=1),
        soundfont=None,
    )
    _write_generation_metadata(
        isolated_history_dir,
        gen_id="newer",
        timestamp=now,
        soundfont="FM-Piano1 20190916.sf2",
    )

    loaded = history.load_history()

    assert [entry.id for entry in loaded] == ["newer", "older"]
    assert loaded[0].soundfont == "FM-Piano1 20190916.sf2"
    assert loaded[1].soundfont is None


def test_load_history_allows_legacy_entries_without_reasoning_settings(isolated_history_dir):
    gen_dir = _write_generation_metadata(
        isolated_history_dir,
        gen_id="legacy",
        timestamp=datetime.now(),
    )
    metadata_path = gen_dir / "metadata.json"
    metadata_json = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata_json.pop("use_thinking")
    metadata_json.pop("effort")
    metadata_path.write_text(json.dumps(metadata_json), encoding="utf-8")

    loaded = history.load_history()

    assert len(loaded) == 1
    assert loaded[0].use_thinking is None
    assert loaded[0].effort is None


def test_load_history_sorts_newest_first(isolated_history_dir):
    now = datetime.now()
    _write_generation_metadata(
        isolated_history_dir, gen_id="older", timestamp=now - timedelta(days=1)
    )
    _write_generation_metadata(isolated_history_dir, gen_id="newer", timestamp=now)

    loaded = history.load_history()

    assert [entry.id for entry in loaded] == ["newer", "older"]


def test_load_history_skips_missing_and_malformed_metadata(isolated_history_dir):
    valid_dir = _write_generation_metadata(
        isolated_history_dir,
        gen_id="valid",
        timestamp=datetime.now(),
    )
    missing_dir = isolated_history_dir / "gen_missing"
    missing_dir.mkdir(parents=True)
    bad_dir = isolated_history_dir / "gen_bad"
    bad_dir.mkdir(parents=True)
    (bad_dir / "metadata.json").write_text("{not-json}", encoding="utf-8")

    loaded = history.load_history()

    assert [entry.id for entry in loaded] == ["valid"]
    assert valid_dir.exists()
    assert missing_dir.exists()
    assert bad_dir.exists()


def test_load_history_skips_metadata_id_mismatch_without_deleting_other_generation(tmp_path):
    root = tmp_path / "generations"
    attacker_dir = _write_generation_metadata(
        root,
        gen_id="attacker",
        timestamp=datetime.now() - timedelta(days=1),
    )
    victim_dir = _write_generation_metadata(
        root,
        gen_id="victim",
        timestamp=datetime.now(),
    )
    attacker_metadata_path = attacker_dir / "metadata.json"
    attacker_metadata = json.loads(attacker_metadata_path.read_text(encoding="utf-8"))
    attacker_metadata["id"] = "victim"
    attacker_metadata_path.write_text(json.dumps(attacker_metadata), encoding="utf-8")
    store = history.FilesystemArtifactStore(root, max_generations=1)

    assert [entry.id for entry in store.load_history()] == ["victim"]

    history._enforce_limit(root, max_generations=1)

    assert attacker_dir.exists()
    assert victim_dir.exists()


def test_copied_history_reconstructs_artifact_paths_under_new_root(tmp_path, monkeypatch):
    source_root = tmp_path / "source"
    source_store = history.FilesystemArtifactStore(source_root, max_generations=None)
    monkeypatch.setattr(history, "_generate_id", lambda: "fixed_id")
    workspace = source_store.create_generation_workspace()
    _write_binary_file(Path(workspace.midi_path), b"midi")
    _write_binary_file(Path(workspace.audio_path), b"audio")
    Path(workspace.messages_path).write_text("[]", encoding="utf-8")
    source_store.finalize_generation(
        workspace=workspace,
        prompt="prompt",
        key="C",
        scale="major",
        model="model",
        provider="OpenAI",
        temperature=0.0,
        soundfont="soundfont.sf2",
    )

    destination_root = tmp_path / "destination"
    shutil.copytree(source_root, destination_root)
    shutil.rmtree(source_root)
    destination_store = history.FilesystemArtifactStore(destination_root)

    loaded = destination_store.load_history()[0]
    generation_dir = destination_root / "gen_fixed_id"

    assert loaded.midi_path == str(generation_dir / "loop.mid")
    assert loaded.audio_path == str(generation_dir / "loop.mp3")
    assert loaded.messages_path == str(generation_dir / "messages.json")
    assert loaded.soundfont == "soundfont.sf2"
    assert destination_store.get_generation("fixed_id") == loaded


def test_get_generation_returns_none_for_missing_or_invalid_metadata(isolated_history_dir):
    missing_result = history.get_generation("missing")

    bad_dir = isolated_history_dir / "gen_broken"
    bad_dir.mkdir(parents=True)
    (bad_dir / "metadata.json").write_text(json.dumps({"id": "broken"}), encoding="utf-8")

    broken_result = history.get_generation("broken")

    assert missing_result is None
    assert broken_result is None


def test_delete_generation_removes_directory_and_handles_missing_id(isolated_history_dir):
    _write_generation_metadata(isolated_history_dir, gen_id="delete_me", timestamp=datetime.now())

    assert history.delete_generation("delete_me") is True
    assert not (isolated_history_dir / "gen_delete_me").exists()
    assert history.delete_generation("delete_me") is False


@pytest.mark.parametrize(
    "gen_id",
    [
        ".",
        "..",
        "anchor/../../victim",
        "anchor\\..\\..\\victim",
    ],
)
def test_store_rejects_generation_ids_with_path_components(tmp_path, gen_id):
    store = history.FilesystemArtifactStore(tmp_path / "generations")
    victim = tmp_path / "victim"
    victim.mkdir()
    (victim / "keep.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(ValueError, match="non-empty path component"):
        store.get_generation(gen_id)

    with pytest.raises(ValueError, match="non-empty path component"):
        store.delete_generation(gen_id)

    assert (victim / "keep.txt").read_text(encoding="utf-8") == "keep"


def test_store_rejects_generation_symlink_that_escapes_root(tmp_path):
    root = tmp_path / "generations"
    root.mkdir()
    victim = tmp_path / "victim"
    victim.mkdir()
    (victim / "keep.txt").write_text("keep", encoding="utf-8")

    try:
        (root / "gen_escape").symlink_to(victim, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlinks are unavailable: {exc}")

    store = history.FilesystemArtifactStore(root)
    with pytest.raises(ValueError, match="direct child"):
        store.delete_generation("escape")

    assert (victim / "keep.txt").read_text(encoding="utf-8") == "keep"


def test_enforce_limit_removes_oldest_generations(isolated_history_dir, monkeypatch):
    monkeypatch.setattr(history, "MAX_GENERATIONS", 2)
    now = datetime.now()
    _write_generation_metadata(
        isolated_history_dir, gen_id="oldest", timestamp=now - timedelta(days=2)
    )
    _write_generation_metadata(
        isolated_history_dir, gen_id="middle", timestamp=now - timedelta(days=1)
    )
    _write_generation_metadata(isolated_history_dir, gen_id="newest", timestamp=now)

    history._enforce_limit()

    remaining = {path.name for path in isolated_history_dir.iterdir()}

    assert remaining == {"gen_middle", "gen_newest"}


def test_filesystem_artifact_store_enforces_its_own_limit(tmp_path, monkeypatch):
    root = tmp_path / "generations"
    store = history.FilesystemArtifactStore(root, max_generations=2)
    ids = iter(["oldest", "middle", "newest"])
    monkeypatch.setattr(history, "_generate_id", lambda: next(ids))

    for index in range(3):
        workspace = store.create_generation_workspace()
        _write_binary_file(Path(workspace.midi_path), b"midi")
        store.finalize_generation(
            workspace=workspace,
            prompt=f"prompt {index}",
            key="C",
            scale="major",
            model="model",
            provider="OpenAI",
            temperature=0.0,
        )

    assert {path.name for path in root.iterdir()} == {"gen_middle", "gen_newest"}


def test_filesystem_artifact_store_allows_unlimited_history(tmp_path, monkeypatch):
    root = tmp_path / "generations"
    store = history.FilesystemArtifactStore(root, max_generations=None)
    ids = iter(["one", "two", "three"])
    monkeypatch.setattr(history, "_generate_id", lambda: next(ids))

    for index in range(3):
        workspace = store.create_generation_workspace()
        _write_binary_file(Path(workspace.midi_path), b"midi")
        store.finalize_generation(
            workspace=workspace,
            prompt=f"prompt {index}",
            key="C",
            scale="major",
            model="model",
            provider="OpenAI",
            temperature=0.0,
        )

    assert {path.name for path in root.iterdir()} == {"gen_one", "gen_two", "gen_three"}


@pytest.mark.parametrize("max_generations", [0, -1])
def test_filesystem_artifact_store_rejects_non_positive_limit(tmp_path, max_generations):
    with pytest.raises(ValueError, match="max_generations must be None or a positive integer"):
        history.FilesystemArtifactStore(
            tmp_path / "generations",
            max_generations=max_generations,
        )


@pytest.mark.parametrize("max_generations", [None, 1])
def test_filesystem_artifact_store_accepts_supported_limits(tmp_path, max_generations):
    store = history.FilesystemArtifactStore(
        tmp_path / "generations",
        max_generations=max_generations,
    )

    assert store.max_generations == max_generations


def test_filesystem_artifact_store_keeps_newest_generation_with_limit_one(tmp_path, monkeypatch):
    root = tmp_path / "generations"
    store = history.FilesystemArtifactStore(root, max_generations=1)
    ids = iter(["oldest", "newest"])
    monkeypatch.setattr(history, "_generate_id", lambda: next(ids))

    metadata = None
    for index in range(2):
        workspace = store.create_generation_workspace()
        _write_binary_file(Path(workspace.midi_path), b"midi")
        metadata = store.finalize_generation(
            workspace=workspace,
            prompt=f"prompt {index}",
            key="C",
            scale="major",
            model="model",
            provider="OpenAI",
            temperature=0.0,
        )

    assert metadata is not None
    assert {path.name for path in root.iterdir()} == {"gen_newest"}
    assert Path(metadata.midi_path).exists()


def test_history_count_and_clear_history_reflect_saved_generations(isolated_history_dir):
    _write_generation_metadata(
        isolated_history_dir, gen_id="one", timestamp=datetime.now() - timedelta(minutes=1)
    )
    _write_generation_metadata(isolated_history_dir, gen_id="two", timestamp=datetime.now())

    assert history.get_history_count() == 2
    assert history.clear_history() == 2
    assert history.get_history_count() == 0


def test_filesystem_artifact_store_uses_instance_root_without_global_mutation(
    tmp_path, monkeypatch
):
    default_generations_dir = history.GENERATIONS_DIR
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_store = history.FilesystemArtifactStore(first_root)
    second_store = history.FilesystemArtifactStore(second_root)
    ids = iter(["one", "two"])
    monkeypatch.setattr(history, "_generate_id", lambda: next(ids))

    first_workspace = first_store.create_generation_workspace()
    second_workspace = second_store.create_generation_workspace()
    _write_binary_file(Path(first_workspace.midi_path), b"first-midi")
    _write_binary_file(Path(second_workspace.midi_path), b"second-midi")

    first_metadata = first_store.finalize_generation(
        workspace=first_workspace,
        prompt="first prompt",
        key="C",
        scale="major",
        model="model-a",
        provider="OpenAI",
        temperature=0.0,
    )
    second_metadata = second_store.finalize_generation(
        workspace=second_workspace,
        prompt="second prompt",
        key="D",
        scale="minor",
        model="model-b",
        provider="Google",
        temperature=0.1,
    )

    assert first_metadata.id == "one"
    assert second_metadata.id == "two"
    assert [entry.id for entry in first_store.load_history()] == ["one"]
    assert [entry.id for entry in second_store.load_history()] == ["two"]
    assert history.GENERATIONS_DIR == default_generations_dir


def test_get_provider_for_model_returns_matching_provider_or_ollama_default():
    model_info = {
        "models": {
            "OpenAI": {"gpt-5-mini": {}},
            "Google": {"gemini-3.5-flash": {}},
        }
    }

    assert history.get_provider_for_model("gpt-5-mini", model_info) == "OpenAI"
    assert history.get_provider_for_model("gemini-3.5-flash", model_info) == "Google"
    assert history.get_provider_for_model("llama3.2:1b", model_info) == "Ollama"
