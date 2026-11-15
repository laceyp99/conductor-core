import json
import os
from datetime import UTC, datetime
from pathlib import Path

import pytest

from conductor_core import (
    AudioCommitRecoveryError,
    AudioUpdateInProgressError,
    storage,
)


@pytest.fixture
def saved_generation(tmp_path):
    store = storage.FilesystemArtifactStore(tmp_path / "generations")
    directory = Path(store.artifact_root) / "gen_saved"
    directory.mkdir(parents=True)
    (directory / "loop.mid").write_bytes(b"midi")
    audio = directory / "loop.mp3"
    audio.write_bytes(b"old audio")
    metadata = storage.GenerationMetadata(
        id="saved",
        timestamp=datetime.now(UTC),
        prompt="piano",
        key="C",
        scale="major",
        model="model",
        provider="OpenAI",
        temperature=0.5,
        midi_path=str(directory / "loop.mid"),
        audio_path=str(audio),
        soundfont="old.sf2",
    )
    (directory / "metadata.json").write_text(
        metadata.model_dump_json(indent=2), encoding="utf-8"
    )
    return store, directory, metadata


def test_stage_commits_audio_and_metadata(saved_generation):
    store, directory, original = saved_generation
    with store.stage_generation_audio("saved") as stage:
        stage_path = Path(stage.path)
        assert stage_path.parent.parent == directory
        stage_path.write_bytes(b"new audio")
        assert (directory / "loop.mp3").read_bytes() == b"old audio"
        assert store.get_generation("saved") == original
        updated = stage.commit(soundfont="new.sf2")
    assert not stage_path.parent.exists()
    assert (directory / "loop.mp3").read_bytes() == b"new audio"
    assert updated == store.get_generation("saved")
    assert updated.audio_path == str(directory / "loop.mp3")
    assert updated.soundfont == "new.sf2"
    assert updated.prompt == original.prompt


@pytest.mark.parametrize("prior_audio", [True, False])
def test_metadata_replace_failure_restores_previous_state(
    saved_generation, monkeypatch, prior_audio
):
    store, directory, original = saved_generation
    audio = directory / "loop.mp3"
    if not prior_audio:
        audio.unlink()
        original.audio_path = None
        original.soundfont = None
        (directory / "metadata.json").write_text(
            original.model_dump_json(), encoding="utf-8"
        )
    original_bytes = (directory / "metadata.json").read_bytes()
    replace = storage.os.replace
    failure = PermissionError("metadata locked")

    def fail_metadata(source, destination):
        if Path(destination) == directory / "metadata.json":
            raise failure
        return replace(source, destination)

    monkeypatch.setattr(storage.os, "replace", fail_metadata)
    with store.stage_generation_audio("saved") as stage:
        Path(stage.path).write_bytes(b"new audio")
        with pytest.raises(PermissionError) as caught:
            stage.commit(soundfont="new.sf2")
        assert caught.value is failure
    assert (directory / "metadata.json").read_bytes() == original_bytes
    assert audio.exists() is prior_audio
    if prior_audio:
        assert audio.read_bytes() == b"old audio"


@pytest.mark.parametrize("failure_point", ["backup", "audio_replace"])
def test_commit_prepublication_failures_preserve_saved_files(
    saved_generation, monkeypatch, failure_point
):
    store, directory, original = saved_generation
    failure = PermissionError(failure_point)

    def fail(*args, **kwargs):
        raise failure

    with store.stage_generation_audio("saved") as stage:
        Path(stage.path).write_bytes(b"new audio")
        if failure_point == "backup":
            monkeypatch.setattr(storage.shutil, "copy2", fail)
        else:
            monkeypatch.setattr(storage.os, "replace", fail)
        with pytest.raises(PermissionError) as caught:
            stage.commit(soundfont="new.sf2")
        assert caught.value is failure
    assert (directory / "loop.mp3").read_bytes() == b"old audio"
    assert store.get_generation("saved") == original


def test_rollback_failure_preserves_recovery_files(saved_generation, monkeypatch):
    store, directory, original = saved_generation
    replace = storage.os.replace
    commit_error = PermissionError("metadata locked")
    rollback_error = OSError("audio restore failed")

    with store.stage_generation_audio("saved") as stage:
        staged_path = Path(stage.path)
        staged_path.write_bytes(b"new audio")

        def fail_publication_and_restore(source, destination):
            if Path(destination) == directory / "metadata.json":
                raise commit_error
            if (
                Path(destination) == directory / "loop.mp3"
                and Path(source) != staged_path
            ):
                raise rollback_error
            return replace(source, destination)

        monkeypatch.setattr(storage.os, "replace", fail_publication_and_restore)
        with pytest.raises(AudioCommitRecoveryError) as caught:
            stage.commit(soundfont="new.sf2")
    error = caught.value
    assert isinstance(error, OSError)
    assert error.commit_error is commit_error
    assert error.rollback_error is rollback_error
    assert error.__cause__ is commit_error
    assert str(commit_error) in str(error)
    assert str(rollback_error) in str(error)
    recovery = Path(error.recovery_directory)
    assert recovery.is_dir()
    assert any(path.read_bytes() == b"old audio" for path in recovery.iterdir())
    assert (directory / "loop.mp3").read_bytes() == b"new audio"
    assert store.get_generation("saved") == original


def test_render_exception_cleans_stage_and_preserves_history(saved_generation):
    store, directory, original = saved_generation
    staged_directories = []

    def render():
        with store.stage_generation_audio("saved") as stage:
            staged_directories.append(Path(stage.path).parent)
            Path(stage.path).write_bytes(b"partial")
            raise ValueError("render failed")

    with pytest.raises(ValueError, match="render failed"):
        render()
    assert not staged_directories[0].exists()
    assert (directory / "loop.mp3").read_bytes() == b"old audio"
    assert store.get_generation("saved") == original


def test_cleanup_failure_does_not_mask_success(saved_generation, monkeypatch, caplog):
    store, directory, _ = saved_generation

    def fail_cleanup(*args, **kwargs):
        raise OSError("cleanup failed")

    with store.stage_generation_audio("saved") as stage:
        Path(stage.path).write_bytes(b"new audio")
        monkeypatch.setattr(storage.shutil, "rmtree", fail_cleanup)
        updated = stage.commit(soundfont=None)
    assert (directory / "loop.mp3").read_bytes() == b"new audio"
    assert store.get_generation("saved") == updated
    assert "cleanup failed" in caplog.text


def test_missing_or_invalid_metadata_is_reported(saved_generation):
    store, directory, _ = saved_generation
    with pytest.raises(FileNotFoundError), store.stage_generation_audio("missing"):
        pytest.fail("missing generation entered")
    (directory / "metadata.json").write_text("{invalid", encoding="utf-8")
    with (
        pytest.raises(json.JSONDecodeError),
        store.stage_generation_audio("saved"),
    ):
        pytest.fail("invalid generation entered")


def test_stages_exclude_other_stores_and_release_lock(saved_generation):
    store, _, _ = saved_generation
    other = storage.FilesystemArtifactStore(Path(store.artifact_root) / ".")
    with (
        store.stage_generation_audio("saved"),
        pytest.raises(AudioUpdateInProgressError),
        other.stage_generation_audio("saved"),
    ):
        pytest.fail("overlapping stage entered")
    with other.stage_generation_audio("saved"):
        pass


def test_stage_lifecycle_is_single_use(saved_generation):
    store, _, _ = saved_generation
    context = store.stage_generation_audio("saved")
    with context as stage:
        Path(stage.path).write_bytes(b"new audio")
        stage.commit(soundfont="new.sf2")
        with pytest.raises(RuntimeError):
            stage.commit(soundfont="again.sf2")
    with pytest.raises(RuntimeError), context:
        pytest.fail("context entered twice")
    with pytest.raises(RuntimeError):
        stage.commit(soundfont=None)


def test_staging_failure_releases_lock(saved_generation, monkeypatch):
    store, directory, original = saved_generation
    mkdir = storage.tempfile.mkdtemp

    def fail_staging(*args, **kwargs):
        raise PermissionError("staging denied")

    monkeypatch.setattr(storage.tempfile, "mkdtemp", fail_staging)
    with (
        pytest.raises(PermissionError, match="staging denied"),
        store.stage_generation_audio("saved"),
    ):
        pytest.fail("stage entered")
    monkeypatch.setattr(storage.tempfile, "mkdtemp", mkdir)
    with store.stage_generation_audio("saved"):
        pass
    assert store.get_generation("saved") == original
    assert (directory / "loop.mp3").read_bytes() == b"old audio"


def test_missing_staged_audio_consumes_commit(saved_generation):
    store, directory, original = saved_generation
    with store.stage_generation_audio("saved") as stage:
        with pytest.raises(FileNotFoundError):
            stage.commit(soundfont=None)
        Path(stage.path).write_bytes(b"late audio")
        with pytest.raises(RuntimeError):
            stage.commit(soundfont=None)
    assert (directory / "loop.mp3").read_bytes() == b"old audio"
    assert store.get_generation("saved") == original


def test_legacy_audio_update_rolls_back_metadata_failure(
    saved_generation, tmp_path, monkeypatch
):
    store, directory, original = saved_generation
    source = tmp_path / "rendered.mp3"
    source.write_bytes(b"new audio")
    replace = storage.os.replace

    def fail_metadata(source, destination):
        if Path(destination) == directory / "metadata.json":
            raise PermissionError("metadata locked")
        return replace(source, destination)

    monkeypatch.setattr(storage.os, "replace", fail_metadata)
    with pytest.raises(PermissionError, match="metadata locked"):
        store.update_generation_audio("saved", str(source), soundfont="new.sf2")
    assert source.read_bytes() == b"new audio"
    assert (directory / "loop.mp3").read_bytes() == b"old audio"
    assert store.get_generation("saved") == original


def test_legacy_update_accepts_hard_linked_source(saved_generation, tmp_path):
    store, directory, _ = saved_generation
    source = tmp_path / "rendered.mp3"
    source.write_bytes(b"linked audio")
    linked = tmp_path / "linked.mp3"
    os.link(source, linked)
    updated = store.update_generation_audio(
        "saved", str(linked), soundfont="linked.sf2"
    )
    assert (directory / "loop.mp3").read_bytes() == b"linked audio"
    assert source.read_bytes() == linked.read_bytes() == b"linked audio"
    assert updated == store.get_generation("saved")
    assert updated.soundfont == "linked.sf2"


def test_legacy_update_rejects_empty_source_before_promotion(
    saved_generation, tmp_path
):
    store, directory, original = saved_generation
    source = tmp_path / "empty.mp3"
    source.touch()
    with pytest.raises(ValueError, match="must not be empty"):
        store.update_generation_audio("saved", str(source), soundfont="empty.sf2")
    assert (directory / "loop.mp3").read_bytes() == b"old audio"
    assert store.get_generation("saved") == original


def test_legacy_update_reports_busy_generation(saved_generation):
    store, _, _ = saved_generation
    with (
        store.stage_generation_audio("saved"),
        pytest.raises(AudioUpdateInProgressError),
    ):
        store.update_generation_audio("saved", None)
