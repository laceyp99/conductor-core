import wave
from contextlib import contextmanager
from pathlib import Path

import pytest

from conductor_core import playback as audio


@pytest.fixture(autouse=True)
def reset_extra_soundfont_dirs(monkeypatch):
    monkeypatch.setattr(audio, "_EXTRA_SOUNDFONT_DIRS", [])


def _write_file(path: Path, content: bytes = b"data") -> Path:
    path.write_bytes(content)
    return path


def _write_wav(path: str | Path, frames: bytes = b"\x00\x00") -> None:
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(44100)
        wav_file.writeframes(frames)


def test_list_soundfonts_returns_sorted_sf2_files(monkeypatch, tmp_path):
    monkeypatch.setattr(audio, "SOUNDFONT_DIR", str(tmp_path))
    _write_file(tmp_path / "zeta.sf2")
    _write_file(tmp_path / "Alpha.sf2")
    _write_file(tmp_path / "notes.txt")

    soundfonts = audio.list_soundfonts()

    assert soundfonts == ["Alpha.sf2", "zeta.sf2"]


def test_get_default_soundfont_prefers_known_candidates(monkeypatch, tmp_path):
    monkeypatch.setattr(audio, "SOUNDFONT_DIR", str(tmp_path))
    _write_file(tmp_path / "custom.sf2")
    preferred = _write_file(tmp_path / "FM-Piano1-20190916.sf2")

    soundfont_path = audio.get_default_soundfont()

    assert soundfont_path == str(preferred)


def test_resolve_soundfont_returns_requested_file_from_soundfont_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(audio, "SOUNDFONT_DIR", str(tmp_path))
    expected = _write_file(tmp_path / "custom.sf2")

    soundfont_path = audio.resolve_soundfont("custom.sf2")

    assert soundfont_path == str(expected)


def test_resolve_soundfont_ignores_cwd_soundfonts(monkeypatch, tmp_path):
    core_soundfonts = tmp_path / "core"
    cwd = tmp_path / "cwd"
    cwd_soundfonts = cwd / "soundfonts"
    core_soundfonts.mkdir()
    cwd_soundfonts.mkdir(parents=True)
    _write_file(cwd_soundfonts / "cwd-only.sf2")
    monkeypatch.setattr(audio, "SOUNDFONT_DIR", str(core_soundfonts))
    monkeypatch.chdir(cwd)

    assert audio.resolve_soundfont("cwd-only.sf2") is None


def test_application_soundfont_directory_is_searchable(monkeypatch, tmp_path):
    packaged_soundfonts = tmp_path / "packaged"
    app_soundfonts = tmp_path / "app"
    packaged_soundfonts.mkdir()
    app_soundfonts.mkdir()
    expected = _write_file(app_soundfonts / "personal.sf2")

    monkeypatch.setattr(audio, "SOUNDFONT_DIR", str(packaged_soundfonts))
    monkeypatch.setattr(audio, "_EXTRA_SOUNDFONT_DIRS", [])
    audio.add_soundfont_search_dir(app_soundfonts)

    assert audio.list_soundfonts() == ["personal.sf2"]
    assert audio.resolve_soundfont("personal.sf2") == str(expected)


def test_packaged_soundfont_is_resolved_lazily_with_as_file(monkeypatch, tmp_path):
    materialized_soundfont = _write_file(tmp_path / "packaged.sf2")
    calls = []

    class FakeResource:
        name = "packaged.sf2"

        def is_file(self):
            return True

    class FakeResourceDirectory:
        def iterdir(self):
            calls.append("iterdir")
            return [FakeResource()]

    class FakeResources:
        def joinpath(self, name):
            assert name == "soundfonts"
            return FakeResourceDirectory()

    @contextmanager
    def fake_as_file(resource):
        calls.append(("as_file", resource.name))
        yield materialized_soundfont

    monkeypatch.setattr(audio, "SOUNDFONT_DIR", None)
    monkeypatch.setattr(audio.resources, "files", lambda package: FakeResources())
    monkeypatch.setattr(audio.resources, "as_file", fake_as_file)
    monkeypatch.setattr(audio, "_PACKAGED_SOUNDFONT_PATHS", {})

    assert calls == []
    assert audio.resolve_soundfont("packaged.sf2") == str(materialized_soundfont)
    assert calls == ["iterdir", ("as_file", "packaged.sf2")]


def test_is_playback_available_reports_missing_requested_soundfont(monkeypatch, tmp_path):
    monkeypatch.setattr(audio, "SOUNDFONT_DIR", str(tmp_path))
    monkeypatch.setattr(audio, "is_fluidsynth_available", lambda: True)
    monkeypatch.setattr(audio, "is_ffmpeg_available", lambda: True)

    available, error = audio.is_playback_available("missing.sf2")

    assert available is False
    assert error == f"Requested SoundFont 'missing.sf2' was not found in '{tmp_path}'."


def test_get_playback_status_message_prioritizes_missing_dependencies(monkeypatch):
    monkeypatch.setattr(
        audio, "is_playback_available", lambda soundfont_name=None: (False, "dependency error")
    )
    monkeypatch.setattr(audio, "is_fluidsynth_available", lambda: False)
    monkeypatch.setattr(audio, "is_ffmpeg_available", lambda: False)
    monkeypatch.setattr(audio, "find_soundfont", lambda soundfont_name=None: None)

    status_message = audio.get_playback_status_message("missing.sf2")

    assert "Install FluidSynth" in status_message
    assert "Install FFmpeg" in status_message
    assert "Add the requested SoundFont" not in status_message


def test_midi_to_mp3_uses_requested_soundfont(monkeypatch, tmp_path):
    midi_path = _write_file(tmp_path / "loop.mid")
    soundfont_path = _write_file(tmp_path / "custom.sf2")
    output_path = tmp_path / "loop.mp3"
    captured = {}

    monkeypatch.setattr(audio, "SOUNDFONT_DIR", str(tmp_path))
    monkeypatch.setattr(audio, "is_playback_available", lambda soundfont_name=None: (True, None))

    class FakeFluidSynth:
        def __init__(self, selected_soundfont):
            captured["soundfont_path"] = selected_soundfont

        def midi_to_audio(self, input_midi_path, temp_wav_path):
            captured["input_midi_path"] = input_midi_path
            _write_wav(temp_wav_path)

    class FakeAudioSegment:
        @staticmethod
        def from_wav(temp_wav_path):
            captured["temp_wav_path"] = temp_wav_path

            class FakeExport:
                def export(self, target_output_path, format, bitrate):
                    captured["output_path"] = target_output_path
                    captured["format"] = format
                    captured["bitrate"] = bitrate
                    Path(target_output_path).write_bytes(b"mp3")

            return FakeExport()

    monkeypatch.setattr(audio, "FluidSynth", FakeFluidSynth)
    monkeypatch.setattr(audio, "AudioSegment", FakeAudioSegment)

    result = audio.midi_to_mp3(
        str(midi_path),
        output_path=str(output_path),
        soundfont_name="custom.sf2",
    )

    assert result.path == str(output_path)
    assert result.error is None
    assert captured["soundfont_path"] == str(soundfont_path)
    assert captured["input_midi_path"] == str(midi_path)
    rendered_temp_path = Path(captured["output_path"])
    assert rendered_temp_path.parent == output_path.parent
    assert rendered_temp_path != output_path
    assert not rendered_temp_path.exists()
    assert output_path.read_bytes() == b"mp3"


def test_midi_to_mp3_removes_partial_output_when_export_fails(monkeypatch, tmp_path):
    midi_path = _write_file(tmp_path / "loop.mid")
    _write_file(tmp_path / "custom.sf2")
    output_path = tmp_path / "loop.mp3"
    partial_path = None

    monkeypatch.setattr(audio, "SOUNDFONT_DIR", str(tmp_path))
    monkeypatch.setattr(audio, "is_playback_available", lambda soundfont_name=None: (True, None))

    class FakeFluidSynth:
        def __init__(self, selected_soundfont):
            pass

        def midi_to_audio(self, input_midi_path, temp_wav_path):
            _write_wav(temp_wav_path)

    class FakeAudioSegment:
        @staticmethod
        def from_wav(temp_wav_path):
            class FailedExport:
                def export(self, target_output_path, format, bitrate):
                    nonlocal partial_path
                    partial_path = Path(target_output_path)
                    partial_path.write_bytes(b"partial")
                    raise RuntimeError("export failed")

            return FailedExport()

    monkeypatch.setattr(audio, "FluidSynth", FakeFluidSynth)
    monkeypatch.setattr(audio, "AudioSegment", FakeAudioSegment)

    result = audio.midi_to_mp3(
        str(midi_path),
        output_path=str(output_path),
        soundfont_name="custom.sf2",
    )

    assert result.path is None
    assert result.error == "RuntimeError: export failed"
    assert partial_path is not None
    assert partial_path.parent == output_path.parent
    assert not partial_path.exists()
    assert not output_path.exists()


def test_midi_to_mp3_reports_invalid_fluidsynth_output(monkeypatch, tmp_path):
    midi_path = _write_file(tmp_path / "loop.mid")
    _write_file(tmp_path / "custom.sf2")
    output_path = tmp_path / "loop.mp3"

    monkeypatch.setattr(audio, "SOUNDFONT_DIR", str(tmp_path))
    monkeypatch.setattr(audio, "is_playback_available", lambda soundfont_name=None: (True, None))

    class FakeFluidSynth:
        def __init__(self, selected_soundfont):
            pass

        def midi_to_audio(self, input_midi_path, temp_wav_path):
            _write_wav(temp_wav_path, frames=b"")

    class UnexpectedAudioSegment:
        @staticmethod
        def from_wav(temp_wav_path):
            pytest.fail("Invalid WAV output should not be passed to pydub")

    monkeypatch.setattr(audio, "FluidSynth", FakeFluidSynth)
    monkeypatch.setattr(audio, "AudioSegment", UnexpectedAudioSegment)

    result = audio.midi_to_mp3(
        str(midi_path),
        output_path=str(output_path),
        soundfont_name="custom.sf2",
    )

    assert result.path is None
    assert result.error == "RuntimeError: FluidSynth produced a WAV file with no audio frames"
    assert not output_path.exists()
