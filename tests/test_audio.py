import subprocess
import wave
from contextlib import contextmanager
from pathlib import Path

import pytest
from mido import MetaMessage, MidiFile, MidiTrack

from conductor_core import AudioRenderingError
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


def _write_midi(path: Path, *, tempo_changes=(), midi_type=1) -> Path:
    midi = MidiFile(type=midi_type, ticks_per_beat=480)
    track = MidiTrack()
    for delta, tempo in tempo_changes:
        track.append(MetaMessage("set_tempo", tempo=tempo, time=delta))
    elapsed_ticks = sum(delta for delta, _ in tempo_changes)
    track.append(MetaMessage("end_of_track", time=7680 - elapsed_ticks))
    midi.tracks.append(track)
    midi.save(path)
    return path


def _fluidsynth_output_path(command: list[str]) -> str:
    return command[command.index("-F") + 1]


def test_list_soundfonts_returns_sorted_sf2_files(monkeypatch, tmp_path):
    monkeypatch.setattr(audio, "SOUNDFONT_DIR", str(tmp_path))
    _write_file(tmp_path / "zeta.sf2")
    _write_file(tmp_path / "Alpha.sf2")
    _write_file(tmp_path / "notes.txt")

    soundfonts = audio.list_soundfonts()

    assert soundfonts == ["Alpha.sf2", "zeta.sf2"]


def test_soundfont_discovery_skips_directories_with_sf2_suffix(monkeypatch, tmp_path):
    monkeypatch.setattr(audio, "SOUNDFONT_DIR", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    (tmp_path / "FM-Piano1-20190916.sf2").mkdir()
    expected = _write_file(tmp_path / "custom.sf2")

    assert audio.list_soundfonts() == ["custom.sf2"]
    assert audio.get_default_soundfont() == str(expected)
    assert audio.resolve_soundfont("FM-Piano1-20190916.sf2") is None
    assert audio.resolve_soundfont(str(tmp_path / "FM-Piano1-20190916.sf2")) is None


def test_is_playback_available_rejects_directory_only_soundfonts(monkeypatch, tmp_path):
    monkeypatch.setattr(audio, "SOUNDFONT_DIR", str(tmp_path))
    monkeypatch.setattr(audio, "is_fluidsynth_available", lambda: True)
    monkeypatch.setattr(audio, "is_ffmpeg_available", lambda: True)
    (tmp_path / "piano.sf2").mkdir()

    available, error = audio.is_playback_available()

    assert available is False
    assert error is not None


def test_get_default_soundfont_prefers_known_candidates(monkeypatch, tmp_path):
    monkeypatch.setattr(audio, "SOUNDFONT_DIR", str(tmp_path))
    _write_file(tmp_path / "custom.sf2")
    preferred = _write_file(tmp_path / "FM-Piano1-20190916.sf2")

    soundfont_path = audio.get_default_soundfont()

    assert soundfont_path == str(preferred)


def test_resolve_soundfont_returns_requested_file_from_soundfont_dir(
    monkeypatch, tmp_path
):
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


def test_is_playback_available_reports_missing_requested_soundfont(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(audio, "SOUNDFONT_DIR", str(tmp_path))
    monkeypatch.setattr(audio, "is_fluidsynth_available", lambda: True)
    monkeypatch.setattr(audio, "is_ffmpeg_available", lambda: True)

    available, error = audio.is_playback_available("missing.sf2")

    assert available is False
    assert error == f"Requested SoundFont 'missing.sf2' was not found in '{tmp_path}'."


def test_get_playback_status_message_prioritizes_missing_dependencies(monkeypatch):
    monkeypatch.setattr(
        audio,
        "is_playback_available",
        lambda soundfont_name=None: (False, "dependency error"),
    )
    monkeypatch.setattr(audio, "is_fluidsynth_available", lambda: False)
    monkeypatch.setattr(audio, "is_ffmpeg_available", lambda: False)
    monkeypatch.setattr(audio, "find_soundfont", lambda soundfont_name=None: None)

    status_message = audio.get_playback_status_message("missing.sf2")

    assert "Install FluidSynth" in status_message
    assert "Install FFmpeg" in status_message
    assert "Add the requested SoundFont" not in status_message


def test_midi_to_mp3_raises_when_playback_is_unavailable(monkeypatch, tmp_path):
    midi_path = _write_file(tmp_path / "loop.mid")
    monkeypatch.setattr(
        audio,
        "is_playback_available",
        lambda soundfont_name=None: (
            False,
            "FluidSynth is not installed or not in PATH",
        ),
    )

    with pytest.raises(
        AudioRenderingError,
        match="FluidSynth is not installed or not in PATH",
    ):
        audio.midi_to_mp3(str(midi_path))


def test_midi_to_mp3_uses_requested_soundfont(monkeypatch, tmp_path):
    midi_path = _write_midi(tmp_path / "input loop.mid")
    soundfont_path = _write_file(tmp_path / "custom piano.sf2")
    output_path = tmp_path / "output loop.mp3"
    calls = []

    monkeypatch.setattr(audio, "SOUNDFONT_DIR", str(tmp_path))
    monkeypatch.setattr(
        audio, "is_playback_available", lambda soundfont_name=None: (True, None)
    )

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        if command[0] == "fluidsynth":
            _write_wav(_fluidsynth_output_path(command))
        else:
            assert command[0] == "ffmpeg"
            Path(command[-1]).write_bytes(b"mp3")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(audio.subprocess, "run", fake_run)

    rendered_path = audio.midi_to_mp3(
        str(midi_path),
        output_path=str(output_path),
        soundfont_name="custom piano.sf2",
    )

    assert rendered_path == str(output_path)
    assert len(calls) == 2
    synthesis_command, synthesis_kwargs = calls[0]
    encoding_command, encoding_kwargs = calls[1]
    assert synthesis_command == [
        "fluidsynth",
        "-ni",
        "-T",
        "wav",
        "-O",
        "s16",
        "-F",
        _fluidsynth_output_path(synthesis_command),
        "-r",
        "44100",
        str(soundfont_path),
        str(midi_path),
    ]
    assert synthesis_kwargs == {
        "capture_output": True,
        "text": True,
        "check": False,
        "timeout": audio._FLUIDSYNTH_TIMEOUT_SECONDS,
    }
    rendered_temp_path = Path(encoding_command[-1])
    assert encoding_command == [
        "ffmpeg",
        "-nostdin",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        _fluidsynth_output_path(synthesis_command),
        "-t",
        "8.000",
        "-b:a",
        "192k",
        "-f",
        "mp3",
        str(rendered_temp_path),
    ]
    assert encoding_kwargs == {
        "capture_output": True,
        "text": True,
        "check": False,
        "timeout": audio._FFMPEG_TIMEOUT_SECONDS,
    }
    assert rendered_temp_path.parent == output_path.parent
    assert rendered_temp_path != output_path
    assert not rendered_temp_path.exists()
    assert not Path(_fluidsynth_output_path(synthesis_command)).exists()
    assert output_path.read_bytes() == b"mp3"


@pytest.mark.parametrize("existing_output", [False, True])
def test_midi_to_mp3_removes_partial_output_when_ffmpeg_fails(
    monkeypatch, tmp_path, existing_output
):
    midi_path = _write_midi(tmp_path / "loop.mid")
    _write_file(tmp_path / "custom.sf2")
    output_path = tmp_path / "loop.mp3"
    if existing_output:
        output_path.write_bytes(b"existing preview")
    partial_path = None

    monkeypatch.setattr(audio, "SOUNDFONT_DIR", str(tmp_path))
    monkeypatch.setattr(
        audio, "is_playback_available", lambda soundfont_name=None: (True, None)
    )

    def fake_run(command, **kwargs):
        nonlocal partial_path
        if command[0] == "fluidsynth":
            _write_wav(_fluidsynth_output_path(command))
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        assert command[0] == "ffmpeg"
        partial_path = Path(command[-1])
        partial_path.write_bytes(b"partial")
        return subprocess.CompletedProcess(
            command, 1, stdout="", stderr="encode failed\n"
        )

    monkeypatch.setattr(audio.subprocess, "run", fake_run)

    with pytest.raises(
        AudioRenderingError, match="FFmpeg exited with code 1: encode failed"
    ):
        audio.midi_to_mp3(
            str(midi_path),
            output_path=str(output_path),
            soundfont_name="custom.sf2",
        )

    assert partial_path is not None
    assert partial_path.parent == output_path.parent
    assert not partial_path.exists()
    if existing_output:
        assert output_path.read_bytes() == b"existing preview"
    else:
        assert not output_path.exists()


@pytest.mark.parametrize("output_stream", ["stderr", "stdout"])
def test_encode_wav_to_mp3_reports_process_failure(monkeypatch, output_stream):
    def fake_run(command, **kwargs):
        assert command[0] == "ffmpeg"
        output = {"stdout": "", "stderr": "", output_stream: "encoder unavailable\n"}
        return subprocess.CompletedProcess(command, 2, **output)

    monkeypatch.setattr(audio.subprocess, "run", fake_run)

    with pytest.raises(
        AudioRenderingError, match="FFmpeg exited with code 2: encoder unavailable"
    ):
        audio._encode_wav_to_mp3("input.wav", "output.mp3", 1234)


def test_encode_wav_to_mp3_reports_timeout(monkeypatch):
    def fake_run(command, **kwargs):
        assert command[0] == "ffmpeg"
        assert kwargs["timeout"] == 30
        assert command[command.index("-t") + 1] == "1.234"
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(audio.subprocess, "run", fake_run)

    with pytest.raises(
        AudioRenderingError, match="FFmpeg timed out after 30 seconds"
    ) as raised:
        audio._encode_wav_to_mp3("input.wav", "output.mp3", 1234)

    assert isinstance(raised.value.__cause__, subprocess.TimeoutExpired)


@pytest.mark.parametrize("midi_type", [0, 1])
@pytest.mark.parametrize(
    ("tempo_changes", "expected_duration_ms"),
    [
        ((), 8000),
        (((0, 1_000_000),), 16000),
        (((3840, 1_000_000),), 12000),
    ],
)
def test_midi_duration_includes_default_and_tempo_changes(
    tmp_path, tempo_changes, expected_duration_ms, midi_type
):
    midi_path = _write_midi(
        tmp_path / "loop.mid", tempo_changes=tempo_changes, midi_type=midi_type
    )

    assert audio._midi_duration_ms(str(midi_path)) == expected_duration_ms


def test_midi_duration_rejects_format_2(tmp_path):
    midi_path = _write_midi(tmp_path / "patterns.mid", midi_type=2)

    with pytest.raises(ValueError, match=r"MIDI format 2.*independent patterns"):
        audio._midi_duration_ms(str(midi_path))


def test_midi_to_mp3_rejects_format_2_before_synthesis(monkeypatch, tmp_path):
    midi_path = _write_midi(tmp_path / "patterns.mid", midi_type=2)
    output_path = _write_file(tmp_path / "preview.mp3", b"existing preview")
    monkeypatch.setattr(
        audio, "is_playback_available", lambda soundfont_name=None: (True, None)
    )

    def unexpected_render(*args):
        pytest.fail("Format 2 must be rejected before synthesis")

    monkeypatch.setattr(audio, "_render_midi_to_wav", unexpected_render)

    with pytest.raises(
        AudioRenderingError, match=r"MIDI format 2.*independent patterns"
    ) as raised:
        audio.midi_to_mp3(str(midi_path), output_path=str(output_path))

    assert isinstance(raised.value.__cause__, ValueError)
    assert output_path.read_bytes() == b"existing preview"
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "patterns.mid",
        "preview.mp3",
    ]


def test_midi_to_mp3_wraps_corrupt_midi_duration_error(monkeypatch, tmp_path):
    midi_path = _write_file(tmp_path / "loop.mid", b"not a MIDI file")
    output_path = tmp_path / "loop.mp3"

    monkeypatch.setattr(
        audio, "is_playback_available", lambda soundfont_name=None: (True, None)
    )

    with pytest.raises(AudioRenderingError) as raised:
        audio.midi_to_mp3(str(midi_path), output_path=str(output_path))

    assert raised.value.__cause__ is not None
    assert not output_path.exists()


def test_midi_to_mp3_reports_fluidsynth_process_failure(monkeypatch, tmp_path):
    midi_path = _write_midi(tmp_path / "loop.mid")
    _write_file(tmp_path / "custom.sf2")
    output_path = tmp_path / "loop.mp3"

    monkeypatch.setattr(audio, "SOUNDFONT_DIR", str(tmp_path))
    monkeypatch.setattr(
        audio, "is_playback_available", lambda soundfont_name=None: (True, None)
    )

    def fake_run(command, **kwargs):
        assert command[0] == "fluidsynth"
        Path(_fluidsynth_output_path(command)).write_bytes(b"apparently valid output")
        return subprocess.CompletedProcess(
            command,
            1,
            stdout="",
            stderr="FluidSynth: failed to load SoundFont\n",
        )

    monkeypatch.setattr(audio.subprocess, "run", fake_run)

    with pytest.raises(
        AudioRenderingError,
        match="FluidSynth exited with code 1: FluidSynth: failed to load SoundFont",
    ):
        audio.midi_to_mp3(
            str(midi_path),
            output_path=str(output_path),
            soundfont_name="custom.sf2",
        )

    assert not output_path.exists()


def test_midi_to_mp3_reports_fluidsynth_timeout(monkeypatch, tmp_path):
    midi_path = _write_midi(tmp_path / "loop.mid")
    _write_file(tmp_path / "custom.sf2")
    output_path = tmp_path / "loop.mp3"

    monkeypatch.setattr(audio, "SOUNDFONT_DIR", str(tmp_path))
    monkeypatch.setattr(
        audio, "is_playback_available", lambda soundfont_name=None: (True, None)
    )

    def fake_run(command, **kwargs):
        assert command[0] == "fluidsynth"
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(audio.subprocess, "run", fake_run)

    with pytest.raises(
        AudioRenderingError,
        match=f"FluidSynth timed out after {audio._FLUIDSYNTH_TIMEOUT_SECONDS} seconds",
    ) as raised:
        audio.midi_to_mp3(
            str(midi_path),
            output_path=str(output_path),
            soundfont_name="custom.sf2",
        )

    assert isinstance(raised.value.__cause__, subprocess.TimeoutExpired)
    assert not output_path.exists()


def test_midi_to_mp3_rejects_wav_without_audio_frames(monkeypatch, tmp_path):
    midi_path = _write_midi(tmp_path / "loop.mid")
    _write_file(tmp_path / "custom.sf2")
    output_path = tmp_path / "loop.mp3"

    monkeypatch.setattr(audio, "SOUNDFONT_DIR", str(tmp_path))
    monkeypatch.setattr(
        audio, "is_playback_available", lambda soundfont_name=None: (True, None)
    )

    def fake_run(command, **kwargs):
        assert command[0] == "fluidsynth"
        _write_wav(_fluidsynth_output_path(command), frames=b"")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(audio.subprocess, "run", fake_run)

    with pytest.raises(
        AudioRenderingError,
        match="FluidSynth produced a WAV file with no audio frames",
    ):
        audio.midi_to_mp3(
            str(midi_path),
            output_path=str(output_path),
            soundfont_name="custom.sf2",
        )

    assert not output_path.exists()


def test_midi_to_mp3_reports_soundfont_discovery_failure(monkeypatch, tmp_path):
    midi_path = _write_file(tmp_path / "loop.mid")

    def fail_discovery(soundfont_name=None):
        raise OSError("resource extraction failed")

    monkeypatch.setattr(
        audio,
        "is_playback_available",
        fail_discovery,
    )

    with pytest.raises(
        AudioRenderingError, match="OSError: resource extraction failed"
    ) as raised:
        audio.midi_to_mp3(str(midi_path))

    assert isinstance(raised.value.__cause__, OSError)
