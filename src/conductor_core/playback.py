"""Audio synthesis utilities for MIDI playback.

This module provides functionality to convert MIDI files to MP3 audio
using FluidSynth for synthesis and pydub for MP3 encoding.

Requires:
    - FluidSynth system library installed
    - FFmpeg installed (for MP3 encoding)
    - At least one SoundFont file in the soundfonts directory
"""

import atexit
import logging
import os
import shutil
import subprocess
import tempfile
import wave
from contextlib import ExitStack, suppress
from importlib import resources
from threading import Lock

from conductor_core.errors import AudioRenderingError

logger = logging.getLogger(__name__)

AudioSegment = None

# Optional filesystem override retained for applications that provide their own
# packaged SoundFont directory. The built-in resource directory is resolved
# lazily so importing Core also works from zipped and frozen distributions.
SOUNDFONT_DIR: str | None = None
_EXTRA_SOUNDFONT_DIRS: list[str] = []
_PACKAGED_SOUNDFONT_PATHS: dict[str, str] = {}
_PACKAGED_SOUNDFONT_STACK = ExitStack()
_PACKAGED_SOUNDFONT_LOCK = Lock()
_FLUIDSYNTH_TIMEOUT_SECONDS = 30
atexit.register(_PACKAGED_SOUNDFONT_STACK.close)
# Preferred SoundFont filenames searched in order.
DEFAULT_SOUNDFONT_CANDIDATES = [
    "FM-Piano1-20190916.sf2",
    "SalamanderGrandPiano.sf2",
    "salamander-grand-piano.sf2",
    "piano.sf2",
    "GeneralUser.sf2",
    "FluidR3_GM.sf2",
]


def _render_midi_to_wav(midi_path: str, wav_path: str, soundfont_path: str) -> None:
    """Render MIDI with FluidSynth and raise when the process reports failure."""
    command = [
        "fluidsynth",
        "-ni",
        "-T",
        "wav",
        "-O",
        "s16",
        "-F",
        wav_path,
        "-r",
        "44100",
        soundfont_path,
        midi_path,
    ]
    try:
        completed_process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=_FLUIDSYNTH_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise AudioRenderingError(
            f"FluidSynth timed out after {_FLUIDSYNTH_TIMEOUT_SECONDS} seconds"
        ) from exc

    if completed_process.returncode != 0:
        detail = (completed_process.stderr or completed_process.stdout or "").strip()
        message = f"FluidSynth exited with code {completed_process.returncode}"
        if detail:
            message = f"{message}: {detail}"
        raise AudioRenderingError(message)


def _validate_wav_has_audio(wav_path: str) -> None:
    """Raise when FluidSynth output is missing, malformed, or has no audio frames."""
    if not os.path.exists(wav_path):
        raise AudioRenderingError("FluidSynth did not produce a WAV file")

    try:
        with wave.open(wav_path, "rb") as wav_file:
            if wav_file.getnframes() == 0 or not wav_file.readframes(1):
                raise AudioRenderingError(
                    "FluidSynth produced a WAV file with no audio frames"
                )
    except (EOFError, wave.Error) as exc:
        detail = f": {exc}" if str(exc) else ""
        raise AudioRenderingError(
            f"FluidSynth produced an invalid WAV file{detail}"
        ) from exc


def _soundfont_search_dirs() -> list[str]:
    """Return SoundFont search directories in priority order."""
    search_dirs = list(_EXTRA_SOUNDFONT_DIRS)
    if SOUNDFONT_DIR is not None:
        search_dirs.append(SOUNDFONT_DIR)
    return search_dirs


def _packaged_soundfont_dir():
    """Return the built-in SoundFont resource directory lazily."""
    if SOUNDFONT_DIR is not None:
        return None
    return resources.files("conductor_core.resources").joinpath("soundfonts")


def _packaged_soundfont_files():
    """Return built-in SoundFont resources without assuming filesystem paths."""
    try:
        soundfont_dir = _packaged_soundfont_dir()
        if soundfont_dir is None:
            return []
        return [
            resource
            for resource in soundfont_dir.iterdir()
            if resource.is_file() and resource.name.lower().endswith(".sf2")
        ]
    except (FileNotFoundError, ModuleNotFoundError, NotADirectoryError):
        return []


def _materialize_packaged_soundfont(resource) -> str:
    """Materialize a packaged SoundFont and keep its path valid until exit."""
    with _PACKAGED_SOUNDFONT_LOCK:
        cached_path = _PACKAGED_SOUNDFONT_PATHS.get(resource.name)
        if cached_path is not None and os.path.exists(cached_path):
            return cached_path

        materialized_path = _PACKAGED_SOUNDFONT_STACK.enter_context(
            resources.as_file(resource)
        )
        resolved_path = os.fspath(materialized_path)
        _PACKAGED_SOUNDFONT_PATHS[resource.name] = resolved_path
        return resolved_path


def _soundfont_location_label() -> str:
    """Return a useful label for SoundFont discovery diagnostics."""
    if SOUNDFONT_DIR is not None:
        return SOUNDFONT_DIR
    return "conductor_core.resources/soundfonts"


def add_soundfont_search_dir(soundfont_dir: str | os.PathLike[str]) -> None:
    """Add an application-owned directory to the SoundFont search path."""
    resolved_dir = os.path.abspath(os.fspath(soundfont_dir))
    if resolved_dir not in _EXTRA_SOUNDFONT_DIRS:
        _EXTRA_SOUNDFONT_DIRS.append(resolved_dir)


def _find_soundfont_file(soundfont_name: str) -> str | None:
    """Find a SoundFont by filename across configured search directories."""
    requested_name = os.path.basename(soundfont_name)
    for soundfont_dir in _soundfont_search_dirs():
        candidate_path = os.path.join(soundfont_dir, requested_name)
        if os.path.exists(candidate_path):
            return candidate_path

    for resource in _packaged_soundfont_files():
        if resource.name == requested_name:
            return _materialize_packaged_soundfont(resource)
    return None


def list_soundfonts() -> list[str]:
    """List the available SoundFont filenames.

    Returns:
        list[str]: Sorted `.sf2` filenames in the soundfonts directory.
    """
    names = set()
    for soundfont_dir in _soundfont_search_dirs():
        if not os.path.exists(soundfont_dir):
            continue
        names.update(
            file for file in os.listdir(soundfont_dir) if file.lower().endswith(".sf2")
        )
    names.update(resource.name for resource in _packaged_soundfont_files())
    return sorted(names, key=str.lower)


def get_default_soundfont() -> str | None:
    """Resolve the default SoundFont path.

    Returns:
        str | None: Path to the preferred available SoundFont, or None.
    """
    available_soundfonts = list_soundfonts()
    if not available_soundfonts:
        return None

    available_lookup = {name.lower(): name for name in available_soundfonts}

    for soundfont_name in DEFAULT_SOUNDFONT_CANDIDATES:
        matched_name = available_lookup.get(soundfont_name.lower())
        if matched_name:
            return _find_soundfont_file(matched_name)

    fallback_soundfont = available_soundfonts[0]
    logger.info(f"Falling back to SoundFont: {fallback_soundfont}")
    return _find_soundfont_file(fallback_soundfont)


def resolve_soundfont(soundfont_name: str | None = None) -> str | None:
    """Resolve an explicit or default SoundFont path.

    Args:
        soundfont_name (str | None): Optional SoundFont filename or path.

    Returns:
        str | None: Path to a SoundFont file if found, None otherwise.
    """
    if soundfont_name:
        if os.path.exists(soundfont_name):
            return soundfont_name

        requested_path = _find_soundfont_file(soundfont_name)
        if requested_path:
            return requested_path

        return None

    return get_default_soundfont()


def find_soundfont(soundfont_name: str | None = None) -> str | None:
    """Backwards-compatible wrapper for SoundFont resolution.

    Args:
        soundfont_name (str | None): Optional SoundFont filename or path.

    Returns:
        str | None: Path to a SoundFont file if found, None otherwise.
    """
    return resolve_soundfont(soundfont_name)


def is_fluidsynth_available() -> bool:
    """Check if FluidSynth is installed and available.

    Returns:
        bool: True if FluidSynth is available, False otherwise.
    """
    return shutil.which("fluidsynth") is not None


def is_ffmpeg_available() -> bool:
    """Check if FFmpeg is installed and available.

    Returns:
        bool: True if FFmpeg is available, False otherwise.
    """
    return shutil.which("ffmpeg") is not None


def is_playback_available(soundfont_name: str | None = None) -> tuple[bool, str | None]:
    """Check if audio playback is available.

    Verifies that all required components are present:
    - FluidSynth installed
    - FFmpeg installed
    - SoundFont file exists

    Returns:
        tuple: (is_available, error_message)
            - is_available (bool): True if playback is fully available
            - error_message (str | None): Description of what's missing, or None if all good
    """
    issues = []

    if not is_fluidsynth_available():
        issues.append("FluidSynth is not installed or not in PATH")

    if not is_ffmpeg_available():
        issues.append("FFmpeg is not installed or not in PATH")

    resolved_soundfont = find_soundfont(soundfont_name)
    if resolved_soundfont is None:
        soundfont_location = _soundfont_location_label()
        if soundfont_name:
            issues.append(
                f"Requested SoundFont '{os.path.basename(soundfont_name)}' was not found in "
                f"'{soundfont_location}'."
            )
        else:
            issues.append(
                f"No SoundFont file found in '{soundfont_location}'. "
                "Add a .sf2 file to enable audio playback."
            )

    if issues:
        return False, "; ".join(issues)

    return True, None


def midi_to_mp3(
    midi_path: str,
    output_path: str | None = None,
    soundfont_name: str | None = None,
) -> str:
    """Convert a MIDI file to MP3 audio using FluidSynth.

    Args:
        midi_path (str): Path to the input MIDI file.
        output_path (str, optional): Path for the output MP3 file.
            If not provided, uses the same name as the MIDI file with .mp3 extension.
        soundfont_name (str, optional): SoundFont filename or path to use.

    Returns:
        str: Path to the generated MP3 file.

    Raises:
        AudioRenderingError: If playback dependencies, discovery, synthesis, or encoding fail.
        FileNotFoundError: If the MIDI file doesn't exist.
    """
    if not os.path.exists(midi_path):
        raise FileNotFoundError(f"MIDI file not found: {midi_path}")

    try:
        # Check if playback is available
        available, error = is_playback_available(soundfont_name)
        if not available:
            logger.warning(f"Audio playback not available: {error}")
            raise AudioRenderingError(error or "Audio playback is not available")

        # Determine output path
        if output_path is None:
            base_name = os.path.splitext(midi_path)[0]
            output_path = f"{base_name}.mp3"

        output_directory = os.path.dirname(os.path.abspath(output_path))

        # Find the soundfont
        soundfont_path = find_soundfont(soundfont_name)
        if soundfont_path is None:
            error = "No SoundFont file available"
            logger.error(error)
            raise AudioRenderingError(error)

        # Create a temporary WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
            temp_wav_path = temp_wav.name

        with tempfile.NamedTemporaryFile(
            prefix=f".{os.path.basename(output_path)}.",
            suffix=".tmp",
            dir=output_directory,
            delete=False,
        ) as temp_mp3:
            temp_mp3_path = temp_mp3.name

        # Use FluidSynth to render MIDI to WAV
        logger.info(f"Rendering MIDI to WAV using SoundFont: {soundfont_path}")
        _render_midi_to_wav(midi_path, temp_wav_path, soundfont_path)
        _validate_wav_has_audio(temp_wav_path)

        global AudioSegment

        if AudioSegment is None:
            from pydub import AudioSegment as _AudioSegment

            AudioSegment = _AudioSegment

        # Convert WAV to MP3 using pydub
        logger.info(f"Converting WAV to MP3: {output_path}")
        audio = AudioSegment.from_wav(temp_wav_path)
        audio.export(temp_mp3_path, format="mp3", bitrate="192k")
        os.replace(temp_mp3_path, output_path)

        logger.info(f"Successfully created MP3: {output_path}")
        return output_path

    except AudioRenderingError as exc:
        logger.error(f"Failed to convert MIDI to MP3: {exc}")
        raise
    except Exception as exc:
        detail = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
        error = AudioRenderingError(detail)
        logger.error(f"Failed to convert MIDI to MP3: {error}")
        raise error from exc

    finally:
        # Clean up temporary WAV file
        if "temp_wav_path" in locals() and os.path.exists(temp_wav_path):
            with suppress(OSError):
                os.remove(temp_wav_path)
        if "temp_mp3_path" in locals() and os.path.exists(temp_mp3_path):
            with suppress(OSError):
                os.remove(temp_mp3_path)


def get_playback_status_message(soundfont_name: str | None = None) -> str:
    """Get a user-friendly status message about playback availability.

    Returns:
        str: A message describing the playback status and any setup required.
    """
    available, _error = is_playback_available(soundfont_name)

    if available:
        soundfont = find_soundfont(soundfont_name)
        sf_name = os.path.basename(soundfont) if soundfont else "Unknown"
        return f"Audio playback ready (using {sf_name})"

    dependency_instructions = []

    if not is_fluidsynth_available():
        dependency_instructions.append(
            "  - Install FluidSynth: https://github.com/FluidSynth/fluidsynth/releases"
        )

    if not is_ffmpeg_available():
        dependency_instructions.append(
            "  - Install FFmpeg: https://ffmpeg.org/download.html"
        )

    if dependency_instructions:
        return "\n".join(
            [
                "Audio playback is not available. Setup required:",
                *dependency_instructions,
            ]
        )

    instructions = ["Audio playback is not available. Setup required:"]

    resolved_soundfont = find_soundfont(soundfont_name)
    if resolved_soundfont is None:
        if soundfont_name:
            instructions.append(
                f"  - Pass an existing SoundFont path or package '{os.path.basename(soundfont_name)}' with Core"
            )
        else:
            instructions.append("  - Package a `.sf2` SoundFont file with Core")

    return "\n".join(instructions)
