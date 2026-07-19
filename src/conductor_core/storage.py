"""Generation history management for Conductor.

This module provides functionality to save, load, and manage
the history of MIDI loop generations. Each generation is stored
with its MIDI file, audio file (if available), and metadata.

Storage structure:
    ~/.conductor/core/generations/
        gen_<timestamp>/
            loop.mid        # Generated MIDI file
            loop.mp3        # Rendered audio (if available)
            messages.json   # Provider message history (if available)
            metadata.json   # Generation parameters and info
"""

import json
import logging
import os
import shutil
import stat
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from conductor_core.paths import resolve_default_artifact_root

logger = logging.getLogger(__name__)

# Resolved default used by module-level compatibility helpers. Resolution is
# side-effect free; directories are still created only by workspace writes.
GENERATIONS_DIR = str(resolve_default_artifact_root())

# Default maximum number of generations retained by module-level helpers.
MAX_GENERATIONS = 20

_DEFAULT_MAX_GENERATIONS = object()


class FilesystemArtifactStore:
    """Filesystem generation store bound to a caller-provided artifact root."""

    def __init__(
        self,
        artifact_root: str | Path | None = None,
        max_generations: int | None = MAX_GENERATIONS,
    ):
        self.artifact_root = _resolve_artifact_root(artifact_root)
        self.max_generations = _validate_max_generations(max_generations)

    def create_generation_workspace(self) -> "GenerationWorkspace":
        return _create_generation_workspace(self.artifact_root)

    def finalize_generation(self, *args, **kwargs) -> "GenerationMetadata":
        return _finalize_generation(
            self.artifact_root,
            *args,
            max_generations=self.max_generations,
            **kwargs,
        )

    def cleanup_generation_workspace(self, workspace: "GenerationWorkspace") -> bool:
        return _cleanup_generation_workspace(self.artifact_root, workspace)

    def update_generation_audio(
        self,
        gen_id: str,
        audio_path: Optional[str],
        soundfont: Optional[str] = None,
    ) -> Optional["GenerationMetadata"]:
        return _update_generation_audio(self.artifact_root, gen_id, audio_path, soundfont=soundfont)

    def load_history(self) -> list["GenerationMetadata"]:
        return _load_history(self.artifact_root)

    def get_generation(self, gen_id: str) -> Optional["GenerationMetadata"]:
        return _get_generation(self.artifact_root, gen_id)

    def delete_generation(self, gen_id: str) -> bool:
        return _delete_generation(self.artifact_root, gen_id)


class GenerationMetadata(BaseModel):
    """Metadata for a single generation.

    Attributes:
        id: Unique identifier (timestamp-based).
        timestamp: When the generation was created.
        prompt: User's description/prompt.
        key: Musical key (C, D, etc.).
        scale: Major or minor.
        model: Model name used for generation.
        provider: API provider (OpenAI, Anthropic, Google, Ollama).
        temperature: Temperature setting used.
        cost: API cost if available.
        midi_path: Path to the MIDI file.
        audio_path: Path to the audio file (None if synthesis failed).
        messages_path: Path to the provider message history file.
        soundfont: SoundFont filename used to render the audio file.
    """

    id: str
    timestamp: datetime
    prompt: str
    key: str
    scale: str
    model: str
    provider: str
    temperature: float
    cost: Optional[float] = None
    midi_path: str
    audio_path: Optional[str] = None
    messages_path: Optional[str] = None
    soundfont: Optional[str] = None


class GenerationWorkspace(BaseModel):
    """Canonical artifact paths for a generation in progress."""

    id: str
    directory: str
    midi_path: str
    audio_path: str
    messages_path: str
    metadata_path: str


def _resolve_artifact_root(artifact_root: str | Path | None = None) -> str:
    selected_root = artifact_root if artifact_root is not None else GENERATIONS_DIR
    return str(selected_root)


def _ensure_generations_dir(artifact_root: str | Path | None = None) -> None:
    """Create the generations directory if it doesn't exist."""
    root = _resolve_artifact_root(artifact_root)
    if not os.path.exists(root):
        os.makedirs(root)
        logger.info(f"Created generations directory: {root}")


def _generate_id() -> str:
    """Generate a unique ID for a generation based on timestamp.

    Returns:
        str: A unique identifier string.
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def _get_generation_dir(gen_id: str, artifact_root: str | Path | None = None) -> str:
    """Get the directory path for a specific generation.

    Args:
        gen_id: The generation ID.

    Returns:
        str: Path to the generation's directory.
    """
    _validate_generation_id(gen_id)

    root = Path(_resolve_artifact_root(artifact_root)).resolve()
    generation_name = f"gen_{gen_id}"
    candidate = (root / generation_name).resolve()
    if candidate.parent != root or candidate.name != generation_name:
        raise ValueError("generation path must be a direct child of the artifact root")

    return str(candidate)


def _validate_generation_id(gen_id: str) -> None:
    """Reject generation IDs that can be interpreted as filesystem paths."""
    if (
        not isinstance(gen_id, str)
        or not gen_id
        or gen_id in {".", ".."}
        or "/" in gen_id
        or "\\" in gen_id
        or "\x00" in gen_id
    ):
        raise ValueError("generation ID must be a non-empty path component")


def _build_workspace(gen_id: str, artifact_root: str | Path | None = None) -> GenerationWorkspace:
    """Build the canonical path set for a generation ID."""
    gen_dir = _get_generation_dir(gen_id, artifact_root)
    return GenerationWorkspace(
        id=gen_id,
        directory=gen_dir,
        midi_path=os.path.join(gen_dir, "loop.mid"),
        audio_path=os.path.join(gen_dir, "loop.mp3"),
        messages_path=os.path.join(gen_dir, "messages.json"),
        metadata_path=os.path.join(gen_dir, "metadata.json"),
    )


def _validate_workspace(
    artifact_root: str | Path, workspace: GenerationWorkspace
) -> GenerationWorkspace:
    """Require every workspace path to match the store's canonical path set."""
    canonical = _build_workspace(workspace.id, artifact_root)
    for field_name in (
        "directory",
        "midi_path",
        "audio_path",
        "messages_path",
        "metadata_path",
    ):
        supplied_path = Path(getattr(workspace, field_name)).resolve()
        canonical_path = Path(getattr(canonical, field_name)).resolve()
        if supplied_path != canonical_path:
            raise ValueError(
                f"workspace {field_name} does not match generation {workspace.id!r} "
                "under this artifact root"
            )

    return canonical


def _validate_artifact_file(path: str, *, required: bool) -> str | None:
    """Return a regular, single-link artifact path without following links."""
    try:
        file_stat = os.lstat(path)
    except FileNotFoundError:
        if required:
            raise
        return None

    reparse_point = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    file_attributes = getattr(file_stat, "st_file_attributes", 0)
    if stat.S_ISLNK(file_stat.st_mode) or file_attributes & reparse_point:
        raise ValueError(f"artifact path must not be a symbolic link or reparse point: {path}")
    if not stat.S_ISREG(file_stat.st_mode):
        raise ValueError(f"artifact path must be a regular file: {path}")
    if file_stat.st_nlink != 1:
        raise ValueError(f"artifact path must not be hard linked: {path}")

    return path


def _write_metadata_file(path: str, metadata: GenerationMetadata) -> None:
    """Atomically replace metadata without following an existing destination link."""
    directory = os.path.dirname(path)
    file_descriptor, temporary_path = tempfile.mkstemp(
        prefix=".metadata-", suffix=".tmp", dir=directory
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as metadata_file:
            metadata_file.write(metadata.model_dump_json(indent=2))
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def _copy_artifact_file(source: str, destination: str) -> None:
    """Copy through a new file and atomically replace the destination entry."""
    directory = os.path.dirname(destination)
    file_descriptor, temporary_path = tempfile.mkstemp(
        prefix=".artifact-", suffix=".tmp", dir=directory
    )
    os.close(file_descriptor)
    try:
        shutil.copy2(source, temporary_path)
        os.replace(temporary_path, destination)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def _load_generation_metadata(artifact_root: str | Path, gen_id: str) -> GenerationMetadata:
    """Load metadata and bind all artifact paths to its validated directory."""
    workspace = _build_workspace(gen_id, artifact_root)
    metadata_path = _validate_artifact_file(workspace.metadata_path, required=True)
    assert metadata_path is not None

    with open(metadata_path, "r", encoding="utf-8") as metadata_file:
        metadata = GenerationMetadata(**json.load(metadata_file))

    if metadata.id != gen_id:
        raise ValueError(
            f"metadata generation ID {metadata.id!r} does not match directory ID {gen_id!r}"
        )

    _validate_artifact_file(workspace.midi_path, required=False)
    audio_path = _validate_artifact_file(workspace.audio_path, required=False)
    messages_path = _validate_artifact_file(workspace.messages_path, required=False)
    metadata.midi_path = workspace.midi_path
    metadata.audio_path = audio_path
    metadata.messages_path = messages_path
    if audio_path is None:
        metadata.soundfont = None
    return metadata


def get_provider_for_model(model: str, model_info: dict) -> str:
    """Determine the provider for a given model name.

    Args:
        model: The model name.
        model_info: The model information dictionary from model_list.json.

    Returns:
        str: The provider name.
    """
    for provider, models in model_info.get("models", {}).items():
        if model in models:
            return provider
    return "Ollama"  # Default to Ollama for local models


def create_generation_workspace() -> GenerationWorkspace:
    """Create a generation directory and return its canonical artifact paths.

    Returns:
        GenerationWorkspace: The newly allocated generation workspace.

    Raises:
        RuntimeError: If a unique generation ID cannot be allocated.
    """
    return _create_generation_workspace(GENERATIONS_DIR)


def _create_generation_workspace(artifact_root: str | Path) -> GenerationWorkspace:
    """Create a generation workspace under an explicit artifact root."""
    _ensure_generations_dir(artifact_root)

    for _ in range(100):
        gen_id = _generate_id()
        workspace = _build_workspace(gen_id, artifact_root)
        try:
            os.makedirs(workspace.directory)
            logger.info(f"Created generation workspace: {workspace.directory}")
            return workspace
        except FileExistsError:
            continue

    raise RuntimeError("Unable to allocate a unique generation workspace")


def finalize_generation(
    workspace: GenerationWorkspace,
    prompt: str,
    key: str,
    scale: str,
    model: str,
    provider: str,
    temperature: float,
    cost: Optional[float] = None,
    soundfont: Optional[str] = None,
    audio_render_succeeded: bool | None = None,
) -> GenerationMetadata:
    """Finalize a generation after its artifacts have been written.

    Args:
        workspace: Generation workspace with canonical artifact paths.
        prompt: User's description/prompt.
        key: Musical key.
        scale: Major or minor.
        model: Model name used.
        provider: API provider.
        temperature: Temperature setting.
        cost: API cost (optional).
        soundfont: SoundFont filename used to render the audio file (optional).
        audio_render_succeeded: Explicit audio render outcome. When False, any
            partial audio artifact is removed. When omitted, audio is detected
            for backwards compatibility.

    Returns:
        GenerationMetadata: The persisted metadata.
    """
    return _finalize_generation(
        GENERATIONS_DIR,
        workspace=workspace,
        prompt=prompt,
        key=key,
        scale=scale,
        model=model,
        provider=provider,
        temperature=temperature,
        cost=cost,
        soundfont=soundfont,
        audio_render_succeeded=audio_render_succeeded,
        max_generations=MAX_GENERATIONS,
    )


def _finalize_generation(
    artifact_root: str | Path,
    workspace: GenerationWorkspace,
    prompt: str,
    key: str,
    scale: str,
    model: str,
    provider: str,
    temperature: float,
    cost: Optional[float] = None,
    soundfont: Optional[str] = None,
    audio_render_succeeded: bool | None = None,
    max_generations: int | None = MAX_GENERATIONS,
) -> GenerationMetadata:
    """Finalize a generation using an explicit artifact root."""
    max_generations = _validate_max_generations(max_generations)
    workspace = _validate_workspace(artifact_root, workspace)

	try:
		_validate_artifact_file(workspace.midi_path, required=True)
	except FileNotFoundError as exc:
		raise FileNotFoundError(
			f"Missing MIDI file for generation: {workspace.midi_path}"
		) from exc

	if audio_render_succeeded is False:
		try:
			os.remove(workspace.audio_path)
		except FileNotFoundError:
			pass
		except OSError as exc:
			logger.warning(f"Failed to remove partial audio artifact: {exc}")

		audio_path = None
	else:
		audio_path = _validate_artifact_file(
			workspace.audio_path,
			required=False,
		)

	messages_path = _validate_artifact_file(
		workspace.messages_path,
		required=False,
	)

    metadata = GenerationMetadata(
        id=workspace.id,
        timestamp=datetime.now(),
        prompt=prompt,
        key=key,
        scale=scale,
        model=model,
        provider=provider,
        temperature=temperature,
        cost=cost,
        midi_path=workspace.midi_path,
        audio_path=audio_path,
        messages_path=messages_path,
        soundfont=soundfont if audio_path else None,
    )

    _write_metadata_file(workspace.metadata_path, metadata)

    logger.info(f"Finalized generation {workspace.id} in history")
    _enforce_limit(artifact_root, max_generations=max_generations)

    return metadata


def cleanup_generation_workspace(workspace: GenerationWorkspace) -> bool:
    """Remove an unfinalized generation workspace.

    Finalized generation directories are left intact so cleanup can be called
    from broad exception handlers without deleting visible history entries.

    Args:
        workspace: Generation workspace to remove.

    Returns:
        bool: True if the workspace was removed, False otherwise.
    """
    return _cleanup_generation_workspace(GENERATIONS_DIR, workspace)


def _cleanup_generation_workspace(
    artifact_root: str | Path, workspace: GenerationWorkspace
) -> bool:
    """Remove an unfinalized generation workspace."""
    workspace = _validate_workspace(artifact_root, workspace)

    if os.path.exists(workspace.metadata_path):
        return False

    if not os.path.isdir(workspace.directory):
        return False

    shutil.rmtree(workspace.directory)
    logger.info(f"Removed unfinalized generation workspace: {workspace.directory}")
    return True


def update_generation_audio(
    gen_id: str,
    audio_path: Optional[str],
    soundfont: Optional[str] = None,
) -> Optional[GenerationMetadata]:
    """Update the stored audio path and SoundFont for an existing generation.

    Args:
        gen_id: The generation ID.
        audio_path: Path to the rendered audio file.
        soundfont: SoundFont filename used to render the audio file.

    Returns:
        GenerationMetadata or None if the generation does not exist.
    """
    return _update_generation_audio(GENERATIONS_DIR, gen_id, audio_path, soundfont=soundfont)


def _update_generation_audio(
    artifact_root: str | Path,
    gen_id: str,
    audio_path: Optional[str],
    soundfont: Optional[str] = None,
) -> Optional[GenerationMetadata]:
    """Update stored audio metadata under an explicit artifact root."""
    metadata = _get_generation(artifact_root, gen_id)
    if metadata is None:
        return None

    gen_dir = _get_generation_dir(gen_id, artifact_root)
    metadata_path = os.path.join(gen_dir, "metadata.json")

    dest_audio_path = metadata.audio_path
    audio_updated = False
    if audio_path and os.path.exists(audio_path):
        dest_audio_path = os.path.join(gen_dir, "loop.mp3")
        if os.path.abspath(audio_path) != os.path.abspath(dest_audio_path):
            _copy_artifact_file(audio_path, dest_audio_path)
        else:
            dest_audio_path = audio_path
        audio_updated = True

    metadata.audio_path = dest_audio_path
    if audio_updated:
        metadata.soundfont = soundfont

    _write_metadata_file(metadata_path, metadata)

    return metadata


def load_history() -> list[GenerationMetadata]:
    """Load all generations from history.

    Returns:
        list: List of GenerationMetadata objects, sorted by timestamp (newest first).
    """
    return _load_history(GENERATIONS_DIR)


def _load_history(artifact_root: str | Path) -> list[GenerationMetadata]:
    """Load generation history from an explicit artifact root."""
    root = _resolve_artifact_root(artifact_root)
    if not os.path.isdir(root):
        return []

    generations = []

    for item in os.listdir(root):
        if not item.startswith("gen_"):
            continue

        try:
            gen_dir = _get_generation_dir(item.removeprefix("gen_"), root)
        except ValueError as exc:
            logger.warning(f"Skipping unsafe generation path {item}: {exc}")
            continue
        if not os.path.isdir(gen_dir):
            continue

        try:
            metadata = _load_generation_metadata(root, item.removeprefix("gen_"))
            generations.append(metadata)
        except FileNotFoundError:
            logger.warning(f"Missing metadata for generation: {item}")
            continue
        except Exception as e:
            logger.warning(f"Failed to load generation {item}: {e}")
            continue

    # Sort by timestamp, newest first
    generations.sort(key=lambda g: g.timestamp, reverse=True)

    return generations


def get_generation(gen_id: str) -> Optional[GenerationMetadata]:
    """Get a specific generation by ID.

    Args:
        gen_id: The generation ID.

    Returns:
        GenerationMetadata or None if not found.
    """
    return _get_generation(GENERATIONS_DIR, gen_id)


def _get_generation(artifact_root: str | Path, gen_id: str) -> Optional[GenerationMetadata]:
    """Get a specific generation from an explicit artifact root."""
    _validate_generation_id(gen_id)
    try:
        return _load_generation_metadata(artifact_root, gen_id)
    except FileNotFoundError:
        return None
    except Exception as e:
        logger.error(f"Failed to load generation {gen_id}: {e}")
        return None


def delete_generation(gen_id: str) -> bool:
    """Delete a generation from history.

    Args:
        gen_id: The generation ID to delete.

    Returns:
        bool: True if deleted successfully, False otherwise.
    """
    return _delete_generation(GENERATIONS_DIR, gen_id)


def _delete_generation(artifact_root: str | Path, gen_id: str) -> bool:
    """Delete a generation from an explicit artifact root."""
    gen_dir = _get_generation_dir(gen_id, artifact_root)

    if not os.path.exists(gen_dir):
        logger.warning(f"Generation not found: {gen_id}")
        return False

    try:
        shutil.rmtree(gen_dir)
        logger.info(f"Deleted generation: {gen_id}")
        return True
    except Exception as e:
        logger.error(f"Failed to delete generation {gen_id}: {e}")
        return False


def _enforce_limit(
    artifact_root: str | Path | None = None,
    max_generations: int | None | object = _DEFAULT_MAX_GENERATIONS,
) -> None:
    """Delete oldest generations if over the limit."""
    if max_generations is _DEFAULT_MAX_GENERATIONS:
        max_generations = MAX_GENERATIONS
    max_generations = _validate_max_generations(max_generations)
    if max_generations is None:
        return

    root = _resolve_artifact_root(artifact_root)
    generations = _load_history(root)

    if len(generations) <= max_generations:
        return

    # Delete oldest generations (they're at the end since list is sorted newest first)
    generations_to_delete = generations[max_generations:]

    for gen in generations_to_delete:
        logger.info(f"Removing old generation {gen.id} to enforce limit")
        _delete_generation(root, gen.id)


def _validate_max_generations(max_generations: int | None) -> int | None:
    """Validate the maximum number of retained generations."""
    if max_generations is not None and (
        isinstance(max_generations, bool)
        or not isinstance(max_generations, int)
        or max_generations <= 0
    ):
        raise ValueError("max_generations must be None or a positive integer")

    return max_generations


def get_history_count() -> int:
    """Get the number of generations in history.

    Returns:
        int: Number of saved generations.
    """
    return len(load_history())


def clear_history() -> int:
    """Clear all generations from history.

    Returns:
        int: Number of generations deleted.
    """
    generations = load_history()
    count = 0

    for gen in generations:
        if delete_generation(gen.id):
            count += 1

    return count
